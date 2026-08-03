"""
Two-specialist proofreading on top of SOTA -- fix splits and merges SEPARATELY so we
escape the single-model split<->merge tradeoff (one model = one operating point on the
curve; every backbone/aux trick we tried just slid along it).

  Specialist A -- SPLIT-FIXER (learned agglomeration):
    Start from a conservative OVER-segmentation of SOTA's affinities (merge-free by
    construction). Build the region-adjacency graph; for each adjacent fragment pair,
    featurize the boundary (mean short-range affinity = attraction, mean long-range
    affinity = repulsion, contact area, sizes). Train a merge classifier (GT: do the
    two fragments belong to the same neuron?) and agglomerate by its score. Because it
    ONLY joins fragments, it can fix splits without ever creating a split.

  Specialist B -- MERGE-FIXER (split/error-correction, TODO next):
    On the agglomerated result, detect segments that span >=2 neurons and cut them.

This module = Specialist A. It consumes affinities we already predict (sota_plus) and
is scored with the same merge-aware ERL / VOI as everything else, so it's a clean
head-to-head vs plain MWS agglomeration.
"""
import numpy as np
from collections import defaultdict


def _oversegment(aff, short_offs, thr):
    """Robust over-segmentation via watershed: seeds = connected components of the
    high-attraction interior (mean short affinity above a data-adaptive threshold), then
    flood the boundary height (1 - attraction). Scale-robust -> many fragments regardless
    of the affinity distribution, essentially merge-free."""
    from scipy.ndimage import label
    from skimage.segmentation import watershed
    ns = len(short_offs)
    a = aff[:ns].mean(0)                                  # attraction: high inside objects
    seed_thr = max(thr, float(np.quantile(a, 0.6)))       # adaptive: interior cores
    seeds, n = label(a > seed_thr)
    if n == 0:
        seeds, n = label(a > float(np.quantile(a, 0.8)))
    if n == 0:
        return np.zeros(a.shape, np.int64)
    return watershed(1.0 - a, seeds)                      # flood boundaries from the cores


def _relabel(lab):
    _, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(lab.shape)


def _frag_lsd(frags, lsd):
    """Per-fragment mean LSD descriptor (10-D). Shape signature of each fragment."""
    nf = int(frags.max()) + 1
    C = lsd.shape[0]
    ff = frags.ravel(); out = np.zeros((nf, C), np.float32)
    cnt = np.bincount(ff, minlength=nf).astype(np.float32) + 1e-6
    for c in range(C):
        out[:, c] = np.bincount(ff, weights=lsd[c].ravel(), minlength=nf) / cnt
    return out


def _frag_gt(frags, seg):
    """Majority GT neuron id for each fragment."""
    nf = frags.max() + 1
    out = np.zeros(nf, np.int64)
    ff = frags.ravel(); ss = seg.ravel()
    for f in range(nf):
        vals = ss[ff == f]
        if len(vals):
            out[f] = np.bincount(vals).argmax()
    return out


def _rag(frags, aff, short_offs, long_offs, lsd=None):
    """Per adjacent fragment-pair edge features: mean short aff (attraction), mean long
    aff (repulsion), contact count, sizes, and (if lsd given) LSD shape agreement across
    the pair -- the signal MWS lacks: do the two fragments' shapes CONTINUE?"""
    ns = len(short_offs); Z, H, W = frags.shape
    offs = list(short_offs) + list(long_offs)
    NF = int(frags.max()) + 1
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(H), np.arange(W), indexing="ij")
    keys, avs, shorts = [], [], []
    for k, (dz, dy, dx) in enumerate(offs):
        m = (zz + dz < Z) & (yy + dy < H) & (xx + dx < W)
        f1 = frags[zz[m], yy[m], xx[m]]; f2 = frags[zz[m] + dz, yy[m] + dy, xx[m] + dx]
        a = aff[k][m]; diff = f1 != f2
        fa = np.minimum(f1[diff], f2[diff]); fb = np.maximum(f1[diff], f2[diff])
        keys.append(fa.astype(np.int64) * NF + fb); avs.append(a[diff])
        shorts.append(np.full(int(diff.sum()), k < ns, bool))
    if not keys or sum(len(x) for x in keys) == 0:
        return [], np.zeros((0, 4), np.float32)
    key = np.concatenate(keys); av = np.concatenate(avs); sm = np.concatenate(shorts)
    uniq, inv = np.unique(key, return_inverse=True); P = len(uniq)   # compact pair index
    ss = np.zeros(P); ns_e = np.zeros(P); sl = np.zeros(P); nl_e = np.zeros(P)
    np.add.at(ss, inv[sm], av[sm]); np.add.at(ns_e, inv[sm], 1.0)
    np.add.at(sl, inv[~sm], av[~sm]); np.add.at(nl_e, inv[~sm], 1.0)
    sizes = np.bincount(frags.ravel(), minlength=NF)
    a = (uniq // NF).astype(np.int64); b = (uniq % NF).astype(np.int64)
    keep = ns_e > 0                                       # need a short-range contact
    mean_short = ss[keep] / ns_e[keep]
    mean_long = np.where(nl_e[keep] > 0, sl[keep] / np.maximum(nl_e[keep], 1), 0.5)
    cols = [mean_short, mean_long, np.log1p(ns_e[keep]),
            np.log1p(np.minimum(sizes[a[keep]], sizes[b[keep]]))]
    if lsd is not None:                                   # LSD shape agreement (the MWS-lacking signal)
        fl = _frag_lsd(frags, lsd); ak, bk = a[keep], b[keep]
        dist = np.linalg.norm(fl[ak] - fl[bk], axis=1)    # shape descriptor distance (low = continue)
        cos = (fl[ak] * fl[bk]).sum(1) / (np.linalg.norm(fl[ak], axis=1) *
                                          np.linalg.norm(fl[bk], axis=1) + 1e-6)
        cols += [dist, cos]
    feats = np.stack(cols, 1).astype(np.float32)
    pairs = list(zip(a[keep].tolist(), b[keep].tolist()))
    return pairs, feats


def _gaec(frags, pairs, weights):
    """Greedy Additive Edge Contraction (multicut heuristic): edges carry SIGNED weights
    (positive = merge-beneficial, negative = repel). Repeatedly contract the highest
    positive edge, SUMMING parallel weights on contraction, until no positive edge
    remains. Parameter-free (like MWS) but optimizes the global multicut objective MWS
    only greedily approximates -> the genuine upgrade from MWS."""
    import heapq
    nf = int(frags.max()) + 1
    parent = list(range(nf))

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r
    adj = defaultdict(lambda: defaultdict(float))
    for (a, b), w in zip(pairs, weights):
        if a != b:
            adj[a][b] += w; adj[b][a] += w
    heap = [(-adj[a][b], a, b) for a in adj for b in adj[a] if a < b]
    heapq.heapify(heap)
    while heap:
        nw, a, b = heapq.heappop(heap)
        w = -nw
        if w <= 0:
            break                                        # no merge-beneficial edge left
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        cur = adj[ra].get(rb)
        if cur is None or abs(cur - w) > 1e-6:
            continue                                     # stale heap entry (a fresher one exists)
        parent[rb] = ra                                  # contract rb -> ra
        adj[ra].pop(rb, None); adj[rb].pop(ra, None)
        for nb, wt in list(adj[rb].items()):
            adj[nb].pop(rb, None)
            if nb == ra:
                continue
            adj[ra][nb] += wt; adj[nb][ra] += wt
            if adj[ra][nb] > 0:
                heapq.heappush(heap, (-adj[ra][nb], min(ra, nb), max(ra, nb)))
        adj[rb].clear()
    return _relabel(np.array([find(i) for i in range(nf)])[frags])


def _agglomerate(frags, pairs, prob, thr):
    """Union fragment pairs with merge prob > thr (descending), producing final labels."""
    nf = frags.max() + 1
    parent = list(range(nf))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for (a, b), p in sorted(zip(pairs, prob), key=lambda t: -t[1]):
        if p <= thr:
            break
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    remap = np.array([find(i) for i in range(nf)])
    return _relabel(remap[frags])


def run(train_affs, train_segs, eval_affs, eval_segs, ctx, thr_over=0.9,
        train_lsds=None, eval_lsds=None, tA=0.5, tB=0.5):
    """Two specialists on the fragment RAG, composed on top of SOTA:
      A (split-fixer)  = attraction/shape classifier -> merge over-segmented fragments.
      B (merge-fixer)  = repulsion/shape classifier -> VETO merges across true boundaries.
    Final merge iff A says merge AND B does not say boundary. Scored vs plain MWS."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        return {"error": f"sklearn unavailable: {e}"}
    SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT); LONG = OFFS[ns:]
    mws = ctx["mutex_watershed"]; seg_metrics = ctx["seg_metrics"]; erl = ctx["erl_proxy"]
    has_lsd = train_lsds is not None
    # feature columns: 0 mean_short,1 mean_long,2 log_contact,3 log_min_size,(4 lsd_dist,5 lsd_cos)
    A_cols = [0, 2, 5] if has_lsd else [0, 2]              # attraction + shape-continuity
    B_cols = [1, 3, 4] if has_lsd else [1, 3]              # repulsion + shape-discontinuity

    X, ysame = [], []
    for i, (aff, seg) in enumerate(zip(train_affs, train_segs)):
        frags = _relabel(_oversegment(aff, SHORT, thr_over))
        pairs, feats = _rag(frags, aff, SHORT, LONG, train_lsds[i] if has_lsd else None)
        if not len(pairs):
            continue
        fg = _frag_gt(frags, seg)
        lbl = np.array([1 if fg[a] == fg[b] and fg[a] != 0 else 0 for a, b in pairs])
        X.append(feats); ysame.append(lbl)
    X = np.concatenate(X); ysame = np.concatenate(ysame)
    if len(np.unique(ysame)) < 2:
        return {"error": "degenerate merge labels"}
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
    clfA = LogisticRegression(class_weight="balanced", max_iter=300).fit(Xs[:, A_cols], ysame)
    clfB = LogisticRegression(class_weight="balanced", max_iter=300).fit(Xs[:, B_cols], 1 - ysame)

    def score_lab(labs, segs):
        v = [seg_metrics(l, s) for l, s in zip(labs, segs)]
        return {"VOI": round(float(np.mean([x[0] for x in v])), 4),
                "adapted_rand_error": round(float(np.mean([x[1] for x in v])), 4),
                "ERL": round(float(np.mean([erl(l, s) for l, s in zip(labs, segs)])), 4)}

    lab_A, lab_AB, lab_MC, lab_M, nfrag = [], [], [], [], []
    for i, (aff, seg) in enumerate(zip(eval_affs, eval_segs)):
        frags = _relabel(_oversegment(aff, SHORT, thr_over)); nfrag.append(int(frags.max() + 1))
        pairs, feats = _rag(frags, aff, SHORT, LONG, eval_lsds[i] if has_lsd else None)
        if len(pairs):
            fs = (feats - mu) / sd
            pA = clfA.predict_proba(fs[:, A_cols])[:, 1]
            pB = clfB.predict_proba(fs[:, B_cols])[:, 1]
            lab_A.append(_agglomerate(frags, pairs, pA, tA))                    # A only (greedy)
            keep = pA * (pB < tB)
            lab_AB.append(_agglomerate(frags, pairs, keep, tA))                 # A AND not-B (greedy)
            p = np.clip(pA * (1 - pB), 1e-4, 1 - 1e-4)                          # combined merge prob
            w = np.log(p / (1 - p))                                            # signed multicut weight
            lab_MC.append(_gaec(frags, pairs, w))                              # learned MULTICUT (GAEC)
        else:
            lab_A.append(frags); lab_AB.append(frags); lab_MC.append(frags)
        lab_M.append(mws(np.clip(aff, 0, 1), OFFS, ns))
    A = score_lab(lab_A, eval_segs); AB = score_lab(lab_AB, eval_segs)
    MC = score_lab(lab_MC, eval_segs); M = score_lab(lab_M, eval_segs)
    return {"method": "two-specialist proofreading + learned multicut (GAEC) vs MWS",
            "lsd_features": has_lsd, "n_train_pairs": int(len(ysame)),
            "same_rate": round(float(ysame.mean()), 3), "mean_fragments": round(float(np.mean(nfrag)), 1),
            "mws_baseline": M, "splitfix_A_only": A, "splitfix_A_plus_mergefix_B": AB,
            "learned_multicut_gaec": MC,
            "MC_dVOI_vs_mws": round(MC["VOI"] - M["VOI"], 4),
            "MC_dERL_vs_mws": round(MC["ERL"] - M["ERL"], 4),
            "best_vs_mws": min([("mws", M), ("A", A), ("AB", AB), ("multicut", MC)],
                               key=lambda kv: kv[1]["VOI"])[0],
            "multicut_beats_mws": bool(MC["VOI"] <= M["VOI"] and MC["ERL"] >= M["ERL"])}
