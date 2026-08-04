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


def _oversegment(aff, short_offs, thr, seed_q=0.6):
    """Robust over-segmentation via watershed: seeds = connected components of the
    high-attraction interior (mean short affinity above a data-adaptive threshold), then
    flood the boundary height (1 - attraction). Scale-robust -> many fragments regardless
    of the affinity distribution, essentially merge-free.

    Granularity is set by seed_q (the interior-core quantile), NOT thr: seed_thr takes the
    MAX of thr and quantile(a, seed_q), and the quantile usually dominates, so raising
    seed_q -> stricter cores -> FEWER, coarser fragments (attacks over-seg); lowering it ->
    finer. (thr only bites when the quantile floor falls below it.)"""
    from scipy.ndimage import label
    from skimage.segmentation import watershed
    ns = len(short_offs)
    a = aff[:ns].mean(0)                                  # attraction: high inside objects
    seed_thr = max(thr, float(np.quantile(a, seed_q)))    # adaptive: interior cores
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


def _pair_agree(frags, feat, ak, bk):
    """Per-fragment mean of a (C,Z,H,W) feature, then pair L2 distance + cosine."""
    fl = _frag_lsd(frags, feat)                          # reuse per-fragment mean
    dist = np.linalg.norm(fl[ak] - fl[bk], axis=1)
    cos = (fl[ak] * fl[bk]).sum(1) / (np.linalg.norm(fl[ak], axis=1) *
                                      np.linalg.norm(fl[bk], axis=1) + 1e-6)
    return dist, cos


def _rag(frags, aff, short_offs, long_offs, lsd=None, jepa=None):
    """Per adjacent fragment-pair edge features. Returns (pairs, feats, names). Base
    features: mean short aff (attraction), mean long aff (repulsion), contact, sizes.
    Optional LSD and JEPA give shape/context AGREEMENT across the pair (dist + cos) --
    the signals MWS lacks: do the two fragments' shape/representation CONTINUE?"""
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
    empty = ([], np.zeros((0, 4 + (2 if lsd is not None else 0) + (2 if jepa is not None else 0)), np.float32))
    names = ["mean_short", "mean_long", "log_contact", "log_min_size"]
    if lsd is not None:
        names += ["lsd_dist", "lsd_cos"]
    if jepa is not None:
        names += ["jepa_dist", "jepa_cos"]
    if not keys or sum(len(x) for x in keys) == 0:
        return (*empty, *empty, names)
    key = np.concatenate(keys); av = np.concatenate(avs); sm = np.concatenate(shorts)
    uniq, inv = np.unique(key, return_inverse=True); P = len(uniq)   # compact pair index
    ss = np.zeros(P); ns_e = np.zeros(P); sl = np.zeros(P); nl_e = np.zeros(P)
    np.add.at(ss, inv[sm], av[sm]); np.add.at(ns_e, inv[sm], 1.0)
    np.add.at(sl, inv[~sm], av[~sm]); np.add.at(nl_e, inv[~sm], 1.0)
    sizes = np.bincount(frags.ravel(), minlength=NF)
    a = (uniq // NF).astype(np.int64); b = (uniq % NF).astype(np.int64)

    def rows(mask):                                       # feature matrix for the selected pairs
        ak, bk = a[mask], b[mask]
        ms = np.where(ns_e[mask] > 0, ss[mask] / np.maximum(ns_e[mask], 1), 0.0)
        ml = np.where(nl_e[mask] > 0, sl[mask] / np.maximum(nl_e[mask], 1), 0.5)
        cols = [ms, ml, np.log1p(ns_e[mask] + nl_e[mask]),
                np.log1p(np.minimum(sizes[ak], sizes[bk]))]
        if lsd is not None:
            d, c = _pair_agree(frags, lsd, ak, bk); cols += [d, c]
        if jepa is not None:
            d, c = _pair_agree(frags, jepa, ak, bk); cols += [d, c]
        return list(zip(ak.tolist(), bk.tolist())), np.stack(cols, 1).astype(np.float32)

    local = ns_e > 0                                      # short-range contact = adjacent (contractible)
    lifted = (ns_e == 0) & (nl_e > 0)                     # long-range-only = LIFTED (constraint, no contract)
    lp, lf = rows(local); lifp, liff = rows(lifted)
    return lp, lf, lifp, liff, names


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


def _lifted_gaec(frags, lpairs, lw, liftpairs, liftw):
    """Lifted multicut (GAEC approximation): LOCAL edges are contractible (adjacency);
    LIFTED edges are long-range same/different CONSTRAINTS that are never contracted but
    modify the effective merge score of the local edge between the same two clusters. A
    long-range 'different' signal (negative lifted) blocks a local merge; 'same' (positive)
    encourages it -- the extra expressiveness local multicut lacks. The CREMI-SOTA agglomerator."""
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
    local = defaultdict(lambda: defaultdict(float)); lift = defaultdict(lambda: defaultdict(float))
    for (a, b), w in zip(lpairs, lw):
        if a != b:
            local[a][b] += w; local[b][a] += w
    for (a, b), w in zip(liftpairs, liftw):
        if a != b:
            lift[a][b] += w; lift[b][a] += w

    def eff(u, v):
        return local[u].get(v, 0.0) + lift[u].get(v, 0.0)
    heap = [(-eff(a, b), a, b) for a in local for b in local[a] if a < b]
    heapq.heapify(heap)
    while heap:
        nw, a, b = heapq.heappop(heap); w = -nw
        if w <= 0:
            break
        ra, rb = find(a), find(b)
        if ra == rb or rb not in local[ra]:
            continue
        if abs(eff(ra, rb) - w) > 1e-6:
            continue                                     # stale
        parent[rb] = ra
        for tbl in (local, lift):                        # merge both adjacencies
            tbl[ra].pop(rb, None); tbl[rb].pop(ra, None)
            for nb, wt in list(tbl[rb].items()):
                tbl[nb].pop(rb, None)
                if nb == ra:
                    continue
                tbl[ra][nb] += wt; tbl[nb][ra] += wt
            tbl[rb].clear()
        for nb in list(local[ra]):                       # re-push affected local edges
            e = eff(ra, nb)
            if e > 0:
                heapq.heappush(heap, (-e, min(ra, nb), max(ra, nb)))
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


ATTRACT = {"mean_short", "log_contact", "lsd_cos", "jepa_cos"}     # merge-favoring (specialist A)
REPEL = {"mean_long", "log_min_size", "lsd_dist", "jepa_dist"}     # boundary-favoring (specialist B)


def run(train_affs, train_segs, eval_affs, eval_segs, ctx, thr_over=0.9, seed_q=0.6,
        train_lsds=None, eval_lsds=None, train_jepas=None, eval_jepas=None, merge_bias=0.0,
        merge_w=3.0):
    """Learned MULTICUT (GAEC) agglomeration with two-specialist signed edge weights,
    vs plain MWS. Runs multiple FEATURE VARIANTS so we can isolate each signal's value:
    affinity-only, +LSD shape, +JEPA context. Fed to GAEC as logit(P_A*(1-P_B))."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        return {"error": f"sklearn unavailable: {e}"}
    SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT); LONG = OFFS[ns:]
    mws = ctx["mutex_watershed"]; seg_metrics = ctx["seg_metrics"]; erl = ctx["erl_proxy"]

    def build(affs, lsds, jepas):
        out = []
        for i, aff in enumerate(affs):
            frags = _relabel(_oversegment(aff, SHORT, thr_over, seed_q))
            lp, lf, lifp, liff, names = _rag(frags, aff, SHORT, LONG,
                                             lsds[i] if lsds else None, jepas[i] if jepas else None)
            out.append((frags, lp, lf, lifp, liff, names))
        return out
    tr = build(train_affs, train_lsds, train_jepas)
    ev = build(eval_affs, eval_lsds, eval_jepas)
    names = next((r[5] for r in tr if r[5]), ["mean_short", "mean_long", "log_contact", "log_min_size"])

    X, ysame = [], []
    for (frags, lp, lf, _lifp, _liff, _), seg in zip(tr, train_segs):
        if not lp:
            continue
        fg = _frag_gt(frags, seg)
        X.append(lf); ysame.append(np.array([1 if fg[a] == fg[b] and fg[a] != 0 else 0 for a, b in lp]))
    if not X:
        return {"error": "no fragment pairs"}
    X = np.concatenate(X); ysame = np.concatenate(ysame)
    if len(np.unique(ysame)) < 2:
        return {"error": "degenerate merge labels"}
    mu, sd = X.mean(0), X.std(0) + 1e-6

    from skimage.metrics import variation_of_information as _voi

    def score_lab(labs):
        v = [seg_metrics(l, s) for l, s in zip(labs, eval_segs)]
        # VOI decomposition: variation_of_information(gt, pred) = (H(gt|pred), H(pred|gt))
        #   = (VI_MERGE / under-seg component, VI_SPLIT / over-seg component)
        comp = [_voi(s, l) for l, s in zip(labs, eval_segs)]
        vmerge = float(np.mean([c[0] for c in comp])); vsplit = float(np.mean([c[1] for c in comp]))
        return {"VOI": round(float(np.mean([x[0] for x in v])), 4),
                "VOI_merge": round(vmerge, 4), "VOI_split": round(vsplit, 4),
                "adapted_rand_error": round(float(np.mean([x[1] for x in v])), 4),
                "ERL": round(float(np.mean([erl(l, s) for l, s in zip(labs, eval_segs)])), 4)}

    def error_types(labs, cover=0.1):
        """Count actual MERGE errors (a predicted segment spanning >=2 GT neurons) and
        SPLIT errors (a GT neuron broken into >=2 predicted segments), by contingency."""
        nmerge = nsplit = ngt = 0
        for lab, seg in zip(labs, eval_segs):
            for g in np.unique(seg):
                if g == 0:
                    continue
                ngt += 1
                pl = lab[seg == g]
                pv, pc = np.unique(pl, return_counts=True)
                if (pc >= cover * pc.sum()).sum() >= 2:
                    nsplit += 1                          # GT neuron split across >=2 predicted
            for p in np.unique(lab):
                gl = seg[lab == p]
                gv, gc = np.unique(gl[gl > 0], return_counts=True) if (gl > 0).any() else (np.array([]), np.array([]))
                if len(gc) and (gc >= cover * max(1, gc.sum())).sum() >= 2:
                    nmerge += 1                          # predicted segment spans >=2 GT neurons
        return {"n_gt_neurons": int(ngt), "split_errors": int(nsplit), "merge_errors": int(nmerge),
                "split_error_rate": round(nsplit / max(1, ngt), 3)}

    def edge_w(clfA, clfB, feats, Ai, Bi):                # signed multicut weight logit(P_A*(1-P_B))
        if not len(feats):
            return np.zeros(0)
        pA = clfA.predict_proba((feats[:, Ai] - mu[Ai]) / sd[Ai])[:, 1]
        pB = clfB.predict_proba((feats[:, Bi] - mu[Bi]) / sd[Bi])[:, 1]
        p = np.clip(pA * (1 - pB), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def multicut(use, lifted=False):
        Ai = [names.index(n) for n in use if n in ATTRACT]
        Bi = [names.index(n) for n in use if n in REPEL]
        clfA = LogisticRegression(class_weight="balanced", max_iter=300).fit((X[:, Ai] - mu[Ai]) / sd[Ai], ysame)
        clfB = LogisticRegression(class_weight="balanced", max_iter=300).fit((X[:, Bi] - mu[Bi]) / sd[Bi], 1 - ysame)
        labs = []
        for frags, lp, lf, lifp, liff, _ in ev:
            if not lp:
                labs.append(frags); continue
            wl = edge_w(clfA, clfB, lf, Ai, Bi) + merge_bias   # +bias -> merge more (attack over-seg)
            if lifted and lifp:
                wlift = edge_w(clfA, clfB, liff, Ai, Bi) + merge_bias
                labs.append(_lifted_gaec(frags, lp, wl, lifp, wlift))
            else:
                labs.append(_gaec(frags, lp, wl))
        return score_lab(labs), labs

    mws_labs = [mws(np.clip(a, 0, 1), OFFS, ns) for a in eval_affs]
    M = score_lab(mws_labs)
    base = ["mean_short", "mean_long", "log_contact", "log_min_size"]
    full = names
    variants, labels = {}, {"mws": mws_labs}
    def add(name, use, lifted=False):
        sc, lb = multicut(use, lifted); variants[name] = sc; labels[name] = lb
    add("multicut_affinity", base)
    if "jepa_cos" in names:
        add("multicut_affinity_lsd_jepa", full)
        add("lifted_multicut_full", full, lifted=True)
    elif "lsd_cos" in names:
        add("multicut_affinity_lsd", full)
        add("lifted_multicut_full", full, lifted=True)
    allm = {"mws": M, **variants}
    best = min(allm.items(), key=lambda kv: kv[1]["VOI"])
    # ERL-FIRST / MERGE-AVERSE selection: in connectomics a MERGE is catastrophic (one
    # merge poisons a whole traced run to length 0) while a split is cheaply proofread.
    # So also select by ERL (higher = longer correct runs, merge-penalized) and by a
    # merge-weighted VOI (VOI_merge/under-seg penalized merge_w x harder than VOI_split).
    best_erl = max(allm.items(), key=lambda kv: kv[1]["ERL"])

    def mwv(d):
        return d["VOI_split"] + merge_w * d["VOI_merge"]        # merge-averse composite (lower better)
    best_mw = min(allm.items(), key=lambda kv: mwv(kv[1]))
    # error types for EVERY method -> pick the merge-MINIMIZING operating point, not just VOI
    err_all = {k: error_types(labs) for k, labs in labels.items()}
    merge_errs = {k: err_all[k]["merge_errors"] for k in err_all}
    # CREMI-style score proxy = geomean(VOI, adapted-Rand); lower better (NOT official protocol)
    cremi = {k: round(float((v["VOI"] * v["adapted_rand_error"]) ** 0.5), 4) for k, v in allm.items()}
    return {"method": "learned multicut (GAEC) feature-variants vs MWS + ERL-first error analysis",
            "features_available": names, "n_train_pairs": int(len(ysame)),
            "same_rate": round(float(ysame.mean()), 3),
            "mean_fragments": round(float(np.mean([r[0].max() + 1 for r in ev])), 1),
            "mws_baseline": M, **variants,
            "best_by_voi": best[0],
            "best_by_erl": best_erl[0],                          # ERL-first (merge-averse) winner
            "best_by_merge_weighted_voi": best_mw[0], "merge_w": merge_w,
            "best_beats_mws": bool(best[0] != "mws" and best[1]["VOI"] <= M["VOI"]),
            "VOI_split_merge_note": "VOI_merge = under-seg (merges), VOI_split = over-seg (splits)",
            "error_types_mws_vs_best": {"mws": err_all["mws"], best[0]: err_all[best[0]]},
            "error_types_all": err_all, "merge_errors_by_method": merge_errs,
            "cremi_score_proxy": cremi,
            "deltas_vs_mws": {k: {"dVOI": round(v["VOI"] - M["VOI"], 4),
                                  "dERL": round(v["ERL"] - M["ERL"], 4)} for k, v in variants.items()}}
