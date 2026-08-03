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
    """Conservative fragments (fast): boundary where mean short-range affinity is low;
    fragments = connected components of the non-boundary interior; boundary voxels filled
    from the nearest fragment. Over-segments -> essentially merge-free. All C-level."""
    from scipy.ndimage import label, distance_transform_edt
    ns = len(short_offs)
    bnd = aff[:ns].min(0) < thr                           # weakest short edge low = at a boundary
    lab, n = label(~bnd)                                  # CC of confident interior
    if n == 0:
        return np.zeros(aff.shape[1:], np.int64)
    _, (iz, iy, ix) = distance_transform_edt(lab == 0, return_indices=True)
    return lab[iz, iy, ix]                                # fill boundary from nearest fragment


def _relabel(lab):
    _, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(lab.shape)


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


def _rag(frags, aff, short_offs, long_offs):
    """Per adjacent fragment-pair edge features: mean short aff (attraction), mean long
    aff (repulsion), contact count, sizes."""
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
    feats = np.stack([mean_short, mean_long, np.log1p(ns_e[keep]),
                      np.log1p(np.minimum(sizes[a[keep]], sizes[b[keep]]))], 1).astype(np.float32)
    pairs = list(zip(a[keep].tolist(), b[keep].tolist()))
    return pairs, feats


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


def run(train_affs, train_segs, eval_affs, eval_segs, ctx, thr_over=0.9, thr_merge=0.5):
    """Train the split-fixer on train subvols, apply on eval, score vs plain MWS."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        return {"error": f"sklearn unavailable: {e}"}
    SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT)
    LONG = OFFS[ns:]
    mws = ctx["mutex_watershed"]; seg_metrics = ctx["seg_metrics"]; erl = ctx["erl_proxy"]

    # ---- build training set: fragment-pair features + merge labels ----
    X, y = [], []
    for aff, seg in zip(train_affs, train_segs):
        frags = _relabel(_oversegment(aff, SHORT, thr_over))
        pairs, feats = _rag(frags, aff, SHORT, LONG)
        if not len(pairs):
            continue
        fg = _frag_gt(frags, seg)
        lbl = np.array([1 if fg[a] == fg[b] and fg[a] != 0 else 0 for a, b in pairs])
        X.append(feats); y.append(lbl)
    X = np.concatenate(X); y = np.concatenate(y)
    if len(np.unique(y)) < 2:
        return {"error": "degenerate merge labels"}
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(class_weight="balanced", max_iter=300)
    clf.fit((X - mu) / sd, y)

    # ---- apply on eval, score agglomeration vs plain MWS baseline ----
    vs_a, rs_a, es_a, vs_m, rs_m, es_m, nfrag = [], [], [], [], [], [], []
    for aff, seg in zip(eval_affs, eval_segs):
        frags = _relabel(_oversegment(aff, SHORT, thr_over))
        nfrag.append(int(frags.max() + 1))
        pairs, feats = _rag(frags, aff, SHORT, LONG)
        if len(pairs):
            prob = clf.predict_proba((feats - mu) / sd)[:, 1]
            lab_a = _agglomerate(frags, pairs, prob, thr_merge)
        else:
            lab_a = frags
        v, r = seg_metrics(lab_a, seg); vs_a.append(v); rs_a.append(r); es_a.append(erl(lab_a, seg))
        lab_m = mws(np.clip(aff, 0, 1), OFFS, ns)                # plain MWS baseline
        v, r = seg_metrics(lab_m, seg); vs_m.append(v); rs_m.append(r); es_m.append(erl(lab_m, seg))

    def mean(a): return round(float(np.mean(a)), 4)
    return {"method": "split-fixer: learned agglomeration on over-segmented SOTA affinities",
            "n_train_pairs": int(len(y)), "merge_rate": round(float(y.mean()), 3),
            "mean_fragments": mean(nfrag),
            "learned_agglo": {"VOI": mean(vs_a), "adapted_rand_error": mean(rs_a), "ERL": mean(es_a)},
            "mws_baseline": {"VOI": mean(vs_m), "adapted_rand_error": mean(rs_m), "ERL": mean(es_m)},
            "dVOI_vs_mws": round(mean(vs_a) - mean(vs_m), 4),
            "dERL_vs_mws": round(mean(es_a) - mean(es_m), 4),
            "beats_mws": bool(mean(vs_a) <= mean(vs_m) and mean(es_a) >= mean(es_m))}
