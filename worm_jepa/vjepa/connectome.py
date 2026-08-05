"""
End-to-end skeleton, stage A->B bridge: turn a SEGMENTATION into a CONNECTOME graph, and
measure how segmentation MERGES corrupt the wiring -- because a merge = a false edge = a
corrupted mechanistic model. Output the graph for stage-2 mechanistic extraction.

Connectome edges from CREMI: each labelled synaptic CLEFT sits at the apposition of two
neurons; dilate the cleft, read the two dominant neuron ids it touches -> an edge. Nodes =
neurons, edge weight = #clefts between them. (Undirected skeleton; the directed pre->post
version needs CREMI's partner annotations -- a later refinement.)

The mechanism-relevant metric: build the GT connectome (GT neuron_ids), then simulate
segmentation merges at increasing rate and measure connectome edge precision/recall + how
many edges are destroyed/created. This says, in wiring units, how merge-safe the
segmentation must be for the extracted mechanism to be correct.
"""
import numpy as np
from collections import defaultdict


def compact(vol):
    """Relabel arbitrary (possibly huge uint64) ids -> 0..K with 0 = background (the MAX
    original value, CREMI's convention). Robust to huge background values (no max-sized LUT)."""
    vol = np.asarray(vol)
    bg = vol.max()
    uniq = np.unique(vol[vol != bg]) if (vol != bg).any() else np.array([], dtype=vol.dtype)
    out = np.zeros(vol.shape, np.int64)
    if len(uniq):
        out = np.clip(np.searchsorted(uniq, vol) + 1, 0, len(uniq))   # non-bg -> 1..K
        out[vol == bg] = 0
    return out, int(len(uniq)), int(bg)


def cleft_regions(clefts, dil=2, pad=3):
    """Precompute, ONCE, each cleft's padded bounding box + dilated boolean mask. The clefts
    never change across simulated segmentation merges, so we pay find_objects/dilation once and
    reuse the regions for every merge rate (the expensive full-volume scan happens a single
    time, not once per merge rate)."""
    from scipy.ndimage import find_objects, binary_dilation
    lab = clefts.astype(np.int32)                            # already compacted: 0=bg, 1..K
    regions = []
    for i, sl in enumerate(find_objects(lab)):               # bounding box per cleft (one pass)
        if sl is None:
            continue
        psl = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in sl)   # pad the bbox
        mask = binary_dilation(lab[psl] == (i + 1), iterations=dil)
        regions.append((psl, mask))
    return regions


def edges_from_regions(seg, regions):
    """Read the two dominant neuron ids each cleft touches -> an undirected weighted edge."""
    edges = defaultdict(int)
    for psl, mask in regions:
        neigh = seg[psl][mask]; neigh = neigh[neigh > 0]
        if len(neigh) < 2:
            continue
        v, ct = np.unique(neigh, return_counts=True)
        top = v[np.argsort(-ct)][:2]
        if len(top) >= 2:
            a, b = int(min(top[0], top[1])), int(max(top[0], top[1]))
            if a != b:
                edges[(a, b)] += 1
    return dict(edges)


def build_edges(seg, clefts, cleft_bg=0, dil=2, pad=3):
    return edges_from_regions(seg, cleft_regions(clefts, dil=dil, pad=pad))


def merge_neurons(seg, rate, rng):
    """Randomly fuse a fraction `rate` of neurons into partners -- simulate seg merges.
    `seg` must be COMPACTED (0=bg, 1..K) so the LUT is small."""
    ids = [int(i) for i in np.unique(seg) if i > 0]
    if len(ids) < 2:
        return seg
    ids = list(ids); rng.shuffle(ids)
    lut = np.arange(int(seg.max()) + 1)
    npairs = int(rate * len(ids) / 2)
    for k in range(npairs):
        lut[ids[2 * k + 1]] = lut[ids[2 * k]]                 # fuse b into a
    return lut[seg]


def _f1(pred, gt):
    p, g = set(pred), set(gt); tp = len(p & g)
    prec = tp / max(1, len(p)); rec = tp / max(1, len(g))
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
            "tp": tp, "n_pred_edges": len(p), "n_gt_edges": len(g)}


def spectral(edges, nodes):
    """Structure of the connectome graph. Restrict to nodes INCIDENT to at least one edge --
    isolated neurons carry no wiring info and building a dense NxN over all ~37k neurons is an
    11GB, O(N^3) trap. The wired subgraph is tiny (a few hundred nodes for CREMI)."""
    from scipy.linalg import eigh
    n_total = len(nodes)
    wired = sorted({a for (a, b) in edges} | {b for (a, b) in edges})
    idx = {n: i for i, n in enumerate(wired)}; N = len(wired)
    if N < 3:
        return {"n_nodes_total": n_total, "n_wired_nodes": N, "n_edges": len(edges),
                "note": "too few wired nodes for a spectral gap"}
    A = np.zeros((N, N))
    for (a, b), w in edges.items():
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = w
    deg = A.sum(1); order = np.argsort(-deg)
    d = np.clip(deg, 1e-6, None); Dm = np.diag(1 / np.sqrt(d))
    ev = np.sort(np.clip(eigh(Dm @ (np.diag(deg) - A) @ Dm, eigvals_only=True), 0, 2))
    return {"n_nodes_total": n_total, "n_wired_nodes": N, "n_edges": len(edges),
            "mean_degree_wired": round(float(deg.mean()), 2), "max_degree": int(deg.max()),
            "spectral_gap_fiedler": round(float(ev[2] - ev[1]), 4),
            "hub_nodes_top5": [int(wired[i]) for i in order[:5]],
            "note": "adjacency of the WIRED subgraph; spectral gap ~ modular separability (stage-2 input)"}


def run(neuron_ids, clefts, merge_rates=(0.0, 0.05, 0.1, 0.2, 0.4)):
    # Compact BOTH inputs first: CREMI stores huge uint64 ids with MAX = background. Compacting
    # gives 0=bg, 1..K and keeps every LUT/array small and int64-safe.
    seg, n_neurons, nbg = compact(neuron_ids)
    cl, n_clefts, cbg = compact(clefts)
    diag = {"cleft_dtype": str(np.asarray(clefts).dtype), "cleft_bg_value": cbg,
            "n_synaptic_clefts": n_clefts, "neuron_bg_value": nbg, "n_neurons": n_neurons}
    if n_clefts == 0:
        return {**diag, "note": ("no synaptic clefts in this chunk after background removal -- "
                                 "either genuinely cleft-sparse or the wrong label channel. "
                                 "Cannot build a connectome from clefts here."),
                "gt_connectome": {"n_edges": 0}, "merge_corruption_curve": []}
    nodes = [int(i) for i in np.unique(seg) if i > 0]
    regions = cleft_regions(cl)                               # find_objects/dilation ONCE
    gt_edges = edges_from_regions(seg, regions)
    struct = spectral(gt_edges, nodes)
    rng = np.random.default_rng(0)
    corruption = []
    for r in merge_rates:
        mseg = seg if r == 0 else merge_neurons(seg, r, rng)
        me = edges_from_regions(mseg, regions)
        # merged seg reuses compacted ids (fused), so edges live in a GT-derived space; compare to GT.
        corruption.append({"merge_rate": r, **_f1(me, gt_edges)})
    return {**diag,
            "gt_connectome": {"n_edges": len(gt_edges), "structure": struct},
            "merge_corruption_curve": corruption,
            "reading": ("connectome edge-F1 vs segmentation merge rate: this is the wiring-level "
                        "cost of merges -> sets how merge-safe the labeling must be for the mechanistic "
                        "model to be correct. The gt_connectome adjacency is the stage-2 input."),
            "edges_sample": [list(k) for k in list(gt_edges.keys())[:50]]}
