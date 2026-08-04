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


def build_edges(seg, clefts, cleft_bg, dil=2, pad=3):
    from scipy.ndimage import find_objects, binary_dilation
    mx = int(clefts.max())
    ids = [int(c) for c in np.unique(clefts) if c != cleft_bg]
    lut = np.zeros(mx + 1, np.int32)                          # compact non-bg clefts -> 1..K
    for i, c in enumerate(ids):
        lut[c] = i + 1
    lab = lut[np.clip(clefts, 0, mx)]
    slices = find_objects(lab)                               # bounding box per cleft (one pass)
    edges = defaultdict(int)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        psl = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in sl)   # pad the bbox
        d = binary_dilation(lab[psl] == (i + 1), iterations=dil)
        neigh = seg[psl][d]; neigh = neigh[neigh > 0]
        if len(neigh) < 2:
            continue
        v, ct = np.unique(neigh, return_counts=True)
        top = v[np.argsort(-ct)][:2]
        if len(top) >= 2:
            a, b = int(min(top[0], top[1])), int(max(top[0], top[1]))
            if a != b:
                edges[(a, b)] += 1
    return dict(edges)


def merge_neurons(seg, rate, rng):
    """Randomly fuse a fraction `rate` of neurons into partners -- simulate seg merges."""
    ids = [int(i) for i in np.unique(seg) if i > 0]
    if len(ids) < 2:
        return seg
    rng.shuffle(ids)
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
    from scipy.linalg import eigh
    idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    if N < 3:
        return {"n_nodes": N, "note": "too few nodes"}
    A = np.zeros((N, N))
    for (a, b), w in edges.items():
        if a in idx and b in idx:
            A[idx[a], idx[b]] = A[idx[b], idx[a]] = w
    deg = A.sum(1); order = np.argsort(-deg)
    d = np.clip(deg, 1e-6, None); Dm = np.diag(1 / np.sqrt(d))
    ev = np.sort(np.clip(eigh(Dm @ (np.diag(deg) - A) @ Dm, eigvals_only=True), 0, 2))
    return {"n_nodes": N, "n_edges": len(edges), "mean_degree": round(float(deg.mean()), 2),
            "max_degree": int(deg.max()), "spectral_gap_fiedler": round(float(ev[2] - ev[1]), 4) if N > 2 else None,
            "hub_nodes_top5": [int(nodes[i]) for i in order[:5]],
            "note": "adjacency of the connectome; spectral gap ~ modular separability (stage-2 input)"}


def run(neuron_ids, clefts, merge_rates=(0.0, 0.05, 0.1, 0.2, 0.4)):
    cleft_bg = int(clefts.max())                              # CREMI: max label = background
    nodes = [int(i) for i in np.unique(neuron_ids) if i > 0]
    gt_edges = build_edges(neuron_ids, clefts, cleft_bg)
    struct = spectral(gt_edges, nodes)
    rng = np.random.default_rng(0)
    corruption = []
    for r in merge_rates:
        mseg = neuron_ids if r == 0 else merge_neurons(neuron_ids, r, rng)
        me = build_edges(mseg, clefts, cleft_bg)
        # map merged edges back to GT-neuron space via the merge LUT is implicit: merged seg
        # uses GT ids (fused), so edges are already in a GT-derived space; compare to GT edges.
        corruption.append({"merge_rate": r, **_f1(me, gt_edges)})
    return {"n_neurons": len(nodes), "n_synaptic_clefts": int(len(np.unique(clefts)) - 1),
            "gt_connectome": {"n_edges": len(gt_edges), "structure": struct},
            "merge_corruption_curve": corruption,
            "reading": ("connectome edge-F1 vs segmentation merge rate: this is the wiring-level "
                        "cost of merges -> sets how merge-safe the labeling must be for the mechanistic "
                        "model to be correct. The gt_connectome adjacency is the stage-2 input."),
            "edges_sample": [list(k) for k in list(gt_edges.keys())[:50]]}
