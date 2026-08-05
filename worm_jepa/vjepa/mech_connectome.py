"""
Second hop of the pipeline: pull MECHANISTIC STRUCTURE out of the connectome graph itself
(anatomy only -- no activity needed). Given the wired adjacency produced by connectome.py, run
graph-theoretic mechanism extraction: connected components (separable circuits), community
structure (candidate functional modules), rich-club (an integrative core), motif/transitivity
(recurrent vs feed-forward wiring), and Guimera-Amaral node roles (which neurons integrate
across modules -- the mechanistically load-bearing ones).

Dependency-free (numpy + scipy.sparse.csgraph only) so it runs anywhere the segmentation runs.
Each block returns a number AND a "reading" that says what the number means for the mechanism.
Small-graph caveat is reported explicitly: CREMI's cleft sampling wires ~100 nodes, so these are
a proof of the graph->mechanism hop, not a whole-brain claim.
"""
import numpy as np
from collections import defaultdict


def _adj(edges, nodes=None):
    if nodes is None:
        nodes = sorted({a for a, b in edges} | {b for a, b in edges})
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N))
    for (a, b), w in edges.items():
        if a in idx and b in idx:
            A[idx[a], idx[b]] = A[idx[b], idx[a]] = float(w)
    return A, nodes


def components(A):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n, lab = connected_components(csr_matrix(A > 0), directed=False)
    sizes = sorted(np.bincount(lab).tolist(), reverse=True)
    return n, sizes, lab


def transitivity_clustering(A):
    Ab = (A > 0).astype(float)
    deg = Ab.sum(1)
    tri_node = np.diag(Ab @ Ab @ Ab)                     # 2*#triangles through node
    denom = deg * (deg - 1)
    cc = np.divide(tri_node, denom, out=np.zeros_like(tri_node), where=denom > 0)   # local clustering
    triangles = tri_node.sum() / 6.0
    open_triads = float((deg * (deg - 1) / 2).sum()) - 3 * triangles
    transit = float(3 * triangles / (3 * triangles + open_triads)) if (triangles + open_triads) > 0 else 0.0
    return float(cc.mean()), transit, int(round(triangles))


def rich_club(A):
    """Unweighted rich-club Phi(k): among nodes of degree > k, the fraction of possible edges
    that exist. Rising Phi(k) with k => hubs preferentially wire to hubs (an integrative core)."""
    Ab = (A > 0).astype(float)
    deg = Ab.sum(1).astype(int)
    out = {}
    for k in range(1, int(deg.max())):
        sel = deg > k
        nk = int(sel.sum())
        if nk < 2:
            continue
        ek = Ab[np.ix_(sel, sel)].sum() / 2.0
        out[int(k)] = round(float(ek / (nk * (nk - 1) / 2)), 3)
    return out


def communities(A):
    """Greedy agglomerative modularity maximization (Clauset-Newman-Moore, weighted). Returns
    the partition, modularity Q, and #communities. O(N^3) but N~100 here."""
    N = A.shape[0]
    m = A.sum() / 2.0
    if m == 0:
        return np.arange(N), 0.0, N
    E = A / (2 * m)                                       # community-community edge fractions
    a = E.sum(1).copy()
    label = np.arange(N)
    active = list(range(N))
    while len(active) > 1:
        best = (1e-12, None, None)
        for ii in range(len(active)):
            i = active[ii]
            row = E[i]
            for jj in range(ii + 1, len(active)):
                j = active[jj]
                if row[j] == 0:
                    continue
                dq = 2 * (row[j] - a[i] * a[j])
                if dq > best[0]:
                    best = (dq, i, j)
        if best[1] is None:
            break
        _, i, j = best                                   # merge j into i
        for k in active:
            if k in (i, j):
                continue
            s = E[i, k] + E[j, k]
            E[i, k] = E[k, i] = s
        E[i, i] = E[i, i] + E[j, j] + 2 * E[i, j]
        a[i] += a[j]
        label[label == j] = i
        active.remove(j)
    # relabel 0..C-1 and compute Q from the final partition
    _, lab = np.unique(label, return_inverse=True)
    Q = 0.0
    k = A.sum(1)
    for c in np.unique(lab):
        sel = lab == c
        Q += A[np.ix_(sel, sel)].sum() / (2 * m) - (k[sel].sum() / (2 * m)) ** 2
    return lab, float(Q), int(lab.max() + 1)


def node_roles(A, lab):
    """Guimera-Amaral roles from within-module degree z-score (z) and participation coeff (P).
    Connector hubs (high z, high P) integrate ACROSS modules -> the mechanistically load-bearing
    neurons; provincial hubs (high z, low P) are within-module drivers."""
    Ab = (A > 0).astype(float)
    deg = Ab.sum(1)
    N = A.shape[0]
    z = np.zeros(N)
    P = np.zeros(N)
    for c in np.unique(lab):
        sel = lab == c
        kin = Ab[np.ix_(sel, sel)].sum(1)                # within-module degree
        mu, sd = kin.mean(), kin.std()
        z[sel] = (kin - mu) / sd if sd > 0 else 0.0
    for i in range(N):
        if deg[i] == 0:
            continue
        ssum = 0.0
        for c in np.unique(lab):
            kis = Ab[i, lab == c].sum()
            ssum += (kis / deg[i]) ** 2
        P[i] = 1 - ssum
    connector_hubs = [int(i) for i in range(N) if z[i] >= 1.5 and P[i] > 0.5]
    return z, P, connector_hubs


def assortativity(A):
    Ab = (A > 0).astype(float)
    deg = Ab.sum(1)
    ii, jj = np.where(np.triu(Ab, 1) > 0)
    if len(ii) < 2:
        return None
    x, y = deg[ii], deg[jj]
    xy = np.concatenate([x, y]); yx = np.concatenate([y, x])   # symmetrize
    if xy.std() == 0 or yx.std() == 0:
        return 0.0
    return float(np.corrcoef(xy, yx)[0, 1])


def run(edges):
    """edges: {(a,b): weight}. Returns the structural mechanism extracted from the connectome."""
    if not edges:
        return {"note": "empty connectome -- nothing to extract"}
    A, nodes = _adj(edges)
    N = len(nodes)
    ncomp, sizes, clab = components(A)
    cc, transit, ntri = transitivity_clustering(A)
    rc = rich_club(A)
    lab, Q, ncomm = communities(A)
    z, P, connectors = node_roles(A, lab)
    # Q over the whole graph is trivially high when the graph is fragmented (components ARE
    # modules). The non-trivial signal is modularity WITHIN the largest connected component.
    lc = int(np.argmax(np.bincount(clab)))
    lc_sel = clab == lc
    Alc = A[np.ix_(lc_sel, lc_sel)]
    _, Q_lc, ncomm_lc = communities(Alc) if lc_sel.sum() >= 3 else (None, None, None)
    assort = assortativity(A)
    deg = (A > 0).sum(1)
    order = np.argsort(-deg)
    rc_rising = (len(rc) >= 2 and rc[max(rc)] > rc[min(rc)])
    return {
        "n_wired_nodes": N, "n_edges": int((A > 0).sum() / 2),
        "components": {"n_components": int(ncomp), "largest_sizes_top8": sizes[:8],
                       "reading": "separable connected components = anatomically distinct circuits"},
        "modularity": {"Q_whole": round(Q, 4), "n_communities": ncomm,
                       "community_sizes": sorted(np.bincount(lab).tolist(), reverse=True)[:8],
                       "Q_within_largest_component": round(Q_lc, 4) if Q_lc is not None else None,
                       "largest_component_n": int(lc_sel.sum()),
                       "n_communities_in_largest": ncomm_lc,
                       "reading": ("Q_whole is inflated by fragmentation (components ARE modules); "
                                   "Q_within_largest_component is the non-trivial signal -- real "
                                   "sub-modules inside a single circuit = candidate functional groups")},
        "recurrence": {"mean_clustering": round(cc, 4), "transitivity": round(transit, 4),
                       "n_triangles": ntri,
                       "reading": "triangles/transitivity = recurrent (vs feed-forward) wiring motifs"},
        "rich_club": {"phi_by_k": rc, "rising_core": bool(rc_rising),
                      "reading": ("rising Phi(k) => hubs interconnect = an integrative CORE that "
                                  "routes signals between modules")},
        "hub_structure": {"max_degree": int(deg.max()),
                          "hub_nodes_top5": [int(nodes[i]) for i in order[:5]],
                          "degree_assortativity": round(assort, 4) if assort is not None else None,
                          "reading": "hubs + (dis)assortativity = who the integrators are and how they wire"},
        "node_roles": {"connector_hub_neurons": [int(nodes[i]) for i in connectors],
                       "n_connector_hubs": len(connectors),
                       "reading": ("connector hubs (high within-module degree AND high cross-module "
                                   "participation) = the mechanistically load-bearing neurons")},
        "caveat": ("CREMI's cleft sampling wires ~%d neurons of ~37k -- this is a PROOF of the "
                   "graph->mechanism hop on fly, not a whole-brain claim. Full extraction wants a "
                   "denser connectome (FlyWire/hemibrain) and, for dynamics, paired activity." % N),
    }
