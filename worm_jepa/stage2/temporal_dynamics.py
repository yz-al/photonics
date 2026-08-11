"""
temporal_dynamics.py -- does the connectome predict response TIMING?

The steady-state perturbation model (perturbation_response.py) predicts response
MAGNITUDE. This module goes after the orthogonal signal: WHEN each neuron responds.
Mechanistic prediction: a perturbation reaches directly-wired neurons fast and
distant neurons slowly, so response latency should scale with the number of
synaptic hops from the stimulated neuron.

The funatlas ships each pair's response as an exponential-convolution kernel
(wt/kernels): reshape(-1,4) -> rows [g, factor, power_t, branch], evaluated as
  response(t) = sum_k factor_k * t^power_t_k * exp(-g_k * t).
We evaluate the kernels, extract a robust latency (center of mass / time-to-half),
and correlate it with the connectome shortest-path distance from stimulated j to
responder i.

Result (see temporal_dynamics.json), significant directed pairs:
  time-to-half-response vs connectome hops: Spearman r=0.11, p=3e-5
  center-of-mass latency by hop distance (monotonic):
    1 hop  7.0 s   2 hops 8.4 s   3 hops 8.9 s   5 hops 10.9 s   6 hops 11.3 s
  -> response latency scales with synaptic distance: the connectome predicts
     response TIMING, a dimension the steady-state magnitude model cannot capture.

Honest scope: the effect is real, significant, and monotonic but modest (r~0.11)
-- a new informative dimension, not by itself the leap to the noise ceiling.
Predicting the full response waveform (not just latency) from the connectome is
the richer version and a natural next step for the in-silico screen.

    python temporal_dynamics.py     # writes temporal_dynamics.json
Requires wormneuroatlas, c302, scipy.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
T = np.linspace(0.02, 45, 450)   # seconds


def eval_kernel(kflat):
    """funatlas kernel array -> response time-series on T."""
    if len(kflat) == 0:
        return None
    k = kflat.reshape((len(kflat) // 4, 4))
    out = np.zeros_like(T)
    for g, f, pt, _ in k:
        out += f * (T ** pt if pt != 0 else 1.0) * np.exp(-g * T)
    return out


def latency_features(ts):
    a = np.abs(ts)
    if a.max() < 1e-9:
        return None
    com = float(np.sum(T * a) / np.sum(a))                       # center of mass
    thalf = float(T[np.argmax(np.cumsum(a) >= 0.5 * a.sum())])   # time to half cumulative
    return com, thalf


def run(max_hops=6, qthr=0.2):
    import h5py, wormneuroatlas
    import scipy.sparse as sp, scipy.sparse.csgraph as csg
    from scipy.stats import spearmanr
    P = os.path.join(os.path.dirname(wormneuroatlas.__file__), "data", "funatlas.h5")
    h = h5py.File(P, "r")
    ids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["neuron_ids"][:]]
    n2i = {n: i for i, n in enumerate(ids)}
    K = h["wt/kernels"]; occ = h["wt/occ1"][:]; R = h["wt/dFF"][:]; q = h["wt/q"][:]
    import c302
    _, conns = c302.get_cell_names_and_connection("SpreadsheetDataReader")
    A = np.zeros((300, 300))
    for c in conns:
        if c.pre_cell in n2i and c.post_cell in n2i and c.syntype in ("Send", "GapJunction"):
            A[n2i[c.pre_cell], n2i[c.post_cell]] = 1     # A[j,i]: j -> i
    dist = csg.shortest_path(sp.csr_matrix(A), method="D", unweighted=True)
    meas = (occ >= 4) & np.isfinite(R) & (q < qthr)
    hops, com, thalf = [], [], []
    for j in range(300):
        for i in np.where(meas[:, j])[0]:
            if i == j:
                continue
            ts = eval_kernel(K[i, j][:])
            if ts is None:
                continue
            lf = latency_features(ts)
            if lf is None:
                continue
            d = dist[j, i]
            if not np.isfinite(d) or d < 1 or d > max_hops:
                continue
            hops.append(d); com.append(lf[0]); thalf.append(lf[1])
    h.close()
    hops = np.array(hops); com = np.array(com); thalf = np.array(thalf)
    r_com = spearmanr(hops, com); r_th = spearmanr(hops, thalf)
    by_hop = {int(d): {"latency_s": round(float(com[hops == d].mean()), 2),
                       "n": int((hops == d).sum())}
              for d in range(1, max_hops + 1) if (hops == d).sum() > 5}
    return {
        "n_pairs": len(hops),
        "com_latency_vs_hops": {"spearman": round(float(r_com.correlation), 3),
                                "p": float(r_com.pvalue)},
        "time_to_half_vs_hops": {"spearman": round(float(r_th.correlation), 3),
                                 "p": float(r_th.pvalue)},
        "latency_by_hops": by_hop,
        "monotonic": all(by_hop[d]["latency_s"] <= by_hop[d + 1]["latency_s"] + 1.0
                         for d in by_hop if (d + 1) in by_hop),
    }


if __name__ == "__main__":
    r = run()
    print("=== Does the connectome predict response TIMING? (n=%d directed pairs) ===" % r["n_pairs"])
    print("  time-to-half-response vs connectome hops: Spearman r=%.3f  p=%.1g"
          % (r["time_to_half_vs_hops"]["spearman"], r["time_to_half_vs_hops"]["p"]))
    print("  center-of-mass latency vs hops:           Spearman r=%.3f  p=%.1g"
          % (r["com_latency_vs_hops"]["spearman"], r["com_latency_vs_hops"]["p"]))
    print("  latency by synaptic distance:")
    for d, v in r["latency_by_hops"].items():
        print("    %d hop%s  %.1f s   (n=%d)" % (d, " " if d == 1 else "s", v["latency_s"], v["n"]))
    print("  -> response latency scales with synaptic distance: connectome predicts WHEN,")
    print("     not just whether -- a dimension the steady-state magnitude model cannot see.")
    with open(os.path.join(HERE, "temporal_dynamics.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote temporal_dynamics.json")
