"""
Visualize connectome recovery: the discovered model's effective operator vs the
true neuron coupling. This is what struct_corr measures, made visible.
Writes figures/connectome_recovery.png.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discover


def recovered_operator(candidate_path, prob):
    build = discover.load_build_from_code(open(candidate_path).read())
    predict = build(prob["Xtr"].copy(), prob["Dtr"].copy())
    A = discover.effective_operator(predict, prob["Xte"])[: prob["N"]]   # x[t] block (N,N)
    sc = discover.struct_corr(A, prob["coupling"])
    Aop = np.abs(A).copy(); np.fill_diagonal(Aop, 0.0)
    return Aop, sc


def norm(M):
    m = M.copy(); np.fill_diagonal(m, 0.0)
    return m / (m.max() + 1e-9)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    prob = discover.get_problem(lags=3)
    N = prob["N"]
    true = norm(prob["coupling"])                       # |W W^T|, off-diagonal
    A_coup, r_coup = recovered_operator(os.path.join(here, "candidates_temporal", "tgen2_mapelites_coupling.py"), prob)
    A_none, r_none = recovered_operator(os.path.join(here, "candidates_temporal", "tgen1_velocity.py"), prob)

    cmap = "magma"
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("Connectome recovery: discovered wiring vs the true connectome\n"
                 "(each panel normalized to its own max; diagonal removed)", fontsize=12)

    im = ax[0, 0].imshow(true, cmap=cmap); ax[0, 0].set_title("TRUE connectome  |W·Wᵀ|")
    ax[0, 1].imshow(norm(A_coup), cmap=cmap)
    ax[0, 1].set_title(f"Recovered — WITH coupling term\nstruct_corr = {r_coup:.3f}")
    ax[1, 0].imshow(norm(A_none), cmap=cmap)
    ax[1, 0].set_title(f"Recovered — no coupling term\nstruct_corr = {r_none:.3f}")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("neuron j"); a.set_ylabel("neuron i")
    fig.colorbar(im, ax=[ax[0, 0], ax[0, 1], ax[1, 0]], shrink=0.6, label="|coupling| (norm.)")

    # the scatter struct_corr actually computes (coupling model)
    off = ~np.eye(N, dtype=bool)
    ax[1, 1].scatter(true[off], norm(A_coup)[off], s=4, alpha=0.25, color="#1f77b4")
    ax[1, 1].set_title(f"What struct_corr measures\n(off-diagonal, r = {r_coup:.3f})")
    ax[1, 1].set_xlabel("true coupling  |W·Wᵀ|"); ax[1, 1].set_ylabel("recovered |A|")

    figdir = os.path.join(here, "figures"); os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "connectome_recovery.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}  | coupling r={r_coup:.3f}  no-coupling r={r_none:.3f}")


if __name__ == "__main__":
    main()
