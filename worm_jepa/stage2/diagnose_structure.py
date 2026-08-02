"""
Why are we at struct_corr ~0.35 when effective-vs-structural is ~0.49 (and
SynapsNet >0.80)? Test whether structure is more recoverable than 0.35 suggests
if we read it off a COUPLING MATRIX directly (like SynapsNet reads its learned FC)
instead of the velocity-blurred effective operator, and with the right PRIORS
(the true coupling W.W^T is low-rank + symmetric).

Reports struct_corr = corr(|M|_offdiag, |W.W^T|_offdiag) for M =
  corr / cov / partial-corr / low-rank corr (rank r) / symmetric low-rank fit.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discover


def sc(M, true):
    N = M.shape[0]; off = ~np.eye(N, dtype=bool)
    a, c = np.abs(M)[off], true[off]
    if a.std() < 1e-12 or c.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, c)[0, 1])


def main():
    prob = discover.get_problem(lags=3, n_worms=160)
    N = prob["N"]
    xt = prob["Xtr"][:, :N]
    true = np.abs(prob["coupling"])                     # |W W^T|, the ground truth

    C = np.cov(xt.T)
    R = np.corrcoef(xt.T)
    P = np.linalg.inv(C + 1e-2 * np.eye(N))
    d = np.sqrt(np.diag(P)); partial = -P / (np.outer(d, d) + 1e-9)

    def lowrank(M, r):
        U, S, Vt = np.linalg.svd(M)
        return (U[:, :r] * S[:r]) @ Vt[:r]

    res = {
        "corr": round(sc(R, true), 4),
        "cov": round(sc(C, true), 4),
        "partial_corr": round(sc(partial, true), 4),
        "corr_lowrank_r6": round(sc(lowrank(R, 6), true), 4),
        "cov_lowrank_r6": round(sc(lowrank(C, 6), true), 4),
        "corr_lowrank_r12": round(sc(lowrank(R, 12), true), 4),
        "reference_effective_operator": 0.35,   # what we've been reporting
        "reference_literature_eff_vs_struct": 0.49,
    }
    for k, v in res.items():
        print(f"  {k:32s} {v}")
    out = os.path.join(os.path.dirname(__file__), "diagnose_structure.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[stage2] wrote {out}")


if __name__ == "__main__":
    main()
