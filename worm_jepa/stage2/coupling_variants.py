"""
Break the struct_corr plateau? Compare coupling-matrix estimators used as the
message-passing term:
  corr        - Pearson correlation of activity (current best)
  cov         - covariance
  precision   - inverse (regularized) covariance  -> conditional structure
  partial     - partial correlation (normalized precision)  -> DIRECT edges

Honest hypothesis: the synthetic ground truth W.W^T is a shared-latent (covariance-
like) coupling with NO direct neuron-neuron edges (neurons are conditionally
independent given the latents), so precision/partial may not help -- they target
direct connectivity that doesn't exist here. Test it.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discover
from discover import softplus


def coupling(xt, kind, reg=1e-2):
    C = np.cov(xt.T)
    N = C.shape[0]
    if kind == "corr":
        M = np.corrcoef(xt.T)
    elif kind == "cov":
        M = C
    elif kind in ("precision", "partial"):
        P = np.linalg.inv(C + reg * np.eye(N))
        if kind == "precision":
            M = -P                              # off-diagonal precision (sign for coupling)
        else:
            d = np.sqrt(np.diag(P))
            M = -P / (np.outer(d, d) + 1e-9)    # partial correlation
    M = np.nan_to_num(M)
    np.fill_diagonal(M, 0.0)
    return M


def make_build(kind, ridge=10.0):
    def build(Xtr, Dtr):
        N = Dtr.shape[1]
        M = coupling(Xtr[:, :N], kind)

        def raw(X):
            xt, xt1 = X[:, :N], X[:, N:2 * N]
            vel = xt - xt1
            return np.concatenate([xt, xt @ M.T, vel @ M.T], axis=1)

        T = raw(Xtr); cmu, csd = T.mean(0), T.std(0) + 1e-8
        Ts = (T - cmu) / csd
        C = np.linalg.solve(Ts.T @ Ts + ridge * np.eye(Ts.shape[1]), Ts.T @ Dtr)
        return lambda X: ((raw(X) - cmu) / csd) @ C
    return build


def main():
    prob = discover.get_problem(lags=3, n_worms=160)
    res = {}
    for kind in ["corr", "cov", "precision", "partial"]:
        m = discover.evaluate(make_build(kind), prob)
        res[kind] = {"struct_corr": round(m["struct_corr"], 4), "pred_r2": round(m["pred_r2"], 4)}
        print(f"  {kind:10s} struct={res[kind]['struct_corr']:+.3f}  pred={res[kind]['pred_r2']:+.3f}")
    out = os.path.join(os.path.dirname(__file__), "coupling_variants.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[stage2] wrote {out}")


if __name__ == "__main__":
    main()
