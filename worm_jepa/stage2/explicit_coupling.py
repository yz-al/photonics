"""
Beat effective-vs-structural (0.49) on synthetic by fixing BOTH the method and
the measurement:

  METHOD (SynapsNet-style): connectivity is an EXPLICIT model parameter used for
  prediction, not a post-hoc operator. We learn a coupling matrix M and read the
  recovered connectome straight off M.

  PRIORS (what the literature says is essential): the true coupling W.W^T is
  symmetric and LOW-RANK (rank = #latents). We bake those in by parameterizing
  M = U @ U.T with U in R^{N x r} -> symmetric, PSD, rank r. M is optimized
  jointly with a linear readout to predict the delta.

Reports, held-out:
  * pred_r2
  * struct_corr read from the explicit coupling M  (vs |W.W^T|)
Compared to: fixed covariance coupling, and the old effective-operator readout.
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discover


def struct_corr(M, true):
    N = M.shape[0]; off = ~np.eye(N, dtype=bool)
    a, c = np.abs(M)[off], np.abs(true)[off]
    return float(np.corrcoef(a, c)[0, 1])


def r2(Y, Yh):
    ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / st).mean())


def frames(X, N):
    xt, xt1 = X[:, :N], X[:, N:2 * N]
    return xt, xt - xt1                       # x_t, velocity


def learned_coupling(prob, r=6, epochs=400, l1=1e-4, seed=0):
    torch.manual_seed(seed)
    N = prob["N"]
    Xtr, Dtr, Xte, Dte = (torch.tensor(prob[k], dtype=torch.float32)
                          for k in ("Xtr", "Dtr", "Xte", "Dte"))
    xt_tr, vel_tr = frames(Xtr, N); xt_te, vel_te = frames(Xte, N)

    # init U from the top-r eigenvectors of the covariance (a warm structural start)
    cov = np.cov(prob["Xtr"][:, :N].T)
    w, V = np.linalg.eigh(cov)
    U0 = V[:, -r:] * np.sqrt(np.clip(w[-r:], 0, None))
    U = nn.Parameter(torch.tensor(U0, dtype=torch.float32))
    C = nn.Parameter(torch.zeros(3 * N, N))
    opt = torch.optim.Adam([U, C], lr=5e-3)

    def feats(xt, vel):
        M = U @ U.T
        return torch.cat([xt, xt @ M, vel @ M], dim=1), M

    for _ in range(epochs):
        F, M = feats(xt_tr, vel_tr)
        pred = F @ C
        loss = ((pred - Dtr) ** 2).mean() + l1 * M.abs().mean() + 1e-4 * C.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        Fte, M = feats(xt_te, vel_te)
        pr = r2(prob["Dte"], (Fte @ C).numpy())
        Mnp = M.numpy(); np.fill_diagonal(Mnp, 0.0)
    return pr, struct_corr(Mnp, prob["coupling"])


def fixed_coupling(prob, transform=None):
    """Explicit coupling M = cov(transform(activity)); message-passing predictor.
    transform=None -> raw (transferable). transform=exp -> undoes the softplus
    observation nonlinearity (a prior; synthetic-specific since we know the
    generator's nonlinearity)."""
    N = prob["N"]
    xt_all = prob["Xtr"][:, :N]
    M = np.cov((transform(xt_all) if transform else xt_all).T); np.fill_diagonal(M, 0.0)
    xt_tr, vel_tr = prob["Xtr"][:, :N], prob["Xtr"][:, :N] - prob["Xtr"][:, N:2 * N]
    xt_te, vel_te = prob["Xte"][:, :N], prob["Xte"][:, :N] - prob["Xte"][:, N:2 * N]
    T = np.concatenate([xt_tr, xt_tr @ M, vel_tr @ M], axis=1)
    cmu, csd = T.mean(0), T.std(0) + 1e-8; Ts = (T - cmu) / csd
    Cc = np.linalg.solve(Ts.T @ Ts + 10 * np.eye(Ts.shape[1]), Ts.T @ prob["Dtr"])
    Tte = (np.concatenate([xt_te, xt_te @ M, vel_te @ M], axis=1) - cmu) / csd
    pr = r2(prob["Dte"], Tte @ Cc)
    return pr, struct_corr(M, prob["coupling"])


def main():
    prob = discover.get_problem(lags=3, n_worms=160)
    res = {"reference_old_effective_operator": 0.35,
           "reference_literature_eff_vs_struct": 0.49}
    pr_f, sc_f = fixed_coupling(prob)
    res["fixed_covariance_coupling_TRANSFERABLE"] = {"pred_r2": round(pr_f, 4), "struct_corr": round(sc_f, 4)}
    pr_e, sc_e = fixed_coupling(prob, transform=np.exp)
    res["exp_corrected_coupling_SYNTHETIC_ONLY"] = {
        "pred_r2": round(pr_e, 4), "struct_corr": round(sc_e, 4),
        "note": "exp inverts the softplus generator nonlinearity; a prior tuned to "
                "the synthetic observation model. On real data the calcium (GCaMP) "
                "nonlinearity differs and must be estimated."}
    print(f"  fixed cov (transferable): pred_r2={pr_f:.3f} struct_corr={sc_f:.3f}")
    print(f"  exp-corrected (synthetic-only): pred_r2={pr_e:.3f} struct_corr={sc_e:.3f}")
    for r in (4, 6, 8):
        pr, sc = learned_coupling(prob, r=r)
        res[f"learned_lowrank_symmetric_r{r}"] = {"pred_r2": round(pr, 4), "struct_corr": round(sc, 4)}
        print(f"  learned r={r}: pred_r2={pr:.3f}  struct_corr={sc:.3f}")
    print(f"  fixed covariance: pred_r2={pr_f:.3f}  struct_corr={sc_f:.3f}")
    out = os.path.join(os.path.dirname(__file__), "explicit_coupling.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[stage2] wrote {out}")


if __name__ == "__main__":
    main()
