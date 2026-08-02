"""
Run our BEST synthetic structure method (learned explicit low-rank symmetric
coupling, which hit 0.68-0.81 on synthetic) on REAL data, scored vs the real Cook
connectome. Compare to the raw correlation/covariance couplings (which gave ~0.11).

Does a coupling matrix LEARNED for prediction recover the real wiring better than
raw correlation?
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2_cc import build_core   # noqa: E402


def struct_corr(M, A):
    N = M.shape[0]; off = ~np.eye(N, dtype=bool)
    a, c = np.abs(M)[off], np.abs(A)[off]
    return float(np.corrcoef(a, c)[0, 1])


def r2(Y, Yh):
    ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / st).mean())


def main():
    torch.manual_seed(0)
    fullw, core, names, A = build_core()
    N = len(core)
    # temporal delta pairs split by worm
    Xt, Vel, D, wid = [], [], [], []
    for k, w in enumerate(fullw):
        a = w.activity[:, core].astype(np.float32)
        Xt.append(a[1:-1]); Vel.append(a[1:-1] - a[:-2]); D.append(a[2:] - a[1:-1])
        wid.append(np.full(len(a) - 2, k))
    Xt = np.concatenate(Xt); Vel = np.concatenate(Vel); D = np.concatenate(D); wid = np.concatenate(wid)
    te = wid >= (len(fullw) - max(1, len(fullw) // 4))

    res = {"n_neurons": N, "n_worms": len(fullw), "connectome": "Cook2019Herm"}
    # fixed couplings (baselines) vs Cook
    cov = np.cov(Xt[~te].T); corr = np.corrcoef(Xt[~te].T)
    res["struct_corr_raw_correlation"] = round(struct_corr(corr, A), 4)
    res["struct_corr_raw_covariance"] = round(struct_corr(cov, A), 4)

    # learned explicit low-rank symmetric coupling M = U U^T, optimized for delta pred
    Xtr, Vtr, Dtr = (torch.tensor(z[~te]) for z in (Xt, Vel, D))
    Xte, Vte, Dte = (torch.tensor(z[te]) for z in (Xt, Vel, D))
    w0, V0 = np.linalg.eigh(cov)
    best = None
    for r in (6, 8, 12):
        U = nn.Parameter(torch.tensor(V0[:, -r:] * np.sqrt(np.clip(w0[-r:], 0, None)), dtype=torch.float32))
        C = nn.Parameter(torch.zeros(3 * N, N))
        opt = torch.optim.Adam([U, C], lr=5e-3)
        for _ in range(500):
            M = U @ U.T
            F = torch.cat([Xtr, Xtr @ M, Vtr @ M], 1)
            loss = ((F @ C - Dtr) ** 2).mean() + 1e-4 * M.abs().mean() + 1e-4 * C.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            M = U @ U.T
            pr = r2(Dte.numpy(), (torch.cat([Xte, Xte @ M, Vte @ M], 1) @ C).numpy())
            sc = struct_corr(M.numpy(), A)
        res[f"learned_r{r}"] = {"pred_r2": round(pr, 4), "struct_corr": round(sc, 4)}
        print(f"  learned r={r}: pred_r2={pr:.3f}  struct_corr(vs Cook)={sc:.3f}")
        if best is None or sc > best:
            best = sc
    res["best_learned_struct_corr"] = round(best, 4)
    res["reference_raw_correlation_real"] = res["struct_corr_raw_correlation"]
    res["reference_synthetic_best"] = 0.806
    print(json.dumps(res, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "stage2_real_learned.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
