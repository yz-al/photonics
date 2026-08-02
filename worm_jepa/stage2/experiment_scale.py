"""
Data lever (only possible because it's synthetic): does MORE data raise stage-2
recovery, or does it plateau? Sweep n_worms and measure the held-out
delta-prediction R^2 and connectome recovery (struct_corr vs true |W@W.T|), at
two operating points:

  S1 = single frame x[t]                 (what stage 2 currently uses)
  S3 = [x[t], x[t-1], x[t-2]]            (temporal context -- the proven lever)

Models: ridge (both), and an MLP on S3 (capacity that can exploit more data).
W is fixed across sizes (same seed) so struct_corr is comparable.

Reading it:
  * curve still climbing at the largest size -> data is a real lever (and tells
    us how much real worm data we'd need).
  * flat -> data is not the bottleneck at this operating point (info/capacity is).
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import make_synthetic_worms   # noqa: E402


def r2(Y, Yh):
    ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / st).mean())


def make(n_worms, N=48, T=512, seed=0):
    worms, names, gt = make_synthetic_worms(n_worms=n_worms, N=N, T=T, seed=seed)
    S1, S3, Y, wid = [], [], [], []
    for w, worm in enumerate(worms):
        a = worm.activity
        for t in range(2, T - 1):
            S1.append(a[t]); S3.append(np.concatenate([a[t], a[t - 1], a[t - 2]]))
            Y.append(a[t + 1] - a[t]); wid.append(w)
    S1 = np.asarray(S1, np.float64); S3 = np.asarray(S3, np.float64)
    Y = np.asarray(Y, np.float64); wid = np.asarray(wid)
    te = wid >= (n_worms - max(1, n_worms // 4))
    coupling = np.abs(gt["neuron_coupling"])
    return S1, S3, Y, te, coupling


def ridge(X, Y, te, lam=1.0):
    Xtr, Ytr, Xte, Yte = X[~te], Y[~te], X[te], Y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    A = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    return r2(Yte, Xte @ A), A


def struct_corr_from_S1(X_S1, Y, te, coupling, lam=1.0):
    _, A = ridge(X_S1, Y, te, lam)          # A is (N,N) for single-frame input
    N = A.shape[0]; off = ~np.eye(N, dtype=bool)
    a = np.abs(A)[off]; c = coupling[off]
    if a.std() < 1e-12 or c.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, c)[0, 1])


def mlp(X, Y, te, hidden=128, epochs=120, seed=0):
    torch.manual_seed(seed)
    Xtr, Ytr, Xte, Yte = X[~te], Y[~te], X[te], Y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xte_t = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss(); bs = 2048
    n = len(Xtr_t)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(net(Xtr_t[idx]), Ytr_t[idx])
            loss.backward(); opt.step()
    with torch.no_grad():
        return r2(Yte, net(Xte_t).numpy())


def main():
    sizes = [10, 20, 40, 80, 160, 320]
    res = []
    for n in sizes:
        S1, S3, Y, te, coupling = make(n)
        r_s1, _ = ridge(S1, Y, te)
        r_s3, _ = ridge(S3, Y, te)
        m_s3 = mlp(S3, Y, te)
        sc = struct_corr_from_S1(S1, Y, te, coupling)
        row = {"n_worms": n, "n_pairs_train": int((~te).sum()),
               "ridge_S1": round(r_s1, 4), "ridge_S3": round(r_s3, 4),
               "mlp_S3": round(m_s3, 4), "struct_corr_S1": round(sc, 4)}
        res.append(row)
        print(f"  n={n:4d} pairs={row['n_pairs_train']:6d}  "
              f"ridge@S1={r_s1:+.3f}  ridge@S3={r_s3:+.3f}  "
              f"mlp@S3={m_s3:+.3f}  struct@S1={sc:+.3f}", flush=True)
    out = os.path.join(os.path.dirname(__file__), "experiment_scale.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[stage2] wrote {out}")


if __name__ == "__main__":
    main()
