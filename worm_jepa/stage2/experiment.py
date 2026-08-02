"""
Diagnostic: is stage-2's modest pred_r2 an EXTRACTION problem (impoverished
input) or a RECREATION problem (search/capacity too weak) -- or just the task's
intrinsic ceiling?

Same held-out target throughout: predict delta = x[t+1]-x[t] on test worms.
We vary two axes independently:

  EXTRACTION (input richness):
    S1    = x[t]                     single frame  (what stage 2 currently uses)
    S3    = [x[t], x[t-1], x[t-2]]   short history (gives velocity/acceleration)
    LAT   = h[t]                     TRUE latents  (oracle extraction, no vel)
    LATV  = [h[t], h[t]-h[t-1]]      TRUE latents + velocity (oracle)
    WIN   = x[t-W+1 .. t]            full window, consumed by a GRU (stage-1 style)

  RECREATION (model capacity), on the raw-activity inputs:
    ridge  = linear
    mlp    = 2-hidden-layer MLP (high-capacity brute force)
    gen4   = our best discovered program (S1 only)
    gru    = recurrent net over WIN

Reading the table:
  * mlp@S1 vs gen4@S1  -> RECREATION gap at fixed info (does more capacity help
                          on the SAME single frame?).
  * S3/WIN vs S1       -> EXTRACTION gap from temporal context.
  * LAT / LATV         -> ceiling given perfect extraction of the generators.
If everything on S1 caps near 0.2 but WIN/LATV are high, the bottleneck is
EXTRACTION (single frame), not the search.
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
import discover                          # noqa: E402


def r2(Y, Yh):
    ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / st).mean())


def build_tables(n_worms=60, N=48, T=512, W=32, seed=0):
    worms, names, gt = make_synthetic_worms(n_worms=n_worms, N=N, T=T, seed=seed)
    lat = gt["latents"]                          # (n_worms, T, k)
    rows = {"S1": [], "S3": [], "LAT": [], "LATV": [], "Y": [], "worm": [], "WIN": []}
    for w, worm in enumerate(worms):
        a = worm.activity
        for t in range(W, T - 1):                # need history W and next step
            rows["S1"].append(a[t])
            rows["S3"].append(np.concatenate([a[t], a[t - 1], a[t - 2]]))
            rows["LAT"].append(lat[w, t])
            rows["LATV"].append(np.concatenate([lat[w, t], lat[w, t] - lat[w, t - 1]]))
            rows["WIN"].append(a[t - W + 1:t + 1])    # (W, N)
            rows["Y"].append(a[t + 1] - a[t])
            rows["worm"].append(w)
    for k in rows:
        rows[k] = np.asarray(rows[k], dtype=np.float32)
    te = rows["worm"] >= (n_worms - n_worms // 4)
    return rows, te


def ridge_r2(X, Y, te, lam=1.0):
    Xtr, Ytr, Xte, Yte = X[~te], Y[~te], X[te], Y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    d = Xtr.shape[1]
    A = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)
    return r2(Yte, Xte @ A)


def mlp_r2(X, Y, te, hidden=128, epochs=300, seed=0):
    torch.manual_seed(seed)
    Xtr, Ytr, Xte, Yte = X[~te], Y[~te], X[te], Y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = torch.tensor((Xtr - mu) / sd); Xte = torch.tensor((Xte - mu) / sd)
    Ytr = torch.tensor(Ytr); Yte_t = torch.tensor(Yte)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(Xtr), Ytr); loss.backward(); opt.step()
    with torch.no_grad():
        return r2(Yte, net(Xte).numpy())


def gru_r2(WIN, Y, te, hidden=128, epochs=120, seed=0):
    torch.manual_seed(seed)
    Wtr, Ytr, Wte, Yte = WIN[~te], Y[~te], WIN[te], Y[te]
    Wtr = torch.tensor(Wtr); Wte = torch.tensor(Wte)
    Ytr = torch.tensor(Ytr)
    N = Y.shape[1]
    gru = nn.GRU(N, hidden, batch_first=True)
    head = nn.Linear(hidden, N)
    params = list(gru.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    lossf = nn.MSELoss()
    bs = 512
    for _ in range(epochs):
        perm = torch.randperm(len(Wtr))
        for i in range(0, len(Wtr), bs):
            idx = perm[i:i + bs]
            h, _ = gru(Wtr[idx]); pred = head(h[:, -1, :])
            opt.zero_grad(); loss = lossf(pred, Ytr[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        h, _ = gru(Wte); pred = head(h[:, -1, :]).numpy()
    return r2(Yte, pred)


def main():
    rows, te = build_tables()
    res = {}
    # extraction x recreation grid on raw-activity inputs
    res["ridge@S1"] = ridge_r2(rows["S1"], rows["Y"], te)
    res["ridge@S3"] = ridge_r2(rows["S3"], rows["Y"], te)
    res["mlp@S1"] = mlp_r2(rows["S1"], rows["Y"], te)
    res["mlp@S3"] = mlp_r2(rows["S3"], rows["Y"], te)
    res["gru@WIN"] = gru_r2(rows["WIN"], rows["Y"], te)
    # oracle extraction (true latents)
    res["ridge@LAT(oracle)"] = ridge_r2(rows["LAT"], rows["Y"], te)
    res["ridge@LATV(oracle)"] = ridge_r2(rows["LATV"], rows["Y"], te)
    res["mlp@LATV(oracle)"] = mlp_r2(rows["LATV"], rows["Y"], te)
    # our discovered program on S1 (recreation reference at fixed single-frame info)
    res["gen4_program@S1"] = discover.eval_program(
        os.path.join(os.path.dirname(__file__), "candidates", "gen4_tuned.py")
    )["pred_r2"]

    order = ["ridge@S1", "gen4_program@S1", "mlp@S1", "ridge@S3", "mlp@S3",
             "gru@WIN", "ridge@LAT(oracle)", "ridge@LATV(oracle)", "mlp@LATV(oracle)"]
    print("\n=== delta-prediction R^2 (held-out worms) ===")
    for k in order:
        print(f"  {k:24s} {res[k]:+.4f}")
    out = os.path.join(os.path.dirname(__file__), "experiment.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[stage2] wrote {out}")
    return res


if __name__ == "__main__":
    main()
