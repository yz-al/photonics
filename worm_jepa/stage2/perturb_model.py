"""
Is the PERTURB model (NPI-style) better at STRUCTURE, or just the best PREDICTOR?

NPI = train a forecaster, then virtually PERTURB each neuron and measure how the
poke PROPAGATES over a multi-step rollout -> directed effective connectivity P.
This is stronger than the single-step Jacobian (which scored 0.25 in close_loop).

Compare, as STRUCTURE estimators (corr vs ground-truth wiring):
  correlation   - raw activity correlation (direct, no model)
  jacobian      - forecaster single-step sensitivity
  perturb (NPI) - forecaster multi-step rollout perturbation
and report the forecaster's prediction R^2 (is perturb just the best predictor?).

Run on synthetic (known W) and real (Cook connectome).
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import make_synthetic_worms   # noqa: E402
from close_loop import Forecaster, build, r2, scorr, L   # noqa: E402


def rollout(fore, win, K):
    cur = win.clone(); outs = []
    for _ in range(K):
        with torch.no_grad():
            d = fore(cur)
        nxt = cur[:, -1, :] + d
        outs.append(nxt)
        cur = torch.cat([cur[:, 1:, :], nxt[:, None, :]], dim=1)
    return torch.stack(outs, dim=1).numpy()          # (B,K,N)


def perturb_conn(fore, win, eps=0.5, K=5):
    N = win.shape[2]
    base = rollout(fore, win, K)
    P = np.zeros((N, N))
    for j in range(N):
        w2 = win.clone(); w2[:, -1, j] += eps
        eff = np.abs(rollout(fore, w2, K) - base).sum(1).mean(0)   # cumulative effect on each i
        P[:, j] = eff / eps
    np.fill_diagonal(P, 0.0)
    return (P + P.T) / 2                              # symmetric magnitude for comparison


def train_forecaster(Xw, D, te, N, epochs=60):
    torch.manual_seed(0)
    fore = Forecaster(N); opt = torch.optim.Adam(fore.parameters(), lr=3e-3)
    tw, dw = torch.tensor(Xw[~te]), torch.tensor(D[~te])
    for _ in range(epochs):
        for i in range(0, len(tw), 512):
            b = slice(i, i + 512)
            loss = ((fore(tw[b]) - dw[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = r2(D[te], fore(torch.tensor(Xw[te])).numpy())
    return fore, pr


def jacobian_conn(fore, Xw, te, N):
    J = np.mean([fore.eff_conn(torch.tensor(Xw[te][i:i + 1])) for i in range(6)], axis=0)
    np.fill_diagonal(J, 0.0); return (J + J.T) / 2


def evaluate(Xw, Xt, D, te, N, trueC, sub=200):
    fore, pr = train_forecaster(Xw, D, te, N)
    corr = np.corrcoef(Xt[~te].T); np.fill_diagonal(corr, 0.0)
    J = jacobian_conn(fore, Xw, te, N)
    idx = np.random.default_rng(0).choice(np.where(te)[0], size=min(sub, int(te.sum())), replace=False)
    P = perturb_conn(fore, torch.tensor(Xw[idx]))
    return {"forecaster_pred_r2": round(pr, 4),
            "struct_correlation": round(scorr(corr, trueC), 4),
            "struct_jacobian": round(scorr(J, trueC), 4),
            "struct_perturb_NPI": round(scorr(P, trueC), 4)}


def main():
    out = {}
    # ---- synthetic (known W) ----
    worms, names, gt = make_synthetic_worms(n_worms=60, N=48, T=400, seed=0)
    Xw, Xt, Vel, D, wid = build(worms)
    out["synthetic"] = evaluate(Xw, Xt, D, wid >= 45, 48, np.abs(gt["neuron_coupling"]))
    print("synthetic:", out["synthetic"])

    # ---- real (Cook connectome) ----
    from stage2_cc import build_core
    fullw, core, names, A = build_core()
    N = len(core)
    Xw2, Xt2, D2, wid2 = [], [], [], []
    for k, w in enumerate(fullw):
        a = w.activity[:, core].astype(np.float32)
        for t in range(L, len(a) - 1):
            Xw2.append(a[t - L:t]); Xt2.append(a[t - 1]); D2.append(a[t] - a[t - 1]); wid2.append(k)
    Xw2 = np.asarray(Xw2); Xt2 = np.asarray(Xt2); D2 = np.asarray(D2); wid2 = np.asarray(wid2)
    te2 = wid2 >= (len(fullw) - max(1, len(fullw) // 4))
    out["real"] = evaluate(Xw2, Xt2, D2, te2, N, np.abs(A))
    out["real"]["reference_learned_coupling"] = 0.334
    print("real:", out["real"])

    with open(os.path.join(os.path.dirname(__file__), "perturb_model.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
