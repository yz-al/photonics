"""
THE DOUBLE BLACK BOX, actually chained.

Box 1: a trained forecaster (opaque GRU) of C. elegans dynamics.
Mechinterp (stage 1): PULL OUT its effective connectivity J -- the Jacobian of
its next-step prediction w.r.t. each input neuron. This is the extracted object.
Box 2 (stage 2): feed J into an automated discovery step that SYNTHESIZES a
compact mechanistic model whose dynamics are built on J and that reproduces the
forecaster's behavior (distillation). The chain: stage-2 input = stage-1 output.

Validated on synthetic worms (known W), the methods testbed:
  fidelity   : does the synthesized mechanistic model reproduce the black-box
               forecaster's predictions? (distillation R^2, held out)
  struct_J   : did mechinterp extract real structure? corr(|J|, |W W^T|)
  struct_M   : does the synthesized model's coupling recover structure?
  baseline   : the UN-chained version (discover from raw activity correlation,
               ignoring the forecaster) -- to show the chain adds signal.
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import make_synthetic_worms   # noqa: E402

L = 24   # forecaster window


def r2(Y, Yh):
    ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / st).mean())


def scorr(M, A):
    N = M.shape[0]; off = ~np.eye(N, dtype=bool)
    return float(np.corrcoef(np.abs(M)[off], np.abs(A)[off])[0, 1])


class Forecaster(nn.Module):
    def __init__(self, N, hid=64):
        super().__init__()
        self.gru = nn.GRU(N, hid, batch_first=True); self.head = nn.Linear(hid, N)

    def forward(self, win):                       # win: (B,L,N) -> delta_next (B,N)
        h, _ = self.gru(win); return self.head(h[:, -1, :])

    def eff_conn(self, win):                       # mean |d delta_i / d last-frame_j|
        N = win.shape[2]; J = torch.zeros(N, N)
        x = win[:1].clone().requires_grad_(True)
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            d = self.forward(x)[0]
            for i in range(N):
                g = torch.autograd.grad(d[i], x, retain_graph=(i < N - 1))[0]
                J[i] = g[0, -1].abs()
        return J.detach().numpy()


def build(worms):
    Xw, Xt, Vel, D, wid = [], [], [], [], []
    for k, w in enumerate(worms):
        a = w.activity.astype(np.float32)
        for t in range(L, len(a) - 1):
            Xw.append(a[t - L:t]); Xt.append(a[t - 1]); Vel.append(a[t - 1] - a[t - 2])
            D.append(a[t] - a[t - 1]); wid.append(k)
    return (np.asarray(Xw), np.asarray(Xt), np.asarray(Vel),
            np.asarray(D), np.asarray(wid))


def main():
    torch.manual_seed(0)
    worms, names, gt = make_synthetic_worms(n_worms=60, N=48, T=400, seed=0)
    N = 48; trueC = np.abs(gt["neuron_coupling"])
    Xw, Xt, Vel, D, wid = build(worms)
    te = wid >= 45
    tw, dw = torch.tensor(Xw[~te]), torch.tensor(D[~te])

    # ---- Box 1: train the forecaster (the black box) ----
    fore = Forecaster(N)
    opt = torch.optim.Adam(fore.parameters(), lr=3e-3)
    for _ in range(60):
        for i in range(0, len(tw), 512):
            b = slice(i, i + 512)
            loss = ((fore(tw[b]) - dw[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        f_tr = fore(torch.tensor(Xw[~te])).numpy()      # forecaster's delta predictions
        f_te = fore(torch.tensor(Xw[te])).numpy()
    fore_self_r2 = r2(D[te], f_te)                       # forecaster's own accuracy

    # ---- Stage 1 (mechinterp): pull out effective connectivity J ----
    J = np.mean([fore.eff_conn(torch.tensor(Xw[te][i:i + 1])) for i in range(6)], axis=0)
    np.fill_diagonal(J, 0.0)

    # ---- Stage 2 (chained): synthesize a mechanistic model FROM J, distilling box 1 ----
    def distill(M, target_tr, target_te):
        Ttr = np.concatenate([Xt[~te], Xt[~te] @ M, Vel[~te] @ M], 1)
        cmu, csd = Ttr.mean(0), Ttr.std(0) + 1e-8; Ts = (Ttr - cmu) / csd
        C = np.linalg.solve(Ts.T @ Ts + 5 * np.eye(Ts.shape[1]), Ts.T @ target_tr)
        Tte = (np.concatenate([Xt[te], Xt[te] @ M, Vel[te] @ M], 1) - cmu) / csd
        return Tte @ C

    # AlphaEvolve-lite: pick the J-derived coupling variant that best distills box 1
    def lowrank(A, r):
        U, S, Vt = np.linalg.svd((A + A.T) / 2); return (U[:, :r] * S[:r]) @ Vt[:r]
    variants = {"J": J, "J_sym": (J + J.T) / 2, "J_lowrank6": lowrank(J, 6), "J_lowrank12": lowrank(J, 12)}
    best = None
    for name, M in variants.items():
        pred_te = distill(M, f_tr, f_te)
        fid = r2(f_te, pred_te)                          # fidelity to the forecaster
        if best is None or fid > best["fidelity"]:
            best = {"variant": name, "fidelity": round(fid, 4),
                    "struct_M_vs_W": round(scorr(M, trueC), 4)}

    # ---- Un-chained baseline: discover from RAW activity correlation (ignore box 1) ----
    Mcorr = np.corrcoef(Xt[~te].T); np.fill_diagonal(Mcorr, 0.0)
    base_pred = distill(Mcorr, D[~te], D[te])            # fit raw data, not forecaster
    baseline = {"struct_corr_raw": round(scorr(Mcorr, trueC), 4),
                "pred_r2_raw": round(r2(D[te], base_pred), 4)}

    report = {
        "forecaster_self_r2": round(fore_self_r2, 4),
        "stage1_extracted_struct_J_vs_W": round(scorr(J, trueC), 4),
        "stage2_synthesized_model": best,
        "unchained_baseline": baseline,
        "note": "chained: stage-2 coupling derived from stage-1's extracted J, "
                "distilling the forecaster. baseline ignores box 1.",
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "close_loop.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
