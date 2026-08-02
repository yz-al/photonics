"""
CC-LVM-style test on REAL data: does CONSTRAINING the coupling to the real
connectome improve prediction? We learn a per-neuron linear dynamics with a
coupling matrix whose support is fixed by a mask, and compare masks:

  none        - no coupling (per-neuron autoregression baseline)
  dense       - unconstrained (learn all off-diagonal connections)
  connectome  - weights ONLY on real Cook 2019 synapses (the CC-LVM constraint)
  shuffled    - same sparsity as connectome but PERMUTED wiring (control:
                isolates whether the *real* circuit matters vs just sparsity)

Model (learned by gradient descent, held-out worms):
  delta_i = a_i*x_i + b_i*vel_i + sum_j W_ij x_j + sum_j V_ij vel_j
  with W,V free only where mask=1.

The decisive comparison is connectome vs shuffled.
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_real_worms   # noqa: E402
from stage2_real import load_connectome   # noqa: E402


def r2(Y, Yh):
    ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / st).mean())


def build_core(max_worms=500):
    worms, canon = load_real_worms(max_worms=max_worms, resample_T=256)
    nodes, A_full = load_connectome()
    node_idx = {n: i for i, n in enumerate(nodes)}
    present = np.zeros(len(canon))
    for w in worms:
        present += w.present
    common = [n for j, n in enumerate(canon) if n in node_idx and present[j] >= 0.2 * len(worms)]
    ci = [canon.index(n) for n in common]
    order = np.argsort(-present[ci])
    for K in (40, 30, 20):
        cand = [ci[j] for j in order[:K]]
        full = [w for w in worms if w.present[cand].all()]
        if len(full) >= 12:
            core, names, fullw = cand, [canon[j] for j in cand], full
            break
    A = A_full[np.ix_([node_idx[canon[j]] for j in core], [node_idx[canon[j]] for j in core])]
    return fullw, core, names, A


def make_masks(A, seed=0):
    N = A.shape[0]
    eye = np.eye(N, dtype=bool)
    conn = (A > 0) & ~eye
    dense = ~eye
    rng = np.random.default_rng(seed)
    # shuffled: permute neuron labels -> same #edges, scrambled wiring
    perm = rng.permutation(N)
    shuf = conn[perm][:, perm] & ~eye
    return {"none": np.zeros_like(conn), "dense": dense,
            "connectome": conn, "shuffled": shuf}


class CCModel(nn.Module):
    def __init__(self, mask):
        super().__init__()
        N = mask.shape[0]
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.a = nn.Parameter(torch.zeros(N)); self.b = nn.Parameter(torch.zeros(N))
        self.W = nn.Parameter(torch.zeros(N, N)); self.V = nn.Parameter(torch.zeros(N, N))

    def forward(self, xt, vel):
        W = self.W * self.mask; Vv = self.V * self.mask
        return self.a * xt + self.b * vel + xt @ W.T + vel @ Vv.T


def fit_eval(mask, xt_tr, vel_tr, D_tr, xt_te, vel_te, D_te, epochs=600, seed=0):
    torch.manual_seed(seed)
    m = CCModel(mask)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(epochs):
        pred = m(xt_tr, vel_tr)
        loss = ((pred - D_tr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return r2(D_te.numpy(), m(xt_te, vel_te).numpy())


def main():
    fullw, core, names, A = build_core()
    N = len(core)
    # build delta pairs split by worm
    Xt, Vel, D, wid = [], [], [], []
    for k, w in enumerate(fullw):
        a = w.activity[:, core].astype(np.float32)
        Xt.append(a[1:-1]); Vel.append(a[1:-1] - a[:-2]); D.append(a[2:] - a[1:-1])
        wid.append(np.full(len(a) - 2, k))
    Xt = np.concatenate(Xt); Vel = np.concatenate(Vel); D = np.concatenate(D)
    wid = np.concatenate(wid)
    te = wid >= (len(fullw) - max(1, len(fullw) // 4))
    to_t = lambda z: torch.tensor(z, dtype=torch.float32)
    args = (to_t(Xt[~te]), to_t(Vel[~te]), to_t(D[~te]),
            to_t(Xt[te]), to_t(Vel[te]), to_t(D[te]))

    masks = make_masks(A)
    res = {"n_core_neurons": N, "n_worms": len(fullw),
           "n_connectome_edges": int(masks["connectome"].sum())}
    for name, mk in masks.items():
        res[name] = round(fit_eval(mk, *args), 4)
        print(f"  {name:12s} pred_r2={res[name]:+.4f}  (edges={int(mk.sum())})")
    out = os.path.join(os.path.dirname(__file__), "stage2_cc.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
