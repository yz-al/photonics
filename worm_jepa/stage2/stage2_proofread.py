"""
RL CONNECTOME PROOFREADER (prototype). Precursor to a human-connectome proofreader:
same architecture, swap the organism.

Setup: most of the Cook connectome is trusted (train edges); a held-out region is
CORRUPTED with injected errors (removed true edges = splits, added false edges =
merges). A policy proofreads the corrupted region using:
  - an ACTIVITY value function (GBM edge-score from neural activity, the gate model)
  - GRAPH CONTEXT (node degrees, common neighbors on the current candidate graph)
trained with REINFORCE to maximize error correction on the held-out region.

Baselines: corrupted (before), greedy (trust the value function), RL policy, and a
SHUFFLED-VALUE control (activity signal destroyed -> should not beat corrupted).
Metric: edge-F1 on the held-out region before vs after proofreading.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_real_worms   # noqa: E402
from stage2_real import load_connectome   # noqa: E402
from stage2_linkpred import lagged_xcorr   # noqa: E402


def f1(pred, true):
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)


def graph_context(G, ii, jj):
    outdeg = G.sum(1); indeg = G.sum(0)
    common = (G @ G)                       # shared neighbors
    return np.stack([outdeg[ii], indeg[jj], common[ii, jj]], axis=1).astype(np.float32)


def main():
    torch.manual_seed(0)
    worms, canon = load_real_worms(max_worms=int(os.environ.get("STAGE2_MAX_WORMS", "500")),
                                   resample_T=256)
    nodes, A_full = load_connectome()
    node_idx = {n: i for i, n in enumerate(nodes)}
    present = np.zeros(len(canon))
    for w in worms:
        present += w.present
    common = [n for j, n in enumerate(canon) if n in node_idx and present[j] >= 0.2 * len(worms)]
    ci = [canon.index(n) for n in common]
    N = len(common)
    A = (A_full[np.ix_([node_idx[n] for n in common], [node_idx[n] for n in common])] > 0).astype(int)

    blocks = []
    for w in worms:
        a = w.activity[:, ci].astype(np.float64); a[:, ~w.present[ci].astype(bool)] = np.nan
        blocks.append(a)
    df = pd.DataFrame(np.concatenate(blocks), columns=common)
    corr = np.nan_to_num(df.corr().values); cov = np.nan_to_num(df.cov().values)
    P = np.linalg.inv(cov + 1e-2 * np.eye(N)); dd = np.sqrt(np.abs(np.diag(P)))
    partial = np.nan_to_num(-P / (np.outer(dd, dd) + 1e-9))
    lag = lagged_xcorr(worms, ci, N, ci)

    ii, jj = np.where(~np.eye(N, dtype=bool))
    feats = np.stack([corr[ii, jj], np.abs(corr)[ii, jj], cov[ii, jj], partial[ii, jj], lag[ii, jj]], 1)
    y = A[ii, jj]
    rng = np.random.default_rng(0); perm = rng.permutation(len(y))
    cut = int(0.7 * len(y)); tr, te = perm[:cut], perm[cut:]

    # activity value function trained on trusted (train) edges
    gbm = GradientBoostingClassifier().fit(feats[tr], y[tr])
    vscore = gbm.predict_proba(feats)[:, 1]                 # per-pair activity edge score

    te_ii, te_jj, te_y, te_v = ii[te], jj[te], y[te], vscore[te]
    # trusted graph = truth on train pairs, unknown (to be filled) on test pairs
    known = np.zeros((N, N)); known[ii[tr], jj[tr]] = A[ii[tr], jj[tr]]

    def corrupt(rate=0.5):
        c = te_y.copy()
        flip = rng.random(len(c)) < rate
        c[flip] = 1 - c[flip]
        return c

    def policy_state(cand, vscore_te):
        G = known.copy(); G[te_ii, te_jj] = cand
        ctx = graph_context(G, te_ii, te_jj)
        s = np.stack([vscore_te, cand.astype(np.float32), (vscore_te - cand)], axis=1)
        s = np.concatenate([s, ctx / (ctx.std(0) + 1e-6)], axis=1)
        return torch.tensor(s, dtype=torch.float32)

    pol = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)

    def run_policy(vscore_te, episodes=250, train=True):
        best = []
        for _ in range(episodes):
            cand = corrupt()
            st = policy_state(cand, vscore_te)
            logits = pol(st).squeeze(1)
            p = torch.sigmoid(logits)
            act = (torch.bernoulli(p) if train else (p > 0.5).float()).detach()
            newc = np.where(act.numpy().astype(bool), 1 - cand, cand)
            reward = f1(newc, te_y) - f1(cand, te_y)        # optimize F1 directly (sparse graph)
            if train:
                logp = (act * torch.log(p + 1e-8) + (1 - act) * torch.log(1 - p + 1e-8)).sum()
                loss = -reward * logp
                opt.zero_grad(); loss.backward(); opt.step()
            best.append(f1(newc, te_y))
        return float(np.mean(best[-50:]))

    # baselines on a fixed set of corruptions
    corrs = [corrupt() for _ in range(50)]
    corrupted_f1 = float(np.mean([f1(c, te_y) for c in corrs]))
    greedy_f1 = float(np.mean([f1((te_v > 0.5).astype(int), te_y) for _ in corrs]))  # trust value fn

    rl_f1 = run_policy(te_v, train=True)
    rl_eval = run_policy(te_v, episodes=50, train=False)
    # shuffled-value control: destroy activity signal
    vsh = te_v.copy(); rng.shuffle(vsh)
    for g in pol:                                          # reset policy
        if isinstance(g, nn.Linear):
            g.reset_parameters()
    opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
    rl_shuffled = run_policy(vsh, train=True)

    res = {"n_neurons": N, "n_test_edges": int(len(te_y)),
           "test_edge_density": round(float(te_y.mean()), 4),
           "corrupt_rate": 0.5,
           "f1_corrupted_before": round(corrupted_f1, 4),
           "f1_greedy_valuefn": round(greedy_f1, 4),
           "f1_rl_policy": round(rl_eval, 4),
           "f1_rl_shuffled_value_CONTROL": round(rl_shuffled, 4)}
    print(json.dumps(res, indent=2))
    with open(os.path.join(os.path.dirname(__file__), "stage2_proofread.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
