"""
The decisive CC-LVM-style test on REAL data: HELD-OUT NEURON IMPUTATION.

For each target neuron i, predict its activity from OTHER neurons' contemporaneous
activity, but restrict the predictors to a set S_i:
  connectome : i's real Cook synaptic neighbors
  shuffled   : the same NUMBER of random other neurons (control)
  dense      : all other neurons (upper bound)

Fit per-neuron ridge on train worms, evaluate R^2 on held-out worms, average over
neurons. If connectome > shuffled, i's real wiring carries functional signal that
random neurons don't -- the real test of whether the connectome helps.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2_cc import build_core   # noqa: E402


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    pred = Xte @ w + ytr.mean()
    ss = ((yte - pred) ** 2).sum(); st = ((yte - yte.mean()) ** 2).sum() + 1e-9
    return 1 - ss / st


def main():
    fullw, core, names, A = build_core()
    N = len(core)
    conn = (A > 0) & ~np.eye(N, dtype=bool)
    rng = np.random.default_rng(0)

    # per-worm activity over core; split worms train/test
    acts = [w.activity[:, core].astype(np.float64) for w in fullw]
    n_te = max(1, len(fullw) // 4)
    tr = np.concatenate(acts[:-n_te]); te = np.concatenate(acts[-n_te:])

    per = {"connectome": [], "shuffled": [], "dense": []}
    used = 0
    for i in range(N):
        nbrs = np.where(conn[i])[0]
        if len(nbrs) < 2:
            continue
        used += 1
        others = np.array([j for j in range(N) if j != i])
        shuf = rng.choice(others, size=len(nbrs), replace=False)
        for name, cols in [("connectome", nbrs), ("shuffled", shuf), ("dense", others)]:
            per[name].append(ridge_r2(tr[:, cols], tr[:, i], te[:, cols], te[:, i]))

    res = {"n_core_neurons": N, "n_worms": len(fullw),
           "n_target_neurons": used,
           "mean_degree": round(float(conn.sum(1)[conn.sum(1) >= 2].mean()), 2)}
    for name in per:
        arr = np.array(per[name])
        res[f"{name}_meanR2"] = round(float(arr.mean()), 4)
        res[f"{name}_medianR2"] = round(float(np.median(arr)), 4)
    # paired advantage of real wiring over the shuffled control
    diff = np.array(per["connectome"]) - np.array(per["shuffled"])
    res["connectome_minus_shuffled_mean"] = round(float(diff.mean()), 4)
    res["connectome_wins_fraction"] = round(float((diff > 0).mean()), 3)

    out = os.path.join(os.path.dirname(__file__), "stage2_heldout_neuron.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
