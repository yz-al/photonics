"""
GATE for the RL connectome-proofreader program: is there enough edge signal in
activity to predict the real connectome at all? Supervised link prediction of
Cook 2019 synapses from pairwise activity features, on held-out EDGES, with the
shuffled-connectome negative control.

Pairwise features per neuron pair (i,j):
  pearson, |pearson|, covariance, partial correlation, and LAGGED cross-correlation
  (lag 1-3, the directional/propagation signal that instantaneous corr misses).

If held-out-edge AUC >> 0.5 (and shuffled ~ 0.5), edge signal exists and the RL
proofreader is worth building. This value function is also the proofreader's core.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_real_worms   # noqa: E402
from stage2_real import load_connectome   # noqa: E402


def lagged_xcorr(worms, ci, N, present_idx, lags=(1, 2, 3)):
    """Average |lag cross-correlation| over worms; per-worm on present neurons."""
    S = np.zeros((N, N)); C = np.zeros((N, N))
    for w in worms:
        p = w.present[ci].astype(bool)
        if p.sum() < 2:
            continue
        a = w.activity[:, ci][:, p].astype(np.float64)
        a = (a - a.mean(0)) / (a.std(0) + 1e-8)
        idx = np.where(p)[0]
        best = np.zeros((len(idx), len(idx)))
        for L in lags:
            X = (a[:-L].T @ a[L:]) / (len(a) - L)     # directed lag-L corr
            best = np.maximum(best, np.abs(X))
        S[np.ix_(idx, idx)] += best; C[np.ix_(idx, idx)] += 1
    return S / (C + 1e-9)


def main():
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
    A = A_full[np.ix_([node_idx[n] for n in common], [node_idx[n] for n in common])]
    edge = (A > 0).astype(int)                          # ground-truth Cook edges
    print(f"[linkpred] {N} neurons, {int(edge.sum())} directed edges "
          f"({edge.mean()*100:.1f}% density)")

    # cross-sectional features (pairwise-complete)
    blocks = []
    for w in worms:
        a = w.activity[:, ci].astype(np.float64)
        a[:, ~w.present[ci].astype(bool)] = np.nan
        blocks.append(a)
    df = pd.DataFrame(np.concatenate(blocks), columns=common)
    corr = df.corr().values
    cov = df.cov().values
    P = np.linalg.inv(np.nan_to_num(cov) + 1e-2 * np.eye(N))
    dd = np.sqrt(np.abs(np.diag(P))); partial = -P / (np.outer(dd, dd) + 1e-9)
    lag = lagged_xcorr(worms, ci, N, ci)

    # build pair table (ordered pairs i!=j; edges are directed)
    ii, jj = np.where(~np.eye(N, dtype=bool))
    feats = np.stack([np.nan_to_num(corr)[ii, jj], np.abs(np.nan_to_num(corr))[ii, jj],
                      np.nan_to_num(cov)[ii, jj], np.nan_to_num(partial)[ii, jj],
                      lag[ii, jj]], axis=1)
    y = edge[ii, jj]

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y)); tr, te = perm[:cut], perm[cut:]

    res = {"n_neurons": N, "n_pairs": int(len(y)), "edge_density": round(float(y.mean()), 4)}
    for name, clf in [("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
                      ("gbm", GradientBoostingClassifier())]:
        clf.fit(feats[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(feats[te])[:, 1])
        res[f"{name}_heldout_edge_AUC"] = round(float(auc), 4)
        # shuffled control: permute edge labels -> destroy the wiring signal
        ysh = y.copy(); rng.shuffle(ysh)
        clf.fit(feats[tr], ysh[tr])
        auc_sh = roc_auc_score(ysh[te], clf.predict_proba(feats[te])[:, 1])
        res[f"{name}_shuffled_AUC"] = round(float(auc_sh), 4)
        print(f"  {name}: held-out edge AUC={auc:.3f}  | shuffled control={auc_sh:.3f}")

    out = os.path.join(os.path.dirname(__file__), "stage2_linkpred.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
