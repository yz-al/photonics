"""
STAGE 2 ON REAL DATA. Score functional-coupling connectome recovery + delta
prediction on the real worm dataset (qsimeon/celegans_neural_data), against the
REAL Cook 2019 hermaphrodite connectome (chemical + gap junctions).

Crucial difference from synthetic: the real connectome is a SPARSE DIRECTED
synaptic graph, not a shared-latent Gram. So (a) our synthetic 0.68/0.81 may not
transfer, and (b) partial correlation (a direct-edge tool) may finally help.

Reports, over neurons common to the activity data and the Cook connectome:
  * struct_corr of each coupling estimator (corr / cov / partial) vs the real
    structural connectome  (pairwise-complete correlation handles missing neurons)
  * pred_r2 (held-out worms) of the explicit-coupling delta predictor
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_real_worms   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_connectome():
    d = np.load(os.path.join(HERE, "data", "cook2019_connectome.npz"), allow_pickle=True)
    nodes = [str(n) for n in d["nodes"]]
    A = np.abs(d["CS"]) + np.abs(d["CS"].T) + np.abs(d["GJ"]) + np.abs(d["GJ"].T)  # symmetric total
    return nodes, A


def r2(Y, Yh):
    ss = ((Y - Yh) ** 2).sum(0); st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / st).mean())


def struct_corr(M, A_true):
    off = ~np.eye(M.shape[0], dtype=bool)
    a, c = np.abs(M)[off], A_true[off]
    ok = np.isfinite(a) & np.isfinite(c)
    if a[ok].std() < 1e-12 or c[ok].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], c[ok])[0, 1])


def main():
    max_worms = int(os.environ.get("STAGE2_MAX_WORMS", "500"))
    worms, canon = load_real_worms(max_worms=max_worms, resample_T=256)
    nodes, A_full = load_connectome()
    node_idx = {n: i for i, n in enumerate(nodes)}

    # neurons common to activity + connectome, present in >= 20% of worms
    present_count = np.zeros(len(canon))
    for w in worms:
        present_count += w.present
    common = [n for j, n in enumerate(canon)
              if n in node_idx and present_count[j] >= 0.2 * len(worms)]
    ci = [canon.index(n) for n in common]
    A_true = A_full[np.ix_([node_idx[n] for n in common], [node_idx[n] for n in common])]
    print(f"[real] worms={len(worms)} common neurons (activity ∩ Cook, >=20% present)={len(common)}")

    # activity over common neurons, NaN for absent (pairwise-complete stats)
    blocks = []
    for w in worms:
        a = w.activity[:, ci].astype(np.float64)
        a[:, ~w.present[ci].astype(bool)] = np.nan
        blocks.append(a)
    X = np.concatenate(blocks, axis=0)
    df = pd.DataFrame(X, columns=common)

    corr = df.corr().values
    cov = df.cov().values
    P = np.linalg.inv(np.nan_to_num(cov) + 1e-2 * np.eye(len(common)))
    dd = np.sqrt(np.abs(np.diag(P))); partial = -P / (np.outer(dd, dd) + 1e-9)
    for M in (corr, cov, partial):
        np.fill_diagonal(M, 0.0)

    struct = {
        "n_common_neurons": len(common),
        "n_worms": len(worms),
        "struct_corr_correlation": round(struct_corr(np.nan_to_num(corr), A_true), 4),
        "struct_corr_covariance": round(struct_corr(np.nan_to_num(cov), A_true), 4),
        "struct_corr_partial": round(struct_corr(np.nan_to_num(partial), A_true), 4),
        "connectome": "Cook2019Herm (chemical+gap, symmetric)",
        "reference_synthetic_covariance": 0.684,
        "reference_literature_eff_vs_struct": 0.49,
    }

    # ---- prediction: delta on worms where all common neurons are present ----
    full = [w for w in worms if w.present[ci].all()]
    pred = {"n_full_worms": len(full)}
    if len(full) >= 8:
        Xs, Ds, wid = [], [], []
        for k, w in enumerate(full):
            a = w.activity[:, ci].astype(np.float64)
            for t in range(2, a.shape[0] - 1):
                Xs.append(np.concatenate([a[t], a[t - 1]])); Ds.append(a[t + 1] - a[t]); wid.append(k)
        Xa, Da, wid = np.asarray(Xs), np.asarray(Ds), np.asarray(wid)
        N = len(common)
        te = wid >= (len(full) - max(1, len(full) // 4))
        M = np.nan_to_num(corr)
        def feats(Z):
            xt, xt1 = Z[:, :N], Z[:, N:2 * N]
            return np.concatenate([xt, xt @ M, (xt - xt1) @ M], axis=1)
        Ttr, Tte = feats(Xa[~te]), feats(Xa[te])
        cmu, csd = Ttr.mean(0), Ttr.std(0) + 1e-8
        C = np.linalg.solve(((Ttr - cmu) / csd).T @ ((Ttr - cmu) / csd) + 10 * np.eye(3 * N),
                            ((Ttr - cmu) / csd).T @ Da[~te])
        pred["pred_r2_heldout"] = round(r2(Da[te], ((Tte - cmu) / csd) @ C), 4)
        pred["reference_CC_LVM_real"] = "0.3-0.5"

    report = {"structure": struct, "prediction": pred}
    with open(os.path.join(HERE, "stage2_real.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
