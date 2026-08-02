"""
Testing the AlphaGo/self-play hypothesis on the WORM connectome, in the
STRONG-PERCEPTION regime (graph topology, not weak activity).

Perception = topological link-prediction on the Cook connectome graph (connectomes
are highly structured, so held-out edges ARE predictable from the rest of the
wiring). No images, no activity -- just the real connectome graph.

Then a PROOFREADER (fixer): given a corrupted graph, predict each edge's true
value from topology features recomputed on the corrupted graph. Train it two ways:
  random     - corruptions are random edges
  self-play  - corruptions come from an ADAPTIVE ADVERSARY that targets the
               current fixer's mistakes (a competitive curriculum, iterated)
Evaluate both on held-out RANDOM and held-out HARD (adversarial) corruptions.

Question: does competitive/self-play training beat single-agent when perception
is strong? (The regime where I argued AlphaGo-style RL should help.)
"""
import os
import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def topo(G):
    deg = G.sum(1).astype(float)
    CN = G @ G
    il = np.zeros_like(deg); m = deg > 1; il[m] = 1.0 / np.log(deg[m])
    AA = G @ (il[:, None] * G)
    RA = G @ ((1.0 / np.maximum(deg, 1))[:, None] * G)
    return CN, AA, RA, deg


def feats(G, pairs):
    CN, AA, RA, deg = topo(G)
    i, j = pairs[:, 0], pairs[:, 1]
    return np.stack([CN[i, j], AA[i, j], RA[i, j], deg[i] * deg[j], G[i, j]], 1)


def main():
    d = np.load(os.path.join(HERE, "data", "cook2019_connectome.npz"), allow_pickle=True)
    Aall = ((np.abs(d["CS"]) + np.abs(d["CS"].T) + np.abs(d["GJ"]) + np.abs(d["GJ"].T)) > 0).astype(int)
    np.fill_diagonal(Aall, 0)
    keep = np.argsort(-Aall.sum(1))[:150]                 # top-150 by degree
    A = Aall[np.ix_(keep, keep)]; N = len(keep)
    iu, ju = np.triu_indices(N, 1)
    allp = np.stack([iu, ju], 1); ytrue = A[iu, ju]
    dens = ytrue.mean()

    # ---- perception gate: topological link prediction (held-out edges) ----
    perm = rng.permutation(len(allp)); cut = int(0.7 * len(perm))
    tr, te = perm[:cut], perm[cut:]
    Gtr = A.copy()
    for k in te[ytrue[te] == 1]:
        a, b = allp[k]; Gtr[a, b] = Gtr[b, a] = 0
    gate = GradientBoostingClassifier().fit(feats(Gtr, allp[tr]), ytrue[tr])
    gate_auc = roc_auc_score(ytrue[te], gate.predict_proba(feats(Gtr, allp[te]))[:, 1])

    # ---- corruption + training-set builders ----
    def corrupt(nflip, adversary=None):
        G = A.copy()
        if adversary is None:
            idx = rng.choice(len(allp), nflip, replace=False)
        else:
            p = adversary.predict_proba(feats(A, allp))[:, 1]
            err = np.abs(p - ytrue)                         # where the fixer is wrong on truth
            cand = np.argsort(-err)[:nflip * 4]
            idx = rng.choice(cand, nflip, replace=False)
        for k in idx:
            a, b = allp[k]; G[a, b] = G[b, a] = 1 - G[a, b]
        return G, idx

    def train_set(nsamp, nflip, adversary=None):
        X, Y = [], []
        for _ in range(nsamp):
            G, idx = corrupt(nflip, adversary)
            un = rng.choice(len(allp), len(idx), replace=False)
            pk = np.concatenate([idx, un])
            X.append(feats(G, allp[pk])); Y.append(ytrue[pk])
        return np.concatenate(X), np.concatenate(Y)

    NF = 200
    Xr, Yr = train_set(30, NF)
    fixer_random = GradientBoostingClassifier().fit(Xr, Yr)

    # self-play: iterate adversary <-> fixer
    fixer_comp = GradientBoostingClassifier().fit(Xr, Yr)
    for _ in range(3):
        Xc, Yc = train_set(30, NF, adversary=fixer_comp)
        fixer_comp = GradientBoostingClassifier().fit(
            np.concatenate([Xr, Xc]), np.concatenate([Yr, Yc]))

    # ---- evaluate: reconstruct true graph from corrupted, F1 over all pairs ----
    def evalf(fixer, adversary):
        f1 = []
        for _ in range(20):
            G, _ = corrupt(NF, adversary)
            pred = (fixer.predict_proba(feats(G, allp))[:, 1] > 0.5).astype(int)
            f1.append(f1_score(ytrue, pred))
        return float(np.mean(f1))

    # common hard-test adversary = the random-trained fixer (fixed reference for fairness)
    res = {
        "n_nodes": N, "edge_density": round(float(dens), 4),
        "perception_gate_AUC_topology": round(float(gate_auc), 4),
        "random_test": {
            "fixer_random": round(evalf(fixer_random, None), 4),
            "fixer_selfplay": round(evalf(fixer_comp, None), 4)},
        "hard_test_vs_ref": {
            "fixer_random": round(evalf(fixer_random, fixer_random), 4),
            "fixer_selfplay": round(evalf(fixer_comp, fixer_random), 4)},
    }
    print(json.dumps(res, indent=2))
    with open(os.path.join(HERE, "stage2_selfplay.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
