"""
AlphaZero-style SINGLE-agent self-play connectome builder on the worm.

One policy-value network (no committee, no diversity -- the same agent, like
AlphaZero). It plays self-play BUILDING episodes: sequentially commit edges into a
graph; committed edges reshape the topology the agent sees next (the construction
dynamic). Reward = edge correctness vs the ground-truth Cook connectome -- which
plays the role Go's rules play (the free win/loss signal, available at TRAIN time
from a known connectome). Trained by actor-critic over episodes with a difficulty
curriculum, then deployed to build a held-out region.

Compared against a static one-shot link predictor (same features, no self-play).
Question: does self-play sequential construction beat the static predictor?
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn

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
    X = np.stack([CN[i, j], AA[i, j], RA[i, j], deg[i] * deg[j]], 1).astype(np.float32)
    return np.log1p(X)                                  # compress heavy tails


class PolicyValue(nn.Module):
    def __init__(self, d=4, h=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.pi = nn.Linear(h, 1); self.v = nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z).squeeze(-1), torch.tanh(self.v(z)).squeeze(-1)


def main():
    d = np.load(os.path.join(HERE, "data", "cook2019_connectome.npz"), allow_pickle=True)
    Aall = ((np.abs(d["CS"]) + np.abs(d["CS"].T) + np.abs(d["GJ"]) + np.abs(d["GJ"].T)) > 0).astype(int)
    np.fill_diagonal(Aall, 0)
    keep = np.where(Aall.sum(1) >= 6)[0]
    A = Aall[np.ix_(keep, keep)]; N = len(keep)
    iu, ju = np.triu_indices(N, 1)
    allp = np.stack([iu, ju], 1); ytrue = A[iu, ju].astype(np.float32)
    perm = rng.permutation(len(allp))
    te = perm[:int(0.25 * len(perm))]; trp = perm[int(0.25 * len(perm)):]
    trainmask = np.zeros(len(allp), bool); trainmask[trp] = True

    def seed_graph(nseed):
        G = np.zeros((N, N))
        pe = trp[ytrue[trp] == 1]
        pick = rng.choice(pe, min(nseed, len(pe)), replace=False)
        for k in pick:
            a, b = allp[k]; G[a, b] = G[b, a] = 1
        return G, set(pick.tolist())

    net = PolicyValue()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    def episode(nseed, train=True, pairs_universe=trp):
        G, used = seed_graph(nseed)
        undec = [int(k) for k in pairs_universe if k not in used]
        R, logs, correct = 6, [], 0
        for rd in range(R):
            if len(undec) < 20:
                break
            pj = np.array(undec)
            X = torch.tensor(feats(G, allp[pj]))
            logit, val = net(X)
            p = torch.sigmoid(logit)
            B = max(1, len(pj) // (2 * (R - rd)))
            inc = torch.topk(p, B).indices                      # commit as EDGES
            exc = torch.topk(-p, B).indices                     # commit as NON-edges
            picks = torch.cat([inc, exc])
            act = torch.cat([torch.ones(B), torch.zeros(B)])
            y = torch.tensor(ytrue[pj[picks.numpy()]])
            rew = (act == y).float() * 2 - 1                     # +1 correct, -1 wrong
            correct += int((act == y).sum())
            if train:
                logs.append((logit[picks], act, rew, val[picks]))
            for k in inc.numpy():                                # construction dynamic
                a, b = allp[pj[k]]; G[a, b] = G[b, a] = 1
            done = set(pj[picks.numpy()].tolist()); undec = [k for k in undec if k not in done]
        if train and logs:
            lo = torch.cat([l[0] for l in logs]); ac = torch.cat([l[1] for l in logs])
            rw = torch.cat([l[2] for l in logs]); vv = torch.cat([l[3] for l in logs])
            adv = (rw - vv).detach()
            bce = nn.functional.binary_cross_entropy_with_logits(lo, ac, reduction="none")
            loss = (bce * adv).mean() + 0.5 * ((vv - rw) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return G

    # ---- self-play training with a curriculum (seed shrinks -> harder) ----
    for ep in range(400):
        nseed = int(np.interp(ep, [0, 400], [400, 60]))
        episode(nseed, train=True)

    # ---- evaluate on held-out test pairs: greedy build, AUC + F1 ----
    def eval_build(nseed):
        G, used = seed_graph(nseed)
        with torch.no_grad():
            logit, _ = net(torch.tensor(feats(G, allp[te])))
            score = torch.sigmoid(logit).numpy()
        from sklearn.metrics import roc_auc_score, f1_score
        auc = roc_auc_score(ytrue[te], score)
        thr = np.quantile(score, 1 - ytrue[te].mean())          # density-matched threshold
        f1 = f1_score(ytrue[te], (score > thr).astype(int))
        return round(float(auc), 4), round(float(f1), 4)

    az_auc, az_f1 = eval_build(200)

    # ---- static one-shot baseline: GBM link predictor, same features/seed ----
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, f1_score
    Gs, used = seed_graph(200)
    seedpairs = np.array([k for k in trp if k not in used])
    clf = GradientBoostingClassifier(n_estimators=80).fit(feats(Gs, allp[seedpairs]), ytrue[seedpairs])
    sc = clf.predict_proba(feats(Gs, allp[te]))[:, 1]
    base_auc = round(float(roc_auc_score(ytrue[te], sc)), 4)
    thr = np.quantile(sc, 1 - ytrue[te].mean())
    base_f1 = round(float(f1_score(ytrue[te], (sc > thr).astype(int))), 4)

    res = {"n_nodes": N, "edge_density": round(float(ytrue.mean()), 4),
           "alphazero_selfplay": {"test_AUC": az_auc, "test_F1": az_f1},
           "static_linkpred_baseline": {"test_AUC": base_auc, "test_F1": base_f1}}
    print(json.dumps(res, indent=2))
    with open(os.path.join(HERE, "stage2_alphazero.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
