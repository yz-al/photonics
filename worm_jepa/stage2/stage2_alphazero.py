"""
AlphaZero-style SINGLE-agent self-play connectome builder on the worm (GPU).

One policy-value network (no committee). Self-play building episodes: sequentially
commit edges; committed edges reshape the topology the agent sees next (the
construction dynamic). Reward = edge correctness vs ground-truth Cook (the "rules").
Actor-critic with entropy + reward normalization, difficulty curriculum, and MANY
episodes -- RL needs long training, so this is built for the GPU.

Env: WORM_AZ_EPISODES (default 400 smoke; use ~30000+ on GPU), WORM_AZ_DEVICE.
Logs test AUC every eval_every episodes so we can watch it learn. Compares to a
static one-shot link predictor at the end.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
DEV = os.environ.get("WORM_AZ_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def topo_t(G):                                    # G: (N,N) torch on DEV
    deg = G.sum(1)
    CN = G @ G
    il = torch.where(deg > 1, 1.0 / torch.log(deg.clamp(min=2)), torch.zeros_like(deg))
    AA = G @ (il[:, None] * G)
    RA = G @ ((1.0 / deg.clamp(min=1))[:, None] * G)
    return CN, AA, RA, deg


def feats_t(G, pi, pj):
    CN, AA, RA, deg = topo_t(G)
    X = torch.stack([CN[pi, pj], AA[pi, pj], RA[pi, pj], deg[pi] * deg[pj]], 1)
    return torch.log1p(X)


class PolicyValue(nn.Module):
    def __init__(self, d=4, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU())
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
    P = torch.tensor(np.stack([iu, ju], 1), device=DEV)
    y = torch.tensor(A[iu, ju].astype(np.float32), device=DEV)
    n = len(y); perm = rng.permutation(n)
    te = torch.tensor(perm[:int(0.25 * n)], device=DEV)
    trp = perm[int(0.25 * n):]
    Atrue = torch.tensor(A, device=DEV, dtype=torch.float32)
    pos_tr = trp[A[iu, ju][trp] == 1]

    def seed_graph(nseed):
        G = torch.zeros(N, N, device=DEV)
        pick = rng.choice(pos_tr, min(nseed, len(pos_tr)), replace=False)
        pk = torch.tensor(pick, device=DEV)
        a, b = P[pk, 0], P[pk, 1]; G[a, b] = 1; G[b, a] = 1
        return G, set(pick.tolist())

    net = PolicyValue().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    trp_t = torch.tensor(trp, device=DEV)

    def episode(nseed):
        G, used = seed_graph(nseed)
        undec = torch.tensor([k for k in trp if k not in used], device=DEV)
        R = 6; logits_all, act_all, rew_all, val_all = [], [], [], []
        for rd in range(R):
            if len(undec) < 20:
                break
            pi, pj = P[undec, 0], P[undec, 1]
            X = feats_t(G, pi, pj)
            logit, val = net(X)
            p = torch.sigmoid(logit)
            B = max(1, len(undec) // (2 * (R - rd)))
            inc = torch.topk(p, B).indices; exc = torch.topk(-p, B).indices
            picks = torch.cat([inc, exc]); act = torch.cat([torch.ones(B, device=DEV), torch.zeros(B, device=DEV)])
            yy = y[undec[picks]]
            rew = (act == yy).float() * 2 - 1
            logits_all.append(logit[picks]); act_all.append(act); rew_all.append(rew); val_all.append(val[picks])
            ia, ib = P[undec[inc], 0], P[undec[inc], 1]; G[ia, ib] = 1; G[ib, ia] = 1
            mask = torch.ones(len(undec), dtype=torch.bool, device=DEV); mask[picks] = False
            undec = undec[mask]
        lo = torch.cat(logits_all); ac = torch.cat(act_all); rw = torch.cat(rew_all); vv = torch.cat(val_all)
        rwn = (rw - rw.mean()) / (rw.std() + 1e-6)
        adv = (rwn - vv).detach()
        bce = nn.functional.binary_cross_entropy_with_logits(lo, ac, reduction="none")
        ent = -(torch.sigmoid(lo) * nn.functional.logsigmoid(lo) +
                (1 - torch.sigmoid(lo)) * nn.functional.logsigmoid(-lo)).mean()
        loss = (bce * adv).mean() + 0.5 * ((vv - rwn) ** 2).mean() - 0.01 * ent
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def eval_auc(nseed=200):
        G, _ = seed_graph(nseed)
        logit, _ = net(feats_t(G, P[te, 0], P[te, 1]))
        return float(roc_auc_score(y[te].cpu().numpy(), torch.sigmoid(logit).cpu().numpy()))

    EP = int(os.environ.get("WORM_AZ_EPISODES", "400"))
    hist = []
    for ep in range(EP):
        episode(int(np.interp(ep, [0, EP], [400, 60])))
        if (ep + 1) % max(1, EP // 20) == 0:
            hist.append({"ep": ep + 1, "auc": round(eval_auc(), 4)})
            print(f"[az] ep {ep+1}/{EP}  test_auc={hist[-1]['auc']:.4f}", flush=True)

    with torch.no_grad():
        G, _ = seed_graph(200)
        sc = torch.sigmoid(net(feats_t(G, P[te, 0], P[te, 1]))[0]).cpu().numpy()
    yt = y[te].cpu().numpy(); az_auc = float(roc_auc_score(yt, sc))
    thr = np.quantile(sc, 1 - yt.mean()); az_f1 = float(f1_score(yt, (sc > thr).astype(int)))

    # static baseline
    Gs, used = seed_graph(200)
    sp = np.array([k for k in trp if k not in used])
    Gn = Gs.cpu().numpy()
    def fnp(G, pk):
        Gt_ = torch.tensor(G, device=DEV, dtype=torch.float32)
        return feats_t(Gt_, P[pk, 0], P[pk, 1]).cpu().numpy()
    clf = GradientBoostingClassifier(n_estimators=80).fit(fnp(Gn, torch.tensor(sp, device=DEV)), A[iu, ju][sp])
    bs = clf.predict_proba(fnp(Gn, te))[:, 1]
    base_auc = float(roc_auc_score(yt, bs)); thr2 = np.quantile(bs, 1 - yt.mean())
    base_f1 = float(f1_score(yt, (bs > thr2).astype(int)))

    res = {"device": DEV, "episodes": EP, "n_nodes": N,
           "alphazero_selfplay": {"test_AUC": round(az_auc, 4), "test_F1": round(az_f1, 4)},
           "static_linkpred_baseline": {"test_AUC": round(base_auc, 4), "test_F1": round(base_f1, 4)},
           "learning_curve": hist}
    print(json.dumps(res, indent=2))
    with open(os.path.join(HERE, "stage2_alphazero.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
