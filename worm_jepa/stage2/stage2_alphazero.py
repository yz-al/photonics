"""
AlphaZero-style SINGLE-agent self-play connectome builder on the worm (GPU).

One policy-value network (no committee). Self-play building episodes: sequentially
commit edges; committed edges reshape the topology the agent sees next (the
construction dynamic). Reward = edge correctness vs ground-truth Cook (the "rules").
Sampled REINFORCE with a value baseline, entropy bonus, reward normalization, a
difficulty curriculum, and a sparsity prior on the edge-logit bias.

FINDING (why this loses to a static link predictor, and why more training doesn't
help): the construction dynamic POISONS its own features. Committing act=1 raises
the degree of the involved nodes, but the pair pool is ~95% non-edges, so early
commits are mostly WRONG (reward -1). High constructed-degree thus becomes a marker
of the agent's own false positives -- anti-correlated with reward -- so REINFORCE
correctly learns to SUPPRESS edges at high-degree nodes. That is exactly backwards
from the real connectome (hubs have more edges: degree AUC ~0.60). The policy
converges to this inverted fixed point (flat test AUC ~0.36, corr(logit,deg)~-0.58),
so longer training converges TO it, not out of it. The static one-shot predictor
wins precisely because it never commits, so its features never invert. Topology-only
features cap ~0.58-0.60 here; beating that needs signal BEYOND topology (perception
/ activity), not a cleverer search over the same features.

Env: WORM_AZ_EPISODES (default 400), WORM_AZ_DEVICE. Logs test AUC each eval step;
compares to a static one-shot link predictor at the end.
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
    # Sparsity prior: initialise the edge-logit bias to the true base rate so the
    # untrained policy commits edges at ~graph density (~5%), not ~50%. Without
    # this, early random commits flood the believed graph with false edges and
    # the topology features the agent learns from are garbage (policy collapses
    # below chance). This is the same prior discipline as METHODOLOGY.md.
    dens = float(y.mean())
    net.pi.bias.data.fill_(float(np.log(dens / (1 - dens))))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    trp_t = torch.tensor(trp, device=DEV)

    def episode(nseed):
        # Proper REINFORCE with a value baseline. Actions are SAMPLED from the
        # policy (Bernoulli over "is this an edge"); we commit the sampled edges
        # into the believed graph (the construction dynamic), reward = correctness
        # vs ground truth, and update with -logp * advantage. (The earlier greedy
        # top-k + BCE-toward-action surrogate was not a valid policy gradient and
        # drove the policy below chance.)
        G, used = seed_graph(nseed)
        undec = torch.tensor([k for k in trp if k not in used], device=DEV)
        R = 6; logp_all, rew_all, val_all, ent_all = [], [], [], []
        for rd in range(R):
            if len(undec) < 20:
                break
            pi, pj = P[undec, 0], P[undec, 1]
            logit, val = net(feats_t(G, pi, pj))
            B = max(1, len(undec) // (R - rd))                 # pairs decided this round
            sel = torch.randperm(len(undec), device=DEV)[:B]   # act on a random subset
            dist = torch.distributions.Bernoulli(logits=logit[sel])
            act = dist.sample()
            yy = y[undec[sel]]
            rew = (act == yy).float() * 2 - 1
            logp_all.append(dist.log_prob(act)); rew_all.append(rew)
            val_all.append(val[sel]); ent_all.append(dist.entropy())
            on = sel[act == 1]                                  # commit sampled edges
            ia, ib = P[undec[on], 0], P[undec[on], 1]; G[ia, ib] = 1; G[ib, ia] = 1
            mask = torch.ones(len(undec), dtype=torch.bool, device=DEV); mask[sel] = False
            undec = undec[mask]
        logp = torch.cat(logp_all); rw = torch.cat(rew_all)
        vv = torch.cat(val_all); ent = torch.cat(ent_all)
        rwn = (rw - rw.mean()) / (rw.std() + 1e-6)
        adv = (rwn - vv).detach()
        loss = -(logp * adv).mean() + 0.5 * ((vv - rwn) ** 2).mean() - 0.01 * ent.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    @torch.no_grad()
    def build_out(nseed):
        # Faithful evaluation of a CONSTRUCTION agent: seed, then greedily commit
        # the edges the policy is confident about over the training pool, so the
        # believed graph reaches the same density regime the agent trained in.
        # Then held-out test pairs are scored on that constructed graph. (Scoring
        # on a bare sparse seed graph is a train/eval mismatch that inverts the
        # one useful feature -- degree -- and drives AUC below chance.)
        G, used = seed_graph(nseed)
        undec = torch.tensor([k for k in trp if k not in used], device=DEV)
        for rd in range(6):
            if len(undec) < 20:
                break
            logit, _ = net(feats_t(G, P[undec, 0], P[undec, 1]))
            on = undec[torch.sigmoid(logit) > 0.5]
            ia, ib = P[on, 0], P[on, 1]; G[ia, ib] = 1; G[ib, ia] = 1
            keepm = torch.sigmoid(logit) <= 0.5
            undec = undec[keepm]
        return G

    @torch.no_grad()
    def eval_auc(nseed=200, reps=3):
        yt = y[te].cpu().numpy(); s = np.zeros(len(yt))
        for _ in range(reps):
            G = build_out(nseed)
            logit, _ = net(feats_t(G, P[te, 0], P[te, 1]))
            s += torch.sigmoid(logit).cpu().numpy()
        return float(roc_auc_score(yt, s / reps))

    EP = int(os.environ.get("WORM_AZ_EPISODES", "400"))
    hist = []
    for ep in range(EP):
        episode(int(np.interp(ep, [0, EP], [400, 60])))
        if (ep + 1) % max(1, EP // 20) == 0:
            hist.append({"ep": ep + 1, "auc": round(eval_auc(), 4)})
            print(f"[az] ep {ep+1}/{EP}  test_auc={hist[-1]['auc']:.4f}", flush=True)

    with torch.no_grad():
        G = build_out(200)
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

    # diagnostic: sign of the policy's dependence on degree vs its true predictive
    # direction (the inverted-feature pathology). Negative corr with positive
    # feature-AUC == the construction dynamic inverted degree.
    with torch.no_grad():
        Xte = feats_t(build_out(200), P[te, 0], P[te, 1])
        lg = net(Xte)[0].cpu().numpy(); degf = Xte[:, 3].cpu().numpy()
    corr_logit_deg = float(np.corrcoef(lg, degf)[0, 1])
    deg_feat_auc = float(roc_auc_score(yt, degf))

    res = {"device": DEV, "episodes": EP, "n_nodes": N,
           "alphazero_selfplay": {"test_AUC": round(az_auc, 4), "test_F1": round(az_f1, 4)},
           "static_linkpred_baseline": {"test_AUC": round(base_auc, 4), "test_F1": round(base_f1, 4)},
           "diagnosis": {"corr_logit_degree": round(corr_logit_deg, 4),
                         "degree_feature_AUC": round(deg_feat_auc, 4),
                         "note": "construction dynamic inverts degree: negative "
                                 "policy-degree corr despite positive degree AUC; "
                                 "converged fixed point, longer training does not help"},
           "learning_curve": hist}
    print(json.dumps(res, indent=2))
    with open(os.path.join(HERE, "stage2_alphazero.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
