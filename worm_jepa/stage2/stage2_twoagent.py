"""
Two-agent competitive connectome CONSTRUCTION on the worm (Cook graph).

Your idea: two agents seeded on DIFFERENT regions of the connectome grow it
outward, competing to build, with AGREEMENT -> confidence and DISAGREEMENT ->
active-learning queries. This is query-by-committee active learning applied to
connectome completion, compared head-to-head against a single agent with the
SAME query budget. Only doable where the full connectome is known (the worm), so
we can score recovery against ground truth.

Each agent is a link predictor over topology features of its own BELIEVED graph
(seed region + committed/queried edges); as it commits edges the believed graph
grows and prediction improves (the construction dynamic).

Question: does the competitive two-agent committee recover the connectome better
(held-out AUC) than a single agent at the same labeling budget?
"""
import os
import json
from collections import deque

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def topo(G):
    deg = G.sum(1).astype(float)
    CN = G @ G
    il = np.zeros_like(deg); m = deg > 1; il[m] = 1.0 / np.log(deg[m])
    AA = G @ (il[:, None] * G)
    RA = G @ ((1.0 / np.maximum(deg, 1))[:, None] * G)
    return np.stack([CN, AA, RA], 0), deg


def feat(G, pairs):
    S, deg = topo(G)
    i, j = pairs[:, 0], pairs[:, 1]
    return np.stack([S[0][i, j], S[1][i, j], S[2][i, j], deg[i] * deg[j]], 1)


def regions(A, N):
    """Assign nodes to the nearer of two far-apart seed nodes (BFS distance)."""
    deg = A.sum(1); s1 = int(np.argmax(deg))
    def bfs(s):
        d = np.full(N, 1e9); d[s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in np.where(A[u] > 0)[0]:
                if d[v] > d[u] + 1:
                    d[v] = d[u] + 1; q.append(v)
        return d
    d1 = bfs(s1); s2 = int(np.argmax(d1 * (d1 < 1e8)))
    d2 = bfs(s2)
    return (d1 <= d2)                                  # True -> region 1


def build_believed(N, edge_pairs, edge_labels):
    G = np.zeros((N, N))
    for (a, b), y in zip(edge_pairs, edge_labels):
        if y:
            G[a, b] = G[b, a] = 1
    return G


def fit_predict(G, lab_pairs, lab_y, query_pairs):
    # class-balance: subsample negatives in training
    pos = lab_y == 1; neg = lab_y == 0
    ni = np.where(neg)[0]
    keep = np.concatenate([np.where(pos)[0], rng.choice(ni, min(len(ni), 3 * pos.sum() + 5), replace=False)])
    clf = GradientBoostingClassifier(n_estimators=60, max_depth=3)
    clf.fit(feat(G, lab_pairs[keep]), lab_y[keep])
    return clf.predict_proba(feat(G, query_pairs))[:, 1]


def main():
    d = np.load(os.path.join(HERE, "data", "cook2019_connectome.npz"), allow_pickle=True)
    Aall = ((np.abs(d["CS"]) + np.abs(d["CS"].T) + np.abs(d["GJ"]) + np.abs(d["GJ"].T)) > 0).astype(int)
    np.fill_diagonal(Aall, 0)
    keep = np.where(Aall.sum(1) >= 4)[0]
    A = Aall[np.ix_(keep, keep)]; N = len(keep)
    iu, ju = np.triu_indices(N, 1)
    allp = np.stack([iu, ju], 1); ytrue = A[iu, ju]

    reg = regions(A, N)                                # node -> region 1 (bool)
    p_reg1 = reg[iu] & reg[ju]                         # pairs inside region 1
    p_reg2 = (~reg[iu]) & (~reg[ju])                   # pairs inside region 2

    perm = rng.permutation(len(allp))
    test = perm[:int(0.25 * len(perm))]; testmask = np.zeros(len(allp), bool); testmask[test] = True
    pool = ~testmask                                   # everything else is buildable
    Q_TOTAL = 1500                                     # labeling budget after seeds
    ROUNDS = 5

    def eval_auc(scores):
        return round(float(roc_auc_score(ytrue[test], scores)), 4)

    # ---------- TWO-AGENT committee (different regions) ----------
    seed1 = np.where(p_reg1 & pool)[0]; seed2 = np.where(p_reg2 & pool)[0]
    lab1 = set(seed1.tolist()); lab2 = set(seed2.tolist())
    poolidx = np.where(pool & ~p_reg1 & ~p_reg2)[0].tolist()   # boundary/unknown to discover
    for _ in range(ROUNDS):
        l1 = np.array(sorted(lab1)); l2 = np.array(sorted(lab2))
        G1 = build_believed(N, allp[l1], ytrue[l1]); G2 = build_believed(N, allp[l2], ytrue[l2])
        pj = np.array(poolidx)
        if len(pj) == 0:
            break
        p1 = fit_predict(G1, allp[l1], ytrue[l1], allp[pj])
        p2 = fit_predict(G2, allp[l2], ytrue[l2], allp[pj])
        disagree = np.abs(p1 - p2)
        q = pj[np.argsort(-disagree)[:Q_TOTAL // ROUNDS]]        # query where they DISAGREE
        for k in q:
            lab1.add(int(k)); lab2.add(int(k))
        poolidx = [k for k in poolidx if k not in lab1]
    l1 = np.array(sorted(lab1)); l2 = np.array(sorted(lab2))
    G1 = build_believed(N, allp[l1], ytrue[l1]); G2 = build_believed(N, allp[l2], ytrue[l2])
    s1 = fit_predict(G1, allp[l1], ytrue[l1], allp[test])
    s2 = fit_predict(G2, allp[l2], ytrue[l2], allp[test])
    two_agent_auc = eval_auc((s1 + s2) / 2)
    n_labels_committee = len(lab1 | lab2)

    # ---------- SINGLE agent, same seeds + same budget (uncertainty sampling) ----------
    lab = set(seed1.tolist()) | set(seed2.tolist())
    poolidx = np.where(pool & ~p_reg1 & ~p_reg2)[0].tolist()
    for _ in range(ROUNDS):
        l = np.array(sorted(lab)); G = build_believed(N, allp[l], ytrue[l])
        pj = np.array(poolidx)
        if len(pj) == 0:
            break
        p = fit_predict(G, allp[l], ytrue[l], allp[pj])
        unc = -np.abs(p - 0.5)                          # most uncertain
        q = pj[np.argsort(-unc)[:Q_TOTAL // ROUNDS]]
        for k in q:
            lab.add(int(k))
        poolidx = [k for k in poolidx if k not in lab]
    l = np.array(sorted(lab)); G = build_believed(N, allp[l], ytrue[l])
    single_auc = eval_auc(fit_predict(G, allp[l], ytrue[l], allp[test]))

    res = {"n_nodes": N, "edge_density": round(float(ytrue.mean()), 4),
           "region1_nodes": int(reg.sum()), "region2_nodes": int((~reg).sum()),
           "query_budget": Q_TOTAL, "rounds": ROUNDS,
           "two_agent_committee_test_AUC": two_agent_auc,
           "single_agent_test_AUC": single_auc,
           "committee_total_labels": int(n_labels_committee)}
    print(json.dumps(res, indent=2))
    with open(os.path.join(HERE, "stage2_twoagent.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
