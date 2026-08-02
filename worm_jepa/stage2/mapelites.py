"""
Full-recipe AlphaEvolve (minus raw scale): a MAP-Elites quality-diversity program
database, multi-objective (pred_r2 x struct_corr), evolving structure-aware
programs. Goal: push connectome recovery (struct_corr) up without collapsing
prediction.

Faithful components (vs the DeepMind/OpenEvolve/CodeEvolve recipe):
  * Program database = MAP-Elites grid over (pred_r2, struct_corr) -> keeps a
    DIVERSE set of trade-offs, one elite per cell (no single-best collapse).
  * Multi-objective: cells are the two metrics; we don't scalarize the search.
  * Inspiration + parent selection: offspring are mutations of archived elites.
  * "Diff" mutation: a genome (feature/coupling switches) is copied and 1-2 genes
    are changed -- a targeted edit, not a full rewrite.
  * Seeds encode the LLM-operator insight: a data-driven neuron-coupling term
    (correlation of activity, which reflects shared latents) used as message
    passing -- this is what makes the effective operator carry real structure.

No LLM API needed to RUN it (the mutation operator is programmatic over a rich
structure-aware program space); llm_evolve.py is the API-backed variant.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discover                                  # noqa: E402
from discover import softplus                    # noqa: E402


# --------------------------------------------------------------------------- #
# Program space (genome -> build function). Input is temporal: [x_t,x_{t-1},x_{t-2}]
# --------------------------------------------------------------------------- #
GENES = {
    "vel": [False, True],
    "acc": [False, True],
    "softplus": [False, True],
    "tanh_vel": [False, True],
    "square": [False, True],
    "coupling": ["none", "corr", "cov", "corr_thresh", "lowrank"],
    "k": [4, 6, 8, 12],                 # rank for lowrank coupling
    "thresh_p": [0.1, 0.2, 0.4],        # keep top-fraction per row for corr_thresh
    "couple_vel": [False, True],        # also pass velocity through the coupling
    "ridge": [0.3, 1.0, 3.0, 10.0, 30.0],
}


def _coupling_matrix(xt, kind, k, thresh_p):
    N = xt.shape[1]
    if kind == "none":
        return None
    if kind == "cov":
        M = np.cov(xt.T)
    else:
        M = np.corrcoef(xt.T)
    M = np.nan_to_num(M)
    np.fill_diagonal(M, 0.0)
    if kind == "lowrank":
        U, S, Vt = np.linalg.svd(M)
        M = (U[:, :k] * S[:k]) @ Vt[:k]
        np.fill_diagonal(M, 0.0)
    if kind == "corr_thresh":
        keep = max(1, int(thresh_p * N))
        Mt = np.zeros_like(M)
        for i in range(N):
            idx = np.argsort(-np.abs(M[i]))[:keep]
            Mt[i, idx] = M[i, idx]
        M = Mt
    return M


def build_from_genome(g):
    def build(Xtr, Dtr):
        N = Dtr.shape[1]
        xt_tr = Xtr[:, :N]
        M = _coupling_matrix(xt_tr, g["coupling"], g["k"], g["thresh_p"])

        def raw(X):
            xt, xt1, xt2 = X[:, :N], X[:, N:2 * N], X[:, 2 * N:3 * N]
            vel = xt - xt1
            acc = xt - 2 * xt1 + xt2
            cols = [xt]
            if g["vel"]:
                cols.append(vel)
            if g["acc"]:
                cols.append(acc)
            if g["softplus"]:
                cols.append(softplus(xt))
            if g["tanh_vel"]:
                cols.append(np.tanh(vel))
            if g["square"]:
                cols.append(xt ** 2)
            if M is not None:
                cols.append(xt @ M.T)          # message passing: coupled-neuron activity
                if g["couple_vel"]:
                    cols.append(vel @ M.T)
            return np.concatenate(cols, axis=1)

        T = raw(Xtr)
        cmu, csd = T.mean(0), T.std(0) + 1e-8
        Ts = (T - cmu) / csd
        C = np.linalg.solve(Ts.T @ Ts + g["ridge"] * np.eye(Ts.shape[1]), Ts.T @ Dtr)

        def predict(X):
            return ((raw(X) - cmu) / csd) @ C
        return predict
    return build


# --------------------------------------------------------------------------- #
# MAP-Elites
# --------------------------------------------------------------------------- #
def _rand_genome(rng):
    return {k: v[rng.integers(len(v))] for k, v in GENES.items()}


def _mutate(g, rng, n=2):
    g = dict(g)
    for _ in range(rng.integers(1, n + 1)):
        key = list(GENES)[rng.integers(len(GENES))]
        g[key] = GENES[key][rng.integers(len(GENES[key]))]
    return g


def _cell(m, bins=10):
    pr = min(bins - 1, max(0, int(m["pred_r2"] * bins)))         # pred_r2 in [0,1]
    sc = min(bins - 1, max(0, int(m.get("struct_corr", 0) / 0.8 * bins)))  # [0,0.8]
    return (pr, sc)


def run(iters=350, seed=0):
    rng = np.random.default_rng(seed)
    prob = discover.get_problem(lags=3)          # fair temporal input
    archive = {}                                 # cell -> (genome, metrics)

    def consider(g):
        m = discover.evaluate(build_from_genome(g), prob)
        if not m.get("ok"):
            return
        c = _cell(m)
        cur = archive.get(c)
        score = m["struct_corr"] + 0.1 * m["pred_r2"]      # tie-break toward structure
        if cur is None or score > cur[1]["struct_corr"] + 0.1 * cur[1]["pred_r2"]:
            archive[c] = (g, m)

    # seeds: the LLM-operator insight (coupling terms) + a plain temporal model
    seeds = [
        {"vel": True, "acc": True, "softplus": True, "tanh_vel": False, "square": False,
         "coupling": "corr", "k": 6, "thresh_p": 0.2, "couple_vel": True, "ridge": 3.0},
        {"vel": True, "acc": False, "softplus": False, "tanh_vel": False, "square": False,
         "coupling": "corr_thresh", "k": 6, "thresh_p": 0.2, "couple_vel": False, "ridge": 3.0},
        {"vel": True, "acc": True, "softplus": True, "tanh_vel": True, "square": True,
         "coupling": "none", "k": 6, "thresh_p": 0.2, "couple_vel": False, "ridge": 3.0},
    ]
    for g in seeds:
        consider(g)
    for _ in range(25):
        consider(_rand_genome(rng))

    for it in range(iters):
        parents = [v[0] for v in archive.values()]
        g = _mutate(parents[rng.integers(len(parents))], rng)
        consider(g)
        if (it + 1) % 50 == 0:
            best_s = max(archive.values(), key=lambda v: v[1]["struct_corr"])
            print(f"[mapelites] it={it+1:4d} cells={len(archive):3d}  "
                  f"best_struct={best_s[1]['struct_corr']:.3f} "
                  f"(pred={best_s[1]['pred_r2']:.3f})", flush=True)

    elites = sorted(archive.values(), key=lambda v: v[1]["struct_corr"], reverse=True)
    best_struct = elites[0]
    valid = [v for v in archive.values() if v[1]["pred_r2"] > 0.6]
    best_struct_valid = max(valid, key=lambda v: v[1]["struct_corr"]) if valid else None
    best_pred = max(archive.values(), key=lambda v: v[1]["pred_r2"])

    report = {
        "iters": iters, "archive_cells": len(archive),
        "baseline_struct_corr": 0.248,   # best from the earlier temporal run
        "best_struct": {"genome": best_struct[0], "metrics": best_struct[1]},
        "best_struct_pred_gt_0.6": (
            {"genome": best_struct_valid[0], "metrics": best_struct_valid[1]}
            if best_struct_valid else None),
        "best_pred": {"genome": best_pred[0], "metrics": best_pred[1]},
        "top5_by_struct": [{"genome": g, "metrics": m} for g, m in elites[:5]],
    }
    out = os.path.join(os.path.dirname(__file__), "mapelites.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[mapelites] best struct_corr = {best_struct[1]['struct_corr']:.3f} "
          f"(pred_r2={best_struct[1]['pred_r2']:.3f})")
    if best_struct_valid:
        print(f"[mapelites] best struct with pred>0.6 = "
              f"{best_struct_valid[1]['struct_corr']:.3f} "
              f"(pred={best_struct_valid[1]['pred_r2']:.3f})")
    print(f"[mapelites] wrote {out}")
    return report


if __name__ == "__main__":
    run()
