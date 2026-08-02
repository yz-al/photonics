"""
Double black box, STAGE 2: automated mechanistic-model discovery.

Stage 1 gave us a black-box dynamics model (the forecaster) + extracted effective
connectivity. Stage 2 synthesizes a *mechanistic* model of those dynamics and
scores it against ground truth.

Testbed: the synthetic worm system (data.make_synthetic_worms) has a KNOWN
coupling matrix W, so every discovered model is scored two ways:
  * pred_r2   -- held-out R^2 predicting the state change  d = x[t+1] - x[t]
                 from x[t]  (predicting the delta, so the trivial "return x"
                 identity scores 0 -- you must recover real dynamics).
  * struct_corr -- correlation of the model's effective linear operator (fit
                 A: predict(X) ~= X @ A) against the true neuron coupling
                 |W @ W.T| (off-diagonal). This is connectome recovery.

Candidate contract (the unit every method + the LLM produce), an AlphaEvolve-style
"program that fits then predicts":

    def build(X_train, D_train):
        # fit whatever you like on the training pairs
        def predict(X):        # (B, N) -> (B, N) predicted delta
            ...
        return predict

Methods here (all pure numpy, no deps): zero baseline, linear ridge, SINDy
(sparse nonlinear regression), and a non-LLM random/evolutionary search. The
LLM-guided method (Claude as the mutation operator) evaluates candidate programs
written to stage2/candidates/ via eval_program() -- that loop is driven by the
agent (or by llm_evolve.py if an ANTHROPIC_API_KEY is available).
"""
from __future__ import annotations

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import make_synthetic_worms   # noqa: E402


# --------------------------------------------------------------------------- #
# Problem: (x[t] -> delta) pairs from the synthetic worms, split by worm
# --------------------------------------------------------------------------- #
def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def get_problem(n_worms=60, N=48, T=512, seed=0, lags=1):
    """
    Build (state -> delta) pairs. lags=1 gives a single frame x[t] as input
    (the hard, phase-ambiguous target). lags=k stacks [x[t], x[t-1], ..., x[t-k+1]]
    so the input carries velocity/history -- the fair operating point for stage 2.
    The prediction target is always the delta x[t+1]-x[t]; N stays the output width.
    """
    worms, names, gt = make_synthetic_worms(n_worms=n_worms, N=N, T=T, seed=seed)
    Xs, Ds, wid = [], [], []
    for i, w in enumerate(worms):
        a = w.activity
        for t in range(lags - 1, len(a) - 1):
            Xs.append(np.concatenate([a[t - j] for j in range(lags)]))   # [x_t, x_{t-1}, ...]
            Ds.append(a[t + 1] - a[t]); wid.append(i)
    X = np.asarray(Xs, dtype=np.float64)
    D = np.asarray(Ds, dtype=np.float64)
    wid = np.asarray(wid)
    n_test_worms = max(1, n_worms // 4)
    test_worms = set(range(n_worms - n_test_worms, n_worms))
    te = np.array([w in test_worms for w in wid])
    coupling = np.abs(gt["neuron_coupling"]).astype(np.float64)
    return {
        "Xtr": X[~te], "Dtr": D[~te], "Xte": X[te], "Dte": D[te],
        "coupling": coupling, "N": N, "lags": lags, "gt": gt,
    }


def _r2(Y, Yhat):
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss_res / ss_tot).mean())


def effective_operator(predict, Xref):
    """Fit A minimizing ||predict(Xref) - Xref @ A|| -> (N,N) effective operator."""
    D = predict(Xref)
    A, *_ = np.linalg.lstsq(Xref, D, rcond=None)
    return A


def struct_corr(A, coupling):
    N = A.shape[0]
    off = ~np.eye(N, dtype=bool)
    a = np.abs(A)[off]
    c = coupling[off]
    if a.std() < 1e-12 or c.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, c)[0, 1])


def evaluate(build, prob, time_budget=20.0):
    """Run a candidate build()->predict on the problem; return metrics (robust)."""
    t0 = time.time()
    try:
        predict = build(prob["Xtr"].copy(), prob["Dtr"].copy())
        Dhat = np.asarray(predict(prob["Xte"].copy()), dtype=np.float64)
        if Dhat.shape != prob["Dte"].shape or not np.all(np.isfinite(Dhat)):
            return {"ok": False, "reason": "bad output shape/nan", "pred_r2": -9.9}
        r2 = _r2(prob["Dte"], Dhat)
        A = effective_operator(predict, prob["Xte"])
        # with lags the operator is (lags*N, N); the instantaneous coupling lives
        # in the x[t] block (first N rows).
        sc = struct_corr(A[: prob["N"]], prob["coupling"])
        return {"ok": True, "pred_r2": r2, "struct_corr": sc,
                "seconds": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "pred_r2": -9.9}


# --------------------------------------------------------------------------- #
# Methods (each is a build function or a factory returning one)
# --------------------------------------------------------------------------- #
def build_zero(Xtr, Dtr):
    N = Xtr.shape[1]
    return lambda X: np.zeros((X.shape[0], N))


def make_linear(ridge=1.0):
    def build(Xtr, Dtr):
        N = Xtr.shape[1]
        A = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(N), Xtr.T @ Dtr)
        return lambda X: X @ A
    return build


def _stlsq(Theta, Y, thresh, iters=10, ridge=1e-2):
    P = Theta.shape[1]
    Xi = np.linalg.solve(Theta.T @ Theta + ridge * np.eye(P), Theta.T @ Y)
    for _ in range(iters):
        small = np.abs(Xi) < thresh
        Xi[small] = 0.0
        for j in range(Y.shape[1]):
            big = ~small[:, j]
            if big.any():
                Tb = Theta[:, big]
                Xi[big, j] = np.linalg.solve(
                    Tb.T @ Tb + ridge * np.eye(big.sum()), Tb.T @ Y[:, j])
    return Xi


def make_sindy(thresh=0.02):
    """Library: [1, x, softplus(x), x^2] (elementwise) -> sparse regression."""
    def build(Xtr, Dtr):
        def lib(X):
            return np.concatenate(
                [np.ones((X.shape[0], 1)), X, softplus(X), X ** 2], axis=1)
        Xi = _stlsq(lib(Xtr), Dtr, thresh=thresh)
        return lambda X: lib(X) @ Xi
    return build


def make_random_search(iters=400, seed=0):
    """
    Non-LLM automated search: evolve a sparse elementwise-nonlinear model. Each
    genome picks, per feature block (x, softplus(x), tanh(x), x^2), a random
    sparsity mask; coefficients are least-squares fit; keep the best on a
    held-out slice of TRAIN. This is the 'dumb evolution' baseline for the LLM.
    """
    def build(Xtr, Dtr):
        rng = np.random.default_rng(seed)
        N = Xtr.shape[1]
        cut = int(0.8 * len(Xtr))
        Xa, Da, Xv, Dv = Xtr[:cut], Dtr[:cut], Xtr[cut:], Dtr[cut:]

        def feats(X):
            return [X, softplus(X), np.tanh(X), X ** 2]

        Fa, Fv = feats(Xa), feats(Xv)
        best_score, best = -1e9, None
        for _ in range(iters):
            masks = [rng.random(N) < rng.uniform(0.05, 0.5) for _ in range(4)]
            cols = [Fa[b][:, masks[b]] for b in range(4) if masks[b].any()]
            colv = [Fv[b][:, masks[b]] for b in range(4) if masks[b].any()]
            if not cols:
                continue
            Ta = np.concatenate(cols, axis=1)
            Tv = np.concatenate(colv, axis=1)
            C = np.linalg.solve(Ta.T @ Ta + 1e-2 * np.eye(Ta.shape[1]), Ta.T @ Da)
            s = _r2(Dv, Tv @ C)
            if s > best_score:
                best_score, best = s, (masks, C)
        masks, C = best

        def predict(X):
            F = feats(X)
            T = np.concatenate([F[b][:, masks[b]] for b in range(4) if masks[b].any()], axis=1)
            return T @ C
        return predict
    return build


# --------------------------------------------------------------------------- #
# LLM-guided (Claude as operator): evaluate a candidate program file/string
# --------------------------------------------------------------------------- #
SAFE_GLOBALS = {"np": np, "softplus": softplus, "__builtins__": {
    "range": range, "len": len, "float": float, "int": int, "abs": abs,
    "min": min, "max": max, "sum": sum, "enumerate": enumerate, "zip": zip,
}}


def load_build_from_code(code: str):
    ns = dict(SAFE_GLOBALS)
    exec(compile(code, "<candidate>", "exec"), ns)
    if "build" not in ns:
        raise ValueError("candidate must define build(X_train, D_train)")
    return ns["build"]


def eval_program(path_or_code, prob=None):
    prob = prob or get_problem()
    code = path_or_code
    if os.path.exists(path_or_code):
        with open(path_or_code) as f:
            code = f.read()
    build = load_build_from_code(code)
    return evaluate(build, prob)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_baselines(prob=None):
    prob = prob or get_problem()
    methods = {
        "zero": build_zero,
        "linear_ridge": make_linear(),
        "sindy": make_sindy(),
        "random_evo": make_random_search(),
    }
    board = {}
    for name, build in methods.items():
        board[name] = evaluate(build, prob)
        print(f"[stage2] {name:14s} {board[name]}")
    return prob, board


def run_candidates(prob):
    """Evaluate every LLM-authored program in candidates/ (Claude-as-operator)."""
    here = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.join(here, "candidates")
    board = {}
    if os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".py"):
                board[fn[:-3]] = eval_program(os.path.join(cdir, fn), prob)
                print(f"[stage2] {fn[:-3]:22s} {board[fn[:-3]]}")
    return board


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.json")
    prob, baselines = run_baselines()
    candidates = run_candidates(prob)
    everyone = {**{f"baseline:{k}": v for k, v in baselines.items()},
                **{f"claude:{k}": v for k, v in candidates.items()}}
    ranked = sorted((v for v in everyone.items() if v[1].get("ok")),
                    key=lambda kv: kv[1]["pred_r2"], reverse=True)
    report = {
        "problem": {"N": prob["N"], "n_train": int(len(prob["Xtr"])),
                    "n_test": int(len(prob["Xte"])),
                    "target": "delta x[t+1]-x[t] from x[t]; struct vs |W@W.T|"},
        "baselines": baselines,
        "claude_candidates": candidates,
        "ranking_by_pred_r2": [{"model": k, "pred_r2": round(v["pred_r2"], 4),
                                "struct_corr": round(v.get("struct_corr", 0), 4)}
                               for k, v in ranked],
    }
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[stage2] wrote {out}")
