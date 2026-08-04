"""
Convert the trained segmentation model's boundary decision into an explicit MECHANISTIC
EQUATION over interpretable IMAGE PRIMITIVES (not opaque NN features), so it can be
manipulated / perturbed / carried to mouse+human EM.

Method = SINDy-style sparse symbolic regression: build a library from interpretable local
image features (intensity, gradient magnitude, local variance, multi-scale blur, high-pass)
+ their squares and pairwise products, then GREEDY-sparse-select a few terms that predict
the membrane (short-range boundary) decision. Report the equation and -- the whole point --
its FIDELITY: accuracy vs GT membranes and agreement with the U-Net's own decision. The gap
to the U-Net is exactly the diffuse, non-compressible part the DBB flagged.
"""
import numpy as np


def _logfit(X, y, l2=1e-3, iters=500, lr=0.3):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w); b -= lr * (p - y).mean()
    return w, b


def _pred(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))


def _dev(X, y, w, b):
    p = _pred(X, w, b)
    return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))


def _auc(p, y):                                            # rank-based AUC (imbalance-robust)
    y = y > 0.5; n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = p.argsort().argsort() + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _bal_acc(p, y):
    yp = p > 0.5; y = y > 0.5
    tpr = (yp & y).sum() / max(1, y.sum()); tnr = (~yp & ~y).sum() / max(1, (~y).sum())
    return float(0.5 * (tpr + tnr))


def run(prims, names, y_gt, y_unet, max_terms=6, min_gain=0.003):
    """prims: (N,P) primitives; names: P names; y_gt: membrane label; y_unet: U-Net's membrane
    decision (for distillation fidelity)."""
    P = prims.shape[1]
    lib, lname = [], []
    for i in range(P):
        lib.append(prims[:, i]); lname.append(names[i])
    for i in range(P):
        lib.append(prims[:, i] ** 2); lname.append(f"{names[i]}^2")
    for i in range(P):
        for j in range(i + 1, P):
            lib.append(prims[:, i] * prims[:, j]); lname.append(f"{names[i]}*{names[j]}")
    L = np.stack(lib, 1); Ls = (L - L.mean(0)) / (L.std(0) + 1e-6)

    chosen = []
    _, b0 = _logfit(np.zeros((len(y_gt), 1)), y_gt); cur = _dev(np.zeros((len(y_gt), 1)), y_gt, np.zeros(1), b0)
    while len(chosen) < max_terms:                          # greedy: pick the term that cuts DEVIANCE most
        best = None
        for k in range(L.shape[1]):
            if k in chosen:
                continue
            cols = chosen + [k]; w, b = _logfit(Ls[:, cols], y_gt); dv = _dev(Ls[:, cols], y_gt, w, b)
            if best is None or dv < best[0]:
                best = (dv, k, w, b)
        if cur - best[0] < min_gain:                        # stop at the L-curve corner
            break
        cur = best[0]; chosen.append(best[1])
    w, b = _logfit(Ls[:, chosen], y_gt) if chosen else (np.zeros(0), 0.0)
    p = _pred(Ls[:, chosen], w, b) if chosen else np.full(len(y_gt), float((y_gt > 0.5).mean()))
    terms = [f"{w[i]:+.2f}*{lname[chosen[i]]}" for i in range(len(chosen))]
    eq = "P(membrane) = sigma( " + f"{b:+.2f} " + " ".join(terms) + " )"
    up = (y_unet > 0.5).astype(float) if y_unet is not None else None
    return {"equation": eq, "n_terms": len(chosen), "terms_by_importance": [lname[c] for c in chosen],
            "coeffs": [round(float(x), 3) for x in w], "intercept": round(float(b), 3),
            "library_size": L.shape[1], "membrane_prevalence": round(float((y_gt > 0.5).mean()), 4),
            "eq_AUC_vs_GT": round(_auc(p, y_gt), 4), "eq_balanced_acc_vs_GT": round(_bal_acc(p, y_gt), 4),
            "unet_AUC_vs_GT": round(_auc(up, y_gt), 4) if up is not None else None,
            "unet_balanced_acc_vs_GT": round(_bal_acc(up, y_gt), 4) if up is not None else None,
            "distillation_agreement_with_unet": round(float((( p > 0.5) == (up > 0.5)).mean()), 4) if up is not None else None,
            "fidelity_note": ("compare eq_AUC to unet_AUC: the equation captures that much of the boundary "
                              "decision; the gap to the U-Net is the diffuse, non-compressible part the DBB flagged")}
