"""
rigor.py -- statistical hardening of the connectome claims. Turns "mean connectome >
mean shuffled" into a defensible inferential statement: a PAIRED per-unit test, a
bootstrap 95% CI on the effect, a standardized effect size, MULTIPLE-COMPARISON
correction across the gates, and a hyperparameter ROBUSTNESS sweep (not cherry-picked).

Why this module
---------------
The benchmarks each report a point estimate with a shuffled control. A reviewer's
first three questions are: (1) is the gap significant per-unit, not just on the mean?
(2) how big is it, with a confidence interval? (3) does it survive testing several
claims at once, and does it hold away from the one hyperparameter you happened to
show? This module answers all three for the three CONNECTOME gates -- the scientific
core, where the "structure recovered" claim lives -- recomputing each at the level of
its natural experimental unit:

  L1  imputation   unit = NEURON        connectome vs shuffled-wiring, paired by neuron
  L2  perturbation unit = PERTURBATION  (I-gA)^-1 vs shuffled operator, paired by stim
  L2* external     unit = ANIMAL        frozen connectome vs shuffled, paired by worm
                                        (Flavell 000776 -- never used to build the model)

For each: one-sided paired Wilcoxon (connectome > shuffled), a bootstrap 95% CI on the
mean paired difference, Cohen's d_z, then Holm-Bonferroni across the three gates. Plus
a robustness grid per gate showing the connectome-minus-shuffled gap stays positive at
every hyperparameter setting -- the "not cherry-picked" evidence.

    python rigor.py     # writes rigor.json
Reuses the exact loaders/operators of l2_imputation_benchmark, perturbation_response,
external_validation -- same data, same models, just the inference done properly.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.RandomState(0)


# ---- generic paired inference ------------------------------------------------
def paired_stats(con, sh, n_boot=5000):
    """con, sh: aligned per-unit scores. Returns paired Wilcoxon p (con>sh), the mean
    paired difference with a bootstrap 95% CI, Cohen's d_z, and n."""
    from scipy.stats import wilcoxon
    con, sh = np.asarray(con, float), np.asarray(sh, float)
    d = con - sh
    n = len(d)
    try:
        p = float(wilcoxon(con, sh, alternative="greater")[1])
    except Exception:
        p = float("nan")
    boot = np.array([d[RNG.randint(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dz = float(d.mean() / (d.std(ddof=1) + 1e-12))
    return {"n_units": n, "connectome_mean": round(float(con.mean()), 4),
            "shuffled_mean": round(float(sh.mean()), 4),
            "mean_diff": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "cohens_dz": round(dz, 3), "wilcoxon_p": p,
            "ci_excludes_zero": bool(lo > 0)}


def holm(pvals):
    """Holm-Bonferroni: return per-gate adjusted p and reject@0.05."""
    idx = np.argsort(pvals); m = len(pvals); adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return [round(a, 5) for a in adj], [bool(a < 0.05) for a in adj]


# ---- L1: held-out-neuron imputation, paired by neuron ------------------------
def _impute_keyed(R, meas, Wmat, min_cols=8):
    from scipy.stats import pearsonr
    N = R.shape[0]; Wc = Wmat + Wmat.T; out = {}
    for i in range(N):
        partners = np.where(Wc[i] > 0)[0]; partners = partners[partners != i]
        if len(partners) < 2:
            continue
        w = Wc[i, partners]; w = w / w.sum()
        cols = np.where(meas[i])[0]
        if len(cols) < min_cols:
            continue
        pred = np.zeros(len(cols)); num = np.zeros(len(cols))
        for pp, wp in zip(partners, w):
            rp = R[pp, cols]; f = np.isfinite(rp); pred[f] += wp * rp[f]; num[f] += wp
        ok = num > 0
        if ok.sum() < min_cols:
            continue
        pv = pred[ok] / num[ok]; av = R[i, cols][ok]
        if np.std(pv) < 1e-9 or np.std(av) < 1e-9:
            continue
        out[i] = float(pearsonr(pv, av)[0])
    return out


def gate_imputation(n_shuffle=5, min_cols=8):
    import l2_imputation_benchmark as L2I
    R, occ, W, ceil, r_sh = L2I.build_data()
    meas = (occ >= 4) & np.isfinite(R)
    con = _impute_keyed(R, meas, W, min_cols)
    sh_dicts = [_impute_keyed(R, meas, W[np.ix_(p, p)], min_cols)
                for p in (np.random.RandomState(s).permutation(R.shape[0]) for s in range(n_shuffle))]
    keys = [i for i in con if all(i in d for d in sh_dicts)]
    c = [con[i] for i in keys]; s = [np.mean([d[i] for d in sh_dicts]) for i in keys]
    return paired_stats(c, s), (R, meas, W)


def robustness_imputation(ctx):
    R, meas, W = ctx
    grid = {}
    for mc in (6, 8, 12):
        con = _impute_keyed(R, meas, W, mc)
        sh = _impute_keyed(R, meas, W[np.ix_(np.random.RandomState(1).permutation(R.shape[0]),
                                             np.random.RandomState(1).permutation(R.shape[0]))], mc)
        keys = [i for i in con if i in sh]
        grid["min_cols=%d" % mc] = round(float(np.mean([con[i] for i in keys]) -
                                               np.mean([sh[i] for i in keys])), 4)
    return grid


# ---- L2: held-out perturbation, paired by perturbation -----------------------
def _score_keyed(P, R, meas, N):
    from scipy.stats import pearsonr
    out = {}
    for j in range(N):
        ii = np.where(meas[:, j])[0]; ii = ii[ii != j]
        if len(ii) < 10:
            continue
        if np.std(P[ii, j]) > 1e-9 and np.std(R[ii, j]) > 1e-9:
            out[j] = float(pearsonr(P[ii, j], R[ii, j])[0])
    return out


def gate_perturbation(n_shuffle=5, g=0.7):
    import perturbation_response as PR
    R, meas, Wc, Wg, ceil = PR.load(); N = PR.N
    con = _score_keyed(PR.dynamical_propagation(Wc, Wg, g), R, meas, N)
    sh_dicts = []
    for s in range(n_shuffle):
        p = np.random.RandomState(s).permutation(N)
        Psh = PR.dynamical_propagation(Wc[np.ix_(p, p)], Wg[np.ix_(p, p)], g)
        sh_dicts.append(_score_keyed(Psh, R, meas, N))
    keys = [j for j in con if all(j in d for d in sh_dicts)]
    c = [con[j] for j in keys]; svals = [np.mean([d[j] for d in sh_dicts]) for j in keys]
    return paired_stats(c, svals), (R, meas, Wc, Wg, N)


def robustness_perturbation(ctx):
    import perturbation_response as PR
    R, meas, Wc, Wg, N = ctx
    grid = {}
    for g in (0.5, 0.6, 0.7, 0.8):
        con = _score_keyed(PR.dynamical_propagation(Wc, Wg, g), R, meas, N)
        p = np.random.RandomState(1).permutation(N)
        sh = _score_keyed(PR.dynamical_propagation(Wc[np.ix_(p, p)], Wg[np.ix_(p, p)], g), R, meas, N)
        keys = [j for j in con if j in sh]
        grid["g=%.1f" % g] = round(float(np.mean([con[j] for j in keys]) -
                                        np.mean([sh[j] for j in keys])), 4)
    return grid


# ---- L2*: external Flavell, paired by animal ---------------------------------
def _external_per_animal(train_frac=0.6, seed=0):
    import external_validation as EV
    d = np.load(EV.DATA, allow_pickle=True)
    wsym = EV.load_connectome(); rng = np.random.RandomState(seed)
    con, sh = [], []
    for X, names in zip(d["acts"], d["names"]):
        names = [str(n) for n in names]; X = np.asarray(X, float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        cut = int(len(X) * train_frac); Xtr, Xte = X[:cut], X[cut:]
        if Xte.shape[0] < 30 or X.shape[1] < 15:
            continue
        c = EV._impute_connectome(Xtr, Xte, names, wsym)
        perm = rng.permutation(len(names)); sn = [names[p] for p in perm]
        s = EV._impute_connectome(Xtr, Xte, sn, wsym)
        if c and s:
            con.append(np.mean(c)); sh.append(np.mean(s))
    return con, sh


def gate_external():
    c, s = _external_per_animal()
    return paired_stats(c, s), None


def robustness_external(_):
    grid = {}
    for tf in (0.5, 0.6, 0.7):
        c, s = _external_per_animal(train_frac=tf)
        grid["train_frac=%.1f" % tf] = round(float(np.mean(c) - np.mean(s)), 4)
    return grid


def run():
    gates = []
    order = [
        ("L1 imputation (per neuron)", gate_imputation, robustness_imputation),
        ("L2 perturbation (per stim)", gate_perturbation, robustness_perturbation),
        ("L2* external Flavell (per animal)", gate_external, robustness_external),
    ]
    for name, gatefn, robfn in order:
        stat, ctx = gatefn()
        stat["gate"] = name
        stat["robustness_connectome_minus_shuffled"] = robfn(ctx)
        stat["robust_all_positive"] = bool(all(v > 0 for v in stat["robustness_connectome_minus_shuffled"].values()))
        gates.append(stat)
    padj, reject = holm([g["wilcoxon_p"] for g in gates])
    for g, pa, rj in zip(gates, padj, reject):
        g["holm_adj_p"] = pa; g["survives_holm"] = rj
    return {
        "method": "paired per-unit Wilcoxon (connectome>shuffled) + bootstrap 95% CI + Cohen's dz, "
                  "Holm-Bonferroni across gates, + hyperparameter robustness grid",
        "gates": gates,
        "all_gates_survive_holm": bool(all(g["survives_holm"] for g in gates)),
        "all_gates_robust": bool(all(g["robust_all_positive"] for g in gates)),
    }


if __name__ == "__main__":
    r = run()
    print("=" * 82)
    print("  STATISTICAL RIGOR -- connectome gates, paired per-unit inference")
    print("=" * 82)
    for g in r["gates"]:
        print("\n%s   (n=%d units)" % (g["gate"], g["n_units"]))
        print("   connectome %.3f  vs  shuffled %.3f   |  paired diff %.3f  95%% CI [%.3f, %.3f]"
              % (g["connectome_mean"], g["shuffled_mean"], g["mean_diff"], g["ci95"][0], g["ci95"][1]))
        print("   Cohen's dz = %.2f   Wilcoxon p = %.1e   Holm-adj p = %.1e   survives Holm: %s"
              % (g["cohens_dz"], g["wilcoxon_p"], g["holm_adj_p"], g["survives_holm"]))
        print("   robustness (connectome-shuffled at each setting): %s  -> all positive: %s"
              % (g["robustness_connectome_minus_shuffled"], g["robust_all_positive"]))
    print("\n" + "=" * 82)
    print("  all gates survive Holm correction : %s" % r["all_gates_survive_holm"])
    print("  all gates robust to hyperparams   : %s" % r["all_gates_robust"])
    print("=" * 82)
    with open(os.path.join(HERE, "rigor.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote rigor.json")
