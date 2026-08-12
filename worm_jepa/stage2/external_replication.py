"""
external_replication.py -- does the out-of-distribution result REPLICATE? Run the
identical frozen-connectome test on a SECOND independent whole-brain dataset the model
never saw, and report both cohorts side by side.

Why
---
One out-of-distribution win (external_validation.py, Flavell 000776) could be a quirk
of one lab's preparation. A regulator -- and a referee -- wants REPLICATION on an
independent cohort collected differently. This module runs the same held-out-neuron
imputation (frozen anatomical connectome, zero parameters fit to the test data,
train/test temporal split) on:

  cohort A  Flavell/Atanas 2023   (DANDI:000776)  spontaneous, freely-moving,  ~85 neu
  cohort B  chemosensory 000981   (DANDI:000981)  odor-stimulated, NeuroPAL,   ~200 neu

Different lab, different microscope, different behavioral regime (spontaneous vs
odor-driven), different labeling method. The connectome is the SAME frozen wiring for
both. The claim REPLICATES only if connectome >> shuffled in BOTH cohorts, per animal.

Result (see external_replication.json): connectome beats the shuffled-wiring floor in
both cohorts, each with a paired per-animal Wilcoxon and a bootstrap 95% CI on the
gap. Two independent out-of-distribution datasets, same frozen model, same direction.

    python external_replication.py     # writes external_replication.json
Reuses external_validation's operator/imputer verbatim; self-contained (committed npz).
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
COHORTS = [
    ("Flavell 000776 (spontaneous)", os.path.join(HERE, "external_flavell.npz")),
    ("chemosensory 000981 (odor)",   os.path.join(HERE, "external_chemo.npz")),
]


def _cohort(path, train_frac=0.6, seed=0):
    """Per-animal connectome vs shuffled-wiring imputation r on one dataset."""
    import external_validation as EV
    from scipy.stats import wilcoxon
    d = np.load(path, allow_pickle=True)
    wsym = EV.load_connectome(); rng = np.random.RandomState(seed)
    con, sh = [], []
    for X, names in zip(d["acts"], d["names"]):
        names = [str(n) for n in names]; X = np.asarray(X, float)
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        good = np.isfinite(sd) & (sd > 1e-9)              # drop dead / all-NaN neurons
        X = np.nan_to_num((X[:, good] - mu[good]) / (sd[good] + 1e-9))
        names = [n for n, g in zip(names, good) if g]
        cut = int(len(X) * train_frac); Xtr, Xte = X[:cut], X[cut:]
        if Xte.shape[0] < 30 or X.shape[1] < 15:
            continue
        c = [x for x in EV._impute_connectome(Xtr, Xte, names, wsym) if np.isfinite(x)]
        perm = rng.permutation(len(names))
        s = [x for x in EV._impute_connectome(Xtr, Xte, [names[p] for p in perm], wsym) if np.isfinite(x)]
        if c and s:
            con.append(float(np.mean(c))); sh.append(float(np.mean(s)))
    con, sh = np.array(con), np.array(sh)
    d_arr = con - sh
    try:
        p = float(wilcoxon(con, sh, alternative="greater")[1])
    except Exception:
        p = float("nan")
    boot = np.array([d_arr[rng.randint(0, len(d_arr), len(d_arr))].mean() for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_animals": len(con),
            "connectome_r": round(float(con.mean()), 4),
            "shuffled_r": round(float(sh.mean()), 4),
            "mean_gap": round(float(d_arr.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "wilcoxon_p": p,
            "beats_shuffle": bool(con.mean() > sh.mean() + 0.02 and p < 0.01 and lo > 0)}


def run():
    cohorts = []
    for name, path in COHORTS:
        if not os.path.exists(path):
            cohorts.append({"cohort": name, "error": "missing %s" % os.path.basename(path)}); continue
        r = _cohort(path); r["cohort"] = name; cohorts.append(r)
    ok = [c for c in cohorts if "error" not in c]
    return {
        "task": "held-out-neuron imputation, frozen anatomical connectome, 0 params fit to test data",
        "cohorts": cohorts,
        "n_datasets": len(ok),
        "replicates": bool(len(ok) >= 2 and all(c["beats_shuffle"] for c in ok)),
    }


if __name__ == "__main__":
    r = run()
    print("=" * 84)
    print("  OUT-OF-DISTRIBUTION REPLICATION -- one frozen connectome, two unseen cohorts")
    print("=" * 84)
    for c in r["cohorts"]:
        if "error" in c:
            print("\n%s : %s" % (c["cohort"], c["error"])); continue
        print("\n%s   (n=%d animals)" % (c["cohort"], c["n_animals"]))
        print("   connectome r=%.3f  vs  shuffled r=%.3f   |  gap %.3f  95%% CI [%.3f, %.3f]  p=%.1e"
              % (c["connectome_r"], c["shuffled_r"], c["mean_gap"], c["ci95"][0], c["ci95"][1], c["wilcoxon_p"]))
        print("   beats shuffled floor: %s" % c["beats_shuffle"])
    print("\n" + "=" * 84)
    print("  REPLICATES across %d independent out-of-distribution datasets: %s"
          % (r["n_datasets"], r["replicates"]))
    print("=" * 84)
    with open(os.path.join(HERE, "external_replication.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote external_replication.json")
