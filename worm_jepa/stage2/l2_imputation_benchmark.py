"""
l2_imputation_benchmark.py -- connectome-constrained held-out-neuron imputation,
the Creamer/Leifer/Pillow 2024 benchmark, and the cleanest instantiation of the
double-black-box methodology (structure shows in imputation, connectome >> shuffled).

Reference
---------
Creamer, Leifer & Pillow (2024), "A connectome-constrained model of the C. elegans
brain" (bioRxiv 2024.09.22.614271). Fit a noisy linear dynamical system whose
weight matrix is sparse where the connectome has synapses, to the Randi 2023
signal-propagation data; predict held-out neurons' perturbation-triggered
responses. Reported: connectome model reaches ~92% of the cross-animal
reproducibility (noise ceiling); a shuffled-connectome model with identical
sparsity does much worse.

Why this benchmark matters most
-------------------------------
This is the "data speaking" side of the trap (METHODOLOGY.md): next-step
prediction is persistence-bound and connectome ~ shuffled there, but held-out
imputation is where real structure shows -- connectome >> shuffled. It is the
worked example the whole methodology is built around.

What this does (all from local data: funatlas.h5 via wormneuroatlas + c302 connectome)
--------------------------------------------------------------------------------
- Response matrix R = Randi wt/dFF (300x300, stim j -> response i).
- Noise ceiling from per-trial dFF_all via split-half reliability (Spearman-Brown).
- Anatomical connectome W (c302 SpreadsheetDataReader) aligned to the 300 neurons.
- Held-out-neuron imputation: predict each neuron's response profile as a
  connectome-weighted average of its partners' profiles (1-hop, leak-free), scored
  as a fraction of the noise ceiling; connectome vs shuffled-connectome control.

Result (see l2_imputation_benchmark.json):
  noise ceiling (split-half)      0.217
  connectome model    r=0.183  = 84% of ceiling   (Creamer's fitted LDS: 92%)
  shuffled connectome r=0.041  = 19% of ceiling
  -> connectome >> shuffled by ~65 points of ceiling: STRUCTURE RECOVERED.

This reproduces Creamer's connectome-vs-shuffled result with a simple 1-hop model
(84% of ceiling vs their fitted-LDS 92%). The numerical SOTA-beat at L2 is the
harder PAIR-holdout generalization (beat_sota_l2 / squeeze_l2: ~2.4x the
connectome-linear class); this module is the held-out-NEURON protocol Creamer used.

    python l2_imputation_benchmark.py     # writes l2_imputation_benchmark.json
Requires wormneuroatlas, c302, scipy.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))


def build_data(min_obs=4, seed=0):
    """R (response matrix), occ (obs count), noise ceiling, connectome W, aligned to 300 neurons."""
    import h5py, wormneuroatlas
    from scipy.stats import pearsonr
    rng = np.random.RandomState(seed)
    p = os.path.join(os.path.dirname(wormneuroatlas.__file__), "data", "funatlas.h5")
    with h5py.File(p, "r") as h:
        ids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["neuron_ids"][:]]
        R = h["wt/dFF"][:]; occ = h["wt/occ1"][:]
        dFF_all = h["wt/dFF_all"]
        a, b = [], []
        for i in range(300):
            for j in range(300):
                if occ[i, j] < min_obs:
                    continue
                v = np.array(dFF_all[i, j], float); v = v[np.isfinite(v)]
                if len(v) < min_obs:
                    continue
                rng.shuffle(v); half = len(v) // 2
                a.append(v[:half].mean()); b.append(v[half:].mean())
    r_sh = float(pearsonr(a, b)[0]); ceil = 2 * r_sh / (1 + r_sh)   # Spearman-Brown
    n2i = {n: i for i, n in enumerate(ids)}
    import c302
    _, conns = c302.get_cell_names_and_connection("SpreadsheetDataReader")
    W = np.zeros((300, 300))
    for c in conns:
        if c.syntype == "Send" and c.pre_cell in n2i and c.post_cell in n2i:
            W[n2i[c.pre_cell], n2i[c.post_cell]] += c.number
    return R, occ, W, ceil, r_sh


def impute(R, meas, Wmat, min_cols=8):
    """Predict each neuron's response profile from its connectome partners' profiles
    (1-hop, leak-free: partner != held-out neuron). Return per-neuron correlations."""
    from scipy.stats import pearsonr
    N = R.shape[0]
    Wc = Wmat + Wmat.T
    corrs = []
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
            rp = R[pp, cols]; f = np.isfinite(rp)
            pred[f] += wp * rp[f]; num[f] += wp
        ok = num > 0
        if ok.sum() < min_cols:
            continue
        pv = pred[ok] / num[ok]; av = R[i, cols][ok]
        if np.std(pv) < 1e-9 or np.std(av) < 1e-9:
            continue
        corrs.append(float(pearsonr(pv, av)[0]))
    return np.array(corrs)


def run(seed=0, n_shuffle=5):
    rng = np.random.RandomState(seed)
    R, occ, W, ceil, r_sh = build_data(seed=seed)
    meas = (occ >= 4) & np.isfinite(R)
    real = impute(R, meas, W)
    sh = []
    for _ in range(n_shuffle):
        perm = rng.permutation(R.shape[0])
        sh.append(float(impute(R, meas, W[np.ix_(perm, perm)]).mean()))
    out = {
        "noise_ceiling": round(ceil, 4),
        "split_half_r": round(r_sh, 4),
        "n_neurons_scored": len(real),
        "connectome": {"r": round(float(real.mean()), 4),
                       "pct_of_ceiling": round(float(real.mean()) / ceil * 100, 1)},
        "shuffled": {"r": round(float(np.mean(sh)), 4),
                     "pct_of_ceiling": round(float(np.mean(sh)) / ceil * 100, 1)},
        "reference_creamer2024_pct_of_ceiling": 92,
        "structure_recovered": bool(real.mean() > np.mean(sh) + 0.02),
    }
    return out


if __name__ == "__main__":
    r = run()
    print("=== L2 connectome-constrained held-out-neuron imputation (Creamer 2024) ===")
    print("  noise ceiling (split-half, SB) = %.3f" % r["noise_ceiling"])
    print("  connectome model     r=%.3f  = %d%% of ceiling  (n=%d neurons)"
          % (r["connectome"]["r"], r["connectome"]["pct_of_ceiling"], r["n_neurons_scored"]))
    print("  shuffled connectome  r=%.3f  = %d%% of ceiling"
          % (r["shuffled"]["r"], r["shuffled"]["pct_of_ceiling"]))
    print("  Creamer 2024 (fitted LDS): connectome 92%% of ceiling, shuffled much lower")
    print("  -> %s (connectome >> shuffled on held-out imputation = the data speaking)"
          % ("STRUCTURE RECOVERED" if r["structure_recovered"] else "n.s."))
    with open(os.path.join(HERE, "l2_imputation_benchmark.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote l2_imputation_benchmark.json")
