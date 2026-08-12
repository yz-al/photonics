"""
l3_benchmark.py -- L3 whole-brain next-step prediction, head-to-head vs worm-graph.

Reference
---------
Simeon, Venancio, Skuhersky, Nayebi, Boyden, Yang (2024), "Scaling Properties for
ANN Models of a Small Nervous System" (worm-graph; bioRxiv 2024.02.13.580186,
IEEE SoutheastCon 2024). Self-supervised next-time-step calcium prediction over a
canonical 300-neuron slot space with a labeled-neuron mask; masked MSE loss.
Reported: naive-persistence validation MSE = 0.03541 (the baseline); their linear
model 0.03533 and only CTRNN/LSTM beat persistence, and only marginally.

Data (public): HuggingFace qsimeon/celegans_neural_data -- 919 worms across 12
source datasets, standardized calcium traces, canonical `slot` neuron index.

What this does
--------------
Reconstructs per-worm (T x 300) matrices in canonical slot space, reproduces the
persistence baseline, and fits a GLOBAL cross-worm linear next-step model (with a
short history) evaluated on held-out worms with the masked-MSE metric.

Result (held-out worms; see l3_benchmark.json):
  persistence baseline              0.0290   (worm-graph reports 0.03541)
  our global linear AR (1 lag)      0.0287   -1.0%
  our global linear + 3-lag history 0.0203   -30%
Two validations here: (1) our single-frame linear barely beats persistence
(-1.0%), reproducing worm-graph's own finding that their linear model barely beats
the baseline (0.03533 vs 0.03541) -- our setup matches theirs; (2) adding temporal
history takes it to -30%, far past the ~0.2% their linear/CTRNN/LSTM report. We
reproduce persistence in the same ballpark as their 0.03541 (residual = metric
aggregation: per-worm mean here vs their sequence-level protocol).

This is also the methodology point (see METHODOLOGY.md): next-step prediction is
persistence-bound -- "the trap." A 30% win here is real but next-step MSE is the
weak task; recovered structure shows in held-out-neuron IMPUTATION (cf. Creamer/
Leifer/Pillow 2024, connectome LDS at 92% of the noise ceiling), not here.

    python l3_benchmark.py     # downloads data (cached), runs, writes l3_benchmark.json
Requires huggingface_hub, pandas, scikit-learn.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = 300
HF_REPO = "qsimeon/celegans_neural_data"
HF_FILE = "worm_data_short.parquet"


def load_worms():
    """Reconstruct per-worm (T, 300) canonical-slot matrices + labeled mask."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    df = pd.read_parquet(path, columns=["source_dataset", "worm", "slot",
                                        "is_labeled_neuron", "calcium_data", "max_timesteps"])
    worms = []
    for (src, w), grp in df.groupby(["source_dataset", "worm"]):
        T = int(grp.max_timesteps.iloc[0])
        if T < 20:
            continue
        X = np.zeros((T, SLOTS), np.float32); lab = np.zeros(SLOTS, bool)
        for _, r in grp.iterrows():
            s = int(r.slot); c = np.asarray(r.calcium_data, np.float32)
            n = min(len(c), T); X[:n, s] = c[:n]
            if r.is_labeled_neuron:
                lab[s] = True
        worms.append((src, w, X, lab))
    return worms


def _pairs(ws, K, rng, sub=None):
    Xs, Ys, Ms = [], [], []
    for _, _, X, lab in ws:
        T = len(X)
        for t in range(K - 1, T - 1):
            Xs.append(X[t - K + 1:t + 1].ravel()); Ys.append(X[t + 1]); Ms.append(lab)
    Xs = np.array(Xs, np.float32); Ys = np.array(Ys, np.float32); Ms = np.array(Ms)
    if sub and len(Xs) > sub:
        sel = rng.choice(len(Xs), sub, replace=False)
        return Xs[sel], Ys[sel], Ms[sel]
    return Xs, Ys, Ms


def _masked_mse(pred, true, mask):
    return float(((pred - true) ** 2)[mask].mean())


def per_worm(seed=0, K=3, sub=150_000):
    """Aligned per-held-out-worm masked MSE for the model vs persistence (predict the
    previous frame), evaluated on the SAME frames -- per-unit arrays for paired
    inference (see rigor.py). Lower MSE is better."""
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(seed)
    worms = load_worms()
    idx = np.arange(len(worms)); rng.shuffle(idx); cut = int(0.8 * len(idx))
    tr = [worms[i] for i in idx[:cut]]; te = [worms[i] for i in idx[cut:]]
    Xtr, Ytr, _ = _pairs(tr, K, rng, sub=sub)
    m = Ridge(alpha=50.0).fit(Xtr, Ytr)
    model, persistence = [], []
    for _, _, X, lab in te:
        T = len(X)
        if T <= K + 1:
            continue
        Xe = np.array([X[t - K + 1:t + 1].ravel() for t in range(K - 1, T - 1)], np.float32)
        Ye = X[K:T]; mask = np.tile(lab, (len(Ye), 1))
        model.append(_masked_mse(m.predict(Xe), Ye, mask))
        persistence.append(_masked_mse(X[K - 1:T - 1], Ye, mask))   # predict previous frame
    return {"model": model, "persistence": persistence}


def run(seed=0):
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(seed)
    worms = load_worms()
    idx = np.arange(len(worms)); rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    tr = [worms[i] for i in idx[:cut]]; te = [worms[i] for i in idx[cut:]]

    pmse = [_masked_mse(X[1:], X[:-1], np.tile(lab, (len(X) - 1, 1)))
            for _, _, X, lab in te]
    out = {"n_worms": len(worms), "reference_worm_graph_baseline": 0.03541,
           "persistence": round(float(np.mean(pmse)), 5), "models": {}}
    for K in (1, 3):
        Xtr, Ytr, _ = _pairs(tr, K, rng, sub=200_000)
        m = Ridge(alpha=50.0).fit(Xtr, Ytr)
        e = []
        for _, _, X, lab in te:
            T = len(X)
            if T <= K + 1:
                continue
            Xe = np.array([X[t - K + 1:t + 1].ravel() for t in range(K - 1, T - 1)], np.float32)
            Ye = X[K:T]; pred = m.predict(Xe)
            e.append(_masked_mse(pred, Ye, np.tile(lab, (len(Ye), 1))))
        mse = float(np.mean(e))
        out["models"]["linear_K%d" % K] = dict(
            masked_mse=round(mse, 5),
            improvement_pct=round(100 * (out["persistence"] - mse) / out["persistence"], 1),
            beats_persistence=bool(mse < out["persistence"]))
    return out


if __name__ == "__main__":
    r = run()
    print("=== L3 next-step prediction vs worm-graph (masked MSE, held-out worms; n=%d) ===" % r["n_worms"])
    print("  persistence baseline              %.5f   (worm-graph reports %.5f)"
          % (r["persistence"], r["reference_worm_graph_baseline"]))
    for k, v in r["models"].items():
        print("  our %-28s %.5f   %+.1f%%  %s"
              % (k, v["masked_mse"], -v["improvement_pct"],
                 "beats persistence" if v["beats_persistence"] else "worse"))
    with open(os.path.join(HERE, "l3_benchmark.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote l3_benchmark.json")
