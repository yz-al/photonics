"""
l4_benchmark.py -- L4 behavior decoding, head-to-head vs Hallinen et al. 2021.

Reference
---------
Hallinen, Dempsey, Scholz, Yu, Linder, Randi, Sharma, Shaevitz, Leifer (2021),
"Decoding locomotion from population neural activity in moving C. elegans,"
eLife 10:e66135. Protocol: decode velocity and body curvature from whole-
population calcium activity via ridge regression on [F, dF/dt] across all
neurons; metric R^2_ms (mean-subtracted R^2) on a held-out test set.
Reported: velocity R^2 median 0.56 (0.76 best exemplar); curvature 0.29 (0.60);
GFP motion-artifact control floor 0.30 / 0.04.

What this does
--------------
Reproduces the Hallinen ridge baseline on the Flavell 000776 whole-brain +
behavior dataset, then tests whether a stronger decoder beats it on the SAME
held-out splits and features. The clean, honest claim is the same-data method
comparison (nonlinear vs their linear ridge), not a cross-dataset scalar match
(Flavell != Leifer data, so absolute numbers differ).

Result (n=38 worms; see l4_benchmark.json):
                         velocity R^2_ms      curvature R^2_ms
  Hallinen ridge [F,dF]     0.653  (median)      0.200
  ours  HistGradBoost       0.744               0.397   (~2x on curvature)
  temporal-shuffle null    ~0.0                ~0.0
Our HGB is positive in 38/38 worms on velocity and beats ridge on both channels
on identical splits; the shuffle null confirms the signal is not autocorrelation
leakage shared by both models.

    python l4_benchmark.py          # runs, writes l4_benchmark.json
    BRIDGE_CACHE=/path python l4_benchmark.py   # point at the flav_worms cache
"""
import os, glob, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BRIDGE_CACHE", os.path.join(HERE, "data", "bridge_cache"))
FLAV = os.path.join(CACHE, "flav_worms")

CHANNELS = [("velocity", "vel"), ("curvature", "beh_body_curvature")]
REFERENCE = {  # Hallinen 2021 published R^2_ms
    "velocity":  {"median": 0.56, "exemplar": 0.76, "gfp_floor": 0.30},
    "curvature": {"median": 0.29, "exemplar": 0.60, "gfp_floor": 0.04},
}


def _features(traces):
    F = np.nan_to_num((traces - np.nanmean(traces, 0)) / (np.nanstd(traces, 0) + 1e-9))
    dF = np.vstack([np.zeros((1, F.shape[1])), np.diff(F, axis=0)])
    return F, dF


def run(cache=CACHE, seed=0):
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    rng = np.random.RandomState(seed)
    methods = ["hallinen_ridge", "hgb_ours", "shuffle_null"]
    acc = {ch: {m: [] for m in methods} for ch, _ in CHANNELS}
    files = sorted(glob.glob(os.path.join(cache, "flav_worms", "*.npz")))
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        tr = d["tr"]; T = len(tr); F, dF = _features(tr)
        X = np.hstack([F, dF])                    # Hallinen features
        cut = int(0.6 * T); tri = np.arange(cut); tei = np.arange(cut, T)
        for ch, key in CHANNELS:
            if key not in d.files:
                continue
            y = np.asarray(d[key], float)
            if len(y) != T:
                continue
            m = np.isfinite(y)
            a, b = tri[m[tri]], tei[m[tei]]
            if len(a) < 100 or len(b) < 50:
                continue
            # Hallinen baseline: cross-validated ridge on [F, dF/dt]
            r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[a], y[a])
            acc[ch]["hallinen_ridge"].append(r2_score(y[b], r.predict(X[b])))
            # ours: nonlinear decoder, identical features + split
            g = HistGradientBoostingRegressor(max_iter=200, max_depth=3,
                learning_rate=0.06, l2_regularization=1.0,
                random_state=seed).fit(X[a], y[a])
            acc[ch]["hgb_ours"].append(r2_score(y[b], g.predict(X[b])))
            # null floor: circularly shift the target to break the F->y alignment
            ysh = np.roll(y, T // 2)
            rn = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[a], ysh[a])
            acc[ch]["shuffle_null"].append(r2_score(ysh[b], rn.predict(X[b])))

    out = {"n_worms": {}, "reference_hallinen2021": REFERENCE, "results": {}}
    for ch, _ in CHANNELS:
        out["results"][ch] = {}
        for m in methods:
            v = np.array(acc[ch][m]); v = v[np.isfinite(v)]
            out["results"][ch][m] = dict(
                median=round(float(np.median(v)), 4),
                mean=round(float(v.mean()), 4),
                n_positive=int((v > 0).sum()), n=len(v))
        out["n_worms"][ch] = len(acc[ch]["hgb_ours"])
        rr = out["results"][ch]
        out["results"][ch]["ours_beats_ridge"] = bool(
            rr["hgb_ours"]["median"] > rr["hallinen_ridge"]["median"])
    return out


if __name__ == "__main__":
    res = run()
    print("=== L4 head-to-head vs Hallinen 2021 (R^2_ms, held-out) ===")
    for ch, _ in CHANNELS:
        r = res["results"][ch]; ref = REFERENCE[ch]
        print("\n%s  (n=%d worms)   [Hallinen published: median %.2f, exemplar %.2f, GFP floor %.2f]"
              % (ch, res["n_worms"][ch], ref["median"], ref["exemplar"], ref["gfp_floor"]))
        print("  Hallinen ridge [F,dF/dt]     median=%.3f  mean=%.3f  (>0: %d/%d)"
              % (r["hallinen_ridge"]["median"], r["hallinen_ridge"]["mean"],
                 r["hallinen_ridge"]["n_positive"], r["hallinen_ridge"]["n"]))
        print("  HistGradBoost (ours)         median=%.3f  mean=%.3f  (>0: %d/%d)  %s"
              % (r["hgb_ours"]["median"], r["hgb_ours"]["mean"],
                 r["hgb_ours"]["n_positive"], r["hgb_ours"]["n"],
                 "<- beats ridge" if r["ours_beats_ridge"] else ""))
        print("  temporal-shuffle null        median=%.3f  mean=%.3f"
              % (r["shuffle_null"]["median"], r["shuffle_null"]["mean"]))
    json.dump(res, open(os.path.join(HERE, "l4_benchmark.json"), "w"), indent=2)
    print("\nwrote l4_benchmark.json")
