"""
perturbation_improved.py -- beating the 38% cap on held-out perturbation prediction.

The long-standing ceiling
-------------------------
perturbation_response.py established that the parameter-free dynamical model
(I - gA)^-1 predicts the DOWNSTREAM response to a HELD-OUT perturbation at ~38% of the
matched noise ceiling (0.371), and that six model classes + the wireless connectome all
plateaued there -- an apparent informational cap.

Three principled levers move it to ~49%, cross-validated
--------------------------------------------------------
Each is motivated, not a free knob, and each is selected on TRAIN perturbations and
scored on HELD-OUT perturbations (2-fold, 5 seeds); the shuffled connectome stays ~0.

  1. GAP-JUNCTION WEIGHT (gap ~2-4, was 0.5). Electrical coupling contributes far more
     to the steady-state propagated response than the default assumed. The CV picks
     gap = 2-4 every time.
  2. HIGHER GAIN (g ~0.85-0.95, was 0.7). The responses are globally integrated /
     long-range; more of the (I-gA)^-1 geometric series is real signal.
  3. A MONOTONIC (isotonic) LINK, fit on train. This is the compressive nonlinearity we
     PROVED this session (dose_response.py: the concentration->response map is ~92%
     compressed). A linear propagation correlated against a saturating response leaves r
     on the table; a monotonic link recovers it. (Diagnostic: Spearman > Pearson.)

Result (see perturbation_improved.json), DOWNSTREAM held-out perturbations, matched ceiling 0.371:
  parameter-free baseline (g=0.7, gap=0.5, linear)   r=0.141  = 38% of ceiling
  CV-tuned + isotonic link (held-out perturbations)  r~0.181  = 49% +/- 3%
  shuffled connectome                                 r~0.002  =  1%   (clean null)

Honest boundary: 49% is still about HALF the ceiling. The remaining ~51% is reproducible
response variance (it is in the ceiling) that the aggregated connectome + steady-state
atlas still does not carry -- the cap is lower, not gone. Closing it further needs signal
this data lacks (per-animal raw dynamics, cell state). But 38 -> 49 is a real, honestly
cross-validated gain, and it is the dose-response nonlinearity + electrical coupling doing
the work.

    python perturbation_improved.py     # writes perturbation_improved.json
Requires wormneuroatlas, c302, scikit-learn.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
G_GRID = [0.85, 0.9, 0.95]
GAP_GRID = [0.5, 1.0, 2.0, 4.0]


def prop(Wc, Wg, g, gap):
    A = Wc / (Wc.sum(1, keepdims=True) + 1e-9) + gap * Wg / (Wg.sum(1, keepdims=True) + 1e-9)
    A = g * A / (np.abs(np.linalg.eigvals(A)).max() + 1e-12)
    return np.linalg.inv(np.eye(A.shape[0]) - A)


def _colscore(P, R, meas, js, iso=None):
    from scipy.stats import pearsonr
    rs = []
    N = R.shape[0]
    for j in js:
        ii = np.where(meas[:, j])[0]; ii = ii[ii != j]
        a = P[ii, j]; b = R[ii, j]
        if iso is not None:
            a = iso.predict(a)
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            rs.append(float(pearsonr(a, b)[0]))
    return np.mean(rs) if rs else 0.0


def _fit_iso(P, R, meas, js):
    from sklearn.isotonic import IsotonicRegression
    xa, ya = [], []
    for j in js:
        ii = np.where(meas[:, j])[0]; ii = ii[ii != j]
        xa += list(P[ii, j]); ya += list(R[ii, j])
    return IsotonicRegression(out_of_bounds="clip").fit(xa, ya)


def _cv(Wc, Wg, R, meas, cols, seed):
    rng = np.random.RandomState(seed); perm = rng.permutation(cols); h = len(perm) // 2
    tests, picks = [], []
    for tr, te in [(perm[:h], perm[h:]), (perm[h:], perm[:h])]:
        best = (-9, None, None, None)
        for g in G_GRID:
            for gap in GAP_GRID:
                P = prop(Wc, Wg, g, gap); iso = _fit_iso(P, R, meas, tr)
                s = _colscore(P, R, meas, tr, iso)
                if s > best[0]:
                    best = (s, (g, gap), iso, P)
        _, gp, iso, P = best
        tests.append(_colscore(P, R, meas, te, iso)); picks.append(gp)
    return float(np.mean(tests)), picks


def run(n_seeds=5):
    import perturbation_response as PR
    R, meas, Wc, Wg, ceil = PR.load(); N = PR.N
    cols = [j for j in range(N) if (meas[:, j] & (np.arange(N) != j)).sum() >= 10]
    base = _colscore(prop(Wc, Wg, 0.7, 0.5), R, meas, cols)
    vals, picks = [], []
    for s in range(n_seeds):
        v, pk = _cv(Wc, Wg, R, meas, cols, s); vals.append(v); picks += pk
    sh = []
    for s in range(3):
        p = np.random.RandomState(s).permutation(N)
        sh.append(_cv(Wc[np.ix_(p, p)], Wg[np.ix_(p, p)], R, meas, cols, s)[0])
    vals = np.array(vals)
    from collections import Counter
    return {
        "metric": "downstream held-out perturbation (self excluded); matched ceiling",
        "noise_ceiling": round(ceil, 3),
        "baseline_paramfree": {"r": round(base, 3), "pct_of_ceiling": round(100 * base / ceil, 1)},
        "improved_cv": {"r": round(float(vals.mean()), 3), "std": round(float(vals.std()), 3),
                        "pct_of_ceiling": round(100 * float(vals.mean()) / ceil, 1),
                        "pct_std": round(100 * float(vals.std()) / ceil, 1)},
        "shuffled": {"r": round(float(np.mean(sh)), 3), "pct_of_ceiling": round(100 * float(np.mean(sh)) / ceil, 1)},
        "levers": ["gap-junction weight 2-4x (electrical coupling)", "gain 0.85-0.95 (global integration)",
                   "isotonic link (compressive nonlinearity; see dose_response.py)"],
        "hyperparam_picks": {k: v for k, v in Counter(map(str, picks)).most_common()},
        "beats_baseline": bool(vals.mean() > base + 0.02 and np.mean(sh) < 0.02),
        "headroom_remaining_pct": round(100 - 100 * float(vals.mean()) / ceil, 1),
    }


if __name__ == "__main__":
    r = run()
    print("=== BEATING THE 38% CAP -- held-out perturbation prediction ===")
    print("  matched noise ceiling = %.3f" % r["noise_ceiling"])
    print("  parameter-free baseline   r=%.3f = %.0f%% of ceiling"
          % (r["baseline_paramfree"]["r"], r["baseline_paramfree"]["pct_of_ceiling"]))
    print("  CV-tuned + isotonic link  r=%.3f = %.0f%% +/- %.0f%%  (HELD-OUT perturbations)"
          % (r["improved_cv"]["r"], r["improved_cv"]["pct_of_ceiling"], r["improved_cv"]["pct_std"]))
    print("  shuffled connectome       r=%.3f = %.0f%%  (clean null)"
          % (r["shuffled"]["r"], r["shuffled"]["pct_of_ceiling"]))
    print("  levers: %s" % "; ".join(r["levers"]))
    print("  hyperparam picks (g,gap): %s" % r["hyperparam_picks"])
    print("  -> %s. Headroom to ceiling still %.0f%% (the cap is lower, not gone)."
          % ("BEATS 38%% CAP" if r["beats_baseline"] else "n.s.", r["headroom_remaining_pct"]))
    with open(os.path.join(HERE, "perturbation_improved.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote perturbation_improved.json")
