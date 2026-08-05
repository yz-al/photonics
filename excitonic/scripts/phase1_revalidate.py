#!/usr/bin/env python3
"""
Phase 1 — honest re-validation.

The random-5-fold R²=0.81 on the 370-material C2DB BSE set overstates what the
surrogate can do, because that set is almost entirely 2D-confined, large-binding
materials and random folds leak chemically-similar entries across the split.
This script runs the checks that actually characterize the model:

  1. Baselines: constant-mean and PBE-gap-only, so "skill" has a reference.
  2. Leave-one-chemistry-out CV (grouped by family and by element-set) vs random.
  3. GaAs out-of-distribution: predict the 3D anchor (E_b≈4 meV) the model never
     saw; report the failure rather than hide it.
  4. Uncertainty calibration curve: observed RMSE per predicted-σ bin.

    python excitonic/scripts/phase1_revalidate.py
Writes excitonic/data/manifests/phase1_revalidation.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

# Import the training module's building blocks without running its main().
_spec = importlib.util.spec_from_file_location(
    "p1train", os.path.join(HERE, "phase1_train_baseline.py"))
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)

from exciton_fm.features import featurize_formula, feature_names, parse_formula  # noqa: E402
from exciton_fm.seed_select import tag_family                                    # noqa: E402
from sklearn.model_selection import GroupKFold, KFold                            # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score                        # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor                       # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "data", "manifests", "phase1_revalidation.json")


def _metrics(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    yt, yp = y[m], yhat[m]
    return {"n": int(m.sum()),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "r2": float(r2_score(yt, yp))}


def grouped_oof(X, y, groups):
    n_groups = len(set(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        good = np.isfinite(y[tr])
        if good.sum() < 20:
            continue
        models = p1.ensemble_fit(X[tr], y[tr], n=10, seed0=2)
        m, _ = p1.ensemble_predict(models, X[te])
        oof[te] = m
    return oof


def single_feature_oof(gap_col, y):
    """PBE-gap-only baseline (one feature)."""
    X1 = gap_col.reshape(-1, 1)
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in kf.split(X1):
        good = np.isfinite(y[tr])
        if good.sum() < 20:
            continue
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, random_state=0)
        m.fit(X1[tr][good], y[tr][good])
        oof[te] = m.predict(X1[te])
    return oof


def main() -> int:
    df = p1.load()
    X, Y, names, _f, _dropped = p1.build_xy(df)
    y = Y["E_b"]
    formulas = df["formula"].astype(str).tolist()

    report = {"target": "E_b (eV, GW-BSE)", "n": int(np.isfinite(y).sum()),
              "checks": {}}

    # --- 1. Baselines --------------------------------------------------------
    ymean = np.full_like(y, np.nanmean(y))
    report["checks"]["baseline_constant_mean"] = _metrics(y, ymean)
    gap = df["gap"].to_numpy(dtype=float)
    report["checks"]["baseline_pbe_gap_only"] = _metrics(y, single_feature_oof(gap, y))

    # --- 2. Random vs leave-one-chemistry-out --------------------------------
    oof_rand, _ = p1.cv_oof(X, y)
    report["checks"]["random_5fold"] = _metrics(y, oof_rand)

    fam = np.array([tag_family(f) for f in formulas])
    report["checks"]["group_by_family"] = {
        **_metrics(y, grouped_oof(X, y, fam)),
        "n_groups": int(len(set(fam))),
        "note": "leave-one-family-out: train on some families, predict held-out families",
    }
    elset = np.array(["-".join(sorted(parse_formula(f).keys())) for f in formulas])
    report["checks"]["group_by_element_set"] = {
        **_metrics(y, grouped_oof(X, y, elset)),
        "n_groups": int(len(set(elset))),
        "note": "no exact composition-family split across folds",
    }

    # --- 3. GaAs out-of-distribution -----------------------------------------
    # Build GaAs's feature row the same way as training: composition features +
    # DFT descriptors. GaAs is 3D -> the 2D descriptors are undefined (NaN).
    comp = featurize_formula("GaAs")
    ncomp = len(feature_names())
    ndft = X.shape[1] - ncomp  # after degenerate-column drop this won't line up;
    # so instead re-derive GaAs on the SAME kept columns by re-running build_xy on
    # a one-row frame appended — simplest: refit a full-data ensemble on X and
    # predict a composition-only nearest surrogate is fragile. Use a robust path:
    # train on all C2DB rows, predict GaAs by matching the kept feature layout.
    # Re-featurize consistently:
    import pandas as pd
    gaas_row = {c: np.nan for c in df.columns}
    gaas_row["formula"] = "GaAs"
    gaas_row["gap"] = 0.5      # PBE gap of GaAs (severely underestimated; ~expt 1.42)
    gaas_row["is_magnetic"] = "No"
    df2 = pd.concat([df, pd.DataFrame([gaas_row])], ignore_index=True)
    X2, Y2, names2, _f2, _d2 = p1.build_xy(df2)
    models = p1.ensemble_fit(X2[:-1], Y2["E_b"][:-1], n=20, seed0=7)
    gaas_pred, gaas_sig = p1.ensemble_predict(models, X2[-1:].reshape(1, -1))
    report["checks"]["gaas_out_of_distribution"] = {
        "E_b_true_eV": 0.004,
        "E_b_pred_eV": float(gaas_pred[0]),
        "sigma_eV": float(gaas_sig[0]),
        "abs_error_eV": float(abs(gaas_pred[0] - 0.004)),
        "note": ("GaAs is a 3D material absent from C2DB; true E_b≈4 meV, a_B≈10 nm. "
                 "A large predicted binding with small σ is the OOD failure — the "
                 "surrogate is a 2D-monolayer interpolator, not a general model."),
    }

    # --- 4. Calibration curve (observed RMSE per predicted-σ bin) -------------
    _, sig_rand = p1.cv_oof(X, y)
    m = np.isfinite(y) & np.isfinite(oof_rand) & np.isfinite(sig_rand) & (sig_rand > 0)
    yt, yp, ys = y[m], oof_rand[m], sig_rand[m]
    order = np.argsort(ys)
    bins = np.array_split(order, 4)
    curve = []
    for b in bins:
        curve.append({"mean_sigma": float(ys[b].mean()),
                      "observed_rmse": float(np.sqrt(np.mean((yt[b] - yp[b]) ** 2))),
                      "n": int(len(b))})
    report["checks"]["calibration_curve"] = {
        "bins_by_predicted_sigma": curve,
        "note": ("If calibrated, observed_rmse should rise with mean_sigma AND be "
                 "≈ mean_sigma. Flat or non-monotonic ⇒ σ is decorative for "
                 "active-learning acquisition."),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[reval] E_b, n={report['n']}")
    print(f"[reval] constant-mean:   R²={report['checks']['baseline_constant_mean']['r2']:.3f} "
          f"MAE={report['checks']['baseline_constant_mean']['mae']:.3f}")
    print(f"[reval] PBE-gap-only:    R²={report['checks']['baseline_pbe_gap_only']['r2']:.3f} "
          f"MAE={report['checks']['baseline_pbe_gap_only']['mae']:.3f}")
    print(f"[reval] random 5-fold:   R²={report['checks']['random_5fold']['r2']:.3f} "
          f"MAE={report['checks']['random_5fold']['mae']:.3f}")
    print(f"[reval] leave-family-out:R²={report['checks']['group_by_family']['r2']:.3f} "
          f"MAE={report['checks']['group_by_family']['mae']:.3f} "
          f"({report['checks']['group_by_family']['n_groups']} families)")
    print(f"[reval] leave-elset-out: R²={report['checks']['group_by_element_set']['r2']:.3f} "
          f"MAE={report['checks']['group_by_element_set']['mae']:.3f}")
    g = report["checks"]["gaas_out_of_distribution"]
    print(f"[reval] GaAs OOD: true={g['E_b_true_eV']} eV  pred={g['E_b_pred_eV']:.3f}±{g['sigma_eV']:.3f} eV  "
          f"abs_err={g['abs_error_eV']:.3f} eV")
    print("[reval] calibration (mean_σ -> obs_RMSE):")
    for c in curve:
        print(f"          σ={c['mean_sigma']:.3f} -> RMSE={c['observed_rmse']:.3f} (n={c['n']})")
    print(f"[reval] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
