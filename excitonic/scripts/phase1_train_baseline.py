#!/usr/bin/env python3
"""
Phase 1 — multi-task baseline surrogate with ensemble uncertainty.

Predicts, from composition + cheap DFT descriptors:
  - E_b   (exciton binding energy, GW-BSE label)      [primary]
  - f_osc (in-plane interband polarizability, DFT proxy for oscillator strength)

Model: per-target bootstrap ensemble of gradient-boosted trees. The ensemble
spread is the (epistemic) predictive uncertainty used later for active learning.
Honest evaluation: 5-fold out-of-fold predictions, compared to a mean-predictor
baseline, plus reproduction of the TMD anchor family and uncertainty calibration.

Deliberately excluded from the feature set to avoid leakage / keep the surrogate
usable on un-GW'd materials:
  - gap_gw (GW; the whole point is to predict GW-quality labels from cheap inputs)
  - alpha*_el / plasmafrequency (these ARE / proxy the f_osc target)

    python excitonic/scripts/phase1_train_baseline.py

Writes metrics + predictions + a saved model under excitonic/data/ and
excitonic/models/ and (if matplotlib present) a parity figure under figures/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.features import featurize_many, feature_names  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor       # noqa: E402
from sklearn.model_selection import KFold                        # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score        # noqa: E402
from scipy.stats import spearmanr                                # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "processed", "c2db_excitonic.csv")
MAN_DIR = os.path.join(ROOT, "data", "manifests")
MODEL_DIR = os.path.join(ROOT, "models")
FIG_DIR = os.path.join(ROOT, "figures")

DFT_FEATURES = [
    "gap", "gap_hse", "emass_cbm", "emass_vbm",
    "thickness", "area", "natoms", "nspecies", "dipz", "ehull", "hform",
    "alphax_lat",
]
N_ENSEMBLE = 20
N_SPLITS = 5
SEED = 0

TMD_ANCHORS = {"MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2"}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    for c in df.columns:
        if c not in ("formula", "layergroup", "is_magnetic"):
            df[c] = _num(df[c])
    df["is_magnetic_bin"] = (df["is_magnetic"].astype(str).str.strip().str.lower()
                             == "yes").astype(float)
    return df


def build_xy(df: pd.DataFrame):
    # Targets
    E_b = df["E_B"].to_numpy(dtype=float)
    f_in = df[["alphax_el", "alphay_el"]].mean(axis=1).to_numpy(dtype=float)
    # log10 of a positive polarizability; guard non-positive
    with np.errstate(invalid="ignore"):
        log_f = np.log10(np.where(f_in > 0, f_in, np.nan))

    # Features: composition + cheap DFT descriptors (+ missing indicators)
    comp = featurize_many(df["formula"].astype(str).tolist())
    comp_names = feature_names()

    dft = df[DFT_FEATURES].to_numpy(dtype=float)
    dft = np.column_stack([dft, df["is_magnetic_bin"].to_numpy(dtype=float)])
    dft_names = DFT_FEATURES + ["is_magnetic"]
    # missing indicators for the sparse ones
    for col in ["gap_hse", "alphax_lat", "emass_cbm"]:
        idx = DFT_FEATURES.index(col)
        miss = np.isnan(dft[:, idx]).astype(float)
        dft = np.column_stack([dft, miss])
        dft_names = dft_names + [f"{col}_missing"]

    X = np.column_stack([comp, dft])
    names = comp_names + dft_names

    # Sanitize non-finite values (C2DB reports inf DOS effective mass for flat
    # bands): map ±inf -> NaN so the tree binner treats them as missing. The
    # flat-band cases are still flagged via the emass_cbm_missing indicator.
    X[~np.isfinite(X)] = np.nan

    # Drop degenerate feature columns: all-/mostly-NaN or zero variance. (Some
    # mendeleev properties are unavailable for whole periods -> all-NaN columns,
    # which also break the tree binner.)
    finite_frac = np.isfinite(X).mean(axis=0)
    with np.errstate(invalid="ignore"):
        var = np.nanvar(X, axis=0)
    keep = (finite_frac >= 0.5) & np.isfinite(var) & (var > 0)
    dropped = [n for n, k in zip(names, keep) if not k]
    X = X[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    return X, {"E_b": E_b, "f_osc": log_f}, names, f_in, dropped


def make_model(seed: int) -> HistGradientBoostingRegressor:
    # HistGBT handles NaNs natively (no explicit imputation needed).
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.06, max_iter=400,
        l2_regularization=1.0, min_samples_leaf=8,
        random_state=seed, early_stopping=False,
    )


def ensemble_fit(X, y, n=N_ENSEMBLE, seed0=0):
    """Bootstrap ensemble; returns list of fitted models (on rows with finite y)."""
    rng = np.random.default_rng(seed0)
    good = np.isfinite(y)
    Xg, yg = X[good], y[good]
    models = []
    n_ = len(yg)
    for k in range(n):
        idx = rng.integers(0, n_, size=n_)  # bootstrap resample
        m = make_model(seed0 * 1000 + k)
        m.fit(Xg[idx], yg[idx])
        models.append(m)
    return models


def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def cv_oof(X, y):
    """5-fold out-of-fold mean + std predictions (NaN where target missing)."""
    n = len(y)
    oof_mean = np.full(n, np.nan)
    oof_std = np.full(n, np.nan)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        ytr = y[tr]
        good = np.isfinite(ytr)
        if good.sum() < 20:
            continue
        models = ensemble_fit(X[tr], ytr, seed0=1)
        m, s = ensemble_predict(models, X[te])
        oof_mean[te] = m
        oof_std[te] = s
    return oof_mean, oof_std


def metrics(y, yhat):
    mask = np.isfinite(y) & np.isfinite(yhat)
    yt, yp = y[mask], yhat[mask]
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2 = float(r2_score(yt, yp))
    # skill vs mean predictor
    base = float(mean_absolute_error(yt, np.full_like(yt, yt.mean())))
    return {"n": int(mask.sum()), "mae": mae, "rmse": rmse, "r2": r2,
            "baseline_mae_meanpredictor": base, "skill_vs_mean": float(1 - mae / base)}


def calibration(y, yhat, ysig):
    mask = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(ysig) & (ysig > 0)
    yt, yp, ys = y[mask], yhat[mask], ysig[mask]
    z = np.abs(yt - yp) / ys
    within1 = float(np.mean(z <= 1.0))
    within2 = float(np.mean(z <= 2.0))
    rho, _ = spearmanr(np.abs(yt - yp), ys)
    return {"n": int(mask.sum()), "frac_within_1sigma": within1,
            "frac_within_2sigma": within2,
            "spearman_abs_err_vs_sigma": float(rho)}


def main() -> int:
    os.makedirs(MAN_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = load()
    X, Y, names, f_in_raw, dropped = build_xy(df)
    print(f"[phase1] dataset: {len(df)} materials, {X.shape[1]} usable features "
          f"({len(dropped)} degenerate columns dropped)")

    report = {"n_materials": int(len(df)), "n_features": int(X.shape[1]),
              "features": names, "dropped_features": dropped,
              "targets": {}, "anchors": {}, "notes": []}

    oof_store = {}
    for tname, y in Y.items():
        oof_m, oof_s = cv_oof(X, y)
        oof_store[tname] = (oof_m, oof_s)
        mt = metrics(y, oof_m)
        cal = calibration(y, oof_m, oof_s)
        report["targets"][tname] = {"cv_metrics": mt, "uncertainty_calibration": cal}
        print(f"[phase1] {tname:6s}  n={mt['n']:3d}  MAE={mt['mae']:.4f}  "
              f"RMSE={mt['rmse']:.4f}  R2={mt['r2']:.3f}  "
              f"skill_vs_mean={mt['skill_vs_mean']:.2f}  "
              f"cal(1σ)={cal['frac_within_1sigma']:.2f}")

    # E_b units note: label & predictions are in eV.
    report["targets"]["E_b"]["unit"] = "eV (GW-BSE label)"
    report["targets"]["f_osc"]["unit"] = "log10(mean in-plane interband polarizability [Å]); DFT proxy"

    # --- anchor reproduction (TMD family) ---
    red = df["formula"].astype(str).str.replace(r"([A-Z][a-z]?)1(?![0-9])", r"\1",
                                                regex=True)
    oof_Eb_m, oof_Eb_s = oof_store["E_b"]
    anchors = []
    for i, f in enumerate(df["formula"].astype(str)):
        # crude reduce: strip digits to compare stoich family, but match exact TMD formulas
        if f in TMD_ANCHORS:
            anchors.append({
                "formula": f, "E_b_true_eV": float(df["E_B"].iloc[i]),
                "E_b_pred_oof_eV": (None if not np.isfinite(oof_Eb_m[i]) else float(oof_Eb_m[i])),
                "E_b_sigma_eV": (None if not np.isfinite(oof_Eb_s[i]) else float(oof_Eb_s[i])),
            })
    report["anchors"]["TMD_family_in_c2db"] = anchors
    report["anchors"]["note"] = (
        "C2DB E_B is the 2D exciton binding (large; TMD monolayers ~0.4–0.6 eV, "
        "consistent with Selig/Moody). GaAs (E_b~4 meV) is a 3D anchor NOT in this "
        "2D set → out-of-distribution by construction; flagged, not predicted here. "
        "Halide-perovskite Fröhlich linewidth is a Phase-2 (EPW) quantity, not in C2DB."
    )
    print(f"[phase1] TMD anchors found in set: {[a['formula'] for a in anchors]}")

    # --- fit final full-data ensembles and persist ---
    try:
        import joblib
        final = {}
        for tname, y in Y.items():
            final[tname] = ensemble_fit(X, y, seed0=7)
        joblib.dump({"models": final, "feature_names": names,
                     "dft_features": DFT_FEATURES},
                    os.path.join(MODEL_DIR, "phase1_baseline_ensemble.joblib"))
        report["notes"].append("Saved full-data ensembles to models/phase1_baseline_ensemble.joblib")
    except Exception as e:  # pragma: no cover
        report["notes"].append(f"model persistence skipped: {e}")

    # --- OOF predictions CSV ---
    out = df[["formula", "E_B"]].copy()
    out["E_b_pred_oof"] = oof_store["E_b"][0]
    out["E_b_sigma_oof"] = oof_store["E_b"][1]
    out["f_osc_true_log10"] = Y["f_osc"]
    out["f_osc_pred_oof"] = oof_store["f_osc"][0]
    out["f_osc_sigma_oof"] = oof_store["f_osc"][1]
    out.to_csv(os.path.join(ROOT, "data", "processed", "phase1_oof_predictions.csv"),
               index=False)

    # --- parity figure (optional) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        for ax, tname, unit in [(axes[0], "E_b", "E_b [eV] (GW-BSE)"),
                                (axes[1], "f_osc", "log10 in-plane α (DFT proxy)")]:
            y = Y[tname]; m, s = oof_store[tname]
            mask = np.isfinite(y) & np.isfinite(m)
            ax.errorbar(y[mask], m[mask], yerr=s[mask], fmt="o", ms=3, alpha=0.5,
                        elinewidth=0.6, capsize=0)
            lo = np.nanmin([y[mask].min(), m[mask].min()])
            hi = np.nanmax([y[mask].max(), m[mask].max()])
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlabel(f"true {unit}"); ax.set_ylabel("predicted (OOF)")
            r2 = report["targets"][tname]["cv_metrics"]["r2"]
            mae = report["targets"][tname]["cv_metrics"]["mae"]
            ax.set_title(f"{tname}: R²={r2:.2f}, MAE={mae:.3f}")
        # highlight TMD anchors on the E_b panel
        for a in anchors:
            if a["E_b_pred_oof_eV"] is not None:
                axes[0].scatter([a["E_b_true_eV"]], [a["E_b_pred_oof_eV"]],
                                c="red", s=40, zorder=5)
                axes[0].annotate(a["formula"], (a["E_b_true_eV"], a["E_b_pred_oof_eV"]),
                                 fontsize=7, color="red")
        fig.suptitle("Phase 1 baseline — out-of-fold parity (C2DB GW-BSE E_b, DFT-proxy f_osc)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "phase1_parity.png"), dpi=130)
        report["notes"].append("Wrote figures/phase1_parity.png")
    except Exception as e:  # pragma: no cover
        report["notes"].append(f"figure skipped: {e}")

    with open(os.path.join(MAN_DIR, "phase1_baseline_metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[phase1] wrote {os.path.join(MAN_DIR, 'phase1_baseline_metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
