# Phase 1 — Multi-task baseline surrogate + validation

**Goal (Phase 1):** train a baseline multi-task surrogate on the *existing* C2DB
labels and validate it against the physics anchors — before any GW-BSE/EPW label
generation (Phase 2).
**Status:** complete. All metrics below are 5-fold **out-of-fold** (no test-set
leakage), reproducible via `excitonic/scripts/phase1_train_baseline.py`.

---

## What is (and isn't) predicted here

From Phase 0, only two of the four targets carry labels in C2DB, so only these
two are trainable now:

| Target | Label | n | Quality | Trainable in P1? |
|--------|-------|--:|---------|:---:|
| `E_b` — exciton binding | `E_B` (BSE) | 370 | **GW-BSE** | ✅ |
| `f_osc` — oscillator strength | mean in-plane `alpha_el` | 369 | **DFT proxy** | ✅ |
| `Γ(300 K)` — Fröhlich linewidth | — | 0 | — | ❌ (Phase 2: EPW) |
| `U` — exciton–exciton | — | 0 | — | ❌ (Phase 2: BSE) |

`Γ`, `U`, and therefore the blockade FOM `U/Γ` are **not predicted** and are
withheld to avoid fabrication — exactly as Phase 0 concluded.

## Model

- **Inputs:** composition features (weighted elemental-property statistics via
  `mendeleev`) **+ cheap DFT descriptors** (PBE & HSE gaps, DOS effective masses,
  thickness, cell area, atom/species counts, out-of-plane dipole, `ehull`,
  `hform`, lattice polarizability, magnetic flag). 54 usable features after
  dropping degenerate columns.
- **Deliberately excluded to prevent leakage / keep the surrogate usable on
  un-GW'd materials:** `gap_gw` (the very GW quantity we want to avoid needing)
  and `alpha*_el`/`plasmafrequency` (these *are* the `f_osc` target).
- **Estimator:** per-target **bootstrap ensemble (20×)** of gradient-boosted
  trees (`HistGradientBoostingRegressor`, native NaN handling). Ensemble spread =
  the epistemic uncertainty that drives Phase-3 active learning.
- This is the composition-level **coGN/ALIGNN-style baseline**, appropriate at
  n≈370. A structure-graph GNN (ALIGNN / MACE fine-tune) is the **Phase-1b**
  upgrade — it needs atomic structures + a Modal GPU and is deferred.

## Results (5-fold out-of-fold)

| Target | n | MAE | RMSE | R² | Skill vs mean-predictor |
|--------|--:|----:|-----:|---:|:--:|
| **`E_b`** (eV, GW-BSE) | 370 | **0.172** | 0.292 | **0.807** | 0.66 |
| **`f_osc`** (log₁₀ α, DFT proxy) | 369 | 0.099 | 0.165 | 0.850 | 0.69 |

Parity plots: `figures/phase1_parity.png`. Both targets track the diagonal with
clear skill over a mean predictor; the high-`E_b` tail (>2.5 eV) shows the
expected small-data regression-to-mean — flagged, not hidden.

## Anchor validation (the physics check)

All six group-VI **TMD monolayers** are in the C2DB BSE set. Out-of-fold
predictions vs the C2DB BSE truth:

| TMD | E_b true (eV) | E_b pred ± σ (eV) | error |
|-----|-------------:|:-----------------:|------:|
| MoS₂  | 0.547 | 0.704 ± 0.083 | +0.157 |
| MoSe₂ | 0.500 | 0.530 ± 0.042 | +0.030 |
| MoTe₂ | 0.455 | 0.482 ± 0.055 | +0.027 |
| WS₂   | 0.516 | 0.699 ± 0.084 | +0.183 |
| WSe₂  | 0.480 | 0.536 ± 0.051 | +0.056 |
| WTe₂  | 0.420 | 0.515 ± 0.051 | +0.095 |

- **All six land in the ~0.4–0.7 eV regime**, consistent with the literature
  anchor (large 2D exciton binding; Selig Nat.Commun. 2016 / Moody
  Nat.Commun. 2015 place TMD monolayer binding at several hundred meV). ✅
- The **sulfides (MoS₂, WS₂) are over-predicted by ~0.16 eV** — a systematic
  edge effect (they sit near the high-binding end of the training distribution).
  Reported honestly.

**Anchors intentionally NOT predicted:**
- **GaAs** (E_b ≈ 4 meV, a_B ≈ 10 nm) is a **3D** material, absent from this 2D
  set → out-of-distribution by construction. The surrogate is a 2D-monolayer
  model; predicting GaAs would be extrapolation and is flagged, not reported.
- **Halide-perovskite Fröhlich linewidth** (~40–70 meV LO; Wright
  Nat.Commun. 2016) is a **Phase-2 EPW** quantity — no label exists to train on.

## Uncertainty calibration (honest)

| Target | within 1σ | within 2σ | Spearman(\|err\|, σ) |
|--------|:---------:|:---------:|:-------------------:|
| `E_b`   | 0.40 | 0.72 | 0.44 |
| `f_osc` | 0.47 | 0.78 | 0.52 |

- The bootstrap ensemble is **under-dispersed** (ideal 1σ coverage ≈ 0.68): the
  σ's are epistemic-only and too tight in absolute scale — do **not** read them
  as calibrated error bars.
- **But** |error| correlates positively with σ (Spearman ≈ 0.44–0.52), so the
  uncertainty **ranks** reliably — which is what Phase-3 active-learning
  acquisition actually needs. A calibration multiplier (or an evidential/NGBoost
  head) is the planned fix before the acquisition loop.

## Reproduce

```bash
python excitonic/scripts/phase1_build_dataset.py    # pull 370-row frame
python excitonic/scripts/phase1_train_baseline.py   # train + CV + anchors + figure
# or on Modal via GitHub Actions:
#   Actions -> "Excitonic Phase 1 (baseline surrogate) on Modal" -> Run workflow
#   (runs `modal run excitonic/modal_app.py::train`)
```

Artifacts: `data/manifests/phase1_baseline_metrics.json`,
`data/processed/phase1_oof_predictions.csv`, `figures/phase1_parity.png`.
The trained ensemble (`models/*.joblib`, ~15 MB) is git-ignored and rebuilt by
the script with a fixed seed.

## Verdict → Phase 2

A cheap-input surrogate predicts **GW-BSE exciton binding to ±0.17 eV (R²=0.81)**
and reproduces the TMD anchor family. That is enough to *steer* label generation,
not to decide blockade. The decision-critical targets (`Γ`, `U`) remain
label-less. **Phase 2** stands up GW-BSE + DFPT/EPW on Modal CPU to generate
`Γ(300 K)` and `U` on a ~100–500 seed set (prioritized by this surrogate's
predictions of large binding + the families with low expected Fröhlich coupling),
after which the FOM becomes computable and Phase-3 active learning can begin.
