# Phase 1 — OOD-aware, calibrated predictive uncertainty (fixing the GaAs blind spot)

The Phase-1 bootstrap-ensemble σ failed the only job that matters for active
learning: it was ~2x under-dispersed in-distribution **and blind to
out-of-distribution (OOD) inputs**. GaAs — 3D, true E_b≈4 meV, absent from the 2D
C2DB set — was predicted **0.283 eV with σ=0.078 eV**: the worst error in the
study drew one of the *lowest* uncertainties. An acquisition loop driven by that σ
would bury exactly the novel chemistries it should probe. This adds a
distance-to-training OOD score and a split-conformal calibration, both in
`src/exciton_fm/uncertainty.py`. Reproducible:
`scripts/phase1_uncertainty.py` → `data/manifests/phase1_uncertainty.json`.

## 1. The GaAs fix

The OOD detector standardizes features (median-impute NaNs, then train mean/std)
and scores novelty as the mean distance to the k=5 nearest training points. GaAs
lands at the **99.7th percentile** of the training-distance distribution — it is
essentially at the frontier. The combined uncertainty
`σ_total = σ_cal · (1 + α · ood_percentile)` (α=1) then inflates it:

| quantity | value |
|----------|------:|
| E_b true | 0.004 eV |
| E_b predicted | 0.283 eV |
| **bootstrap σ (old, OOD-blind)** | **0.078 eV** |
| OOD distance percentile | **99.7 %** |
| calibrated σ | 0.144 eV |
| **OOD-inflated σ_total (new)** | **0.288 eV — 3.68× bootstrap** |

The bare σ said "confident"; the OOD-aware σ_total is now the **largest in the
set** and comparable to the actual error (0.279 eV). The extrapolation is flagged,
not hidden.

## 2. Calibration before vs after (held-out C2DB split)

A single split-conformal multiplier is fit on a 35% held-out split
(nonconformity score |residual|/σ, targeted at 1σ central coverage 0.6827):

| coverage | before | after | nominal |
|----------|-------:|------:|--------:|
| within 1σ | 0.42 | **0.70** | 0.68 |
| within 2σ | 0.73 | **0.96** | 0.95 |

The fitted multiplier **q = 1.84** directly quantifies the ~2x under-dispersion.
After rescaling, 1σ coverage is nominal and 2σ lands near 0.95, so the rescaled
residuals are close to Gaussian on the in-distribution split.

## 3. Acquisition: OOD ranking surfaces different candidates

Ranking the 370-material C2DB set by the two signals gives **zero overlap** in the
top 8:

| rank | by OOD distance (novelty) | by bootstrap σ (in-dist. difficulty) |
|-----:|---------------------------|--------------------------------------|
| 1 | Cr₂O₂ | MnO₂ |
| 2 | Ge₂F₂ | MnCl₂ |
| 3 | CoCl₂ | ScO₂ |
| 4 | OsO₂ | NiCl₂ |
| 5 | BaCl₂ | Ag₂F₂ |

Bootstrap σ favors hard-but-in-distribution 2D materials (Mn/Ni transition-metal
dihalides where the ensemble disagrees); OOD distance surfaces chemically-unusual
compositions (Os, Ge/Ba, mixed-anion CrBiAs). An acquisition function that must
probe novelty needs the second signal, which the old one could not provide.

## 4. Honest limits

- **Distance-based OOD is a heuristic.** It measures novelty in *feature* space,
  not error in the target, and `σ_total` is a monotone inflation, not a
  probabilistic OOD posterior. It restores a defensible, non-blind acquisition
  ordering — it does not certify an interval on a novel material.
- **Calibration is on the C2DB (2D) distribution.** The conformal coverage is
  honest for in-distribution 2D points. True calibration on the target
  distribution — novel chemistries and 3D anchors like GaAs — requires held-out
  **GW-BSE labels there, which do not exist yet**. Until such labels are
  computed, the OOD term is a warning flag, and any absolute σ on an
  extrapolation should be read as "large and untrusted," not as a calibrated bar.
- **α is a design choice** (α=1 ⇒ up to 2× widening at the training frontier).
  It sets how aggressively novelty widens intervals and should be revisited once
  real OOD labels allow tuning it against observed extrapolation error.

## Verdict

The surrogate's uncertainty is no longer blind to extrapolation. GaAs moves from
"confidently wrong" (σ=0.078) to "flagged" (σ_total=0.288, 99.7th-percentile
novelty), in-distribution intervals are calibrated to nominal coverage, and the
acquisition ranking now surfaces novel chemistries. This is the minimum needed
before an active-learning loop spends GW-BSE compute — with the standing caveat
that target-distribution calibration awaits labels that do not yet exist.
