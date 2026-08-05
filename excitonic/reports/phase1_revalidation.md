# Phase 1 — honest re-validation (correcting the Phase-1 overclaim)

The original Phase-1 report led with **E_b R²=0.81 (random 5-fold)** and "reproduces
the TMD anchors." Both overstate what the surrogate can do. This re-validation runs
the checks that actually characterize it. Reproducible: `scripts/phase1_revalidate.py`
→ `data/manifests/phase1_revalidation.json`.

## 1. Baselines put the "skill" in context

| Model | R² | MAE (eV) |
|-------|---:|---------:|
| Constant mean (trivial) | 0.00 | 0.512 |
| **PBE gap only (1 feature)** | **0.335** | 0.339 |
| Full surrogate, random 5-fold | 0.807 | 0.172 |

A single cheap feature (PBE gap) already explains a third of the variance. The full
model adds real signal on top, but "R²=0.81" is not the headline it looked like.

## 2. Random folds leak; leave-one-chemistry-out is the honest split

| CV scheme | R² | MAE (eV) |
|-----------|---:|---------:|
| Random 5-fold | 0.807 | 0.172 |
| **Leave-one-family-out** (8 families) | **0.671** | 0.237 |
| Leave-one-element-set-out | 0.820 | 0.173 |

Grouping by **family** (train on some families, predict held-out ones) drops R² from
0.81 to **0.67** — random folds were leaking chemically-similar entries, exactly as
suspected. (Element-set grouping barely moves because MoS₂ and MoSe₂ land in
different groups yet remain near-duplicates — too fine to prevent leakage; the
family split is the meaningful one.) Cross-chemistry generalization is real but
**modest**.

## 3. GaAs out-of-distribution: the model fails, and doesn't know it

GaAs is the anchor that matters — 3D, ~4 meV binding, ~10 nm Bohr radius, and **not
in C2DB at all** (so a true external test). Prediction:

| | true | predicted |
|-|-----:|----------:|
| E_b | 0.004 eV | **0.283 ± 0.078 eV** |

**~70× too large, with a small uncertainty.** The surrogate is a 2D-monolayer
interpolator; outside that regime it is confidently wrong. The TMD "anchors"
(MoS₂…) are inside the training distribution — reproducing them is interpolation,
not validation. GaAs is the real test and it fails. **This bounds the usable scope:
the model may rank 2D large-binding candidates, but must not be trusted for
absolute E_b outside that regime, and cannot screen fundamentally new chemistries.**

## 4. Uncertainty: ranks in-distribution, blind out of it

Observed RMSE per predicted-σ quartile bin:

| mean σ (eV) | observed RMSE (eV) | n |
|------------:|-------------------:|--:|
| 0.050 | 0.104 | 93 |
| 0.077 | 0.152 | 93 |
| 0.107 | 0.239 | 92 |
| 0.217 | 0.502 | 92 |

Two facts, both matter for active learning:
- **Monotonic** — σ ranks difficulty in-distribution (usable for ordering), better
  than the earlier Spearman 0.44 suggested.
- **Under-dispersed ~2×** (RMSE ≈ 2σ), and — the killer — **blind to OOD**: GaAs, the
  worst error in the whole study, got a *low* σ (0.078). Bootstrap-ensemble σ measures
  disagreement among models trained on the same 2D distribution; it cannot see
  extrapolation, which is exactly where active learning must probe.

**Consequence for Phase 3.** Selecting the next (expensive) GW-BSE jobs by this σ
would preferentially pick hard-but-in-distribution 2D materials and *miss* the novel
chemistries the search actually needs. Before any acquisition loop spends CPU, the
uncertainty needs (a) a ~2× calibration and, more importantly, (b) an **OOD-aware**
component — distance-to-training in feature space, a conformal wrapper, or a
diversity-promoting ensemble — so the acquisition function is not close to random on
exactly the candidates that matter.

## Corrected verdict

Phase 1 delivered a **within-2D-distribution interpolator** with modest cross-family
skill (LOCO R²≈0.67), a **documented OOD failure** (GaAs 70×), and **uncertainty that
ranks in-distribution but is uncalibrated and OOD-blind**. That is a legitimate
starting point for *prioritizing* 2D candidates — and explicitly **not** a validated
predictor of the blockade targets. The engineering (workflows, Modal path) ran ahead
of this; it should have waited for these checks, and the effort belongs in Phase 2
where the compute is actually expensive.
