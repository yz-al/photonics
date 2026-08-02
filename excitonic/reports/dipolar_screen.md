# Phase 2 — ranked dipolar candidate list (the actionable seed)

The first concrete output of the dipolar pivot: **209 type-II heterostructures**
(spatially-indirect / interlayer excitons → permanent out-of-plane dipole) from
live HetDB, ranked by a geometric dipolar prior for which ones deserve the
expensive GW-BSE + EPW. Reproducible: `scripts/phase2_dipolar_screen.py` →
`data/manifests/phase2_dipolar_candidates.json`.

## Method (ranking only)

- **Candidate set:** HetDB heterostructures whose C2DB monolayer band edges give a
  **type-II (staggered)** alignment by Anderson's rule → the fingerprint of a
  charge-transfer exciton with a permanent dipole. 209 / 336.
- **Prior:** geometric dipolar strength `g_dip = e²·d²/(ε₀·ε_eff)` from the
  interlayer distance `d` (100% coverage), plus a semiconducting gap-in-window
  bonus and a low-strain (commensurability/stability) bonus.
- **Model tier only.** `ε_eff` is a placeholder; `d` is the *interlayer distance*,
  an **upper bound** on the CT-weighted exciton dipole `d_exc` (a BSE observable).

## Top candidates

| Heterostructure | d (Å) | gap (eV) | g_dip (eV·nm²) |
|-----------------|------:|---------:|---------------:|
| MoS₂ / MoSe₂ | 6.61 | 1.03 | 2.4 |
| MoSe₂ / AsClSe | 6.64 | 0.89 | 2.4 |
| WSe₂ / AsBrTe | 6.90 | 0.85 | 2.5 |
| WSe₂ / GaSe | 7.56 | 1.05 | 2.7 |
| MoTe₂ / CH | 6.10 | 0.97 | 2.2 |
| MoTe₂ / WTe₂ | 7.76 | 0.64 | 2.8 |

Sanity: MoS₂/WSe₂-family and MoTe₂/WTe₂ type-II stacks are exactly the systems where
interlayer excitons are experimentally established; the gaps sit in the ~0.6–1.5 eV
(near-IR / telecom-adjacent) window. This is a physically credible shortlist, not a
random pick.

## What is NOT claimed (rigor)

- **No blockade FOM.** Γ(300 K) is not included (no EPW) and U is a geometric prior,
  not a computed dipolar U. `U/Γ` is deliberately absent.
- **`d` ≠ `d_exc`.** The real dipole is the CT-weighted electron–hole separation from
  the BSE exciton wavefunction, which `d` only bounds from above.
- **These are OOD for the surrogate.** The C2DB-monolayer surrogate cannot predict
  heterostructure excited states (all would flag high OOD under the new uncertainty)
  — confirming these are **label-generation** targets, not fit-from-existing-data.

## Verdict / next real step

This ranked list is the seed for Phase-2 GW-BSE + EPW: run the **top few** type-II
heterostructures end-to-end to get the real `d_exc` (BSE CT character), `Γ(300 K)`
(EPW), and the first genuine dipolar `U/Γ` — after the one-material cost measurement
(`reports/phase2_cost_plan.md`) confirms feasibility. The shortlist means that
measurement lands on a physically motivated candidate (e.g. MoS₂/MoSe₂), not an
arbitrary one.
