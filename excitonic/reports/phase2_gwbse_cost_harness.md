# GW-BSE cost-measurement harness — caps & convergence criteria (for approval)

The first expensive run's **primary deliverable is a cost number**, not physics.
An unconverged BSE can burn indefinitely, so this fixes — in advance — a hard
spend cap, a hard wall-clock cap, and the convergence criteria we will accept.
Nothing launches until these are approved.

## Engine change (unblock)

BerkeleyGW's source is distribution-gated (can't fetch unattended). **Yambo 5.3.0
is on conda-forge** and does G₀W₀ + BSE (exciton binding, oscillator strengths,
and the exciton wavefunction / charge-transfer character the interlayer test
needs). The GW-BSE branch therefore runs entirely from conda-forge:
`qe` (DFT/DFPT/EPW) → `p2y` → `yambo`. No gated download.

## Hard caps (proposed)

| Cap | Value | Enforcement |
|-----|-------|-------------|
| **Wall-clock** | **6 h** | Modal function `timeout=21600` — the job is killed at the cap. |
| **Cores** | **16** | Modal `cpu=16`. |
| **Compute ceiling** | **96 core-hours** | = 6 h × 16. |
| **Spend** | **≤ ~$14** (≈ $5 at low Modal CPU rate) | 96 core-h × ~$0.05–0.15/core-h. |

**This stays under the $25 rule.** If a converged result needs more than 96
core-hours, the run is **rejected** (reported as "did not converge within cap"),
not extended — that non-convergence *is* the cost finding.

## First material

**Monolayer MoS₂** (3 atoms), not the MoS₂/MoSe₂ heterostructure (6+ atoms), for
the cost measurement — it's the cheapest member of the family that still exercises
the full G₀W₀+BSE chain and has the best-known reference (E_b ~0.5 eV). Once its
per-atom cost is measured, the heterostructure cost follows by scaling. (Running the
6-atom hetero first would risk blowing the cap before any number lands.)

## Pre-declared convergence criteria (accept/reject, decided now)

A number is **banked only if ALL of these pass** within the cap; otherwise the run
is reported as unconverged and the number is discarded:

1. **G₀W₀ gap** converged w.r.t. #bands and the dielectric (screening) cutoff to
   **< 0.1 eV** (two successive parameter steps agree). This is the expensive knob.
2. **BSE E_b** converged w.r.t. the k-grid to **< 10%** (two successive k-grids).
3. **Vacuum-truncation plateau** (`acceptance.vacuum_truncation_plateau`): E_b at the
   last two vacuum spacings agree to **< 2%**. If it keeps climbing, truncation is
   off and every 2D number is untrustworthy → reject.
4. **BSE ↔ GW energy reference** (`acceptance.bse_energy_reference`): lowest BSE
   eigenvalue = GW gap − E_b to **< 20 meV**.
5. **Anchor value**: MoS₂ E_b lands in **0.4–0.7 eV** (the established range). Outside
   ⇒ reject and diagnose, do not bank.

For a **heterostructure** later, criterion (6) is the project-critical gate
(`acceptance.interlayer_exciton_signature`): the interlayer exciton must be lower in
energy than both intralayer excitons AND far weaker in oscillator strength.

## What the run records (the deliverable)

- Per-stage wall-clock (DFT scf/nscf, `p2y`, yambo screening/GW, yambo BSE) → total
  core-hours → **$ at the measured rate**.
- Convergence status per criterion (pass/reject).
- Extrapolation: measured per-atom cost × {heterostructure size, seed-set size} →
  the feasibility verdict for any fan-out (feeds `phase2_cost_plan.md`).

## Gate before launch

- [x] Container MPI fixed (fake-ssh) — QE runs multi-rank.
- [x] DFPT phonon pipeline validated (Si non-polar Z*≈0; GaAs polarity after the
      Born-charge parser fix — re-run in flight).
- [x] Physics acceptance gates codified (`acceptance.py`); the BSE gates above are
      wired to run on this output.
- [ ] **Your approval of the caps + criteria above.** On approval, launch is
      `modal run excitonic/modal_phase2.py::gwbse_cost` (to be added), capped as stated.

No GW-BSE run launches until the last box is checked.
