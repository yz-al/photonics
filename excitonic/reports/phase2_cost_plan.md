# Phase 2 — cost first, scaffold second

The scaffold was built before the compute was costed. That was the wrong order.
Before any fan-out, we measure one material end-to-end and multiply. This document
sets the estimates and the go/no-go; the numbers get replaced by a **measured**
single-material run.

## Per-material estimate (converged, non-trivial 2D system)

From the GW-BSE / EPW literature and the fact that C2DB itself has only ~370 BSE
entries (because each is expensive):

| Stage | Code | Core-hours (order) |
|-------|------|-------------------:|
| DFT scf + nscf (many bands) | QE pw.x | 10²–10³ |
| Dielectric matrix (ε, many bands, q-grid) | BerkeleyGW epsilon | 10²–10³ |
| G₀W₀ self-energy | BerkeleyGW sigma | 10²–10³ |
| BSE kernel + absorption | BerkeleyGW kernel/absorption | 10²–10³ |
| **GW-BSE subtotal** | | **~10³–10⁴** |
| DFPT phonons + Born/ε | QE ph.x | 10²–10³ |
| Wannier + e-ph + fine grids (exciton linewidth) | EPW | 10³–10⁴ |
| **EPW subtotal** | | **~10³–10⁴** |
| **Total end-to-end** | | **~5×10³ – 2×10⁴ core-hours** |

## Dollars and wall-clock (Modal CPU)

At a representative ~\$0.05–0.15 / core-hour:

| | low | high |
|-|----:|-----:|
| Per material | **~\$250** | **~\$3,000** |
| Seed set ×100 | ~\$25k | ~\$300k |
| Seed set ×500 | ~\$125k | ~\$1.5M |

Wall-clock on a 16-core job: **~13–50 days per material**; even at 64 cores,
several days each. **The active-learning loop would not reach iteration 1** within
any reasonable budget or timeline. This is the critique's concern, quantified — and
it is very likely fatal to the "100–500 material seed set" as written.

## What this forces (decisions)

1. **Measure one material first.** `modal_phase2.py::run_dft_dfpt` now records
   wall-clock; the first real GW-BSE+EPW material will record the true per-stage
   cost. No seed-set run is launched until that single number exists.
2. **The full converged GW-BSE + EPW seed set is almost certainly infeasible.**
   Plan for **single-digit** flagship materials at full quality, not 100–500.
3. **Use the cheap-but-honest path for breadth** (ties into the U decision):
   - **Γ:** DFPT (ε∞, Born charges, ω_LO) — **hours, not weeks** — feeding the
     Fröhlich-model Γ (tier `frohlich_model`, DFPT-grounded), instead of the
     full EPW exciton-linewidth for every material. Full EPW is reserved for the
     handful of flagship candidates.
   - **U:** the **dipolar/interlayer** U (excited-state dipole from geometry +
     BSE CT character) is far cheaper than biexciton BSE and is the *independent*
     quantity anyway (see `u_definition_decision.md`). Saturation-U is free
     (effective mass), reported as the capped floor.
4. **Re-scope the seed set** around this: a broad, cheap DFPT+geometry screen over
   BiDB/HetDB dipolar candidates, with full GW-BSE+EPW spent only on the few that
   survive.

## Go/no-go

- **Go** to a small flagship set only after one measured end-to-end material
  confirms per-material cost within budget for ~single digits.
- **No-go** on any 100–500 full-quality fan-out; if the physics needs it, the
  honest output is "the search is compute-bound beyond feasibility," which — with
  the saturation-U cap — is itself a bounded-moonshot result.

## Status of the first measurement

The DFPT plumbing run (`run_dft_dfpt`, GaAs anchor) was wired and dispatched on
Modal but has **not yet produced a converged number**: the QE binary hit MPI
process-launch friction in the locked-down Modal container (OpenMPI needs a root
daemon/ssh; documented across four iterations). The rigor guards held throughout —
every failed run emitted only `not_run`, no fabricated values. Resolving the
container MPI (serial `nompi` QE build or an MPI image with a working launcher) is
the immediate next step, after which the DFPT wall-clock is the first real cost
datapoint. **No cost number here is measured yet; all are literature-order
estimates and are labeled as such.**
