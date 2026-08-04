# The dipolar branch rests on one unmeasured exponent — and the flagship must return it

## The chain, and the number at the end of it

- **The bind.** Saturation-sourced U scales with the Bohr radius squared, so raising
  it means a spread-out exciton — but a spread-out exciton is weakly bound and ionises
  at 300 K. GaAs (10 nm, 4 meV, dead above cryo) and TMDs (1 nm, 0.5 eV, thermally fine
  but interaction suppressed ~2 orders) are the two ends. This is why saturation-U ∝ 1/μ
  was proven and rejected.
- **The escape.** Put the electron in one layer, the hole in the other. U now comes from
  a static dipole set by the layer separation d, not by wavefunction overlap — so U rises
  **without** the exciton spreading, and binding stays 300 K-stable. This is why we moved
  to type-II heterostructures.
- **The unpriced cost.** Separating the charges also separates the wavefunctions that
  produce the transition dipole, so oscillator strength **f falls, ~exp(−d/λ_f)**. And f
  is not optional: it sets the coupling g ∝ √f. You need Ω = √N·g > Γ to strong-couple at
  all, and enough exciton fraction for U to survive the Hopfield weighting, since the
  usable FOM ~ |X|²·U_exc/Γ_x. Lose too much f and you either fail to couple or must go so
  excitonic you cannot read the state out.

Both U and f depend on the same d: **U rises with d, f falls with d.** Which wins is one
exponent — how fast f falls relative to how fast U rises. f slow ⇒ a window at moderate d
where U is boosted and coupling still works ⇒ the branch is real. f fast ⇒ the window is
empty and the branch closes for the same structural reason the mesh and crossbar did.

**Nobody has measured or computed that exponent for the systems we screen.** So the
209-material dipolar shortlist is ranked by a geometric prior (g_dip ∝ d²/ε_eff) that
**assumes the tradeoff is favorable without checking it.** That assumption is now flagged
in `dipolar_screen.md` and `family_screen.md`.

## Why this beats every search-efficiency improvement

Dimensional analysis, level-set acquisition, better OOD — all improve how efficiently we
search the space. **The exponent decides whether the space has anything in it.** One
heterostructure at two or three separations, returning U and f at each, gives the slope of
both curves and therefore whether a window exists. One calculation validates or closes the
entire branch.

## The flagship deliverable (reprioritized)

The expensive run's top-line output is no longer a single E_b + a cost number. It is:

> **MoS₂/MoSe₂ heterobilayer GW-BSE at ≥2 interlayer separations d, recording at each d:
> the interlayer-exciton energy (→ E_b and the dipolar U), and its oscillator strength f.**

The BSE already computes f as a residue (now parsed — `oscillator_strength` in the
`gwbse_cost` return). Adding a second separation is nearly free relative to the run's cost
and yields two points on both curves.

## What consumes it

`src/exciton_fm/dipolar_window.py` fits both slopes and returns the verdict:

- `fit_f_decay` → oscillator-strength decay length **λ_f** (small = branch-killing).
- `fit_U_rise` → U rise exponent **p**.
- `window_verdict` → net figure **N(d) = U·f^α** (α=0.5 for g ∝ √f); **N rising with d ⇒
  window OPEN**, **N falling ⇒ window CLOSED**. One point ⇒ `undetermined` (no slope).

Self-tested (`scripts/phase2_dipolar_window_demo.py`): f decaying slowly (λ_f=2 nm) →
`window_open`; f decaying fast (λ_f=0.25 nm) → `window_closed`; a single separation →
`undetermined`. The real verdict is filled from `phase2_gwbse_dsweep.json` when the
flagship runs.

## Sequencing (honest)

The heterobilayer f(d) sweep cannot run until the Yambo chain completes a single cheap
monolayer run — that debug is still in progress. So the order is: (1) get the chain green
on the monolayer debug (truncation verified), (2) run the MoS₂/MoSe₂ heterobilayer at ≥2
separations as the flagship, (3) feed U(d), f(d) to `window_verdict`. The instrumentation
(f-parsing) and the decision logic (window module) are wired **now**, so step 2 produces
the exponent the moment the chain works — not a re-run later.
