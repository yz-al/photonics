# Decision: how U is computed — and why the saturation proxy is rejected

**This is decided before any Phase-2 compute is spent, because it determines
whether the project is a materials search at all.**

U — the exciton–exciton interaction — is the numerator of the blockade FOM
`U/Γ`. Unlike E_b, Γ, and oscillator strength, U has **no routine
first-principles method at scale**. How we define it decides everything.

## The trap: saturation-U is not an independent quantity

The scaffold's `exciton_u.u_saturation` uses the standard saturation/exchange
form (Ciuti/Tassone-Yamamoto):

  g_xx ≈ 6 · E_b · a_B²  (interaction strength),   U = g_xx / A_mode.

For a hydrogenic exciton, E_b ∝ μ/ε² and a_B ∝ ε/μ, so

  g_xx ∝ E_b · a_B² ∝ (μ/ε²)·(ε/μ)² = **1/μ**.

Verified numerically (`scripts`/inline): across any (μ, ε) grid, g_xx tracks 1/μ
**exactly and is independent of ε**. So the saturation interaction strength is
fixed by the reduced effective mass alone — a **ground-state DFT** quantity.

**Consequences, all bad for the premise:**
1. U_sat adds no information beyond μ (and the chosen mode area A). The GW-BSE
   exciton binding and the trained surrogate are **irrelevant** to it.
2. The FOM `U_sat/Γ` collapses to a function of **(μ, A, Γ)** — computable with a
   phonon/EPW calculation and an effective mass, **no BSE, no surrogate**.
3. Worse, Γ's Fröhlich coupling also scales with μ (α ∝ √μ), so U_sat and Γ are
   *correlated through the same parameter*. The saturation `U/Γ` is therefore
   nearly pinned — which is very likely **why the best measured U/Γ clusters at
   ~0.05 across different materials**: the saturation channel is physics-capped,
   not materials-limited.

**Decision 1 — reject saturation-U as the FOM numerator.** Using it would make
the entire apparatus (surrogate, GW-BSE seed runs, active learning) decorative:
the answer is a fixed-ish number set by μ and geometry. We will still *compute*
U_sat, but only as the **physics-capped floor**, reported as tier
`saturation_model`, never as the search objective.

## The corollary result (a real one, per the brief)

If saturation-sourced blockade is physics-capped at U/Γ ~ O(0.05) because U_sat
and Γ share the μ dependence, then **saturation blockade cannot reach RT by
materials choice** — that bounds the moonshot from the saturation side without
running a single BSE, and the brief explicitly counts a defensible "no" as a
result. The remaining question is whether a *different* U channel escapes the cap.

## The only independent U: the dipolar / interlayer channel

A genuinely independent U comes from a **permanent excited-state dipole** — a
spatially-indirect (interlayer) exciton with electron and hole separated by
distance d:

  U_dip ≈ e² d² · n  (oriented-dipole/capacitor interaction), per exciton U = e²d²/(ε₀ε_eff A).

This U is set by **d**, not by a_B or μ, so it **decouples from E_b and Γ** — it is
the one lever that can be made large while the exciton stays small and thermally
stable (large binding). This is exactly the "dipolar decouples U from the Bohr
radius" note in the brief, now load-bearing rather than a footnote.

**Is dipolar-U computable at scale?** Yes, and independently of E_b:
- For **interlayer excitons in vdW bilayers/heterostructures**, d ≈ the interlayer
  charge-transfer separation — largely **geometric** (interlayer spacing × CT
  character of the lowest exciton). The CT character is a genuine BSE observable
  (the exciton wavefunction's electron–hole layer localization), not a restatement
  of E_b.
- The relevant databases are **BiDB / HetDB** (vdW bilayers / heterostructures on
  2dhub), not monolayer C2DB. The search target moves there.

**Decision 2 — the search objective's U is the dipolar/interlayer U**, from the
excited-state dipole d (BSE exciton CT character × interlayer geometry). Biexciton
BSE (the other route to an independent U) is acknowledged as an **open methods
problem** — there is no routine, converged, scalable biexciton-binding pipeline for
non-trivial 2D systems — and is out of scope until the dipolar route is exhausted.

## What this changes downstream

- **Targets:** the FOM numerator is `U_dip` (independent), with `U_sat` reported
  only as the capped floor. Phase 0's "separate saturation- vs dipolar-sourced U"
  is now a hard requirement, not an aside.
- **Dataset:** the seed search shifts toward **interlayer/dipolar-exciton stacks
  (BiDB/HetDB)**; monolayer C2DB gives E_b/Γ context but cannot host a large
  independent U.
- **Surrogate:** must predict the excited-state dipole / CT character, not just
  E_b — a target C2DB does not even tabulate, reinforcing that this is a
  label-generation problem (Phase 2), not a fit-to-C2DB problem.
- **No Phase-2 compute** is spent on a material whose only U is saturation-sourced,
  because its U/Γ is predetermined.

## One-line statement of record

> U for the blockade FOM is the **dipolar/interlayer** exciton–exciton interaction
> (from the excited-state dipole d, a BSE observable), because the saturation
> proxy is algebraically ∝ 1/μ — a ground-state quantity that makes the FOM
> surrogate-redundant and is physics-capped. Saturation-U is kept only as the
> reported floor; biexciton-BSE U is deferred as an unsolved scalable-methods
> problem.
