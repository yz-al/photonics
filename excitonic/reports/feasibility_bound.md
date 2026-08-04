# Feasibility bound — close the question with inequalities, not a search

A search that finds nothing is vulnerable to "you looked in the wrong place." A bound
is not. If the largest U any thermally-stable material can have is below 0.71× the
smallest Γ(300 K) it can have, the feasible set is **empty** and no search is required —
the shape of the Kramers-Kronig result that closed the Kerr route (causality caps
nonlinear-phase-shift / loss; not a failure to find a good Kerr material, a *proof*).

Code: `src/exciton_fm/bound.py`. Populated: `scripts/phase2_bound.py` →
`data/manifests/phase2_bound.json`.

## The three inequalities (all in DFPT-computable parameters)

| Leg | Statement | Status |
|-----|-----------|:------:|
| **Γ floor** | thermal-stable binding ⟹ ionicity ≥ ι_min ⟹ Fröhlich α ≥ α_min ⟹ Γ(300 K) ≥ Γ_floor | **partly derived** |
| **U_sat cap** | saturation U ∝ 1/μ; E_b>threshold with bounded ε forces μ ≥ μ_min | **confinement-dependent** |
| **U_dip cap** | dipolar U rises with d, f falls ~exp(−d/λ_f); coupling needs f ≥ f_min ⟹ d ≤ d_max ⟹ U_dip capped | **needs λ_f (flagship)** |

**Feasibility:** max(U_sat, U_dip) < 0.71·Γ_floor ⟹ CLOSED.

## Current numbers (from the DFPT gate — real ω_LO, Z*, ε∞)

- **Γ floor ≈ 6.2 meV** (min over the four TMDs, at a coupling floor α ≥ 0.3):
  MoS₂ 6.2, WS₂ 6.6, MoSe₂ 8.6, WSe₂ 9.9 meV. So blockade needs **U > 0.71·Γ_floor ≈
  4.4 meV**.
- **U_sat leg:** g_xx at the single-exciton scale is ~1050 meV, but the *usable*
  per-polariton value is that suppressed by mode-area/a_B² = N: ~**10.5 meV at a 10×10 nm
  mode**, falling as the mode delocalizes. So the saturation leg is **not** a clean cap —
  it rises toward g_xx as you localize (moiré traps) and pays in inhomogeneous broadening.
  It is a confinement↔broadening tradeoff, structurally parallel to the dipolar f-penalty.
- **U_dip leg:** needs the oscillator-strength decay length **λ_f** from GW-BSE at ≥2
  separations. Unknown ⟹ this leg is open.
- **Verdict: UNDETERMINED** — and honestly so. Two legs are soft and one is missing.

## What is actually proven vs asserted (no overclaiming)

The framework is a scaffold, not yet a proof. To become one it needs:

1. **Γ floor — derive α_min.** The phonon half (α → Γ_LO) is a genuine lower bound. The
   *stability ⟹ α ≥ α_min* link is currently **asserted** (α_min=0.3 by TMD analogy), not
   derived. Proving it — that any exciton bound hard enough for 300 K must be ionic enough
   (via Z*, LO-TO) to carry α ≥ α_min — is the crux, and it is the most provable piece
   because the polarity↔binding↔phonon-coupling physics is real. The DFPT screen supplies
   Z*, ω_LO at cents/material to pin the curve. Empirical support already exists across
   regimes: GaAs (E_b~4 meV, weakly ionic, tiny RT Γ) vs TMDs (E_b~0.5 eV, Z*~0.4–1.2,
   Γ~6–15 meV) — binding↑ tracks ionicity↑ tracks Γ↑.
2. **U_sat — derive the confinement–broadening slope.** How fast does inhomogeneous
   broadening grow as you localize to raise per-polariton U? That slope caps the usable
   U_sat exactly as λ_f caps U_dip. (This is where moiré confinement lives.)
3. **U_dip — measure λ_f.** The flagship GW-BSE at ≥2 interlayer separations.

## Preliminary read (stated as a caveat, not a result)

The early numbers do **not** obviously close: the delocalized saturation U (10.5 meV)
already exceeds the 4.4 meV target at moderate confinement, so the bound may come back
**open** unless the broadening penalty is steep. That would mean the search still matters —
but the bound will have told us *which leg* stays open (confinement-broadening and/or
λ_f), which is exactly where the physics is. **If instead the derived legs overlap, the
question is closed and Phase 3 never runs.**

## Why this is worth real effort before Phase 3

The bound-derivation is theory + the cent-per-material DFPT quantities — **cheaper than the
search**, and robust to "wrong place." It also reframes the flagship: the $300 run is no
longer "test one material," it is "supply λ_f, the one exponent that closes inequality #3."
Either the bound closes (definitive no, no Phase 3) or it isolates the single open leg.

## The yes-list (mechanisms that could keep it open)

All raise U rather than lower Γ (Γ is phonon-set at 300 K, little room), and all pay in
oscillator strength, coherence, or tunability — the conserved trade seen across six
architectures and three substrates. The out, if any, raises U through something **other
than spatial extent**:

- **Rydberg n≥2 excitons** — interaction ∝ n^high (Cu₂O). Check per screened material
  whether n=2 binding (≈E₁/9 in 2D) stays above the thermal threshold; large-E_b TMDs are
  the candidates.
- **Fermi polarons / gate doping** — tunable U enhancement (search moves to *tuning range*,
  a larger space); adds broadening, net sign unsettled.
- **Purcell narrowing** — only helps if a material is radiatively- not dephasing-limited at
  300 K (almost nothing is); a per-material check, not an assumption.
- **Moiré confinement** — localizes excitons → larger effective U, closer to the single-
  emitter limit blockade needs, sidesteps 1/N; pays in inhomogeneous broadening. This *is*
  the U_sat confinement leg above.

## Track-C update — the Γ-floor leg, now data-derived (not assumed)

`scripts/phase2_bound_alpha.py` tests the floor's crux against the 356-material C2DB
excitonic set instead of asserting α_min. Findings (Spearman ρ):

| Link | ρ | Expectation | Verdict |
|------|:--:|-------------|:------:|
| E_b vs reduced mass μ | **+0.47** | >0 (Wannier) | ✅ |
| E_b vs electronic screening | **−0.84** | <0 (bind ⟹ low screening) | ✅ strong |
| E_b vs Fröhlich proxy α̃ = √μ·(1/ε∞−1/ε0) | **+0.62** | >0 (bind ⟹ couple) | ✅ strong |
| E_b vs ionicity fraction | +0.16 | >0 | weak |

**The chain is real**: thermal-stable binding forces stronger Fröhlich coupling, dominated
by the low-screening (small ε∞ ⟹ large 1/ε∞) enhancement. **But the floor is NOT flat.**
A tail of *marginally*-stable materials (E_b just above threshold) has weak coupling, so a
constant α_min ≥ 0.3 for all stable materials is not justified by the data. The honest
statement is a **binding-dependent floor Γ_floor(E_b)**: robustly bound ⟹ high Γ; a low Γ is
reachable only near the stability edge — which is exactly the corner a low-Γ candidate would
hide in. This **replaces** the earlier flat-α_min assumption with a weaker, data-backed one —
the correct direction for a real proof. Closing it rigorously needs DFPT ω_LO/Z* across a
binding-spanning set (cheap: cents/material via the existing DFPT screen), not the flat
TMD-analogy value. Net: the bound's most-provable leg is now honestly characterized, and it
is *looser* than first assumed — the question is not yet closed on the Γ side either.
