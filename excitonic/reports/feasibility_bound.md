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

## Track-C synthesis — the whole question is now Γ_floor + two exponents

Assembling the legs (`scripts/phase2_bound_synthesis.py`, `src/exciton_fm/bound.py:saturation_window`),
the room-temperature blockade feasibility bound collapses to three quantities — one
data-derived, two single-measurement exponents:

| Leg | Decisive quantity | Closes iff | Status |
|-----|-------------------|-----------|:------:|
| Γ floor | Γ_floor(E_b) | — (it's the target) | **data-derived**, ~6–15 meV robustly bound → need U > 4–11 meV |
| Saturation / moiré | **q_sat** (Γ_inh ∝ L^−q vs U ∝ L^−²) | **q_sat > 2** | unmeasured |
| Dipolar / interlayer | **λ_f** (f-decay vs U-rise) | f falls faster than U rises | unmeasured (needs interlayer BSE) |

Both U channels have the *same* structure: raise U by localizing (moiré) or separating
(interlayer), and pay a readout penalty (inhomogeneous broadening, or oscillator-strength
loss). Each reduces to **one exponent against a critical value** — q_crit = 2 for the
saturation channel (exact, since U_sat ∝ L^−²), and the f-vs-U slope for the dipolar
channel. The saturation demo confirms the switch: q = 1.5 → leg *open* (tight confinement
wins), q = 3.0 → leg *closed* (broadening outruns U).

**Net verdict: UNDETERMINED — but no longer vague.** The entire moonshot now rests on two
numbers: the moiré confinement-broadening exponent q_sat, and the interlayer
oscillator-strength decay λ_f, both measured against the data-derived Γ floor. If both legs
close, the feasible set is empty and Phase 3 never runs; if either stays open, that corner
*is* the search space. q_sat may already be bounded by published moiré-exciton linewidth
data; λ_f needs the interlayer flagship once the BSE toolchain is repaired. That is the
sharpest honest statement the bound supports today — and it is two targeted measurements
from a decisive answer, not a search.

## Track-C follow-through — both open doors, examined

**q_sat (saturation/moiré leg) — from the moiré-exciton literature.** The decisive
exponent is *not* >2. Measured moiré inhomogeneous broadening is ~20–30 meV, set by
twist-angle/strain/electrostatic **disorder**, not by confinement — deeper traps (smaller
twist) *narrow* the line. So q_sat ≈ 0: the leg does **not** close by the exponent
(confinement formally wins), and moiré effects survive to room temperature (MoS₂/WSe₂).
The real obstacle is the **absolute ~20 meV disorder floor** (≫ the ~6 meV phonon floor):
blockade would need U > 0.71·(Γ_phonon+Γ_disorder) ≈ 18 meV. That reframes the saturation
leg from "unknown fundamental exponent" to a **disorder-limited fabrication problem** —
materially different, and improvable with sample quality. (`phase2_qsat_literature.json`.)

**Rydberg escape — mostly closes on character inspection.** Of the 158 naive n=2-RT
survivors, 75 have E_b > 1.5 eV — Frenkel/charge-transfer d-electron halides/oxides where
the hydrogenic (n−½)² series is invalid. Restricting to genuine **Wannier** excitons
(E_b < 1 eV, non-magnetic) leaves only **26**, all at the Wannier/Frenkel edge
(E_b ≈ 0.95–1.0 eV → n=2 barely at 0.10–0.11 eV). So it is not a robust open door — ~26
marginal cases worth a BSE check, not a population. (`phase2_rydberg_screen.json`.)

**Updated live picture.** Of the escape routes that raise U without the Γ penalty: the
**dipolar** leg still awaits λ_f (needs a working BSE); the **moiré** leg is open but
disorder-gated (an engineering, not fundamental, obstacle); the **Rydberg** door largely
closes. So the bound has tightened toward "no" on two of three, with the dipolar exponent
the one genuinely undetermined quantity left — and it is the one that needs the repaired
GW-BSE toolchain.

## Track-C follow-through (2) — moiré disorder is not fundamental; fork A launched

**Moiré disorder can be pushed below the U scale — at the single-trap level.** Individual
moiré-trapped interlayer excitons show <1 meV linewidth (low T), and hBN encapsulation
reaches the homogeneous limit (~2–5 meV). So the ~20 meV floor is *ensemble* (trap-to-trap)
broadening, not a single-trap limit. Blockade needs a **single emitter** (one trap), where
the inhomogeneous disorder is removed and the RT linewidth reverts to the **phonon floor**
(the same Γ ~6–15 meV the bound already rests on) — which tight moiré confinement's U can
plausibly beat. So the moiré/saturation leg is **genuinely open**. The real catch is that
this is a **single-emitter architecture** (quantum-dot-like), trading against the spec's
scalability/CMOS conditions — a device-integration question, not a materials cap. Concrete
experiment: blockade at one moiré trap at 300 K. (`phase2_moire_disorder.json`.)

**Fork A (toolchain) launched.** The λ_f leg needs a working BSE. Since the default Yambo's
BSE hangs even serial at 2×2/KS with OMP_NUM_THREADS=1 (ruling out a thread deadlock), it is
a version bug — so the image now pins **yambo=5.1.2** (a long-stable BSE series) plus BLAS
single-thread guards. A debug run tests whether that build's BSE completes and returns λ_f.

## Fork A — CONCLUDED: the conda-forge Yambo BSE is unusable in this environment

Two major versions tested, both fail the BSE stage while every other stage (DFT,
screening, G₀W₀ — gap 2.66 eV at 5.3.0, 2.87 eV at 5.2.x) runs clean:

| Yambo | GW | BSE |
|-------|:--:|-----|
| 5.3.0 (default) | ✅ | **hangs** (serial, even 2×2/KS) / **SIGABRT** (MPI, any rank) |
| 5.2.x (`yambo<5.3`) | ✅ | **SIGABRT** (serial), no Yambo `[ERROR]` — a library-level abort |

A SIGABRT with no Yambo-level error is a **linked-library abort** (ScaLAPACK/BLAS/HDF5
ABI), specific to the BSE code path. It is not fixable by a version pin. **Conclusion:
λ_f cannot be obtained from the conda-forge Yambo in this Modal image.** Getting it needs
a heavier lift — a Yambo built from source against a controlled MPI/ScaLAPACK/FFTW/HDF5
stack, or BerkeleyGW (its source is distribution-gated). That is a deliberate
infrastructure decision, not an autonomous continuation, so it is parked here.

**Deliverable state of the bound.** λ_f (the dipolar leg) is the one genuinely open
number and it is now blocked on toolchain, not physics. The rest stands: the Γ floor is
data-derived (binding-dependent), the Rydberg escape mostly closes on character, and the
moiré/saturation leg is open but single-emitter (a device-architecture question). So the
room-temperature blockade question is **UNDETERMINED but sharply localized** to (a) λ_f —
needs a source-built BSE — and (b) whether a single-emitter moiré architecture is
acceptable under the spec. That is the honest current answer.

## Another algo: the dipolar-leg EXPONENTS don't need a BSE (λ_f proxy)

Before rebuilding the BSE, I checked whether λ_f is really a BSE-only number. It is not —
and this reframes the leg. The two slopes it turns on are **ground-state / electrostatic**:

- **U(d) rise** is electrostatics. An interlayer exciton is a permanent dipole `p = e·d`;
  the on-site (blockade) dipole–dipole energy goes as **U ∝ d²**. No BSE for the exponent.
- **λ_f (f decay)** is a **tunneling/overlap** problem: the spatially-indirect optical
  matrix element decays as the electron/hole Bloch tails overlap across the vdW gap,
  `f ∝ exp(−2κd)`, `κ = √(2m*Φ)/ħ` — Φ (band offset) and m* are **DFT ground-state**
  quantities. Two independent routes agree:

| route | inputs | λ_f |
|-------|--------|-----|
| DFT tunneling | Φ≈1.0–2.0 eV, m*≈0.4–0.6 | 0.09–0.15 nm |
| measured hBN-spacer decay | ~1 decade of f per inserted hBN monolayer (Rivera 2015, Nagler 2017, Jauregui 2019) | 0.11–0.15 nm |

They land on **λ_f ≈ 0.14 nm** — a ~30 % cross-check, no BSE required.

**Consequence (the reframe), corrected against literature.** Because an exponential always
beats a power law, the net figure `N(d)=U·f^α` (α=0.5) has a finite optimum — and with
λ_f≈0.14 nm it sits **at native contact** (`2λ_f/α ≈ 0.56 nm < d₀`): adding any spacer only
darkens f. My first pass used an *electrostatic* prefactor that put `U ≈ 100 meV` at contact
and concluded "U is not the bottleneck." **A literature check corrected that number.** The
**measured** on-site dipolar interaction is only **≈4–20 meV** (biexciton blueshift 8.4 meV;
tri/quad/quint 12.4/15.5/18.2 meV; density blueshift up to ~20 meV — Kremser/Nagler et al.,
*npj 2D Mater. Appl.* **4**, 8 (2020)) — i.e. **comparable to the 6–15 meV Γ floor, not 15×
above it**. So U is **marginal** (U/Γ ≈ 0.3–3, straddling the 0.71 threshold), not plentiful.
And the darkness is *worse* than I claimed: interlayer radiative lifetime ≈0.4 ns vs
intralayer ≈1.8–2 ps (Palummo et al., *Nano Lett.* **15**, 2794 (2015)) ⇒ **~200–1000×
darker**, not 10–100×. **Corrected verdict:** the dipolar leg is **tight on both axes at a
single trap** — U barely at threshold, f the harder gate — not "U-plentiful, f-only." The
remaining unknown is still the **absolute f of the native interlayer exciton** (the BSE
number), but U is no longer a comfortable margin. (`phase2_lambda_f_proxy.json`, tier
`dft_proxy + measured`; the `literature_check_2026_08` block records each correction.)

**The applied-math route has precedent.** The proposed way to close the leg *without* compute
— a Kramers-Kronig / sum-rule inequality bounding usable nonlinearity — is a real result for
the analogous problem: a noninstantaneous χ³ causally precludes high-fidelity single-photon
Kerr phase shifts (Shapiro 2006; Gea-Banacloche 2010). So "close it with a KK bound" is
grounded, not speculative — a genuine alternative to the BSE for the dipolar leg.

## The sum-rule bound, worked out (the analytic attempt)

I took the crack. The KK linkage has an exact discrete form — a **two-site oscillator-
strength-borrowing model**. An interlayer exciton has the hole in layer B and the electron in
`|ψ_e⟩ = √(1−x)|A⟩ + √x|B⟩`, where `x` is the electron weight in the hole's layer (set by
tunneling/offset). Two consequences follow at leading order:

- **Brightness is borrowed:** `f(x) = x·f₀` — the transition is bright only through the
  layer-B amplitude (a sum-rule budget: f is taken from the intralayer transition).
- **At the cost of the dipole:** `d_eff = (1−x)·d₀ ⇒ U(x) = (1−x)²·U_max`.

So brightening (needed to strong-couple) directly kills the dipole (needed to blockade) — the
KK trade-off, in one parameter. A usable exciton needs **both**: strong coupling
`Ω₀√x > sc·Γ` (⇒ `x > (sc·Γ/Ω₀)²`) and blockade `(1−x)²U_max > 0.71·Γ`
(⇒ `x < 1−√(0.71Γ/U_max)`). A window exists iff

> **(★)  (sc·Γ/Ω₀)² < 1 − √(0.71·Γ/U_max)**

with a hard **necessary condition**: even the max-dipole `x→0` exciton needs `U_max > 0.71·Γ`,
or the leg is closed at *every* brightness.

**Result** (literature box: Γ∈[6,15], U_max∈[8,20], Ω₀∈[20,50] meV; `phase2_dipolar_sumrule.json`):

| strong-coupling criterion | open fraction of box | nominal (Γ10,U14,Ω35) |
|---|:--:|:--:|
| bare onset `Ω>Γ` (sc=1) | **76 %** | OPEN (window x≈0.09–0.27) |
| resolved doublet `Ω>2Γ` (sc=2, device-real) | **37 %** | **CLOSED** |

So the sum rule does **not** cleanly close the dipolar leg — but it converts "undetermined"
into a sharp, criterion-sensitive knife-edge: the route survives only on the favorable corner
(low Γ≈6, high U_max≈20, high Ω₀≈50 meV) **and** only if a bare Rabi onset counts as usable;
demand a resolved polariton and the nominal case closes. The pessimistic corner is closed
outright (`U_max < 0.71·Γ`). This is a stronger statement than the search could make: the leg
is closed *except on a thin, explicitly named corner*, and (★) says exactly which two numbers
keep it alive — `U_max` (the dark, max-dipole interaction) and whether an interlayer exciton
in the x≈0.1–0.3 sliver actually reaches `Ω>Γ` at 300 K. Those are precisely what the source-
built BSE (absolute f, hence Ω) and a DFPT dipole (U_max) would nail. Tier: **model + measured**,
not GW-BSE.

**Alternative engines** if the absolute f is wanted rigorously: Yambo-from-source (pinned
MPI/ScaLAPACK/FFTW/HDF5 — in progress), BerkeleyGW (gated source, layer wired), or **ABINIT
BSE** (conda-forge, an independent second engine).

## Fork B — WALLED: Yambo-from-source impractical in this proxy'd environment

Building Yambo 5.1.2 from source (to escape the conda-binary BSE abort) cleared the FPP
configure hurdle and the fake-ssh/MPI issues, but hit a hard wall: **Yambo's build system
insists on compiling its own bundled `libxc-5.1.5` from a source tarball it fetches at build
time**, and the agent proxy blocks that download (`gzip: not in gzip format`). Pinning a
compatible external `libxc=5.2.3` and passing explicit `--with-libxc-libs`/`--with-libxc-
includedir` did **not** override it — a known Yambo trait (its libxc coupling ignores the
external-lib flags unless the exact Fortran `.mod` files are present). Four build iterations
(~50–80 min each on a slow Modal builder) all ended the same way. **Conclusion: Yambo-from-
source is not practical here.** The remaining compute route for the absolute interlayer `f`
is ABINIT's BSE (clean conda-forge install, no source build) — but its marginal value is now
low, because the **analytic sum-rule bound** (above) already localizes the dipolar leg to a
thin, named corner without any BSE. The honest recommendation is to bank the analytic bound
as the dipolar-leg deliverable and treat a hard `f` as optional refinement, not a gate.

## Solve-for-it — inverting the bound to name the 1–2 survivors

Run in reverse: assume a rare solution exists and solve for the corner it must occupy
(`scripts/phase2_solve_for_it.py`, screening the 133 stable/non-magnetic/E_B≥0.3 eV C2DB
materials). Two archetypes fall out — the "one or two."

**Route B — intrinsic-dipolar Janus monolayers (the real find).** The interlayer dipolar
leg closed because separating e/h to get a dipole makes the exciton dark. A **Janus TMD**
(broken top/bottom mirror symmetry — different chalcogens) carries a *built-in* out-of-plane
dipole while the exciton stays **intralayer and bright** — it sits on the opposite, favorable
side of the U–f trade-off that the sum rule is built on. Top screened hits are **synthesized**
materials: **WSSe (WSeS)** and **MoSSe (MoSeS)** — real Janus TMDs (Lu et al. 2017), dipz≈0.037,
E_B≈0.5 eV, bright, non-magnetic, stable. The one deciding measurement: a GW-BSE exciton
**dipole + f** for MoSSe/WSSe — is the built-in dipole enough for U>0.71·Γ while f stays bright?
A Janus *homobilayer* (aligned dipoles) or Janus/TMD stack tunes U up if the monolayer is short.

**Route A — moiré single-emitter (the fallback).** A tightly-bound, weakly-polar monolayer
localized to ONE moiré/defect trap so Γ→phonon floor and confinement→U. Robust pick: a
**WS₂-class TMD** (bright, weakly polar, well-characterized), run as a single emitter in a
strong-coupling cavity. (The screen's exotic high-U/Γ hits — H₂C₂, ZrI₂, TiX₂ — are partly a
proxy artifact: zero *polar* coupling ≠ zero linewidth; an acoustic/deformation floor remains.)

**So the answer we solved for:** not a list — **a Janus-TMD intrinsic-dipolar exciton (MoSSe/
WSSe), and/or a moiré-trapped single emitter in a Γ-minimal TMD**, both living in the bound's
one open corner, each decided by a single GW-BSE number. Tier: proxy (C2DB + inverted bound),
hypotheses to confirm — not a verdict.
