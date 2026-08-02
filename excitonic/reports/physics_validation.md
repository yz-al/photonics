# Physics validation — gates before any GW-BSE spend

Per review: before spending on GW-BSE, test whether the pipeline reproduces
physics we already know and breaks the way materials physics says it should.
These are trend/consistency checks the code cannot accidentally pass.
Reproducible: `scripts/physics_validation.py` → `data/manifests/physics_validation.json`.
Assertions live in `src/exciton_fm/acceptance.py`.

## Runnable now — ALL PASS

| Check | Result | Detail |
|-------|:------:|--------|
| **Γ(T) shape** | ✅ | monotonic; high-T ~linear (slopes 0.073, 0.089); low-T suppressed (Γ(10)/Γ(300)≈0). Rises with the LO Bose occupation and saturates below ħω_LO — the T-dependence the whole FOM rests on, verified as a curve. |
| **TMD chemical trend** | ✅ | MoS₂(0.547) > MoSe₂(0.500) > MoTe₂(0.455) eV (binding falls down the chalcogen); \|MoS₂−WS₂\|=0.031 eV (small). The C2DB labels reproduce the established ordering. |
| **Reduced-mass direction** | ✅ | Spearman(μ, E_b) = 0.47 (p=5×10⁻²¹, n=356). E_b tracks reduced mass as Wannier-Mott (E_b ∝ μ/ε²) requires — not broken upstream. |

## Gated checks (must pass before their tier's output is accepted)

No "accept whatever finished" on a single expensive run — these are pass/fail gates:

| Check | Gate | What it catches |
|-------|------|-----------------|
| **Born-charge polarity** (Si≈0 vs GaAs≈2.2) | DFPT | a Fröhlich contribution for non-polar Si ⇒ the phonon pipeline is wrong. *(DFPT run in flight; auto-resolves.)* |
| **Dimensionality trend** (mono > bi > bulk binding) | GW-BSE | screening treatment wrong if binding doesn't fall as layers/screening are added. |
| **Vacuum-truncation plateau** | GW-BSE | without Coulomb truncation the 2D binding diverges (log) with vacuum; if MoS₂ binding keeps climbing, truncation is off — poisoning *every* 2D number. |
| **BSE ↔ GW energy reference** (lowest BSE = GW gap − E_b) | GW-BSE | misaligned energy references. |
| **Interlayer-exciton signature** | GW-BSE | **the project-critical one.** A type-II interlayer exciton must be lower in energy than both intralayer excitons **and** far weaker in oscillator strength (spatial separation). If f_inter ≈ f_intra, the charge separation — the entire source of the dipolar U — is not being captured. |

## Anchor spread (chosen for range, not convenience)

| Anchor | E_b | Regime |
|--------|-----|--------|
| GaAs | ~4 meV, a_B ~10 nm | 3D, cold, polar |
| MoS₂ | ~0.5 eV, a_B ~1 nm | 2D |
| halide perovskite | — | Fröhlich LO ~40–70 meV |

Four orders of magnitude in binding and the full polarity range. **A pipeline that
gets all three is doing physics; one that only gets TMDs is tuned to TMDs — exactly
the failure the surrogate already showed at Phase 1.** The interlayer-exciton gate is
the highest-value check because it tests the specific physics this project depends on,
not the general correctness of the code.

## Gate on the expensive run

The MoS₂/MoSe₂ GW-BSE cost measurement does not proceed until: (1) the DFPT
Born-charge polarity gate passes, and (2) the BSE-gated checks are wired to run on
the first (cheap-as-possible) BSE output and are treated as acceptance criteria —
so a run that "finished" but fails a gate is rejected, not banked.
