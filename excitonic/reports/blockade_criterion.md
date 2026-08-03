# Blockade criterion — 300 K locked, reported as a Γ(T) curve

**Decision (locked): the pass condition is 300 K. The per-material headline is the
crossover temperature T\*** — the temperature at which `U = 0.71·Γ(T)` — reported with
the full Γ(T) curve, not a bare 300 K pass/fail.

Code: `src/exciton_fm/fom.py` (`crossover_temperature`, `blockade_vs_temperature`).
Demonstration + self-test: `scripts/phase2_crossover_demo.py` →
`data/manifests/phase2_crossover_demo.json`.

## Why temperature is not a knob to relax

Temperature is not one of the seven spec conditions because it exists to serve two of
them: **pump-free operation** and **CMOS integrability**. Relaxing it fails on three
independent grounds:

1. **Energy accounting.** A 77 K system needs continuous cryogenic cooling, and cooling
   power dominates the ledger by orders. This program killed architectures on 44 pJ/MAC
   of *thermal hold* — standing power that runs whether or not the network computes. A
   cryostat is that same argument at much larger scale, and it does not amortize. A
   77 K "yes" fails the energy accounting the whole program is built on → not a win
   under our own criteria.
2. **Ranking capture.** At 77 K the Γ denominator drops ~10×, so materials clear the FOM
   by having *quiet phonons*, not *large U*. The search would return the lowest-linewidth
   materials rather than the strongest-interaction ones — ranked by the variable we are
   trying to hold fixed. We would learn about phonon dephasing (already well
   characterized) instead of about the interaction (the unmeasured quantity).
3. **The cryogenic corner is not empty.** Best measured anywhere is U/Γ ≈ 0.05 at a few
   kelvin — ~14× short of threshold at a temperature far below 77 K. Relaxing temperature
   entirely, to conditions more forgiving than 77 K, still does not reach blockade. A
   77 K target re-asks a question the literature has already answered negatively at
   lower temperature.

## What we report instead — the Γ(T) curve and T\*

Because U is ~temperature-independent (the interaction) and
`Γ(T) = Γ0 + γ_LO·n_LO(T)` rises monotonically, blockade holds for `T < T*` and fails
above. So for each material we invert the FOM and report:

- **T\*** — the crossover temperature (or **"never"** if `U/0.71 ≤ Γ0`, i.e. the
  interaction is below the T→0 linewidth floor — the strongest possible negative).
- **300 K headroom** — `U/Γ(300 K)` and the **shortfall factor** (how many× U must grow
  to clear at 300 K).
- The **sampled Γ(T) curve** (4, 77, 150, 200, 250, 300 K).

This gives the information a relaxed target would have given — *at what temperature would
this material cross* — without letting the denominator capture the ranking, and it makes
the negative result stronger: not just "nothing clears at 300 K" but **by how much and at
what temperature anything would.** A material that crosses at 250 K is genuinely
interesting and shows up as `T*≈250 K`; most will read `T*` at absurd temperatures or
`never`.

## Demonstration (criterion only — not a physics verdict)

| Case | U | T\* | 300 K ratio | Shortfall @300 K |
|------|---|:---:|:---:|:---:|
| Best-measured cryo anchor (U/Γ≈0.05 @ 4 K) | ~0.10 meV | **never** | 0.021 | 33.9× |
| Illustrative near-RT crosser | ~2.7 meV | **250 K** | 0.567 | (crosses at 250 K) |

The inversion self-test confirms `Γ(T*) = U/0.71` to < 0.05 meV. The best-measured
cryogenic point **reproduces the known negative** — it never crosses at any temperature
and is ~34× short at 300 K — which is exactly why relaxing T is not a new region to
search. The illustrative case shows a near-RT crosser would be surfaced as `T*≈250 K`.

## Rigor

`T*` is only as good as its inputs: U from **GW-BSE**, and Γ(T) = Γ0 + γ_LO·n_LO(T) from
**EPW** (γ0, γ_LO) + **DFPT** (ħω_LO). Until those land per material, `blockade_vs_temperature`
returns a model-tier `T*` flagged as a criterion demonstration, never a first-principles
result. The Γ(T)-curve report is now the standard output for every candidate — replacing
the single-point 300 K verdict.
