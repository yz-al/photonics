# Phase 2 — GW-BSE + EPW label-generation pipeline (scaffold)

**Goal:** generate the two decision-critical labels that Phase 0 showed have
**zero** coverage anywhere — the 300 K Fröhlich linewidth **Γ** and the
exciton–exciton interaction **U** — on a surrogate-prioritized seed set, using
first-principles codes run as many-core **Modal CPU/MPI** jobs.

**Status: scaffold complete and runnable end-to-end; no production GW-BSE/EPW
labels generated yet.** Every piece that can run without a supercomputer job runs
and is verified here. The heavy first-principles runs are deliberately gated —
and until one completes, Γ/U are reported as `not_run` or as explicitly
`*_model` estimates, **never** as fabricated first-principles numbers.

---

## Pipeline

```
seed select ──► structures ──►  QE  e-ph chain:  pw.x → ph.x → epw.x        ──► Γ(300 K)   [tier: epw]
    │                        └►  BerkeleyGW GW-BSE: pw.x → pw2bgw → epsilon
    │                             → sigma → kernel → absorption             ──► E_b, U     [tier: gw_bse]
    └► (model floor) Fröhlich Γ  &  saturation/dipolar U                    ──► Γ, U        [tier: *_model]
```

| Stage | Module / script | Runs now? |
|-------|-----------------|:--:|
| Seed selection | `seed_select.py`, `scripts/phase2_seed_select.py` | ✅ |
| 2D structures | `structures.py` (ASE prototypes; anchors) | ✅ |
| QE e-ph decks | `qe_inputs.py` (scf/ph/epw) | ✅ |
| BerkeleyGW decks | `bgw_inputs.py` (pw2bgw/epsilon/sigma/kernel/absorption) | ✅ |
| Model Γ estimate | `frohlich.py` (`frohlich_model`) | ✅ |
| Model U estimate | `exciton_u.py` (`saturation_model`) | ✅ |
| FOM w/ provenance | `fom.py` | ✅ |
| Modal QE+BGW image | `modal_phase2.py` (`smoke_test`) | ▶ dispatch |
| Production Γ/U labels | `modal_phase2.py::label_material` | ⏸ structure+pseudo-gated |

## Seed set (`data/manifests/phase2_seed_set.json`)

Ranked by a physics-aware score — large `E_b` (thermal stability), low Fröhlich
proxy (small lattice/IR-to-electronic polarizability ratio), useful optical-gap
window, oscillator strength, and stability — across the brief's target families.
Top candidates are large-binding transition-metal halides (TiCl₂, ZrBr₂…) and
noble-metal chalcogenides (Pd₂S₄, Au₂Se₂…); family breakdown spans TMD, halide,
halide-perovskite-like, chalcogenide, oxide, and organic. In Phase 3 this static
score is replaced by the surrogate's predictive **uncertainty** (active learning).

## First-principles methods

- **Γ(300 K):** QE DFPT (`ph.x`, with `epsil=.true.` → Born charges + dielectric
  tensor) → `epw.x` with `lpolar=.true.` (2D Fröhlich treatment, Sohier/Verdi/
  Giustino). The 300 K homogeneous linewidth comes from the imaginary
  exciton/carrier–phonon self-energy. Decks: `qe_inputs.py`.
- **E_b, U:** BerkeleyGW G₀W₀ (`epsilon`→`sigma`) + BSE (`kernel`→`absorption`,
  2D truncation). `E_b` cross-checks the C2DB label; the two-particle kernel /
  biexciton channel gives the saturation `U`, and an interlayer/dipolar treatment
  gives the a_B-decoupled dipolar `U`. Decks: `bgw_inputs.py`.
- **Compute:** GW-BSE and EPW are MPI/CPU-bound → Modal `cpu=16` functions, not
  GPU (`modal_phase2.py`), launched via GitHub Actions (`modal run`).

## Model-tier floor + anchor sanity check (`data/manifests/phase2_anchor_estimates.json`)

The analytic Fröhlich/saturation models give an honest prioritization floor and,
usefully, **reproduce the anchors** (they are calibrated to them, so this is a
consistency check, not independent validation):

| Anchor | Γ(300 K) model | a_B model | lit. anchor |
|--------|---------------:|----------:|-------------|
| MoS₂  | 12.3 meV | 0.98 nm | Γ 5–15 meV, a_B ~1 nm (Selig/Moody) |
| WS₂   | 12.8 meV | 1.20 nm | ″ |
| MoSe₂ | 16.5 meV | 1.00 nm | ″ |
| WSe₂  | 17.5 meV | 1.21 nm | ″ |

Model `U/Γ ≈ 0.02–0.035` at a 100 nm reference mode area — same order as the best
measured `U/Γ ≈ 0.05`. **These are `*_model` tier, not EPW/BSE**, and `U` is
mode-area-dependent (a device parameter). They exist to prioritize, not to decide.

## Rigor (enforced in code)

- `provenance.py` defines the tier ladder; `Label` forbids a value on tier
  `not_run` and only reports `computed=True` for a finished first-principles run.
- `parse_epw_linewidth` / `parse_bse_biexciton` return `not_run` if the output is
  missing or the job did not reach `JOB DONE` — they never guess.
- `fom.blockade_fom` carries the **weakest** input tier into the FOM, and stamps a
  caveat whenever the FOM is built from model estimates.

## Run it

```bash
python excitonic/scripts/phase2_seed_select.py 150     # seed set
python excitonic/scripts/phase2_gen_inputs.py          # decks + model floor (anchors)

# Verify the QE+BerkeleyGW container on Modal (via GitHub Actions):
#   Actions → "Excitonic Phase 2 (GW-BSE + EPW) on Modal" → Run workflow (smoke)
#   → modal run excitonic/modal_phase2.py::smoke
```

## To turn the scaffold into real labels (Phase 2 execution)

1. Stage **relaxed C2DB structures** for the seed set (download the C2DB ASE db)
   and a **pseudopotential library** (SSSP/ONCV) into `label_material`'s workdir.
2. Converge grids/cutoffs per material (the deck parameters are placeholders).
3. Run `modal run excitonic/modal_phase2.py::generate`; parse Γ (`epw`) and U
   (`gw_bse`); write labels with real provenance.
4. Feed the new labels back into the surrogate → **Phase 3** active learning.

Until step 3 completes for a material, its Γ/U stay `not_run` — that is the honest
state, and the deliverable here is the pipeline that will fill them, not the
numbers themselves.
