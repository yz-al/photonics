# Excitonic-property foundation model

An ML surrogate that predicts **excited-state / many-body optical properties**
from crystal structure, to search for (or rule out) a **room-temperature
optical-nonlinearity** material — the one physics-permitted lever that would
reopen general optical neural-network compute via single-photon/polariton
**blockade**.

Blockade opens when **U/Γ ≳ 0.7**, i.e. the figure of merit
**U_exc > 0.71·Γ_x(300 K)** at high exciton fraction, large binding (thermal
stability), and telecom-band integrability. Best measured to date is U/Γ ≈ 0.05
(cryogenic); the RT gap is ~1–2 orders and unquantified on the upside.

## Why a new model
Existing materials foundation models (GNoME, MatterGen, MACE-MP, Orb) predict
**ground-state** DFT properties. The properties that decide blockade are
**excited-state and many-body** — exciton binding, Fröhlich linewidth,
exciton–exciton U, oscillator strength — requiring GW-BSE + electron-phonon
(DFPT/EPW), and not available at foundation scale. This project builds that tool.

## Targets (predict from structure)
1. Exciton binding energy `E_b` (eV) — want large (>0.2 eV RT stability)
2. RT homogeneous linewidth `Γ(300 K)` (meV) — Fröhlich/LO dephasing; want ~10 meV
3. Exciton–exciton interaction `U` — want large (saturation vs dipolar/Rydberg)
4. Oscillator strength / transition dipole — strong-coupling |X| tradeoff
Derived: `U/Γ(300 K)` and the blockade FOM.

## Phase 0 — done (see `reports/phase0_data_inventory.md`)
Live C2DB inventory. **17,001 materials.** Label coverage:

| Target | Label | Coverage | Quality |
|--------|-------|---------:|---------|
| `E_b` | `E_B` (BSE) | 370 | GW-BSE |
| `Γ(300 K)` | — | **0** | **GAP** |
| `U` | — | **0** | **GAP** |
| `f_osc` | `alpha*_el` | 5352 | DFT proxy |

**The FOM is computable for 0 / 17001 materials on existing data** — both Γ and U
have zero labels. That gap is the reason this project exists, now measured.
The GW-BSE core of C2DB is ~283–370 materials, not thousands.

## Layout
```
excitonic/
  src/exciton_fm/        c2db_client.py · targets.py · inventory.py
  scripts/               phase0_inventory.py       (local run)
  data/manifests/        c2db_key_inventory.json · phase0_coverage.json
  modal_app.py           Modal job (Phase 0 inventory; GHA -> Modal proof)
  reports/               phase0_data_inventory.md  (the deliverable)
.github/workflows/       exciton-phase0-modal.yml
```

## Reproduce
```bash
python excitonic/scripts/phase0_inventory.py     # local, plain HTTPS to C2DB
# or via GitHub Actions -> "Excitonic Phase 0 ... on Modal" -> Run workflow
#   (runs `modal run excitonic/modal_app.py`)
```

## Compute — Modal
Training the GNN surrogate → Modal GPU. GW-BSE (BerkeleyGW/Yambo) + EPW label
generation → Modal **CPU/MPI** (these are not GPU workloads). `modal run` is
routed through a GitHub Action because the local sandbox blocks Modal's gRPC
egress.

## Rigor rules
Every predicted property carries an uncertainty and an explicit label-quality
tier (`none`/`dft_proxy`/`gw_bse`/`epw`/`measured`). No GW-BSE or EPW result is
reported unless it actually ran; Γ/U/FOM are withheld until Phase 2 generates them.

## Roadmap
- **P1** multi-task surrogate on `E_b` + `f_osc` + aux; validate vs anchors (GaAs, TMD, perovskite).
- **P2** stand up GW-BSE + EPW on Modal CPU; generate Γ and U on a ~100–500 seed set.
- **P3** active learning: surrogate → propose high-U/Γ → GW-BSE → retrain.
- **P4** screen for RT-blockade candidates; rank with uncertainty; honest reachability verdict.
