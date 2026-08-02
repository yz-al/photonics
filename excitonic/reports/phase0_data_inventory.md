# Phase 0 — Data Inventory & the Excitonic Label Gap

**Project:** Excitonic-property foundation model for a room-temperature optical-nonlinearity (single-photon/polariton **blockade**) material search.
**This milestone:** pull C2DB, inventory which of the four targets already have labels, and quantify the *exact* size of the Fröhlich-linewidth + exciton–exciton-U data gap — **before any model code**.
**Status:** complete. All numbers below are live queries against C2DB, reproducible via `excitonic/scripts/phase0_inventory.py`. Nothing here is fabricated; where a label does not exist, the count is literally zero and is reported as a gap.

---

## TL;DR

| # | Target | C2DB label | Coverage | Quality | Verdict |
|---|--------|-----------|---------:|---------|---------|
| 1 | **Exciton binding energy** `E_b` | `E_B` (BSE) | **370 / 17001** (2.2%) | **GW-BSE** | ✅ have (small but real) |
| 2 | **RT homogeneous linewidth** `Γ(300 K)` | — none — | **0 / 17001** | — | ❌ **THE GAP** |
| 3 | **Exciton–exciton interaction** `U` | — none — | **0 / 17001** | — | ❌ **THE GAP (deepest)** |
| 4 | **Oscillator strength / dipole** `f_osc` | `alpha*_el`, `plasmafrequency_*` | 5352 / 17001 (31%) | **DFT proxy** (PBE/RPA) | ⚠️ proxy only |

> **The gap, in one line:** the two properties that *decide* blockade — the 300 K Fröhlich linewidth Γ and the exciton–exciton interaction U — have **zero labels in the largest existing excited-state 2D database**. The FOM `U_exc > 0.71·Γ_x(300 K)` cannot be evaluated for a single one of the 17,001 C2DB materials from existing data. That is the entire justification for this project, now quantified rather than asserted.

---

## 1. What C2DB actually is (as of this pull)

- Source: **C2DB**, served through its live ASE-db web front-end at `https://c2db.fysik.dtu.dk/` (the DB moved to `https://2dhub.org/`; the front-end is the public, no-login access path). First-principles results from the GPAW code.
- **Total materials currently in C2DB: 17,001.** (The project brief cited "~4000"; C2DB has grown ~4× since. Reported here for honesty — the headline count is bigger, but see the crucial caveat next.)
- **Method:** per-property coverage is measured with the ASE-db filter API — `filter=<key>` returns "N rows out of M", i.e. the number of materials carrying a non-null value for that key. No scraping of individual rows is needed to inventory coverage. See `src/exciton_fm/c2db_client.py`.

### Crucial caveat on "4000 with GW-BSE"
The brief's mental model — "C2DB has GW-BSE for ~4000 materials" — does **not** survive contact with the data. GW-BSE is expensive and was run **selectively**:

- `E_B` (BSE exciton binding): **370** materials.
- `gap_gw` (G₀W₀ gap): **339** materials.

So the GW-quality excited-state core of C2DB is **~340–370 materials, not thousands.** Everything else in C2DB is PBE/HSE-level ground-state or DFPT data. This makes the bootstrap set *smaller and more precious* than assumed, and sharpens the case for generating our own labels.

---

## 2. Target-by-target inventory

### Target 1 — Exciton binding energy `E_b` ✅ (have; GW-BSE)
- **Label:** `E_B` = "Exciton binding energy (BSE) [eV]". **370 materials.**
- **Quality:** genuine **GW-BSE**. This is the highest-value existing signal and our primary validation anchor.
- Nearly all of these are large-binding (2D confinement): **364 / 370 have E_b > 0.2 eV** — i.e. C2DB's BSE set is almost entirely in the RT-thermally-stable regime already. Good news for coverage of the "want large binding" axis; bad news for *variance* (little training signal at small E_b except the anchors we add by hand, e.g. GaAs).

### Target 2 — RT homogeneous linewidth `Γ(300 K)` ❌ **GAP**
- **Label in C2DB: none. Zero keys, zero materials.**
- Probed candidate key names (`linewidth`, `gamma`, `frohlich`, `omega_LO`, `E_LO`, `born_charge`, …) — all return 0 / unknown-key.
- What C2DB *does* carry that is *adjacent*: `alphax_lat/alphay_lat/alphaz_lat` = "Static polarizability (phonons)" (DFPT) on **2001** materials. This is a **Fröhlich ingredient** (the IR/lattice dielectric response), **not** the temperature-dependent exciton dephasing linewidth. Γ(300 K) requires the full exciton–phonon coupling and a finite-T dephasing evaluation (**DFPT + EPW**, or a Fröhlich-model treatment on top of the phonon + exciton data).
- **This label must be generated. 0 exist; up to 17,001 could in principle be computed; the seed target is ~100–500 (Phase 2).**

### Target 3 — Exciton–exciton interaction `U` ❌ **GAP (deepest)**
- **Label in C2DB: none. Zero keys, zero materials.** Not tabulated anywhere at scale.
- Requires biexciton binding / exciton-saturation from BSE-level many-body, or a dipolar/interlayer treatment for dipolar-sourced U (which decouples U from the Bohr radius and thus from the thermal-stability tradeoff — a key search direction).
- Only a **crude proxy** is constructible from existing C2DB features: saturation-U scales roughly as (area / a_B²), and a_B follows from `emass_cbm`/`emass_vbm` + `E_B`. `E_B ∧ emass_cbm` covers **369** materials, so a proxy-U is computable for the BSE set — but it is a proxy, must be flagged as such, and says nothing about *dipolar* U.
- **This is the property the whole moonshot turns on, and it is the least labeled.**

### Target 4 — Oscillator strength / transition dipole `f_osc` ⚠️ proxy
- **Labels:** `alphax_el/alphay_el/alphaz_el` = "Interband polarizability" (**5352** materials) and `plasmafrequency_x/y` (**3931**).
- **Quality:** PBE/RPA electronic response — a **DFT-level proxy** for oscillator strength, **not** a GW-BSE transition dipole. Usable as a feature and a weak label; must be tier-flagged `dft_proxy`, never treated as GW-quality.

---

## 3. Auxiliary features present (useful, not targets)

| Key | Coverage | Use |
|-----|---------:|-----|
| `gap_gw` / `gap_dir_gw` | 339 / 339 | GW electronic structure (pairs with E_B) |
| `gap_hse` | 3363 | hybrid-level gap (mid-tier) |
| `gap` (PBE) | 8795 | DFT gap (broad, cheap) |
| `emass_cbm` / `emass_vbm` | 3664 / 3664 | → exciton reduced mass → Bohr radius → saturation-U proxy |
| `alphax_lat` (phonon polariz.) | 2001 | Fröhlich ingredient (not the linewidth) |
| `dipz` | 8795 | out-of-plane dipole (dipolar-exciton hint) |
| `ehull`, `dyn_stab` | 17001 / — | stability filters for candidate screening |

**Usable GW-BSE training core (intersections):**

| Set | Count |
|-----|------:|
| `E_B` | 370 |
| `E_B ∧ gap_gw` | 283 |
| `E_B ∧ emass_cbm` | 369 |
| `E_B ∧ alphax_el` | 369 |
| `E_B ∧ alphax_el ∧ emass_cbm` | 368 |
| `E_B ∧ alphax_lat` (has Fröhlich ingredient too) | 262 |
| `E_B > 0.2 eV` (RT-stable) | 364 |

So the clean multi-property GW core is **~283–370 materials**. That is the entire high-quality label budget the field has handed us for free.

---

## 4. The gap, quantified (the deliverable)

Of the **4** targets that decide the blockade FOM:

- **2 are labeled** in C2DB: `E_b` (370, GW-BSE) and `f_osc` (5352, DFT proxy).
- **2 are the gap**, with **exactly zero labels** across all 17,001 materials:
  - **Γ(300 K)** — Fröhlich/LO-phonon homogeneous linewidth → **0 labels → must generate (DFPT + EPW).**
  - **U** — exciton–exciton interaction → **0 labels → must generate (BSE biexciton/saturation, + dipolar treatment).**

**The FOM `U_exc > 0.71·Γ_x(300 K)` is therefore computable for 0 / 17001 C2DB materials today.** Both of its inputs are missing. Ground-state foundation models (GNoME, MatterGen, MACE-MP, Orb) do not close this because these are excited-state / many-body / finite-T electron-phonon quantities, not ground-state DFT properties. **This is precisely the gap the project exists to fill, now measured.**

### What "generate the labels" means concretely
- **Γ(300 K):** stand up **QE DFPT + EPW** (CPU/MPI-bound Modal jobs) to get exciton–phonon coupling and the 300 K dephasing linewidth. Seed set ~100–500, prioritizing large-binding / low-Fröhlich families (TMDs, halide perovskites, organics/J-aggregates, 2D chalcogenides, interlayer/dipolar stacks).
- **U:** **BerkeleyGW / Yambo BSE** for saturation/biexciton U on the same seed set, plus a dipolar/interlayer model where the physics decouples U from a_B.
- **Active learning** then chooses which material to run next from surrogate uncertainty, so we cover chemical space without GW-BSE on all 17k.

---

## 5. Validation anchors locked in (for Phase 1)

These are literature reference points the surrogate must reproduce (not predictions):

| Material | Family | E_b | a_B | Γ(300 K) | Source |
|----------|--------|-----|-----|----------|--------|
| GaAs QW | III–V | ~4 meV | ~10 nm | — (cold only) | textbook |
| Monolayer MoS₂/WSe₂ | TMD | ~0.5 eV | ~1 nm | ~5–15 meV | Selig Nat.Commun. 2016; Moody Nat.Commun. 2015 |
| MAPbI₃ perovskite | halide perovskite | — | — | Fröhlich LO ~40–70 meV | Wright Nat.Commun. 2016 |

Encoded in `src/exciton_fm/targets.py::ANCHORS`.

---

## 6. Reproduce this

```bash
# Local (plain HTTPS to C2DB; no login):
python excitonic/scripts/phase0_inventory.py
#   -> excitonic/data/manifests/phase0_coverage.json

# On Modal, via the GitHub Actions path (proves the compute pipeline):
#   Actions tab -> "Excitonic Phase 0 (C2DB inventory) on Modal" -> Run workflow
#   which runs:  modal run excitonic/modal_app.py
```

Artifacts:
- `excitonic/data/manifests/c2db_key_inventory.json` — full C2DB key schema with provenance tiers.
- `excitonic/data/manifests/phase0_coverage.json` — live coverage counts (regenerated by the script).

---

## 7. Honest verdict going into Phase 1

- The "key existing resource" (C2DB GW-BSE) is **~300–370 materials**, not thousands — the bootstrap is small.
- Two of the four decision-critical targets have **no labels at all**; the blockade FOM is **currently unevaluable** on public data.
- Phase 1 can legitimately train a multi-task surrogate on `E_b` (GW-BSE) + `f_osc` (DFT proxy) + auxiliary GW/DFT features and validate against the anchors — **but it cannot predict Γ or U until Phase 2 generates those labels.** Any Γ/U or FOM number before then would be fabrication, and will be labeled as such or withheld.

*Every predicted property will carry an uncertainty and an explicit label-quality tier (`none` / `dft_proxy` / `gw_bse` / `epw` / `measured`). No GW-BSE or EPW result will be reported unless it actually ran.*
