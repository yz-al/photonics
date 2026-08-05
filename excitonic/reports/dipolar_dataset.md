# Phase 2 — Dipolar / Interlayer-Exciton Dataset Inventory (BiDB + HetDB)

**Project:** Excitonic-property surrogate for a room-temperature optical-nonlinearity (single-photon **blockade**) material search.
**This milestone:** the U-definition decision (`reports/u_definition_decision.md`) moved the search objective's U from the saturation channel to the **dipolar / interlayer** channel — a permanent excited-state dipole `d` from a **spatially-indirect** exciton. Those excitons do not live in monolayer C2DB; they live in vdW **bilayers** and **heterostructures**. This report pulls and inventories the two databases that host them — **BiDB** and **HetDB** — the same way Phase 0 inventoried C2DB, and quantifies coverage of the fields a dipolar `U_dip` is built from.
**Status:** complete. Every count below is a live query against the public ASE-db front-ends, reproducible via `excitonic/scripts/phase2_dipolar_inventory.py`. Where a label does not exist, the count is reported as such — no fabrication.

---

## TL;DR

| Database | What it is | Total entries (live) | Dipolar-capable subset | The gap |
|----------|-----------|---------------------:|-----------------------:|---------|
| **BiDB** | vdW **homo**bilayers (same monolayer, all stackings) | **12,138** | **6,109** with broken inversion symmetry | no dipole `d`, no CT character, no U |
| **HetDB** | vdW **hetero**bilayers (two different monolayers) | **336** | **209** derived type-II (staggered) | no *stored* alignment label; no dipole/U |

> **The dipolar gap, in one line:** BiDB and HetDB give the *geometry* of interlayer separation (`interlayer_distance` d, 100% covered in both) and — in HetDB — the band-alignment *ingredients* (CBM/VBM wrt vacuum) needed to identify type-II staggered stacks. But **neither database tabulates the excited-state dipole `d_exc`, the exciton charge-transfer character, or U** — the very quantities the dipolar `U_dip` is made of. Exactly as in Phase 0, the decision-critical excited-state labels have **zero coverage at scale** and are the Phase-2 generation target. What BiDB/HetDB buy us is a **pre-screened seed pool** (dipolar-capable stacks) and a **geometric prior** on d.

---

## 1. Live endpoints found

Both databases are served by the **same ASE-db htmx web app** as C2DB (`c2db.fysik.dtu.dk`), so the Phase-0 client machinery transfers verbatim — get a session id from the landing page, then `GET /table?sid=…&filter=<expr>` returns a fragment whose header reads "N rows out of M"; `filter=<key>` counts materials that *have* that key.

| DB | Live front-end | Total rows (live `filter=` count) | Schema help |
|----|----------------|----------------------------------:|-------------|
| BiDB | `https://bidb.fysik.dtu.dk/` | **12,138** | `/help` |
| HetDB | `https://hetdb.fysik.dtu.dk/` | **336** | `/help` |
| C2DB (join target) | `https://c2db.fysik.dtu.dk/` | 17,001 | used for monolayer band edges |

Client: `excitonic/src/exciton_fm/bidb_client.py` (`StackDBClient`, with `bidb_client()` / `hetdb_client()` factories) — subclasses `C2DBClient`, reusing every network method unchanged; only the base URL, the column-header→key map, and the parse anchor differ per DB.

---

## 2. BiDB — vdW homobilayers (12,138)

BiDB stacks **two copies of the same C2DB monolayer** in every symmetry-distinct registry. Because both layers are the same material, **there is no intrinsic type-II band offset** — a homobilayer cannot have a staggered alignment between two identical gaps. A permanent **out-of-plane dipole is only symmetry-allowed when the stacking breaks inversion symmetry**; the resulting `U_dip` comes from *stacking-induced* charge transfer across `d`, not from a band offset.

### Schema keys relevant to dipolar U (live coverage)

| Key | Meaning | Coverage | Role in U_dip |
|-----|---------|---------:|---------------|
| `interlayer_distance` | Interlayer distance **d** [Å] | **12,138 / 12,138** | the geometric lever: `U_dip ∝ d²` |
| `ebind` | Interlayer binding energy [meV/Å²] | **12,138 / 12,138** | stacking cohesion (stability screen) |
| `has_inversion_symmetry` | Inversion symmetry (Yes/No) | **12,138 / 12,138** | broken ⇒ out-of-plane dipole allowed |
| `area` | Unit-cell area [Å²] | 12,138 / 12,138 | areal density n = 1/A ⇒ `U_dip ∝ 1/A` |
| `thickness` | Slab thickness [Å] | 12,138 / 12,138 | geometry |
| `number_of_layers` | = 2 for all | 12,138 / 12,138 | (confirms all are bilayers) |
| `slide_stability` | Stacking (slide) stability | 12,138 / 12,138 | mechanical stability screen |
| `gap_pbe` | PBE band gap [eV] | **6,886 / 12,138** | semiconducting screen (metals excluded) |
| `magnetic` | Magnetic | 12,138 / 12,138 | screen |
| `c2db_uid` / `monolayer_uid` | Parent monolayer id | 12,138 / 12,138 | join to C2DB E_b/edges |

### Derived (dipolar-capable) subsets — live

| Filter | Count | Meaning |
|--------|------:|---------|
| `has_inversion_symmetry=False` | **6,109** | broken inversion ⇒ **dipolar-capable** |
| `has_inversion_symmetry=True` | 6,029 | inversion-symmetric ⇒ dipole-forbidden |
| `interlayer_distance ∧ ebind` | 12,138 | full geometry + cohesion |
| `interlayer_distance ∧ broken inversion` | 6,109 | dipolar geometry present |
| `gap_pbe>0.1 ∧ broken inversion` | **2,548** | **semiconducting *and* dipolar-capable** (the tightest BiDB seed) |
| `slide_stability=Stable` | 3,141 | mechanically stable stackings |

**Read:** ~50% of BiDB (6,109) is symmetry-eligible for an out-of-plane dipole; requiring a real semiconducting gap narrows the seed pool to **2,548**. Every one of these has `d`, `ebind`, `area` at 100% coverage, so the *geometric* dipolar prior is complete — the missing piece is the exciton itself.

---

## 3. HetDB — vdW heterobilayers (336)

HetDB is the physically richer set: **two *different* C2DB monolayers** stacked, so a genuine **type-II staggered offset can exist**. type-II is the fingerprint of the dipolar exciton: electron and hole localise on **different layers** ⇒ spatially-indirect / charge-transfer exciton ⇒ permanent dipole `d ≈ interlayer_distance`.

### Schema keys relevant to dipolar U (live coverage)

| Key | Meaning | Coverage |
|-----|---------|---------:|
| `interlayer_distance` | Interlayer distance **d** [Å] | **336 / 336** |
| `uid_a`, `uid_b` | C2DB uids of the two layers (join keys) | 336 / 336 |
| `twist_angle` | Twist angle [°] | 336 / 336 |
| `maxstrain` | Max strain to commensurate cell [%] | 336 / 336 |
| `pbe_gap_soc` / `scs_gap_soc` | Heterostructure gap (PBE / LAPS) [eV] | 336 / 336 |
| `pbe_cbm_soc`, `pbe_vbm_soc` | **CBM/VBM wrt vacuum (PBE+SOC)** — band-alignment ingredient | **303 / 336** |
| `scs_cbm_soc`, `scs_vbm_soc` | CBM/VBM wrt vacuum (LAPS+SOC) | 310 / 336 |
| `pbe_evac` | Vacuum level [eV] | 336 / 336 |
| `area` | Unit-cell area [Å²] | 336 / 336 |

### Band-alignment TYPE — not a stored label; **derived**

There is **no explicit alignment-type key in HetDB**. Live probes `type_of_alignment`, `alignment`, `band_alignment`, `type` all return **0 / 336**. So "how many are type-II" cannot be read off a column — it must be **derived** from the band edges.

**Derivation (Anderson's rule):** for each heterostructure, take the two *isolated* monolayers' vacuum-referenced band edges from C2DB (join by `uid_a`/`uid_b`; the 336 stacks use only **38 unique monolayers**, all 38 having C2DB `vbm`/`cbm`). Then:
- heterostructure VBM = the higher of the two monolayer VBMs (highest occupied);
- heterostructure CBM = the lower of the two monolayer CBMs (lowest unoccupied);
- **type-II** if VBM and CBM come from **different** layers (staggered) and the gap stays positive;
- type-I if both from the same layer (straddling); type-III if the gap is broken (overlap).

**Derived counts (live, all 336 classifiable):**

| Alignment type | Count | Dipolar meaning |
|----------------|------:|-----------------|
| **type-II (staggered)** | **209** | **spatially-indirect exciton ⇒ permanent dipole ⇒ dipolar-capable** |
| type-I (straddling) | 78 | direct exciton, no interlayer dipole |
| type-III (broken gap) | 49 | semimetallic overlap, not a clean exciton |

> **209 of 336 HetDB heterostructures (62%) are type-II** — the dipolar-capable seed pool from HetDB. This is a **first-pass estimate** from isolated-monolayer PBE edges (Anderson's rule); it neglects interfacial charge transfer, band bending, and relaxation, so it is flagged as `derived`, not a measured/stored label. It is exactly good enough to *prioritise* which stacks to send to Phase-2 BSE.

---

## 4. How a geometric dipolar-U estimate is computed from these fields

The dipolar interaction (oriented-dipole / capacitor form, per `u_definition_decision.md`):

```
U_dip  =  e² d² / (ε₀ · ε_eff · A)
```

where every symbol maps onto a field these databases already carry:

| Symbol | Meaning | Source field (live coverage) |
|--------|---------|------------------------------|
| `d` | electron–hole (interlayer) separation | `interlayer_distance` — **BiDB 12,138/12,138, HetDB 336/336** (geometric upper bound on d) |
| `A` | mode / unit-cell area | `area` — BiDB 12,138, HetDB 336 |
| `ε_eff` | effective interlayer dielectric | from constituent-monolayer polarizabilities in C2DB (`alpha*_el`), joined via `c2db_uid` / `uid_a,uid_b` |
| — | type-II gate (is there a dipole at all?) | HetDB: **derived** from CBM/VBM-wrt-vacuum (209 type-II). BiDB: broken-inversion gate (6,109) |

So a **purely geometric** `U_dip` prior — using d = `interlayer_distance` as the full charge-transfer separation — is computable **today for all 12,138 BiDB + 336 HetDB entries**, with the type-II / broken-inversion gate selecting the ~209 (HetDB) + ~2,548 (semiconducting BiDB) dipolar-capable seeds.

### What is present vs. the gap (mirroring the Phase-0 analysis)

**Present (geometry + alignment ingredients):**
- `d` (`interlayer_distance`): **100% in both DBs.**
- Interlayer binding `ebind`: 100% of BiDB.
- Band edges to classify type-II: 303–310 / 336 in HetDB (+ C2DB monolayer edges for the join).
- Symmetry gate `has_inversion_symmetry`: 100% of BiDB.

**The gap (the excited-state dipole itself — 0 labels, must be generated):**
- **Excited-state exciton dipole `d_exc`** — the *actual* electron–hole separation of the lowest exciton, i.e. `d` weighted by the exciton's **charge-transfer character** (the fraction of the BSE wavefunction with electron and hole on opposite layers). `interlayer_distance` is only the *geometric* upper bound; the real d_exc = (CT fraction) × d. **Neither BiDB nor HetDB tabulates the exciton CT character or any BSE quantity.** Coverage: **0.**
- **U (dipolar exciton–exciton interaction):** **not present in either DB.** Coverage: **0.**
- The type-II classification itself is **not stored** (0 keys); it is derived here.

This is the same shape as the Phase-0 verdict: the databases provide *ground-state* structure and band alignment cheaply and at scale, but the **excited-state / many-body label that the FOM turns on (here: the exciton CT character → d_exc → U_dip) has zero coverage** and is the Phase-2 generation target. The BSE exciton CT character is a genuine observable (the layer-localisation of the exciton wavefunction) and, per the U-definition decision, is **independent of E_b** — so generating it is not a restatement of anything C2DB already has.

---

## 5. Phase-2 seed inventory (the deliverable)

| Seed pool | Count | Definition |
|-----------|------:|------------|
| **HetDB type-II heterostructures** | **209** | derived staggered alignment ⇒ intrinsic interlayer exciton |
| **BiDB semiconducting + broken-inversion bilayers** | **2,548** | `gap_pbe>0.1 ∧ has_inversion_symmetry=False` |
| BiDB broken-inversion (all, incl. metals) | 6,109 | out-of-plane dipole symmetry-allowed |
| Combined dipolar-capable geometric seed | **~2,757** (209 + 2,548) | pre-screened for Phase-2 BSE CT-character runs |

**Prioritisation for Phase-2 compute:** rank this pool by (largest `interlayer_distance` d) × (type-II confidence / broken-inversion) × (semiconducting gap) × (stable `ebind`/`slide_stability`), then run BSE on the top seeds to get the true exciton CT character → d_exc → U_dip. HetDB's 209 type-II stacks are the highest-value starting point because the interlayer exciton is *intrinsic* there (band offset), whereas BiDB's dipole is the weaker stacking-induced kind.

---

## 6. Honest verdict

- **The dipolar datasets exist and are reachable:** BiDB (**12,138** bilayers) and HetDB (**336** heterostructures), both on live ASE-db front-ends, both with `interlayer_distance` at **100% coverage**.
- **A dipolar-capable seed pool is real and pre-screened:** **209** derived type-II heterostructures + **2,548** semiconducting broken-inversion bilayers.
- **But the load-bearing excited-state labels are absent, exactly as in Phase 0:** the exciton **charge-transfer character**, the excited-state dipole **d_exc**, and **U_dip** itself have **zero coverage** in either database. `interlayer_distance` is a geometric *upper bound* on d, not the CT-weighted exciton dipole.
- **type-II is not even a stored label** — it is derived here by Anderson's rule (an estimate, flagged as such), because HetDB ships the band-edge ingredients but no alignment classification.

So Phase 2 gets a **geometric U_dip prior and a screened seed pool for free**, but the independent U the whole search turns on still has to be **generated** (BSE exciton CT character on the seed set). Any U_dip number reported before that generation runs will be labeled a geometric prior, not a computed dipolar U.

---

## 7. Reproduce this

```bash
python excitonic/scripts/phase2_dipolar_inventory.py
#   -> excitonic/data/manifests/dipolar_dataset_inventory.json   (live counts)
```

Artifacts:
- `excitonic/src/exciton_fm/bidb_client.py` — BiDB/HetDB ASE-db client (`StackDBClient`).
- `excitonic/data/manifests/dipolar_dataset_inventory.json` — live coverage counts (regenerated by the script).

*All counts are live queries; where a dipolar label does not exist it is reported as zero/absent, and the type-II figure is explicitly a derived Anderson's-rule estimate, not a stored or measured label.*
