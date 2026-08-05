# Independent literature verification (external triple-check)

Every load-bearing number behind the blockade verdict, checked against primary literature
by an independent pass. Two outcomes: the **bulk-NO** side is measured-confirmed and solid;
the **single-emitter-YES** side is a real platform but its *compute-relevant* (array) claim is
weaker than the summary implied, and one upstream DFPT value is mislabeled (verdict-neutral).

## 1. Bulk-NO side — CONFIRMED on measured data

| Quantity (project value) | Literature check | Status |
|---|---|---|
| Γ floor ≈ 10–15 meV (RT) | MoS₂ homogeneous linewidth **~14 meV at 300 K** (2 meV at 5 K), Selig/Chalmers ([pdf](https://publications.lib.chalmers.se/records/fulltext/250484/local_250484.pdf)); Fröhlich exciton–phonon [Nat. Commun. 2019](https://www.nature.com/articles/s41467-019-08764-3) | ✅ confirmed |
| Dipolar/biexciton U = 8.4 meV (4–20) | Interlayer biexciton blueshift **ΔE = 8.4 ± 0.6 meV**; tri 12.4 / quad 15.5 / **quint 18.2 meV**, [npj 2D Mater. 2020](https://www.nature.com/articles/s41699-020-0141-3) | ✅ confirmed exactly |
| U comparable to Γ (not ≫) | 8–18 meV vs 12–14 meV → **U/Γ ≈ 0.6–1.3** | ✅ confirmed — the crux |
| MoS₂ E_b ≈ 0.72 eV (GW-BSE) | Freestanding GW-BSE 0.5–0.9 eV; ~150–300 meV *on substrate* (screened) | ✅ confirmed for freestanding; note dielectric-environment dependence |

The self-correction made during the project (dipolar U from a ~100 meV estimate down to the
measured 8–20 meV) is **vindicated** by the primary source. The bulk verdict — every channel to
raise U closes against the Γ floor — rests on measurements, not just the model.

## 2. ω_LO mislabel — CORRECTED (verdict-neutral)

`data/manifests/phase2_tmd_gamma.json` reports `omega_LO = 56.428 meV` (≈455 cm⁻¹) for MoS₂,
taken as the **"top Γ mode."** That is the **A₂″ out-of-plane (ZO) mode**, not the
**Fröhlich-active in-plane E′ LO**, which sits at **~384 cm⁻¹ ≈ 48 meV** (E₂g 47.7 / A₁g 50.6
meV; [arXiv:1603.05172](https://arxiv.org/pdf/1603.05172)). The Fröhlich exciton–phonon
interaction couples to the polar **longitudinal** mode, i.e. the E′ LO, so ~48 meV is the
physically correct input — which is exactly what `frohlich.py`'s calibration anchor already
uses (`ħω_LO ≈ 48 meV`). All four TMD values are the top A₂″ mode, systematically ~15% high.

**Impact on the verdict: none.** The Γ floor is *calibrated to the measured 10–14 meV RT
linewidth*, so it is robust to the mode choice:

| ω_LO used | Γ(300 K) model |
|---|---|
| 56.4 meV (mislabeled A₂″ top mode) | 10.3 meV |
| **48.0 meV (correct E′ LO)** | **12.3 meV** |

Both land inside the measured 10–14 meV band. **Fix for hygiene** (re-extract the E′ LO from the
ph.x dynamical matrix, or use the Raman/literature 48 meV); the bound `need U > ~4–11 meV` and
every downstream conclusion are unchanged.

## 3. Single-emitter-YES side — real platform, but two caveats the summary understated

- **RT single-emitter *blockade* has never been demonstrated.** What exists is RT *strong
  coupling* — ~4 meV Rabi with an hBN emitter + BIC metasurface ([Nat. Commun. 2024](https://www.nature.com/articles/s41467-024-46544-w)), plus Purcell arrays and RT cooperative
  emission. **Strong coupling (Ω > Γ) ≠ blockade (U > Γ)**; U ≪ Ω. So the "single-emitter YES"
  is *model-permitted, not experimentally shown* — consistent with the parent thesis.
- **The array-uniformity escape is on weak ground — and it is the compute-relevant claim.** The
  proposed fix was "identical-by-chemistry color centers remove the disorder wall." The
  literature says the disorder is **environmental, not chemical**: hBN ZPLs span **1.5–2.2 eV**
  from local strain, dielectric environment, and Stark shifts of trapped charges ([Sci. Rep.
  2021](https://www.nature.com/articles/s41598-021-90804-4); [arXiv:1605.04445](https://arxiv.org/pdf/1605.04445)). Best-demonstrated uniformity — structured-defect engineering,
  [ACS Nano 2024](https://pubs.acs.org/doi/10.1021/acsnano.4c11413) — reaches 85% within
  580 ± 10 nm, i.e. **±37 meV**, still several × the ~10 meV linewidth. Identical *chemistry*
  does not remove the dominant *environmental* disorder.

## 4. New result — the array-uniformity bound (Phase 3)

(`scripts/phase3_array_uniformity_bound.py`, `data/manifests/phase3_array_uniformity.json`)

A shared-cavity blockade array works only if the site-to-site inhomogeneous ZPL spread
σ_inhom ≲ Γ (a resonance can only blockade what it is resonant with). Against measured hBN:

| Uniformity | σ_inhom | σ_inhom / Γ (Γ≈10 meV) | Verdict |
|---|---|---|---|
| Best demonstrated (ACS Nano 2024) | ±37 meV | **3.7×** | FAILS |
| Typical (Sci. Rep. 2021) | ~700 meV | **~70×** | FAILS |

**The array fails by ~4× (best) to ~70× (typical), and the dominant disorder (strain/Stark) is
not the one the identical-by-chemistry proposal removes.** This is the compute-relevant bound,
mirroring the bulk bound: single-emitter blockade does not scale to a uniform compute array on
any current hBN platform.

## 5. Net effect

- **Optical-compute close: firmer.** The bulk-NO side is measured-confirmed; the single-emitter
  route (the only physics-permitted escape) does not scale — RT single-emitter blockade is
  unbuilt, and the array hits a documented ~±37 meV uniformity wall the proposed chemistry does
  not address.
- **Single-emitter blockade remains a quantum-optics-device prospect** (single-photon switch /
  photon–photon gate), not an AI-accelerator one, and even that has reached only RT *strong
  coupling*, not blockade.

## Not independently verified (next verification pass)
- The moiré single-trap U ≈ 8–60 meV (model/proxy tier) — the number behind the single-emitter
  U/Γ > 1. Worth a primary-source check of moiré-trapped-exciton on-site interaction energies.
- The next-phase hBN defect-supercell cooperativity C = g²/κγ — and, more importantly for
  compute, the *distribution* of ZPL/coupling across a realistic strained ensemble (the array
  question), not a single perfect emitter.
