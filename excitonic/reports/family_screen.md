# Cross-channel family screen — widening the search where the cancellation is weakest

The likely bounded-no comes from a near-cancellation: the physics that makes the
exciton–exciton interaction **U** large (tight excitons, polar lattice, big dipoles)
also makes the room-temperature Fröhlich linewidth **Γ(300 K)** large. Blockade needs
`U/Γ > ~0.7`; the best measured is ~0.05. To raise the odds we deliberately search the
places where that cancellation breaks — where a family wins on one axis of the
trilemma (U↑, Γ↓, oscillator-strength f↑) without a fatal loss on another.

Reproducible: `scripts/phase2_family_screen.py` → `data/manifests/phase2_family_screen.json`.

## What is and isn't computed (rigor)

- **Computed** (from *measured* phonon energies, tier `measured`): the LO-phonon
  occupation at 300 K, `n_LO(300 K) = 1/(exp(ħω_LO/kT)−1)` — the **thermal driver of
  Γ**. Its ratio to MoS₂ quantifies the denominator lever exactly.
- **NOT computed** (`not_run` for every family): U, f, Γ(300 K), and the FOM U/Γ.
  Those need GW-BSE (U, f) + EPW (the coupling γ_LO, hence Γ). This is a
  **prioritization of where to spend**, not a verdict, and it is labelled as such.

## The map

| Family | ħω_LO (meas.) | n_LO(300 K) | Γ_thermal / MoS₂ | U mechanism | f | Optical window |
|--------|:---:|:---:|:---:|-------------|:--:|----------------|
| **TMD intralayer (MoS₂)** — baseline | 48 meV | 0.185 | 1.00 | saturation ∝E_b·a_B² (a_B-tied, small) | high | visible ✓ |
| **Interlayer dipolar** (type-II bilayer) | ~48 meV | 0.185 | 1.00 | **dipolar** ∝d²/(εA) (a_B-decoupled) | **low** | near-IR ✓ |
| **hBN** (high-ħω_LO) | 180 meV | 0.0009 | **0.005** | Frenkel; weak nonlinearity | mid | deep-UV ✗ |
| **GaN** (III-nitride) | 92 meV | 0.029 | 0.16 | Wannier; strongly polar (γ_LO high) | high | UV ~edge |
| **Cu₂O Rydberg** (yellow series) | 79 meV | 0.049 | 0.27 | **Rydberg blockade ∝n⁷** (giant) | low (quadrupole) | visible ✓ |

*(ħω_LO and E_b values are literature-measured with sources in the manifest;
Molina-Sánchez & Wirtz 2011, Chernikov 2014, Cassabois 2016, Kazimierczuk 2014, etc.)*

## Reading it — each family wins one axis, loses another

- **TMD intralayer** — balanced but capped. This is the cancellation regime; the ~0.05
  record lives here. Nothing to gain by searching more of it.
- **hBN** — the denominator lever taken to the limit: n_LO(300 K) is **~200× below**
  MoS₂, so at fixed coupling the thermal Γ is crushed. But the exciton is Frenkel
  (tiny nonlinearity) and the gap is deep-UV — the numerator and the optical window
  both fail. A cautionary anchor, not a candidate.
- **GaN** — partial denominator win (~6×), but it is strongly polar so the *coupling*
  γ_LO stays high; the occupation drop is offset. Useful to **separate** the two parts
  of Γ (occupation vs coupling) with one EPW run.
- **Cu₂O Rydberg** — the numerator lever: the n⁷ Rydberg-blockade interaction is the
  strongest known excitonic mechanism, the direct analog of Rydberg-atom blockade. The
  catch is temperature: high-n Rydberg excitons thermally ionize and the yellow line is
  quadrupole (low f). The open question worth a calculation: **how fast does U/Γ fall
  with T, and is the RT value fatal or merely small?**
- **Interlayer dipolar** — the **only family that decouples U from Γ without a fatal
  third-axis loss.** It pays in oscillator strength f — but f is the one deficit a
  high-Q, small-mode-volume cavity can recover. This is why it ranks first.

## Spend ranking (prioritization, FOM not computed)

1. **Interlayer-dipolar heterobilayer BSE** — U decoupled from Γ; the f penalty is
   cavity-recoverable. 209 type-II candidates already ranked (`phase2_dipolar_candidates.json`).
2. **Cu₂O Rydberg** — measure U/Γ vs n and vs T; test whether the RT loss is fatal.
3. **High-ħω_LO (hBN/nitride)** — denominator anchor; one EPW run splits γ_LO from
   occupation and calibrates the Fröhlich model across a 200× occupation range.

## Why this raises the odds honestly

It doesn't manufacture a winner. It (a) **quantifies** the denominator lever from
measured phonons (hBN 200×, GaN 6×, Cu₂O 4× below MoS₂), (b) names the one channel
(interlayer) that escapes the cancellation without a fatal loss, and (c) turns the
Rydberg family — the only known giant-U mechanism — into a concrete T-dependence
question. The FOM still has to come from GW-BSE + EPW; this decides **which** runs are
worth it. Nothing here is presented as a blockade result.
