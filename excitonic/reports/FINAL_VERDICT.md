# Final verdict — can a material do room-temperature single-photon blockade?

Post-verification synthesis (all load-bearing numbers checked against primary literature; see
`verification_literature.md`). The question decides whether a low-power all-optical nonlinearity
— the one physics-permitted lever to reopen general optical neural-network compute — exists.
Gate: **U/Γ(300 K) > 0.71**, U = exciton–exciton interaction, Γ = RT linewidth.

## The answer, in three levels

**1. Scalable, room-temperature blockade in a BULK material — NO (measured bound).**
Every bulk channel to raise U closes against the phonon Γ floor (~10–14 meV, measured):
- interlayer/dipolar U is only ~8–18 meV (measured biexciton blueshift 8.4 meV) — *comparable
  to* Γ, not ≫ it; and the dipole that gives U also makes the exciton dark (oscillator-strength
  trade-off);
- Janus (MoSSe) has a bright exciton but a tiny dipole → U ~ 0.005–0.09 meV, 2–4 orders below need;
- Rydberg/saturation routes lose on character or on array disorder.
Strong multi-ground bound (measured + DFPT + model tiers), not a theorem. **The bulk route is
bounded out on real data.**

**2. Single-emitter blockade — YES cold, NOT at room temperature (on measured numbers).**
- **Cold:** a moiré-trapped exciton has measured U ≈ 2 meV against a 100 µeV linewidth → U/Γ ~ 20,
  antibunching demonstrated. Cold single-emitter blockade is real (long known).
- **Room temperature:** the phonon floor lifts Γ to ~10 meV, so the same measured U ≈ 2 meV gives
  **U/Γ ~ 0.2 — fails.** RT single-emitter blockade is *unbuilt*; what exists is RT *strong
  coupling* (~4 meV Rabi, hBN + cavity), which is not blockade. The project's 8–60 meV RT-open
  figure is an **extrapolation** to deeper-than-demonstrated traps.

**3. A compute ARRAY — NO (new bound).**
Even granting a single RT emitter, a neural net needs a spectrally-uniform *array* addressable by
a shared cavity: σ_inhom ≲ Γ. Best-demonstrated hBN uniformity is ±37 meV; typical spread ~700 meV
(ZPL 1.5–2.2 eV). The array **fails by ~4× (best) to ~70× (typical)**, and the dominant disorder
is strain/Stark — *not* the defect chemistry the "identical-by-chemistry" proposal removes.

## What this means

| Question | Verdict | Basis |
|---|---|---|
| **Reopen general optical COMPUTE** with a bulk RT nonlinearity | **NO** | measured bulk bound + array-uniformity bound |
| Single-photon nonlinear DEVICE (switch / photon–photon gate), COLD | **YES** (known) | measured moiré U/Γ ~ 20 |
| Same device at ROOM temperature | **Open, unbuilt** | RT strong coupling shown; RT blockade not; moiré RT U/Γ ~ 0.2 |

**Optical neural-net compute stays closed** — now on a materials bound, not just architecture.
The single-emitter escape is a *cold* quantum-optics result that does not survive to RT on
measured numbers and does not scale to a uniform compute array on any current platform. This is
as close to "impossible for scalable RT compute" as the evidence honestly gets: a data-backed
multi-ground bound, not a theorem.

## Recommendation on the next phase

The planned hBN defect-supercell cooperativity calc (C = g²/κγ) computes a **single-emitter
device** figure — worth doing for the quantum-optics story, but it does **not** bear on the
compute question, which is settled NO by the array-uniformity bound. If the goal is compute, the
only calc that would move the needle is the **ensemble ZPL/coupling distribution under realistic
strain** — i.e. can any growth/engineering route push σ_inhom below ~10 meV. Absent that, the
compute verdict does not change.

## The durable asset

A validated conda-ABINIT GW-BSE pipeline (MoS₂ E_b = 0.72 eV, within the accepted 0.5–0.9 eV) —
reusable for defect-supercell excited-state calculations regardless of the verdict above.
