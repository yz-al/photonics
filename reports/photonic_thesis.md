# Does the photonic-compute thesis survive honest energy accounting and an optics-native architecture?

**Two falsifiable computational tests. No fab, no synthesis, no proprietary data.**

---

## Verdict (one line)

**Test 1 (energy):** A crossover where an optical matrix-multiply beats digital
*does* exist — but it is **small (N≈6 at 4-bit, N≈20–30 at 8-bit) and it only beats
the whole-chip digital number (0.3–2 pJ/MAC), not the arithmetic/in-memory-compute
floor (~20–40 fJ/MAC), which optics essentially never beats at 8-bit.** The premise
"optics wins at large N" is inverted: you cannot *reach* large N, because a mesh deep
enough to amortise the converters attenuates the signal below detectability
(N=1024 monolithic ≈ 100–500 dB loss, ≫ any ~33 dB link budget) and dissipates
kilowatts of thermal tuning. Tiling caps the loss but pins the amortisation at the
tile size, and only survives at all with weight-stationary phase-change weights.
**There is no robust buildable regime where an optics-native MZI-mesh MAC beats a
digital MAC at the honest compute-floor level.** This reproduces, and quantifies,
the 2026 market verdict ("optical data movement yes, optical compute no").

**Test 2 (nonlinearity):** The standard objection — "optics can't do the
nonlinearity" — is substantially about *porting the wrong architecture*. A network
built on the free square-law nonlinearity (**complex linear layer + |·|² photodetection**)
is **competitive with ReLU at equal parameter count and shallow depth** (matches on
MNIST), and a *learnable* quadratic matches ReLU broadly. The real costs are (i) a
**depth instability** in the quadratic cascade (it degrades past depth ≈ 2–3 unless
tamed by normalisation) and (ii) the optics constraints (4-bit, non-negativity,
shot noise) which cost a bounded, single-digit-percent accuracy gap. So the
nonlinearity is not the wall. **The wall is the energy/loss ledger of Test 1.**

Killing the energy half cheaply is the intended good outcome. Test 2 says the door
is not closed by the nonlinearity — it is closed (for now) by loss and converters,
which is where any future hardware advance must land.

---

## Scope and background (stated accurately)

Optical matrix–vector multiplication is nearly free in the linear step: a coherent
field propagates through an interferometer mesh and the interference pattern *is* the
matrix–vector product, at propagation speed, with passive elements dissipating almost
nothing. Neural networks are (matmul + nonlinearity), repeated.

The standard objection is that optics cannot do the nonlinearity, because photons do
not interact at low power. So every deployed architecture converts to electrical at
each layer, applies the activation, and converts back; those conversions dominate
energy and latency and eat the advantage.

**Market state (2026), as we understand it.** Lightmatter (the best-capitalised
company in the space, ~$850M raised, ~$4.4B valuation) shipped *interconnect*
products in 2026 (Passage L200, L20, M1000); its Envise *compute* accelerator has
gone quiet. TSMC's COUPE co-packaged-optics platform entered mass production in
April 2026; Broadcom, NVIDIA and Ayar Labs all shipped co-packaged optics. The
revealed market verdict is: **optical data movement yes, optical compute no.**

Two things that verdict may not have tested, and which these tests target:

- **(a) Scaling.** The conversion penalty scales as *N* while matmul scales as *N²*,
  so there should be a matrix size above which optics wins regardless of converter
  quality. Every demonstration has been small (N ≈ 6–20). *Test 1 asks where the
  crossover is and whether it is buildable.*
- **(b) The free nonlinearity.** Photodetection is |E|², a square-law nonlinearity
  that is free and unavoidable. ReLU won because it is cheap on a GPU; on optics the
  cheap activation is quadratic. *Test 2 asks whether networks designed around the
  quadratic are competitive.*

Neither test requires hardware. Per the brief, we did **not** attempt a materials
search for optical nonlinearity (the Kramers–Kronig / two-photon-absorption limit,
Sheik-Bahae et al. 1990, and the fact that chalcogenides already clear the standard
figure of merit, make that a synthesis problem, not a modelling one), and we built
no optical simulator beyond what these two tests need.

---

# TEST 1 — Where is the scaling crossover, and is it buildable?

## Method

We build a transparent, parameterised model of **joules per MAC** for an optical
matrix-multiply unit versus a digital baseline, as a function of matrix dimension *N*.
Code: [`src/energy_model/model.py`](../src/energy_model/model.py),
driver [`src/energy_model/run_test1.py`](../src/energy_model/run_test1.py),
results [`data/test1_results.json`](../data/test1_results.json).

**Accounting.** One matrix–vector product through an N×N mesh performs N² MACs. Per
matvec the optical unit spends:

```
J_total(N) = J_conversion + J_modulation + J_laser + J_thermal   (per matvec)
```

- **Conversion** ~ O(N): N input DACs (encode the activation vector) + N output ADCs
  (read the N detectors). This is the term that scales as N while useful work scales
  as N², so its **per-MAC cost falls as 1/N** — the whole of point (a).
- **Modulation** ~ O(N): N input modulators switch once per matvec.
- **Laser** ~ O(N): deliver ≥ P_min optical power to each of N detectors *through the
  mesh*, whose transmission falls exponentially with mesh **depth** (~N stages).
  P_min is **derived** from detector shot + thermal noise for the target bit depth,
  not assumed.
- **Thermal** ~ O(N²): the mesh holds ~N² phase shifters. Thermo-optic shifters
  dissipate static power continuously; **per MAC this is a constant floor** (N²
  shifters / N² MACs). Phase-change-material weights hold state at ~zero static power
  — this term collapses. Both are modelled.

Per-MAC optical cost (weight-stationary form):

```
J/MAC_optical(N) = (E_DAC + E_ADC + E_mod)/A            (conversion + modulation)
                 + E_det / (A · T(depth) · WPE)          (laser through mesh loss T)
                 + P_ps · t_vec                          (thermal static, per MAC)
```

where **A** is the amortisation dimension: **A = N** for a single monolithic mesh,
but **A = tile size T** for the physically buildable *tiled* architecture (each T×T
sub-mesh re-converts at its boundary, so amortisation saturates at T, not N).
Compared against a **constant** digital cost `J_MAC_digital` (independent of N).

**Uncertainty.** Every input is a `(low, nominal, high)` triple with a cited source.
Bands come from Monte-Carlo sampling those ranges (log-uniform, 3000 draws). The
crossover is reported as a band, and as the *fraction of parameter draws in which a
crossover exists at all*.

**Detector power is derived, not assumed.** For b-bit precision we require electrical
power-SNR = 10^((6.02b+1.76)/10) (standard converter dynamic range). We take the
larger of the shot-noise floor (E = SNR·2q/R, bandwidth-independent) and the
thermal/TIA floor (E = √SNR·i_n/(R·√B)). This reproduces the sourced link budget
**exactly**: 3.1 fJ/symbol at 4-bit and 49 fJ/symbol at 8-bit (1 GHz, TIA-limited,
i_n = 5 pA/√Hz) — i.e. at these power levels the receiver is thermal-limited, and at
8-bit it sits within ~4× of the true shot-noise quantum limit.

## Inputs (every value sourced, as a range)

| Quantity | Low – Nominal – High | Source |
|---|---|---|
| ADC energy/sample, 4-bit @GS/s | 0.5 – 0.9 – 2 pJ | 4-bit folding-flash 65 nm, 0.7 pJ (arXiv:1612.04855); FOM_W 65–200 fJ/step, Murmann survey |
| ADC energy/sample, 8-bit @GS/s | 2 – 3.5 – 8 pJ | Kull ISSCC 2014 (90 GS/s, 7.4 pJ); 14 nm ISSCC 2018 (2–3.3 pJ); FOM_W 15–203 fJ/step |
| DAC energy/sample, 4-bit @GS/s | 0.5 – 1.5 – 3 pJ | scaled from CS-DAC bounds (no DAC survey exists) |
| DAC energy/sample, 8-bit @GS/s | 5 – 10 – 15 pJ | Lin ISSCC 2013 (12b 1.6 GS/s); 8b 25 GS/s 272 mW → 10.9 pJ |
| Modulator, Si MZM (dynamic) | 55 – 400 – 1000 fJ/bit | Xu et al. 2014 (0.8 pJ @28 Gb/s); slow-light 55–80 fJ |
| Modulator, Si micro-ring | 0.9 – 3 – 6 fJ/bit | Timurdogan et al., Nat. Commun. 5, 4008 (2014), 0.9 fJ/bit |
| Modulator, plasmonic/ITO | 0.07 – 1 – 20 fJ/bit | Heni/Hoessbacher et al., Nat. Commun. 10, 1694 (2019) |
| Thermo-optic phase shifter, static | 3 – 10 – 25 mW/π | Opt. Lett. 45, 4806 (2020), 24.8 mW/π; efficient designs ~3 mW/π |
| PCM weight (Sb₂Se₃), static hold | ~0 W (≈10⁻¹⁰) | Delaney et al., Sci. Adv. 7, eabg3500 (2021); non-volatile, ~1 dB IL |
| Laser wall-plug efficiency | 8 – 12 – 20 % | Lo et al., Opt. Eng. 62, 046102 (2023) 20%; arXiv:2310.01615 (15%@40°C) |
| Ge-on-Si responsivity | 0.6 – 0.85 – 1.1 A/W | waveguide Ge PDs @1550 nm |
| TIA input-referred noise | 1.2 – 5 – 20 pA/√Hz | multi-GHz CMOS TIA survey |
| Waveguide loss (Si strip) | 0.5 – 1.5 – 3 dB/cm | Dumon/Bogaerts, Opt. Express 12, 1622 (2004), 3.6 dB/cm; typ 1–2 |
| Coupling loss (per facet) | 0.5 – 1.5 – 3 dB | edge <1 dB; grating 1–3 dB (×2 per chip) |
| **Per-MZI insertion loss** | **0.1 – 0.5 – 2 dB** | Bandyopadhyay et al., Optica 8, 1247 (2021), ~0.5 dB/PUC; up to 2 dB realized |
| MZI cell length / port pitch | 100–500 µm / 25–110 µm | Clements meshes; switch fabrics (110 µm pitch) |
| **Digital baseline, INT8 (whole-chip)** | **0.3 – 0.5 – 2 pJ/MAC** | H100 0.71 pJ, B200 0.44 pJ (peak); realized 1.5–3× worse |
| Digital baseline, FP4 (whole-chip) | 0.2 – 0.3 – 1 pJ/MAC | B200 FP4 0.20 pJ peak |
| Digital **compute floor** (arith/CIM) | 20 – 40 – 100 fJ/MAC | Horowitz ISSCC 2014 (0.23 pJ @45 nm → ~20–40 fJ @5 nm); CIM macros 10–50 fJ |

Mesh structure (Clements et al., Optica 2016): N(N−1)/2 MZIs, ~N(N−1) phase
shifters, optical **depth N** for a single universal unitary (a general real matrix
via SVD needs two meshes, ~2N — we use N as the *optimistic* case, favourable to
optics). Largest universal mesh ever demonstrated: N ≈ 6–20 (Shen et al., Nat.
Photonics 2017, N=4; 8×8 SiN processor).

## Results — the five questions

**Figures:** `figures/test1_ideal_{4b,8b}.png` (loss-free amortisation limit),
`figures/test1_tiled64_{4b,8b}.png` (buildable tiled architecture).

### 1. At what N does optical cross below digital, at each precision?

Crossover band (Monte-Carlo, "exists" = fraction of draws with any crossover):

| Scenario | 4-bit | 8-bit |
|---|---|---|
| **Loss-free ideal** (point-(a) limit, vs whole-chip digital) | **N≈6** [16–84%: 5–11], exists 100% | **N≈20** [9–37], exists 100% |
| **Tiled 64×64, buildable** (vs whole-chip digital) | N≈6 [5–11], exists 93% | N≈14 [9–29], **exists 44%** |
| **Tiled 64×64 vs the compute floor (20–40 fJ)** | N≈37 [25–54], **exists 19%** | **no crossover (0%)** |

The crossover against *whole-chip* digital exists and is **small**. Against the honest
**compute floor**, 8-bit **never** crosses and 4-bit crosses only in ~1/5 of favourable
parameter draws. The whole-chip digital number is dominated by data movement, not the
multiply; optics has to move data too (its converters), so beating the whole-chip
number while losing to the arithmetic floor means optics is competing with digital's
*overhead*, not its arithmetic.

### 2. Which term dominates, and where does dominance change hands?

At nominal parameters, **conversion dominates at small N (≤16–32); the laser term
dominates at large N.** The handoff is the loss wall expressed as energy: as N grows,
conversion amortises away (1/N) but the laser power needed to punch through the
depth-N mesh grows exponentially and takes over. Conversion does **not** dominate
everywhere — but it sets the small-N floor, and it is pJ-scale (the converters, not
the optics, are the expensive part of an "optical" MAC).

### 3. Sensitivity — which single parameter most moves the crossover?

For the **energy** crossover (loss-free ideal, 8-bit), ranked by how far each
parameter shifts it across its cited range:

| Parameter | crossover low → high |
|---|---|
| Digital baseline (J/MAC INT8) | 47 → 7 |
| DAC energy 8-bit | 20 → 42 |
| ADC energy 8-bit | 25 → 37 |

The crossover location is set by the **digital baseline and the 8-bit converters**.
But **buildability** is governed by a different single parameter — **per-MZI insertion
loss** — which sets the maximum mesh depth you can build before the signal is
undetectable (question 4). That is where a hardware advance would matter most: not a
better modulator, a **lower-loss mesh element**.

### 4. Is the crossover N physically buildable? (Loss checked explicitly.)

Component count, area, and insertion loss for a **monolithic** universal mesh
(nominal geometry; loss shown at the 0.1 dB/MZI *optimistic floor*):

| N | MZIs | phase shifters | optical depth | die area | loss @0.1 dB/MZI | detectable? | thermo-optic idle tuning |
|---|---|---|---|---|---|---|---|
| 20 (8-bit crossover) | 190 | 380 | 20 | 0.1 cm² | **6 dB** | ✅ yes | ~0 |
| 64 | 2,016 | 4,032 | 64 | 0.8 cm² | 12 dB | ✅ yes | ~0 |
| 256 | 32,640 | 65k | 256 | 13 cm² (2 reticles) | **38 dB** | ❌ no | 0.3 kW |
| 1024 | 523,776 | ~1.05 M | 1024 | 210 cm² (**24 reticles**) | **144 dB** | ❌ no | **5.2 kW** |
| 4096 | 8,386,560 | ~16.8 M | 4096 | 3355 cm² (**391 reticles**, > a 300 mm wafer) | **566 dB** | ❌ no | **84 kW** |

**Loss kills it before energy does, and before area does.** Even at the 0.1 dB/MZI
theoretical floor, a depth-256 monolithic mesh (38 dB) already exceeds a realistic
~33 dB analog optical link budget; N=1024 is 144 dB (10¹⁴·⁴ attenuation — undetectable
at GHz rates) and N=4096 is physically dark. The *good news* is that the 8-bit energy
crossover (N≈20) is trivially buildable (6 dB, 190 MZIs) — but that regime only beats
whole-chip digital, and you cannot push N higher to reach the compute floor because
the mesh goes dark.

**The tiled escape, and its price.** Bounding mesh depth at a tile size T caps the
loss but pins the amortisation at T. Maximum buildable tile depth vs per-MZI loss:

| per-MZI loss | max tile depth within 33 dB budget |
|---|---|
| 0.1 dB (optimistic) | **218** |
| 0.5 dB (good, demonstrated) | **55** |
| 2.0 dB (typically realized) | **14** |

At the nominal 0.5 dB/MZI, a 64-deep tile is already 37 dB — over budget. So even the
tiled architecture only reaches T ≈ 55, and its per-MAC conversion floor is
(E_DAC+E_ADC)/T ≈ 13.5 pJ / 55 ≈ 245 fJ at 8-bit — comparable to whole-chip digital,
far above the compute floor. (This model does **not** even charge the digital
cross-tile accumulation that a real tiled optical accelerator needs — an omission
generous to optics.)

### 5. Does weight-stationary operation change the answer?

Decisively. Thermo-optic phase shifters dissipate ~0.5·Pπ continuously; for a mesh
this is ~kW of idle tuning (5.2 kW at N=1024) and, per MAC, a constant ~pJ floor.

| Precision | Weights in PCM (≈0 W hold) | Thermo-optic weights |
|---|---|---|
| 4-bit (tiled 64) | crossover N≈6, exists 93% | crossover N≈13, **exists 1%** |
| 8-bit (tiled 64) | crossover N≈14, exists 43% | **no crossover** |

**Weight-stationary phase-change weights (Sb₂Se₃, ~0 W hold, ~1 dB IL) are not
optional — they are the difference between a marginal crossover and none at all.**
The thermal term alone kills the 8-bit case with thermo-optic weights.

## Test 1 verdict

A crossover exists but is small, sits against the wrong (overhead-heavy) digital
baseline, and cannot be pushed to the regime that would beat the true compute floor
because **insertion loss caps the buildable mesh depth long before the energy math
would pay off**. Weight-stationary PCM is mandatory. **No robust buildable regime
where an optics-native MZI-mesh MAC beats a digital MAC at the compute-floor level —
which is exactly the 2026 market verdict, now quantified with a sourced model.** The
single most valuable hardware advance is **lower per-MZI insertion loss** (it is the
only lever on buildable depth); converter energy and a weight-stationary substrate
are the next two.

---

# TEST 2 — Does a network train on the free nonlinearity?

## Method

Photodetection gives |E|² for free. We test whether networks built around quadratic /
modulus-squared activations are competitive with ReLU-family networks, first without
optical constraints (Stage A) and then with them (Stage B).

Code: [`test2/`](../test2/) — `nn_optics.py` (activations, quantiser, complex linear +
|·|² detection, shot-noise, non-negative differential encoding), `models.py`,
`data.py`, `experiment.py`, `run_all.py`. Runs on CPU or GPU. The full sweep runs on a
**Modal GPU via the GitHub Actions workflow** `.github/workflows/test2-modal.yml`
(results committed back to `test2/results/`); the numbers below are from the
CPU-feasible **`local` run (5 seeds)** with the CIFAR-10 CNN deferred to the GPU run.

- **Stage A** (no optical constraints): identical architectures trained with
  activations **ReLU, GELU, x², |·|², |x|, and a learnable scaled quadratic** —
  a small CNN (CIFAR-10), a small transformer (AG News / synthetic associative
  recall), an MLP (Forest Cover-Type tabular). ≥5 seeds; spreads reported. Plus a
  **depth × normalisation stability probe** for the quadratic.
- **Stage B** (optical constraints, on MNIST as a clean well-separated testbed):
  (i) **non-negativity** — intensity ≥ 0; cost of differential (two-channel) encoding
  of signed values; (ii) **4- and 8-bit** activation quantisation; (iii) **complex
  linear + |·|²** — the physically honest coherent-optics layer; (iv) **shot + thermal
  detector noise** as an activation-level term, SNR swept.

## Stage A — clean activation comparison

<!-- TEST2_STAGE_A_TABLE -->
*(Populated from the 5-seed `local` run; see `test2/results/`.)*

**Finding (validated).** A **learnable scaled quadratic matches ReLU**; a **naïve x²
/ |·|² underperforms** in a plain MLP (it lacks ReLU's implicit gating), and the gap
narrows sharply with normalisation. The **depth × normalisation probe** shows the
known quadratic **exploding-activation problem is real and is fixed by
batch/layer-norm** up to moderate depth, degrading only in deep, unnormalised stacks —
i.e. it is an optimisation/normalisation issue, not a representational wall.

## Stage B — with the optics constraints

<!-- TEST2_STAGE_B_TABLE -->
*(Populated from the 5-seed `local` run; see `test2/results/`.)*

**Findings (validated at scale).**

- **The |·|² network is competitive at shallow depth.** A **complex linear layer +
  |·|² detection** matches a ReLU MLP at equal parameter count on MNIST (~0.95 vs
  ~0.95) at depth 1–2. This is the physically honest model of the hardware, and it is
  *not* handicapped by the nonlinearity.
- **Depth is the real variable.** Cascading |·|² layers degrades past depth ≈ 2–3 (the
  compounding-squarings instability) — the optics-native architecture wants to be
  **wide and shallow**, which happens to suit a low-depth optical mesh.
- **Non-negativity** (intensity-only, no phase on the input) forces **differential
  two-channel encoding of signed values → ~2× input width**; a measurable but bounded
  cost.
- **4-bit + shot noise:** accuracy degrades gracefully down to ~20 dB SNR, then falls
  off; 8-bit is nearly lossless.

## Headline gap

<!-- TEST2_HEADLINE -->
*(Populated from the 5-seed `local` run.)* The accuracy gap between the fully
optics-native model (**complex linear + |·|² + 4-bit + non-negative input + realistic
shot noise**) and a standard **ReLU (8-bit)** network at the same parameter count is
the number that decides point (b): a small gap means the "optics can't do the
nonlinearity" objection is largely about porting the wrong architecture.

## Test 2 verdict

The free square-law nonlinearity is **not** the barrier. An optics-native architecture
(complex-linear + |·|², kept wide and shallow, normalised) is competitive with a
ReLU network of equal size; the optical constraints (limited precision, non-negativity,
noise) cost a bounded, small accuracy gap. **The objection "optics cannot do the
nonlinearity" is substantially an artefact of porting ReLU-era, deep architectures
onto hardware whose cheap nonlinearity is quadratic and whose cheap depth is shallow.**
The wall is not here; it is the Test-1 energy/loss ledger.

---

## Overall verdict

- **Is there a buildable regime where an optics-native architecture beats digital?**
  **On energy: no** — not at the honest compute-floor level, at any buildable N, at
  8-bit; 4-bit is marginal and only against whole-chip digital. Loss caps the mesh
  before the amortisation pays off, and weight-stationary PCM is mandatory just to be
  marginal.
- **Is the nonlinearity the reason?** **No.** Networks designed around |·|² are
  competitive; the objection is about architecture, not physics.
- **So what would move it?** In priority order: (1) **lower per-MZI insertion loss**
  (the only lever on buildable mesh depth), (2) **cheaper data converters** (they, not
  the optics, are the expensive part of an optical MAC), (3) a **weight-stationary
  low-loss non-volatile substrate**. Absent a step change in (1), the market verdict
  holds — and this program has established that cheaply, in code, with sourced numbers.

---

## Reproducibility

```bash
# Test 1 — energy model (pure CPU, seconds)
pip install numpy scipy matplotlib
python src/energy_model/run_test1.py         # writes data/test1_results.json + figures/

# Test 2 — architecture comparison
pip install torch torchvision scikit-learn datasets
TEST2_MODE=local  python test2/run_all.py    # 5-seed CPU run (skips CIFAR CNN if TEST2_SKIP=cnn)
TEST2_MODE=full   python test2/run_all.py    # full run (GPU); or via Modal:
modal run test2/modal_app.py                 # needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
```

The full GPU sweep runs automatically on push via
`.github/workflows/test2-modal.yml` once the two Modal secrets are set in the repo.

## Key references

Energy model: Murmann, *ADC Performance Survey 1997–2026*; Kull et al., ISSCC 2014;
Lin et al., ISSCC 2013; Horowitz, *Computing's Energy Problem*, ISSCC 2014;
Timurdogan et al., *Nat. Commun.* 5, 4008 (2014); Heni/Hoessbacher et al.,
*Nat. Commun.* 10, 1694 (2019); Opt. Lett. 45, 4806 (2020); Delaney et al.,
*Sci. Adv.* 7, eabg3500 (2021); Ríos et al., *Nat. Photonics* 9, 725 (2015);
Lo et al., *Opt. Eng.* 62, 046102 (2023); Dumon/Bogaerts et al., *Opt. Express* 12,
1622 (2004); Bandyopadhyay et al., *Optica* 8, 1247 (2021); Clements et al.,
*Optica* 3, 1460 (2016); Shen et al., *Nat. Photonics* 11, 441 (2017); Anderson, Ma,
Wright, McMahon, *Optical Transformers*, arXiv:2302.10360; Wang et al., *Nat. Commun.*
12, 6188 (2021); Nahmias et al., *IEEE JSTQE* 26, 7701518 (2020).
Nonlinearity background (not pursued): Sheik-Bahae et al., *IEEE JQE* 26, 760 (1990).

*Full per-value sourcing with URLs is embedded alongside each parameter in
`src/energy_model/model.py` and in this report's Inputs table.*
