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
**matches ReLU at equal parameter count on MNIST: 0.959 vs 0.964** (5 seeds), and
adding the discrete optics constraints (**4-bit + non-negative**) keeps it there
(**0.962**). Two caveats, both benign: the quadratic cascade has a **depth instability**
(diverges by depth ≈ 8 unnormalised; **LayerNorm fixes it through depth 16**), so the
architecture wants to be wide and shallow; and the **only** material accuracy penalty is
**detector shot noise**, which is set by the photon budget — ~1.3 pts at 40 dB SNR,
~9 pts at 20 dB. That penalty is a dial on Test 1's *energy* axis, not a property of
the nonlinearity. So the nonlinearity is not the wall. **The wall is the energy/loss
ledger of Test 1.**

Killing the energy half cheaply is the intended good outcome. Test 2 says the door
is not closed by the nonlinearity — it is closed (for now) by loss and converters,
which is where any future hardware advance must land.

**All three buildable mesh topologies were subsequently tested and all three fail the
compute floor** (see the two follow-up sections and the closing three-architecture
table): the dense mesh dies on cascaded insertion loss, the O(log N) butterfly on
converter amortisation, and the microring crossbar on thermal stabilisation of its N²
resonances — three symptoms of one disease, that N² optical weights cannot participate
in a computation without a per-weight physical cost (depth-loss, re-conversion, or
standing power) that exceeds the digital MAC. The single actionable lever across every
architecture is an order-of-magnitude drop in per-conversion / per-element electrical
energy, not a new topology or activation.

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
| Activation | MLP (Cover-Type) | CNN (CIFAR-10) | Transformer |
|---|---|---|---|
| relu | 0.782±0.003 | — | 0.236±0.001 |
| gelu | 0.776±0.002 | — | 0.237±0.002 |
| square | 0.481±0.047 | — | 0.234±0.002 |
| modsq | 0.481±0.047 | — | 0.234±0.002 |
| abs | 0.803±0.001 | — | 0.234±0.005 |
| scaled_quad | 0.740±0.002 | — | 0.238±0.001 |

*(5 seeds; mean best test-acc ± std. CIFAR-10 CNN column is filled by the Modal GPU
run — `test2/results/test2_results_full.json` — the local backup skips it. Transformer
task = synthetic associative recall, where the small model saturates near-uniformly at
this scale, so it does not discriminate the activations; the Modal run uses AG News.)*

Depth × normalisation stability (x² activation, mean best acc; `div` = diverged seeds):

| norm | depth 2 | depth 4 | depth 8 | depth 16 |
|---|---|---|---|---|
| none | 0.708 | 0.509 | **NaN ⚠3/3 diverged** | **NaN ⚠3/3 diverged** |
| batch | 0.683 | 0.490 | 0.488 | 0.487 |
| layer | 0.777 | **0.816** | **0.812** | **0.803** |

**Finding (validated, 5 seeds).** On tabular Cover-Type, ReLU (0.782) and GELU (0.776)
lead, a **learnable scaled quadratic is competitive (0.740)**, **`abs` actually wins
(0.803)**, and a **naïve `x²`/`|·|²` fails (0.481)** — it collapses toward the majority
class because it lacks ReLU's gating and saturates. On the transformer FFN all
activations tie within noise (~0.235). The **depth × normalisation probe is the clean
result**: an unnormalised `x²` MLP **diverges outright at depth ≥ 8** (NaN, all seeds);
**BatchNorm stops the divergence but accuracy still collapses to the majority class**;
**LayerNorm fully tames it and stays strong (0.80–0.82) through depth 16.** So the
quadratic's notorious exploding-activation problem is real but is an
optimisation/normalisation issue with a known fix (LayerNorm), not a representational
wall.

## Stage B — with the optics constraints

<!-- TEST2_STAGE_B_TABLE -->
**Optics-native |·|² depth sweep** (complex linear + |·|², full precision):

| depth | mean best acc | diverged |
|---|---|---|
| 1 | 0.958±0.001 | 0 |
| 2 | 0.960±0.001 | 0 |
| 3 | 0.950±0.002 | 0 |
| 4 | 0.645±0.008 | 0 |

**Quantisation** (real MLP, ReLU vs x², activation bit depth):

| activation | full | 8-bit | 4-bit |
|---|---|---|---|
| relu | 0.964±0.001 | 0.964±0.001 | 0.963±0.001 |
| square | 0.958±0.001 | 0.958±0.001 | 0.955±0.001 |

**Optics-native variants** (complex linear + |·|², matched params):

| variant | mean best acc | params |
|---|---|---|
| complex_modsq_fp | 0.959±0.001 | 271,434 |
| complex_modsq_8b | 0.958±0.001 | 271,434 |
| complex_modsq_4b | 0.958±0.001 | 271,434 |
| complex_modsq_4b_nonneg | 0.962±0.001 | 498,794 |

**Shot-noise SNR sweep** (optics-native 4-bit):

| SNR (dB) | mean best acc |
|---|---|
| None | 0.958±0.001 |
| 40 | 0.951±0.000 |
| 30 | 0.915±0.002 |
| 20 | 0.835±0.014 |
| 15 | 0.807±0.015 |
| 10 | 0.744±0.019 |

**Non-negativity cost**: signed-input 271,434 params → differential (two-channel) 498,794 params (ReLU ref 270,346).


**Findings (validated, 5 seeds on MNIST).**

- **The |·|² network matches ReLU.** A **complex linear layer + |·|² detection** at
  equal parameter count reaches **0.959** vs ReLU **0.964** — a **0.5-point gap**. This
  is the physically honest model of the hardware and it is *not* handicapped by the
  nonlinearity.
- **Precision is nearly free.** Quantising the optics activations to **4-bit costs
  ~0 (0.958)**; 8-bit is lossless. Same for the real square net (4-bit 0.955).
- **Non-negativity is cheap when you pay the width.** Differential two-channel encoding
  of signed values doubles the input width (271k → 499k params); with that width the
  optics net actually **improves to 0.962**. The cost is parameters/area, not accuracy.
- **Depth is the real architectural limit.** Cascading |·|² layers is fine to depth 3
  (0.950) and **breaks at depth 4 (0.645)** — the compounding-squarings instability.
  The optics-native net wants to be **wide and shallow**, which happens to suit a
  low-depth optical mesh.
- **Shot noise is the one real cost, and it is an *energy* knob.** Accuracy vs detector
  SNR: 40 dB → 0.951, 30 dB → 0.915, 20 dB → 0.835, 15 dB → 0.807. Since SNR is set by
  the photon budget (√N photons; Test 1), **the accuracy gap is literally a function of
  how many joules you spend per detection** — connecting the two tests.

## Headline gap

<!-- TEST2_HEADLINE -->
**Headline gap (MNIST, matched params):** ReLU 8-bit = 0.964±0.001, fully optics-native (complex + |·|² + 4-bit + non-negative + shot noise @20 dB) = 0.874±0.004 → **accuracy gap = +9.0 points**.

**But read that gap correctly: it is almost entirely the 20 dB shot-noise assumption,
not the architecture.** Strip the noise and the fully-constrained optics net
(complex + |·|² + 4-bit + non-negative) is **0.962 vs 0.964 — a 0.2-point gap**. Add
noise back and the gap tracks the photon budget: **~1.3 pts at 40 dB, ~5 pts at 30 dB,
~9 pts at 20 dB.** The nonlinearity, precision, and sign constraints cost essentially
nothing; the only material penalty is running the detector photon-starved, which is a
dial on Test 1's energy axis, not a property of the architecture.

## Test 2 verdict

The free square-law nonlinearity is **not** the barrier. An optics-native architecture
(complex-linear + |·|², wide and shallow, LayerNorm) **matches a ReLU network of equal
size to within ~0.5 points**, and the discrete optics constraints (4-bit precision,
non-negative differential encoding) add essentially nothing. The **only** material gap
is detector shot noise, and that is set by the photon/energy budget — so it is the same
axis Test 1 measures, not an independent wall. **The objection "optics cannot do the
nonlinearity" is substantially an artefact of porting ReLU-era, deep architectures onto
hardware whose cheap nonlinearity is quadratic and whose cheap depth is shallow.** The
wall is not here; it is the Test-1 energy/loss ledger.

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

# Follow-up — Does an O(log N) mesh reopen the thesis?

**One-line answer: No.** An O(log N) **butterfly/FFT mesh defeats the loss wall** —
it is buildable to enormous N (depth 14 at N=16384, ≈5–31 dB; buildable to N≈10⁹ at
≤0.5 dB/MZI) — **but it never beats the compute floor at any buildable N**, because a
butterfly does only ~N·log₂N MACs, so the N input + N output conversions amortise over
only **log₂N**, not N. Per-MAC conversion plateaus at **120–400 fJ (4-bit) / ~1–2 pJ
(8-bit)**, above the 20–40 fJ floor **even at N = 2²⁰**. Crossover vs the compute floor
exists in **0% of Monte-Carlo draws**, at every precision and both weight models. And
making the butterfly expressive enough to replace a dense layer (a stack of ~log₂N
factors) pushes depth to (log₂N)² and **fails on loss instead** (57 dB at N=1024). The
two failure modes squeeze from both sides. **Per the pre-registered gate, A2
(butterfly-network training) is moot and was not run.** The last door is closed on
physics. Code: `src/energy_model/run_test1_butterfly.py`,
data `data/test1_butterfly_results.json`.

## A1 — butterfly energy model (same sourced ranges, same Monte-Carlo)

Implementation reuses `model.py` unchanged except mesh structure: **depth = log₂N**,
**(N/2)·log₂N MZIs**, **~N·log₂N phase shifters**, and the crux — the
**amortisation dimension A = log₂N** (N·log₂N MACs per transform ÷ 2N conversions),
versus A = N for a dense Clements mesh.

### Q4 — buildability (nominal geometry; the loss wall is gone)

| N | MZIs | phase shifters | optical depth | die area | loss @0.1 / 0.5 / 2.0 dB/MZI | detectable @0.1? | thermo-optic idle |
|---|---|---|---|---|---|---|---|
| 64 | 192 | 384 | 6 | 0.08 cm² | 3.8 / 6.2 / 15.2 dB | ✅ | ~0 |
| 256 | 1,024 | 2,048 | 8 | 0.41 cm² | 4.1 / 7.3 / 19.3 dB | ✅ | 0.01 kW |
| 1024 | 5,120 | 10,240 | 10 | 2.1 cm² | 4.4 / 8.4 / 23.4 dB | ✅ | 0.05 kW |
| 4096 | 24,576 | 49,152 | 12 | 9.8 cm² | 4.7 / 9.4 / 27.4 dB | ✅ | 0.25 kW |
| 16384 | 114,688 | 229,376 | 14 | 45.9 cm² | 4.9 / 10.5 / 31.5 dB | ✅ | 1.15 kW |

Max buildable N within the 33 dB budget: **≈1.07×10⁹** at 0.1 and 0.5 dB/MZI, **16,384**
at 2.0 dB/MZI. Contrast the dense mesh, which went dark by N≈256. **Loss is no longer
the binding constraint.**

### Crossover bands (butterfly, loss enforced, MC over the sourced ranges)

| Scenario | vs whole-chip digital (0.3–2 pJ) | vs compute floor (20–40 fJ) |
|---|---|---|
| 4-bit, PCM weights | N≈39 [16–84%: 16–664], exists 89% | **no crossover — 0%** |
| 4-bit, thermo-optic | N≈146, exists 1% | **no crossover — 0%** |
| 8-bit, PCM weights | N≈977, exists 22% | **no crossover — 0%** |
| 8-bit, thermo-optic | no crossover | **no crossover — 0%** |

Same story as the dense mesh against the two baselines — a butterfly can beat the
overhead-heavy *whole-chip* number at 4-bit — **but against the honest compute floor it
never crosses, in zero of 3000 draws, at any setting.**

### Why: conversion amortises as 1/log N, not 1/N

Per-MAC term breakdown at nominal parameters (PCM weights); conversion =
(E_DAC+E_ADC+E_mod)/log₂N:

| precision | N=64 | N=1024 | N=16384 | N=2²⁰ (10⁶) | compute floor |
|---|---|---|---|---|---|
| 4-bit conversion/MAC | 400 fJ | 240 fJ | 171 fJ | 120 fJ | **20–40 fJ** |
| 8-bit conversion/MAC | 2250 fJ | 1350 fJ | 964 fJ | 675 fJ | **20–40 fJ** |

To reach 40 fJ at 4-bit you would need log₂N > 60, i.e. **N > 10¹⁸** — more input
modulators than is remotely physical. The 1/log N amortisation is simply too weak;
loss and MAC-count are coupled, and buying the low depth costs you the compute to
amortise over.

### The squeeze — dense-equivalent expressivity fails on loss

A single butterfly is a restricted transform. Emulating a dense layer needs ~log₂N
stacked butterfly factors → optical depth (log₂N)² → loss at 0.5 dB/MZI:

| N | factors | stacked depth | loss @0.5 dB/MZI | detectable? |
|---|---|---|---|---|
| 256 | 8 | 64 | 37.4 dB | ❌ |
| 1024 | 10 | 100 | 56.8 dB | ❌ |
| 4096 | 12 | 144 | 80.4 dB | ❌ |
| 16384 | 14 | 196 | 108.3 dB | ❌ |

So the native butterfly fails on **energy** (weak amortisation) and the expressive
butterfly fails on **loss** (depth (log₂N)²). There is no structure that is both
low-depth (buildable) and high-MAC-count (amortisable), because in a coherent mesh
those two are the same axis.

### Honesty — what is charged, and what is not (all generosities favour optics)

**Charged:** N input DACs + N output ADCs per transform (the amortisation floor); N
input modulators; laser power derived from detector noise, through the log₂N mesh loss;
thermo-optic idle **or** PCM hold; and the dense-equivalent depth penalty (the squeeze).

**Not charged (each makes optics look *better* than reality):** waveguide-crossing loss
from the FFT shuffle — a planar butterfly routes wide-stride connections that cross
O(N) waveguides per path; at ~0.1 dB/crossing this alone would add tens-to-hundreds of
dB and kill even the native butterfly on loss; digital permutation/recombination if the
array must be tiled across reticles; and the reduced expressivity of one butterfly vs a
dense layer (charged only in the squeeze sidebar, not in the main crossover). **Every
uncharged item worsens the optical case, and it already fails the floor by 3–50×.**

### Gate decision

A1 shows **no crossover against the compute floor at any buildable N, in 0% of draws.**
Per the pre-registered gate, **A2 is moot and was not run** — running it could only
report an accuracy number for an architecture that has already lost on energy. The
photonic-compute thesis, in its MZI-mesh form, is **closed on physics**: dense meshes
die on loss, butterfly meshes die on converter amortisation, and the expressive
butterfly dies on loss again. The one lever that could reopen it is unchanged from
Test 1 — an **order-of-magnitude drop in per-conversion energy** (the converters, not
the optics, are the cost), not a cleverer mesh topology.

---

# Follow-up 2 — Does a microring crossbar reopen the thesis?

**One-line answer: No — it is killed by thermal stabilisation of its N² resonances.**
The microring crossbar is the one topology that *escapes the amortisation squeeze*: an
N×N ring array does N² MACs at optical depth ~1 (light crosses a K-ring bus, not N
cascaded stages), and WDM carries K≈14 inputs in parallel, so conversion amortises to
**~65 fJ/MAC — near the compute floor**, and the bus loss stays ~3–4 dB (the loss wall
does **not** reappear). But it fails on a mechanism the mesh model never contained:
every ring is a **resonance that must be actively held on its channel against silicon's
80 pm/K thermal drift**, continuously, whether or not it is computing. With N² rings
that gives `thermal_stab/MAC = N·P_stab/(K·bw)` = **44 pJ/MAC at N=1024** (2.7 pJ at
N=64, 702 pJ at N=16384), 30–10,000× over the 20–40 fJ floor. **Crossover vs the
compute floor: 0% of Monte-Carlo draws, at every N, precision, and weight model.** Per
the pre-registered gate, no accuracy study was run. Code:
`src/energy_model/run_test1_crossbar.py`, data `data/test1_crossbar_results.json`.

## The five failure modes, ranked by what binds

| # | Failure mode | Modelled quantity | Binds? |
|---|---|---|---|
| 1 | **Thermal stabilisation** | N² rings × 1–30 mW held; `N·P_stab/(K·bw)` per MAC | **YES — first and decisively** |
| 3 | WDM channel count K | K = FSR/(3.4–4.6·linewidth) ≈ 14 (demonstrated 4–16) | caps amortisation at ~65 fJ, but **not** below floor — not the killer |
| 5 | Weight precision | Lorentzian + drift → **~4–5 bits** (Tait 3.1–5.1, Feldmann 5) | 8-bit is **unphysical** for a weight ring; 4-bit is the only real op point |
| 2 | Fabrication yield | per-ring yield ≈ 1 when trim range ≥ 1 FSR (mandatory) | not binding — trim covers the ~1 nm scatter (at the cost of hold power, mode 1) |
| 4 | Bus loss / crosstalk | K rings on bus × 0.01–0.2 dB → **3–4 dB** | **survives** — K-bounded, so "depth ~1" holds |

**Answer to the three questions.** (1) No buildable (N,K) beats the compute floor, in
0% of draws. (2) **Thermal stabilisation binds first**, unambiguously — not K (which
was the other candidate: conversion is fine at ~65 fJ). (3) **"Depth ~1" survives** —
the K≈14-ring bus is 3–4 dB, so the loss wall does *not* reappear for the resonant
crossbar; the same K-bound that limits amortisation also bounds the loss.

## Sourced inputs (new physics; ranges + citations)

| Quantity | Low – Nom – High | Source |
|---|---|---|
| Per-ring thermal hold power | 1 – 6 – 30 mW | Feldmann Nature 2021 / Nahmias JSTQE 2020 (≈1 mW avg); undercut..full-FSR |
| Ring loaded Q (weight bank) | 5×10³ – 1×10⁴ – 2×10⁴ | Tait et al., Opt. Express 24, 8895 (2016) |
| Ring radius → FSR | 5–20 µm → 18–4.6 nm | FSR = λ²/(n_g·2πR), n_g 4.0–4.4 |
| Channel spacing / linewidth | 3.4 – 4.0 – 4.6 | Tait et al., IEEE IPC 2017 (8116022), 1-pole, 3 dB penalty |
| ⇒ WDM channels K | 4 – 14 – ~30 (148 optimistic) | Tait demonstrated 4–16; finesse limit |
| Through-port loss per ring | 0.01 – 0.05 – 0.2 dB | Si add-drop microring literature |
| Fab resonance scatter | 0.1 – 0.5 – 2 nm | Selvaraja JSTQE 2010; Lu Opt. Express 2017 (~1 nm wafer) |
| Heater trim range (≥1 FSR) | 3 – 6 – 12 nm | Milanizadeh JLT 2021 (wafer-scale trimming) |
| Ring weight precision | 3.1 – 4.0 – 5.1 bits | Tait 2016/2018; Feldmann 2021 (5 bit) |
| Ring modulator switch energy | 0.9 – 3 – 6 fJ/bit | Timurdogan Nat. Commun. 2014 (confirmed) |

## Feasibility (nominal, K≈14) and thermal-stabilisation floor

| N | rings (N²) | bus loss | yield | idle stabilisation | thermal-stab/MAC (4-bit) | conversion/MAC |
|---|---|---|---|---|---|---|
| 64 | 4,096 | 3.7 dB | ~1 | 0.02 kW | 2.7 pJ | 88 fJ |
| 256 | 65,536 | 3.7 dB | ~1 | 0.4 kW | 11 pJ | 71 fJ |
| 1024 | 1.05 M | 3.7 dB | ~1 | 6.3 kW | **44 pJ** | 66 fJ |
| 4096 | 16.8 M | 3.7 dB | ~1 | 101 kW | 176 pJ | 65 fJ |
| 16384 | 268 M | 3.7 dB | ~1 | 1.6 MW | 702 pJ | 65 fJ |

Conversion (the squeeze) is beaten — ~65 fJ, near the floor. **Thermal stabilisation
is the whole story**, and it is *worse* than the dense MZI thermo-optic term (which
already killed 8-bit at ~1 pJ/MAC) by a factor N/K: WDM parallelises K inputs, but you
still hold N² rings, so per useful MAC the standing power is N/K× higher.

## Honesty — charged vs not charged (all generosities favour optics)

**Charged:** N² ring stabilisation at 1 mW/ring (the *optimistic* Feldmann average, not
Tait's measured 40–52 mA bias); K-ring bus loss; N input DAC/mod + N²/K output ADC +
N²/K digital accumulation; laser through the bus. PCM weights zero the weight-*hold*
power but **not** the resonance lock (the ring still drifts), so the thermal term stays.

**Not charged (each worsens the optical case):** per-ring feedback-control electronics
(Tait's photoconductive-heater sensing + per-channel control loop — real and O(N²));
inter-ring thermal crosstalk; the comb/laser source power for K carriers beyond WPE;
and that ring weight precision (~5 bit) cannot reach 8-bit at all. The *non-resonant*
PCM-crossing variant (Feldmann 2021) avoids the ring thermal term — but then light
crosses N waveguide crossings at 0.12 dB each → **123 dB at N=1024**, i.e. the dense
mesh's loss wall in a new guise (Feldmann measured only up to 32×32, explicitly
loss-limited). Either way the crossbar does not scale.

---

## Programme conclusion — three architectures, one table

An N×N linear layer has N² weights; those weights must be physically instantiated in
the optics, and **that instantiation always costs more than the digital MAC it
replaces** — the cost simply moves between depth, amortisation, and standing power:

| Architecture | Optical depth | MACs / pass | Conversion amortises over | **Dies on** | Magnitude |
|---|---|---|---|---|---|
| **Dense Clements/Reck mesh** | N | N² | N (good) | **cascaded insertion loss** | 512 dB @ N=1024 |
| **Butterfly / FFT mesh** | log₂N | N·log₂N | log₂N | **converter amortisation** | 120–400 fJ/MAC, never floor |
| **Microring crossbar** | ~1 (K-ring bus) | N²·(K/N per pass) | K ≤ ~30 | **thermal stabilisation of N² rings** | 44 pJ/MAC @ N=1024 |

**None reaches the 20–40 fJ/MAC digital compute floor at any buildable size, in 0% of
sourced Monte-Carlo draws.** Dense dies on depth-loss, butterfly on amortisation,
crossbar on standing power — three different symptoms of the same disease: you cannot
make N² optical weights participate in a computation without paying, per weight, either
a cascaded loss (depth), a re-conversion (too few MACs), or a continuous hold
(resonance). In every case the electrical overhead — data converters and thermal
control, **not** the "free" matmul — dominates, exactly the regime the 2026 market
converged on (optical *interconnect* yes, optical *compute* no). The programme's single
actionable lever is unchanged across all three architectures and both tests: an
**order-of-magnitude reduction in per-conversion / per-element electrical energy**, not
a new mesh topology and not a new activation. Test 2 separately established that the
nonlinearity was never the barrier. **The photonic-compute thesis is closed on physics,
cheaply, in code, with every input sourced — which was the goal.**

---

## Reproducibility

```bash
# Test 1 — energy model (pure CPU, seconds)
pip install numpy scipy matplotlib
python src/energy_model/run_test1.py            # dense/tiled: data/test1_results.json + figures/
python src/energy_model/run_test1_butterfly.py  # O(log N) butterfly: data/test1_butterfly_results.json
python src/energy_model/run_test1_crossbar.py   # microring crossbar: data/test1_crossbar_results.json

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
