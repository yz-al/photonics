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

# Follow-up 3 — PCM weights + mode-multiplexing: the one variant that reaches the floor

**One-line answer: a qualified yes, narrowly.** Non-volatile PCM weights in a
*non-resonant* geometry remove the thermal-stabilisation term that killed the ring
crossbar, mode-multiplexing raises the amortisation to K_eff = K_λ·M without adding bus
loss (modes are a *parallel* lossless fan-in, not a series cascade), and low-loss
Sb₂Se₃ keeps per-element loss bounded. The result is the **first architecture in the
programme with a non-zero floor-crossing fraction: 4-bit, non-resonant, ~13% of sourced
Monte-Carlo draws (N≈110)**. But it is a heavily-conditioned crack, not an open door: it
holds **only** at 4-bit, **only** with low-loss (Sb₂Se₃, not GST) PCM, **only** in a
non-resonant geometry, and **only** for write-once inference (endurance ~27 cycles). At
8-bit it fails (2%); with resonant weights it fails (≤1%). Code:
`src/energy_model/run_test1_pcm_crossbar.py`, data `data/test1_pcm_crossbar_results.json`.

## What changed vs the thermal-ring crossbar (a parameter variant, not a new model)

| Term | Thermal-ring crossbar | PCM + mode-mux variant | Source for the change |
|---|---|---|---|
| Weight hold power | N² rings × ~6 mW → **44 pJ/MAC** | **0** (non-volatile, non-resonant weight) | Feldmann Nature2021; Sun NatCommun2025 (weights on straight waveguides / MZI, not rings) |
| Amortisation dim | K_λ ≤ ~30 | **K_eff = K_λ·M** (M spatial modes, parallel) | Sun NatCommun2025 "lossless mode fan-in" (M=3 demo) |
| Bus loss | K_λ ring through-loss | K_λ·(PCM IL) + M·(mode IL) — **modes don't add series loss** | Sun 2025 (per-mode 0.2–0.32 dB, parallel) |
| Weight-update | n/a (thermal tuning per op) | µJ/write, but **write-once** → ≈0 per MAC | Sun 2025 (~0.1–8 µJ, endurance ~27 cycles) |

## Sourced inputs (PCM + mode-mux)

| Quantity | Low – Nom – High | Source |
|---|---|---|
| PCM per-element insertion loss | 0.1 – 0.4 – 1.0 dB | Delaney SciAdv2021 (Sb₂Se₃ ~0.5 dB, 0.006 dB/µm); GST would be 0.8–3 dB and kills it |
| PCM write energy | 1 nJ – 0.5 µJ – 8 µJ | Sun NatCommun2025 (foundry doped-Si heater, derived); Fang NatNano2022 (graphene, nJ) |
| PCM weight precision | 3 – 6 – 9 bit | Sun2025 (3 b foundry, 7 levels); Gong ACSPhot2024 (6 b N-Sb₂Se₃); Zhou2026 (9 b, sim) |
| Cycling endurance | 27 – 1000 – 10⁴ cycles | Sun2025 (27, foundry); Sb₂Se₃ sub-cell (>10⁴) — **write-once at foundry fidelity** |
| Spatial mode count M | 2 – 4 – 10 | Sun2025 (M=3 demo); SciRep2019 (6 TE); crosstalk-limited, 16 exotic |
| Per-mode fan-in loss | 0.2 – 0.35 – 0.5 dB | Sun NatCommun2025 (0.2–0.32 dB, near-lossless, parallel) |
| Weight geometry | non-resonant (0 hold) | Feldmann/Sun/Gong/Zhou — straight waveguide / MZI / metasurface, not rings |

## Crossover bands (MC, loss enforced, same protocol/baselines)

| Scenario | vs whole-chip digital | vs compute floor (20–40 fJ) |
|---|---|---|
| 4-bit, non-resonant PCM | N≈21, exists 67% | **N≈109 [51–269], exists 13%** |
| 4-bit, resonant PCM (ring weight) | N≈16, exists 32% | N≈61, exists 1% |
| 8-bit, non-resonant PCM | N≈34, exists 69% | N≈584, exists 2% |
| 8-bit, resonant PCM | N≈23, exists 33% | no crossover (0%) |

## Where the cost went (per-MAC, nominal, non-resonant) — and the honest caveats

With the thermal term gone and update energy amortised away, the residual per-MAC cost
is **conversion (16–40 fJ at 4-bit, at the floor) plus a WDM ring-addressing residual
(~150 fJ if the wavelength routing uses rings; →0 with a non-resonant AWG demux, Zhang
Nanophotonics2024)**. The 13% of draws that clear the floor are exactly those with
low-loss PCM, low ring-addressing power (or AWG), high K_eff, and 4-bit — the favourable
tail, not the nominal. The conditions that make it work are also its limits:

- **Material:** requires **low-loss Sb₂Se₃/Sb₂S₃**; GST (0.8–3 dB/element) reintroduces
  the cascaded loss wall immediately.
- **Regime:** endurance **~27 foundry cycles → write-once inference only.** Fine for a
  fixed deployed model; **rules out training and frequent reconfiguration.**
- **Precision:** foundry PCM is **~3–6 bit**, which *matches* the 4-bit operating point
  but means 8-bit (where it fails on energy anyway) is also unsupported.
- **Generosities still in optics' favour:** non-resonant AWG addressing charged at 0 in
  the optimistic draws; PCM array fabrication yield of N² elements not charged;
  mode-crosstalk-induced precision loss (M>3) not charged against accuracy.

## Gate decision — a crossover exists, so scope (do not run) the accuracy study

Unlike the butterfly and thermal-ring crossbar (0% at the floor), this variant crosses
in a non-zero fraction, so the pre-registered gate says report it and **scope** what an
accuracy study would test — it does **not** authorise running one, and the constraints
are severe enough that the honest deliverable is the scoping, not a training run. An
accuracy study for this hardware would need: (1) **weight precision at 3–6 bit**
(PCM multi-level, quantisation-aware training), (2) **write-once weights** — no
gradient updates to the optical weights after deployment, i.e. train-then-freeze or
train digitally and program once, (3) **mode-crosstalk as a fixed weight-mixing matrix**
(−13 to −17 dB between modes at M=3, growing with M) applied to the linear layer, and
(4) **PCM level-drift / relaxation noise** on the held weights. Test 2 already showed the
optics-native forward model (complex-linear + |·|², 4-bit) matches ReLU; the open
question this scoping isolates is whether **3–6-bit write-once weights with fixed
mode-crosstalk** hold that accuracy — a bounded, well-posed study, deferred here per the
gate.

---

# Follow-up 4 — Photonic spiking / time-domain encoding: attack the converter floor itself

**One-line answer: no for every fabricated device; the converter floor moves, but the
energy just relocates to the laser/neuron.** Spiking is the only track that changes the
*encoding* rather than the amortisation: a spike is 1-bit (arrived/not), so the receiver
is a **comparator (~8 fJ), not an ADC (~0.9–3.5 pJ)** — a real, sourced **~100–440×**
receiver saving, and 1-bit detection needs ~20–1000 photons vs ~2²ᵇ for a b-bit sample.
But every *fabricated* photonic spiking device sits **25–35,000× above the 20–40 fJ
floor**, because the energy is in the **neuron laser bias + spike generation**, not the
readout. The 1/N bias amortisation that would reach the floor needs **10–164 W of
on-chip laser bias** (not buildable); inside a realistic ~3 W budget only **projected
sub-mW nanolaser neurons** (unfabricated) cross. Code:
`src/energy_model/run_test1_spiking.py`, data `data/test1_spiking_results.json`.

## Model validation — Xiang OEA 2026 reproduced

Xiang et al. (Opto-Electronic Advances 2026, arXiv:2512.00419) report a 16-channel
DFB-SA spiking array at **987.65 GOPS/W (1.01 pJ/op)** and an MZI mesh at 1.39 TOPS/W.
Our model, using their accounting (16 ch × 8 equiv-ops/neuron × 5 GHz, 0.648 W ≈ 40 mW
per neuron), returns **1000 GOPS/W = 1.00 pJ/op** — a match. The number is bias-dominated:
0.64 W of laser standing power over 640 GOPS. Both their numbers are ~25–70× the floor.

## Sourced inputs (spiking)

| Quantity | Low – Nom – High | Source |
|---|---|---|
| Comparator energy / decision | 3 – 8 – 15 fJ | Kala et al. IET 2020 (10.7 fJ @6.25 GHz); ~100–440× < ADC |
| Neuron laser bias (standing) | 0.5 – 10 – 40 mW | Xiang OEA2026 (40 mW DFB-SA); VCSEL lower; nanolaser proj sub-mW |
| Spike generation energy | 10 fJ – 8 pJ – 130 pJ | Xiang (8 pJ/neuron); self-pulsating DFB-SA 67–130 pJ/SOP; nanolaser proj 10 fJ |
| Photons / spike (1-bit) | 20 – 50 – 1000 | quantum limit ~20/bit; 1-photon PSA receiver (Optica) |
| Spikes / neuron / inference | 0.4 – 0.9 – 2.3 | DIET-SNN 0.4; Sengupta 2.35; break-even <1 (Sengupta 2019) |
| Timesteps T | 5 – 16 – 2500 | direct-trained 5–10 (DIET-SNN); rate coding ~2500 (Sengupta) |
| On-chip laser power budget | 1 – 3 – 10 W | thermal ceiling for N biased lasers on-chip |

## Where the energy is — and the buildability wall

Per equivalent-MAC (nominal): **bias dominates** and both bias and generation amortise
as 1/N, but the bias *power* grows with N:

| N | J/eqMAC | bias term | on-chip bias power |
|---|---|---|---|
| 64 | 365 fJ | 250 fJ | 0.64 W |
| 256 | 91 fJ | 63 fJ | 2.6 W |
| 1024 | 23 fJ | 16 fJ | **10 W** |
| 4096 | 5.7 fJ | 3.9 fJ | **41 W** |

So J/eqMAC only reaches the floor where the laser bias is already **10–164 W** — beyond
any on-chip thermal budget. At a 3 W budget the reachable floor value is
**P_bias²·T/(budget·bw)**, which depends only on per-neuron bias:

| Device class | P_bias | N_max @3 W | bias/eqMAC | reaches floor? |
|---|---|---|---|---|
| DFB-SA (measured, Xiang) | 40 mW | 75 | **853 fJ** | ❌ (21×) |
| VCSEL-class | 10 mW | 300 | 53 fJ | ❌ (marginal) |
| nanolaser (projection) | 0.5 mW | 6000 | 0.1 fJ | ✅ but unfabricated |

**Crossover vs the compute floor (MC):** unbounded 88% (at N≈1635, infeasible power);
**power-budget-limited 34% (N≈349) — entirely the sub-10 mW draws.** Every fabricated
device (DFB-SA at 40 mW) is 0% buildable.

## Accuracy — priced from the primary SNN literature (not re-derived)

The metric for spiking is **J per inference at matched accuracy**, so the accuracy gap
must be priced. Rather than reimplement well-established results, we take them from the
sources: **MNIST SNN-vs-ANN gap ≈ 0** (Rueckauer 2017, lossless); **CIFAR-10 ≈ 1 pt at
usable low latency** (DIET-SNN T=5: 92.7% vs ANN 93.7%), closing to ~0 only at ~2500
timesteps (Sengupta), which multiplies energy; **ImageNet ≈ 5 pt** (Sengupta). At matched
accuracy the low-latency ~1-pt CIFAR-10 penalty means a spiking net needs more
neurons/timesteps to equal the ANN — pushing the already-marginal energy the wrong way.
This is the honest tradeoff: even the 34% buildable-crossover (sub-mW) regime pays a
1-pt CIFAR accuracy tax that a fair J/inference comparison must absorb. (A from-scratch
5-seed SNN run would reproduce this documented gap; per the "do not reimplement published
work" rule and the gate, it is priced from the literature.)

## Gate + verdict

A buildable floor-crossing exists (34% of draws) but **only for sub-10 mW VCSEL/nanolaser
neurons that are projected, not fabricated**; the measured DFB-SA device class is 0%.
Changing the encoding to comparators **does move the readout cost** (~100× saving, real)
but **not the dominant term** — the laser bias/generation — so it does not move the floor
for any fabricated device. **Spiking confirms the disease is not topological: a genuine
change of encoding attacks the converter floor and the energy simply reappears in the
optical source.**

---

# Follow-up 5 — Spiking at biological sparsity: gated source, passive weights, single readout

**One-line answer: the source still dominates — because it cannot be gated.** Biological
sparsity only saves source energy if the laser can be switched off between spikes, but the
laser-dynamics literature is unambiguous that a semiconductor spiking neuron must hold a
**continuous near-threshold bias** — both for sub-nanosecond turn-on and to maintain the
excitable regime — so its energy scales with **time, not events**, and sparsity buys almost
nothing on the dominant term. In the model, the gated and continuous regimes give the
**identical** floor-crossing fraction (65% = 65%), which is the direct proof. Code:
`src/energy_model/run_test1_spiking_bio.py`, data `data/test1_spiking_bio_results.json`.

### Sourced inputs (bio-variant)

| Quantity | Low – Nom – High | Source |
|---|---|---|
| Near-threshold bias held continuously (gating buys 1−frac) | 0.80 – 0.95 – 1.0 | Coldren & Corzine (turn-on ∝ prebias); Opt. Lett. 36, 4476 (excitability needs continuous pump); arXiv:2012.08516 (130→67 pJ/spike rate-dependence) |
| Neuron laser bias (standing) | 0.5 – 10 – 40 mW | VCSEL ~2 mW (1 mA×1.8 V); DFB 10–30 mW (up to ~100 mW high-speed / ~50 mA excitable); nanolaser nW–µW (unfabricated array) |
| SNN activation (firing / timestep) | 5 – 10 – 20 % | VGG16 CIFAR-10 T=6: 5.8% (arXiv:2409.08290); ResNet-19 ~15% (arXiv:2511.13050) |
| Low-latency timesteps T | 4 – 6 – 10 | DIET-SNN (arXiv:2008.03658); T=4–32 near-lossless conversion (2205.07473) |
| Gated-pulse drive energy | 10 fJ – 0.1 – 1 pJ | gain-switched pulse ~0.96 pJ/200 ps; nanolaser 1 fJ/bit (arXiv:2212.05148) |

**Task 1 — audit of the prior spiking model.** It already assumed the pessimistic case:
source continuously on (`bias = P_bias·T/bw`, independent of spike count), per-neuron
per-timestep comparator readout, weights not separately held. So sparsity, passive weights
and single readout are genuinely *new* levers here, not double-counted.

**Task 2 — can the source be gated per event? (the crux, sourced).** No, for any fabricated
device:
- Sub-ns turn-on requires prebias *near threshold*; cold-start turn-on rises to the
  carrier-lifetime scale (~2–3 ns) with large jitter (Coldren & Corzine).
- Excitability itself needs it: "the continuous pump current to the gain section is
  essential for maintaining the excitable regime" (Opt. Lett. 36, 4476; two-section InP
  neuron literature).
- The smoking gun: a fabricated DFB laser-neuron reports **130 pJ/spike at 230 MHz →
  67 pJ/spike at 730 MHz** (arXiv:2012.08516) — per-spike energy *falls as rate rises*,
  the unmistakable fingerprint of a continuous standing bias amortised over spikes, i.e.
  time-scaling. Encoding this (gating removes <20% of the near-threshold bias) makes the
  **gated regime numerically identical to continuous**. Gating fully off (gain-switching)
  is possible only to a few GHz and reintroduces standing power via the stabilising
  bias/seed. The only escape is architectural — nanolaser (nW–µW threshold),
  event-driven optoelectronic neurons, or <1 laser/neuron sharing — none fabricated as a
  compute array.

**Task 3 — sparsity sweep (sourced: CIFAR-10 SNNs fire 5–20%/step at T=4–10, 92–94% acc;
arXiv:2409.08290, DIET-SNN 2008.03658; cortex <1% but far sparser than any trained SNN).**
Across 0.5–20% activation the **source term dominates** the per-eqMAC breakdown at every
usable sparsity (it only yields to per-spike generation above ~20%). Sparsity moves the
total by <2×, never by the order of magnitude the floor requires.

**Task 4 — single-readout loss wall.** Detecting only at the final layer works: with PCM
weights at 0.1–1 dB/element plus routing, **13–20 layers** can propagate optically before
the signal falls below the 33 dB budget — so readout elimination is real and not tightly
bounded. But it removes comparator cost, not the source, so it does not change the verdict.

**Honest nuance (flagged, not buried).** At the *realistic low-latency* regime (T≈6, not
the rate-coded T≈2500), the source term `P_bias·T/(N·bw)` shrinks, so a **low-threshold
VCSEL** neuron (~2 mW, fabricated) with 1/N amortisation at N≳30–100 does cross the floor
in ~65% of draws. This is **not a sparsity effect** (gating failed; gated ≡ continuous) —
it is the *same low-threshold-laser + amortisation corner* the spiking track already
identified, enlarged only by using realistic low-latency timesteps. And it rests on three
accountings generous to optics: (i) energy is charged per *equivalent dense MAC* while the
SNN does only `s·N²` sparse SOPs — a fair digital baseline could exploit the same sparsity;
(ii) the synaptic weight-mesh insertion loss is abstracted, not charged per element as it
was for the crossbars; (iii) it assumes matched accuracy, which for such an SNN is
unproven. Per the pre-registered gate — the source still dominates and the gated escape is
physically closed — **the accuracy study is moot and was not run**; the residual VCSEL
corner is reported as the narrow, device-limited, generously-accounted regime it is, the
same way the PCM 13% was flagged.

**Verdict.** The specific escape — biological sparsity via a gated source — is **closed on
physics**: the source cannot be gated, so sparsity buys nothing and the source still
dominates. Every physically-motivated escape (topology, encoding, non-volatile weights,
mode-multiplexing, sparsity) has now been tried, and the cost is conserved.

---

# Follow-up 6 — Shared source: stop giving each neuron its own gain medium

**One-line answer: sharing removes the per-neuron *laser* but not the standing
*optical-power* floor — it relocates and divides the bias by the share factor, but the
share factor is capped at ~16 by the passive elements' bus-loss wall, and the fabricated
per-spike energy stays ~60–125× the compute floor.** The apparent sub-floor per-MAC that
sharing produces is an **equivalent-dense-MAC / fan-in accounting** effect that a *digital*
sparse-SNN accelerator enjoys equally — not a photonics-specific win. Code:
`src/energy_model/run_test1_spiking_shared.py`, data `data/test1_spiking_shared_results.json`.

The escape is real and correctly aimed: the source cost was `N·P_bias` only because each
neuron carried its own gain medium. One shared comb feeding N passive modulators divides
the standing laser hardware and bias overhead by N. But the model, reconciled against the
one architecture that costs this end-to-end (**SEPhIA**, Hejda et al., arXiv:2510.07427),
shows the cost reappearing on exactly the two axes flagged in advance:

- **Bus-loss wall caps the share factor.** SEPhIA's own power law is `2·N·IL_MRM +
  10·log(N)` dB; at 0.2 dB/modulator this caps one shared tile at **N_T ≤ 16 neurons**
  before an amplifier is needed (and that excludes propagation loss and crosstalk, so 16
  is optimistic). Our share-factor sweep at N=1024 reproduces it: S=8 → ~7 fJ/eqMAC (bus
  6 dB); **S=256 → 105 dB, undetectable.** So you get tiles of ~16, not one laser for all.
- **The standing optical drive does not shrink.** SEPhIA's per-neuron energy is **99.8%
  optical drive power** (P_λ ≈ 2.5 mW vs electronic P_E ≈ 4.6 µW). The passive element
  does not make its own light; the µW–mW it modulates must be supplied continuously by the
  shared source, *per element*, independent of N. Fabricated: **2.5 pJ/spike** (60–125×
  the 20–40 fJ floor); idealized minimum (−14 dBm, losses excluded): **44 fJ/spike**.

**Why the model still shows floor-crossings (24–86% of draws), and why they don't count.**
Per *equivalent dense MAC*, one spike is amortised over its fan-in (N synaptic ops), so
2.5 pJ/spike ÷ N=1024 ≈ 2.4 fJ — below the floor. That is the entire "win," and it is the
standard SNN sparsity/fan-in accounting (spikes-per-neuron × fan-out), which is
**contested and, crucially, equally available to a digital sparse-SNN accelerator** — so
it is not a win *for photonics over digital*. Under the honest fabricated per-event metric
(SEPhIA 2.5 pJ/spike) the source still dominates and the floor is missed by ~2 orders of
magnitude, exactly as for every other track. The floor-crossing fraction also depends on
allowing the idealized 44 µW drive; at the realistic 2.5 mW it collapses.

**Accuracy (sourced, not run).** At the sparsity this needs (5–20% firing, T=4–10), trained
CIFAR-10 SNNs reach 92–94% — a 1–2 pt penalty vs the ANN (DIET-SNN arXiv:2008.03658;
arXiv:2205.07473). Usable, not free; and the digital SNN it is compared against gets the
same sparsity benefit.

### Sourced inputs (shared-source)

| Quantity | Low – Nom – High | Source |
|---|---|---|
| Neurons per shared source | 8 – 16 – 64 | SEPhIA Op-Tile N_T≤16, power-limited (arXiv:2510.07427) |
| Optical drive per element | 44 µW – 0.5 – 2.5 mW | SEPhIA P_λ 2.5 mW real / 44 µW ideal; µring self-pulse 80–223 µW |
| Per-element bus IL | 0.2 – 0.4 – 1.0 dB | Si MRM 0.2 dB (SEPhIA); Sb₂Se₃ 0.4–0.65; GST several |
| Photons per 1-bit spike | 1 – 20 – 100 | quantum limit ~1 (Ma et al. Nat.Commun.2023); reliable ~10–20 |
| Fabricated energy/spike | 44 fJ – 2.5 pJ | SEPhIA idealized min / realistic |

**Verdict.** The shared-source escape is the best-aimed of the programme — it attacks the
one term (`N·P_bias`) that sharing can legitimately divide — and it is why photonic SNNs
report their best numbers. But it does **not** produce a general, fabricated,
photonics-specific win: sharing is bus-loss-capped at ~16, the optical drive floor is
untouched (99.8% of SEPhIA's energy), the fabricated 2.5 pJ/spike is ~60–125× the floor,
and the only sub-floor per-MAC comes from fan-in/sparsity accounting a digital SNN shares.
Reported as a narrow, generously-accounted corner — the same flag as the PCM 13%.

---

# Follow-up 7 — Inverting the model: the spec sheet for an optical nonlinearity that would reopen photonic compute

**One line: an all-optical net would need a nonlinear element that switches at ≲ 1 fJ per
activation (sub-fJ for useful depth/width), with ≤ ~1 dB loss, at GHz–THz speed and room
temperature — and the nearest *fabricated, fast, room-temperature* material is ~10–100×
away on energy, while the *direct Kerr/χ³ route is Kramers-Kronig-forbidden at 1550 nm by
6–8 orders of magnitude. The only escape that survives causality is the matter-mediated
one — exciton-polaritons. Room-temperature operation and a single-photon-*triggered*
nonlinearity are both now demonstrated (Corrections, below), so the remaining gap is three
specific unmet requirements: a cascadable, pump-free (blockade, not pump-powered
stimulation) nonlinearity, CMOS-integrable at telecom wavelength and ≤1 dB loss.**

Six escapes were tested and the per-element electrical/optical cost was conserved every
time. The surviving diagnosis is that light must become electricity at every layer, because
the activation is applied electrically — so the last lever is a low-power optical
nonlinearity that removes the per-layer tax. This section runs the model *backward*: it
solves for what that element must be, then measures the gap to every published candidate.
Code: `src/energy_model/run_test1_alloptical.py`, data `data/test1_alloptical_results.json`.
**Scope, stated plainly: this — like every crossover in the programme — is inference-only
(a network that never measures intermediate activations cannot produce gradients), and
bounded to shallow depth (below).**

## Task 1 — the architecture and the cascading gate

Convert once at the input, once at the output; between them, per layer, a passive PCM
linear stage (~0 hold, sourced loss) and a nonlinear element per neuron whose properties
are the free unknowns; one input laser. Because conversion is now amortised over `L·N²`
MACs it becomes negligible — so the binding term is the nonlinear element's **operating
power** `P_op = E_nl·bw`, which the single input laser must hold at every one of N neurons
through the accumulated per-layer loss `t^{-L}`, within a bounded on-chip input power.

**Cascading gate (checked first, per pre-registration): passes only at shallow depth.**
With sourced passive-layer loss (~2 dB/layer nominal), the signal stays inside the 33 dB
link budget for **L ≤ ~8 layers**; at **L = 16 the total loss is 35–67 dB → inter-layer
optical amplification is unavoidable, and that is a source cost by another name → the
escape fails at depth.** So the architecture is real but **depth-bounded to ~8 layers**;
this is not a full stop, but it is a hard scope limit and it is stated as one.

## Task 2 — the required switching-energy spec (solved backward)

The spec is a surface, not a number; the binding constraint is **always the power/cascading
limit**, never energy-amortisation. Max tolerable switching energy per activation:

| | N=64 | N=256 | N=1024 |
|---|---|---|---|
| **L=4** | **7 fJ** (weakest) | 1.2 fJ | 0.29 fJ |
| **L=8** | 0.47 fJ | 0.12 fJ | **0.03 fJ** (useful, deep+wide) |
| L=16 | — amplification needed (gate fails) — | | |

Monte-Carlo over the sourced ranges: a nonlinearity at **1 fJ/activation clears the floor
in 85% of draws; 10 fJ in only 40%; 100 fJ in 5%.** So the **robust target is ≲ 1 fJ**,
the *weakest sufficient* corner is ~7 fJ (a small N=64, L=4, 4-bit net), and a *useful*
deep+wide net needs **~0.03–0.1 fJ (30–100 aJ)**.

**Required material n₂** (Kerr π-shift, A_eff≈0.1 µm², L≈100 µm, τ≈100 ps): **1.1×10⁻¹¹
m²/W** (weakest) to **7.8×10⁻¹⁰ m²/W** (useful) — for reference, chalcogenide As₂S₃ is
~3×10⁻¹⁸, ITO-ENZ ~10⁻¹⁴, graphene ~10⁻¹³.

## Task 3 — the gap to every published candidate (ranked by smallest max-gap)

**Lead (revised on review): room-temperature operation and a single-photon-*triggered*
nonlinearity are both now demonstrated, so the remaining gap is neither of those — it is
(1) cascadability without a per-stage optical pump, (2) a pump-free blockade rather than a
pump-powered stimulation mechanism, and (3) CMOS/photonic integration at telecom wavelength
and ≤1 dB loss.** The energy metric that matters is *per operation including the pump*, not
the single-photon seed.

Spec to beat: **≤1 fJ/op (incl. pump), ≤1 dB loss, ≥GHz speed, room temperature, cascadable.**

| Rank | Candidate (mechanism) | Energy/op (incl. pump) | Speed | Loss | Temp | Gap to spec | Source |
|---|---|---|---|---|---|---|---|
| 1 | **RT organic polariton, single-photon-*triggered*** (bosonic stimulation, MeLPPP) | seed ~1 photon (sub-aJ) **but ~300 pJ pump/op** | not measured (150 fs pulses, 500 Hz) | not reported | **RT (real)** | RT+single-photon-control **solved**; **energy ~10⁴–10⁷× (pump), speed/loss unreported, not cascaded** | Zasedatelev, Nature 597, 493 (2021) |
| 2 | **QD-in-cavity polariton blockade** (cavity-QED) | **14 aJ** | 8.4 GHz | low | **~39 K (cryo)** | meets energy+speed; **temperature** (but RT polariton strong-coupling now shown separately, below) | Sridharan/Waks, Opt. Express 19, 5551 (2011) |
| 3 | **GaAs 0D polariton blockade** | **~0.6 aJ** | ps | — | **<10 K (cryo)** | energy far under; **temperature + U/Γ≈0.42 (blockade not fully reached)** | Delteil, Nat. Mater. 18, 219 (2019) |
| — | *RT polariton strong-coupling / condensation (context)* | — | — | — | **RT (real)** | proves RT strong coupling is solved (not itself a switch): CsPbBr₃ QD condensation; ~840 plasmonic single-QD (electrical); topological CsPbCl₃ | Nat. Commun. 2025 s41467-025-60553-3; 2024 s41467-024-51170-7; Nat. Nanotechnol. 19, 1283 (2024) |
| 4 | **χ²-cascade LiNbO₃** | 80 fJ | **46 fs** | low | RT | fast+RT+low-loss; **~80× energy** (best conventional-material candidate) | Guo/Marandi, Nat. Photon. 16, 625 (2022) |
| 5 | **Plasmonic graphene** | 35 fJ | 260 fs | **dB/µm (fails)** | RT | ~35× energy; **loss-disqualified** | all-optical plasmonic switch, 2021 |
| 6 | **Chalcogenide As₂S₃** | **pJ–nJ** | fs Kerr | 0.05 dB/cm | RT | FOM_T≫2 (clears FOM) but **10³–10⁶× energy** — *the proof FOM isn't binding, energy is* | Lamont, Opt. Express 2007 |
| 7 | **Free-carrier Si (all-optical, Nozaki)** | 0.42 fJ | ~10–30 GHz (**carrier recovery**) | low | RT | energy+RT meet but **recombination-speed-capped, resonant** — and the fast fJ/bit p-n devices are **electrical modulators, not all-optical** (excluded) | Nozaki, Nat. Photon. 4, 477 (2010); (cf. Timurdogan, Nat. Commun. 5, 5008 (2014), *electrical*) |
| 8 | **ITO-ENZ** | GW/cm² intensities | 360 fs | **α~10⁴ cm⁻¹ (fails)** | RT | n₂ and loss peak together; **loss-disqualified, no device** | Alam/Boyd, Science 352, 795 (2016) |
| — | **BIC / slow-light** | Q- or n_g²-enhanced | **capped by Q/n_g lifetime** | — | RT | enhancement ∝ speed⁻¹; **speed-disqualified** | Koshelev; Corcoran, Nat. Photon. 2009 |
| — | **Rydberg-EIT** | 0.25 aJ (single-photon) | µs (slow light) | — | cold atoms | strongest known, **but not a chip: UHV+MOT, ~10¹⁰× overhead** | Peyronel, Nature 488, 57 (2012) |
| — | **PCM-as-Kerr / thermal / optomech.** | — | µs; g₀/κ≈10⁻³ | — | — | **speed-disqualified** (structural/thermal/mechanical) | Delaney 2020; Aspelmeyer RMP 2014 |

## Task 4 — the Kramers-Kronig verdict (the decider)

n₂ and two-photon absorption β are the real and imaginary parts of the same causal
susceptibility (Sheik-Bahae, IEEE JQE 26, 760 (1990); two-band model, Hutchings 1992).
TPA is **identically zero only below half the bandgap** — at 1550 nm (ħω=0.80 eV) that
requires **E_g > 1.60 eV** — and n₂ then scales as **E_g⁻⁴**, capping the TPA-free n₂ at
the **As₂Se₃/As₂S₃-class ceiling ~1×10⁻¹⁷ m²/W**.

**The required n₂ (1.1×10⁻¹¹ … 7.8×10⁻¹⁰) is 6–8 orders of magnitude above that
causality ceiling.** Every route to bridge it fails on a *different* axis:
- **Resonant enhancement** would need Q ≈ 10⁶–10⁷ to make up the n₂ deficit → cavity
  lifetime τ = Qλ/2πc ≈ 1–58 ns → **sub-GHz to few-GHz, too slow.**
- **High-n₂ materials** (ITO-ENZ 10⁻¹⁴, graphene 10⁻¹³) have E_g < 1.6 eV or are metallic →
  TPA turns on or linear loss reaches α~10⁴ cm⁻¹ → **≤1 dB budget violated by orders.**

So for a **Kerr/χ³ material the spec is causality-forbidden at 1550 nm**: you cannot have
the required n₂, ≤1 dB loss, and GHz speed simultaneously — it is a hard n₂–loss–speed
trilemma set by Kramers-Kronig, not an engineering gap. **The only escape is a nonlinearity
that is *not* bound-electron χ³** — exciton-polaritons, where the interaction comes from
real exciton-exciton scattering (the matter half). That bypasses the KK bound, and it is
why polaritons rank 1–3 above. But the strong polariton nonlinearities are **cryogenic**
(GaAs, QD-cavity, <40 K), and the one **room-temperature** case (perovskite, ~6 fJ) sits at
the weakest-spec edge, is saturation- rather than blockade-based, and **has never been built
as a cascadable multilayer network.**

## Is room-temperature blockade forbidden, or just unbuilt? (the one open door)

The table sets up a question it does not answer. Row 2 — QD-in-cavity blockade — **already
meets energy (14 aJ) and speed (8.4 GHz); its only failing axis is the 39 K temperature.**
And the context row shows **room-temperature polariton strong coupling is now demonstrated
in three separate systems.** So the live question is whether the two can be combined —
**strong blockade at room temperature** — and whether that is forbidden by physics or is a
materials-integration problem.

The blockade condition is **U/Γ > 1**, where U is the single-quantum anharmonicity (the
energy to add a second polariton to an already-occupied mode) and Γ is the linewidth. The
two respond to temperature oppositely, and that is what decides the question:
- **U is essentially athermal** — set by exciton binding, oscillator strength, and mode
  volume (U ∝ 1/V for a single emitter), not by T.
- **Γ has an irreducible phonon-dephasing floor that rises monotonically with T:**
  Γ(T) = Γ_rad + Γ_ac·T (acoustic phonons) + Γ_LO/(e^{ħω_LO/kT}−1) (Fröhlich/LO-phonon
  dephasing). Between 40 K and 300 K a single quantum-emitter homogeneous line broadens from
  ~µeV to meV–tens of meV (kT ≈ 25 meV at RT).

So **U/Γ falls with temperature — but there is no thermodynamic no-go theorem here, no hard
bound like the Kramers-Kronig one.** RT blockade is not forbidden; it reduces to a
**materials figure of merit, U > Γ_phonon(300 K)** — a single (or few) emitter whose
intrinsic anharmonicity beats its own room-temperature phonon-broadened linewidth. Nothing
in physics forbids such a material; none is yet shown.

**But it is not "just integration" either, and this is the tension the RT strong-coupling
demonstrations hide.** Strong coupling needs **Ω > Γ**, and the vacuum Rabi splitting
Ω = √N·g is **collectively enhanced** — large N makes Ω hundreds of meV in organics and
perovskites, which is exactly how it beats RT phonon broadening. Blockade needs **U > Γ**,
and U is the *per-particle* interaction, **suppressed as 1/N** by the same collectivity. The
ensemble trick that delivers RT strong coupling therefore actively destroys the RT
nonlinearity: the demonstrated RT systems live in the large-N (big Ω, tiny U) limit, and
blockade lives in the N→1 (big U, no collective Ω) limit. The nearest RT single-emitter case
— a single molecule in a plasmonic nanogap (Chikkaraddy et al., Nature 535, 127 (2016)) —
reaches Ω > Γ but has **not** shown U > Γ; a room-temperature *single-emitter photon
blockade* has never been demonstrated.

**What the measured data says (RT gap unquantified, a continuous FOM, not a no-go).**
Both sides of U/Γ have been measured, and they bracket the door quantitatively:
- **Γ_phonon(T) is measured and rises ~10²–10³× from cryo to RT** in every candidate
  RT-exciton material. Monolayer TMDs: residual homogeneous linewidth ~1.6 meV (T₂≈0.4 ps)
  extrapolated to zero temperature/density, broadening to ~5–15 meV at 300 K via acoustic +
  LO-phonon dephasing (Moody et al., Nat. Commun. 6, 8315 (2015); Selig et al., Nat. Commun.
  7, 13279 (2016)). Halide perovskites are worse: a single LO mode at ~11–20 meV with Fröhlich
  coupling ~40–70 meV drives the RT homogeneous linewidth to tens of meV (Wright et al., Nat.
  Commun. 7, 11755 (2016); "Fröhlich interaction dominated by a single phonon mode in CsPbBr₃,"
  Nat. Commun. 12, s41467-021-26192-0 (2021)). GaAs — the material of the best blockade demos —
  is **not** on this list: its ~few-meV exciton binding lets the exciton ionize above cryo,
  which is *why* those demos stop below 40 K.
- **The directly-measured antibunching is weak, and even the cold interaction ratio is below
  threshold.** Delteil et al. (Nat. Mater. 18, 219 (2019); arXiv:1805.04020) confined 0D GaAs
  polaritons to ~3 µm² and observed a **~5% two-polariton suppression — i.e. g²(0) ≈ 0.95**, an
  order of magnitude from blockade (g²(0) < 0.5). The often-quoted "**U/Γ ≈ 0.42**" is an
  *inferred* interaction/linewidth ratio from the analysis, **not the observed antibunching**;
  in a consistent weak-drive convention the measured g²(0) ≈ 0.95 corresponds to **U/Γ ≈ 0.05**,
  so the honest cold anchor is **~10–14× short of threshold even at few K.** The TMD-nanocavity
  route is orders further: MoSe₂ switches at ~4 fJ (N~10⁴ photons), Γ_LP≈1.8 meV at 4 K,
  "several orders of magnitude" from the single-polariton regime (arXiv:2411.16635).

Put together: the best *measured* cold antibunching is already **~14× short** of blockade
(g²(0)=0.95 ↔ U/Γ≈0.05, threshold ≈0.71). A Hopfield-ceiling estimate at RT (|X|→1) gives a
**~12× floor for a TMD (Γ_x≈10 meV) and ~47× for a perovskite (Γ_x≈40 meV)** — **but these are
*not* in-material numbers**: they carry a **generous GaAs-class saturation U_exc onto a
TMD/perovskite RT linewidth** (a transfer, since no TMD-measured single-mode U_exc is used), so
they are *illustrative floors*, not the achieved gap. **The honest RT gap is unquantified** —
these floors, and the two structural penalties below (Hopfield, oscillator strength), all cut the
same way. What the data establishes is the *shape*, not a number: this is
a **continuous figure of merit, not a measured no-go.** The active levers (Rydberg / dipolar /
interlayer excitons and Fermi-polarons to *raise* U; large-binding, low-Fröhlich TMD-class
materials whose RT Γ is ~10 meV to *lower* the target) push the right way, but — as the next
two paragraphs show — the polariton-basis weighting and the oscillator-strength cost mean the
usable RT gap is **not reliably quantified on the upside; the earlier "~25×" was an artifact
and is withdrawn.**

**Is there a joint bound tying U's ceiling to Γ's floor? (the closest thing to a no-go — and
the escape).** This is the question that would *close* the programme if answered yes, and the
recent theory gets partway there. The interaction that **dominates every measured polariton
system is oscillator-strength saturation**, not exciton–exciton scattering ("Excitonic
oscillator-strength saturation dominates polariton-polariton interactions," Phys. Rev.
Research (2025), arXiv:2501.07899: ~96% saturation-dominated, measured nonlinear scale
E_nl ≈ 50–300 µeV, GaAs open-cavity FOM **U/γ ≈ 0.15**). That mechanism scales as
**U_sat ∝ a_B²** (saturation density n_sat ∝ 1/a_B²), so it pulls in *exactly the opposite
direction* to room-temperature stability, which demands a **large exciton binding energy →
small Bohr radius a_B**. GaAs (a_B ≈ 10 nm, binding ≈ 4 meV) has the large U but ionizes above
cryo; a TMD (a_B ≈ 1 nm, binding ≈ 0.5 eV) survives RT but its *saturation* U is intrinsically
~10²× weaker by the a_B² scaling. **This is a genuine physical tension — a *soft* joint bound,
not a proven no-go: the same parameter (a_B) that buys thermal robustness suppresses the
dominant nonlinearity.** The door is not shut because the tension binds only for
*saturation*-sourced U; sourcing U instead from a **static dipole** decouples it from a_B, and
that is the fastest-moving lever — interlayer/dipolar excitons already show a **~10×
nonlinearity boost** over intralayer A excitons ("Highly nonlinear dipolar exciton-polaritons
in bilayer MoS₂," 2022), with Rydberg-exciton and Fermi-polaron routes adding more. So the
honest statement is: **no one has proven the joint bound, and the one theory that gets closest
(saturation ∝ a_B²) is precisely what the dipolar/Rydberg mechanisms are engineered to
bypass.** Proving a floor on Γ_phonon(300 K) *jointly* with a ceiling on the *dipole-enhanced*
U — not just the saturation U — is the single result that would convert this open door into a
closed one.

**A modelling correction, and why the model was retired.** A first pass here ran a driven-
dissipative Kerr g²(0) model with a Monte-Carlo over U and Γ ranges. **It has been removed from
the repository.** The reason is decisive: **g²(0) is a monotone function of U/Γ, so propagating
input ranges through it returns the inputs.** "0% of 4,000 draws reach RT blockade" restates "the
best corner is short" — the same fact printed twice. It had *no outcome that would have changed
the verdict*; it was illustration, not a test. And the criterion that *does* matter (below) is a
two-number inequality checkable by hand — no Lindblad solve is required to evaluate it — so
keeping code that only restates an inequality would invite the same illustration-as-evidence
failure later. What survives is the following, all verifiable without the model. Three errors in
that first pass, all optimistic, now corrected:
1. **Circular calibration.** It fed U/Γ = 0.42 into its own g²(0), read off 0.66, and called
   that agreement with Delteil. Delteil *measured* g²(0) ≈ 0.95; in a consistent convention that
   is U/Γ ≈ 0.05. The pipeline was never validated against the paper.
2. **Convention-dependent threshold.** g²(0) < 0.5 needs U/Γ ≈ 0.71 in the master-equation
   (population-decay = energy-FWHM) convention used here, but ≈ 0.5 in the common closed form
   g²(0)=1/(1+(2U/Γ)²) — a factor ~√2 that shifts every ratio. (The Lindblad solve and the
   analytic weak-drive ladder agree *exactly* within a convention; the number is meaningful only
   with its convention stated and only comparable to a measurement quoted the same way.)
3. **The "25×" was a cross-material composite doing all the work.** It multiplied a **200×
   dipolariton enhancement measured in cryogenic GaAs coupled quantum wells (PRL 121, 227402
   (2018))** onto a **TMD room-temperature linewidth** — a system no one has grounds to expect.
   Remove the composite and **no in-material number remains**: the RT gap is *unquantified*, and
   the ~12× floor quoted below is *illustrative* — it rests on a generous GaAs-class U_exc, not a
   TMD-measured one (see the trace below). **The 25× is withdrawn.**

**Two pieces of physics the first model omitted — both cut against optics:**
- **Hopfield weighting.** In the polariton basis U scales as **|X|⁴U_exc** and Γ as
  **|X|²Γ_x + |C|²Γ_c**, so the ratio U/Γ → |X|²·U_exc/Γ_x (good cavity) is **maximised at
  |X|→1, the bare-exciton limit.** The cavity buys coupling and readout, *not* a better ratio —
  so the earlier "add Purcell narrowing" idea was backwards: going more photonic lowers U_pol
  faster than Γ_pol. Any g²(0) model without Hopfield weights quotes an exciton ratio and calls
  it a polariton one. The real RT spec is therefore **U_exc > ~0.7·Γ_x(300 K)** — a *bare-
  exciton* anharmonicity (µeV-scale for saturation) exceeding a ~10 meV RT homogeneous linewidth.
- **Oscillator-strength cost.** The dipolar/interlayer routes raise U precisely by separating
  electron and hole — which suppresses oscillator strength, hence g, hence the achievable |X|
  (or strong coupling at all). The 200× on U is not free; its cost lands on exactly the |X| the
  Hopfield weight then rewards. Until that trade is carried, the enhanced-U corner is optimistic
  by an unknown factor.

**Falsification criterion — a hand-checkable inequality, not a model.** The verdict "RT blockade
unreachable with known materials" flips **iff a single material is measured with
U_exc > ~0.7·Γ_x(300 K), both quantities in that material, at an exciton fraction |X|² high
enough to strong-couple and read out** — a bare-exciton anharmonicity exceeding its own RT
homogeneous linewidth. This is a comparison between two measured numbers; nothing computes it but
arithmetic, which is exactly why the g²(0) model was retired. It now lives as one line in the
spec sheet. The **U/Γ ≳ 0.71** threshold (weak-drive Kerr, FWHM convention; ≈0.5 in the
1/(1+(2U/Γ)²) convention) is a one-time quantum-optics fact, quoted, not re-simulated.

**Blockade imports its own ledger — the spec sheet must price it:**
- **Shot noise sets the real speed limit, not Γ.** Operating at single-photon amplitudes sits at
  the shot-noise floor, so resolving a signal costs integration time or repetition (SNR ∝ √N),
  and the operation rate is then set by the **readout electronics and that integration time** — a
  direct tax on effective throughput that a per-MAC-energy comparison must carry. *(Errata: an
  earlier draft claimed a Γ/ħ "bandwidth vs blockade" tension and misquoted it as ~1.5 GHz; the
  arithmetic is ~240 GHz at Γ=1 meV. Both numbers sit far above the actual readout/integration
  limit, so Γ/ħ is **not** the binding speed constraint in either direction — the correction is a
  fixed error, not a real tension. The speed constraint lives in the shot-noise/readout line
  above.)*

**Verdict: necessary, not sufficient — and unquantified on the upside.** RT single-emitter
blockade is not causality- or thermodynamically forbidden (no one has proven the joint bound),
so this axis stays *open*. But the honest state is soberer than the first pass implied: the best
*measured* cold antibunching is ~14× short; an RT Hopfield-ceiling *estimate* is ~12–47× short
but is a **transfer, not an in-material measurement** (generous GaAs-class U_exc); and the two
structural penalties (Hopfield, oscillator strength) cut the same way — so the **RT gap is
unquantified**, not a tidy ×N. More decisively, **clearing it
would not reopen the programme.** Grant U/Γ = 2 at room temperature tomorrow and not one
conserved electrical cost — converter amortisation, thermal hold, laser bias, the 99.8%
optical-drive share — moves; blockade is a *necessary* enabler of the one all-optical escape,
never a *sufficient* one, and it arrives with its own bandwidth-vs-Γ and shot-noise costs. The
materials question is real and worth watching. **It is not the deliverable — the joint condition
below is.**

## The joint condition — all seven at once (the deliverable that survives)

The requirements below are a **conjunction, not a menu**: the nonlinearity must satisfy *all* of
them *simultaneously, in one device*, and several are **mutually antagonistic**, so progress on
one row can regress another. That structure — not any single ×N gap — is what survives review.

| # | Requirement | Best demonstrated | In tension with |
|---|---|---|---|
| 1 | Switching energy ≤ 1 fJ/op **incl. pump** | 14 aJ (cryo blockade); RT case pump-dominated ~300 pJ | #7 (single-photon ⇒ shot-noise) |
| 2 | Insertion loss ≤ ~1 dB/element | **unreported** for every polariton candidate | #6/#7 (dipolar-U ↓ oscillator strength) |
| 3 | Speed ≥ GHz (ideally THz) | 8.4 GHz (cryo QD-cavity) | #1/#7 (shot-noise integration time) |
| 4 | Depth ≤ ~8, **cascadable, no per-stage pump**, inference-only | not shown (RT case re-pumps each stage) | #1 (a pump *is* the energy) |
| 5 | Temperature = RT | RT strong coupling ✓ (**not** RT blockade) | #7 (Γ_phonon(300 K) ⇒ larger U needed) |
| 6 | Causality: non-χ³ (polariton) | polariton mechanism ✓ | #2 (matter fraction ⇒ loss) |
| 7 | Blockade quality U/Γ above threshold at RT | **U/Γ ≈ 0.05 measured** (cryo) | **#1, #2, #3** (shot-noise, oscillator strength) |

**No known mechanism satisfies the conjunction, and three pairs are physically opposed:**
- **#7 ⇄ #1 and #3 (blockade vs energy-accuracy *and* speed).** Single-photon operation sits at
  the shot-noise floor, so resolving a result costs integration time or repetition (SNR ∝ √N) —
  taxing both accuracy (#1) and the achievable operation rate (#3). *This* — the readout /
  integration budget, not the exciton linewidth — is the real speed limit.
- **#6/#7 ⇄ #2 (dipolar-U vs loss and coupling).** Raising U by separating electron and hole
  suppresses oscillator strength → higher loss (#2) and weaker coupling → lower |X| (which the
  Hopfield weight then penalises).
- **#4 ⇄ #1 (cascadability vs energy).** The only demonstrated RT nonlinearity restores signal
  with a per-stage pump — the inter-layer amplification the energy budget forbids.

**So the reusable result is the conjunction and its contradictions, not a single "gap ×N."** A
lab can advance one row; the deliverable is that **advancing all seven at once, in one device,
has no known path, and at least three of the rows trade against each other** — achieving RT
blockade (row 7) actively *worsens* rows 1–3. This statement is independent of the withdrawn
blockade estimate. And it is **necessary-but-not-sufficient for the programme**: every row here
concerns the *nonlinearity only*. The conserved per-element *electrical* costs — conversion,
thermal hold, laser bias, the 99.8% optical-drive share, the load-bearing converter-floor result
— sit entirely *outside* this table and are untouched by any nonlinearity, RT blockade included.

## The spec sheet (the reusable output)

> **Requirements for an optical nonlinearity that reopens photonic compute**
> - **Switching energy:** ≤ 1 fJ/activation for robust (85% of draws); ~7 fJ in the easiest
>   corner (N=64, L=4, 4-bit); **30–100 aJ for a useful deep+wide net.**
> - **Insertion loss:** ≤ ~1 dB per element.
> - **Speed:** ≥ GHz (ideally THz); must not be Q- or carrier-lifetime-limited.
> - **Depth:** ≤ ~8 all-optical layers before inter-layer amplification (a source cost)
>   reappears; **inference only.**
> - **Temperature:** room temperature for a practical accelerator.
> - **Causality:** for a Kerr/χ³ material this is **forbidden at 1550 nm** (required n₂ is
>   6–8 orders above the TPA-free ceiling). A non-χ³ mechanism (polariton/matter interaction)
>   is required.
> - **Blockade quality (the pass/fail line, checkable by hand):** the nonlinearity must reach
>   **U_exc > ~0.71·Γ_x(300 K)** — a *bare-exciton* anharmonicity exceeding its own *room-
>   temperature* homogeneous linewidth, *both measured in the same material*, at an exciton
>   fraction |X|² high enough to strong-couple and read out. Best measured to date: U/Γ ≈ 0.05
>   (cryo). The RT gap is **unquantified** (see *Corrections*): the Hopfield weighting caps the
>   ratio at the bare-exciton value, and the dipolar enhancement that might close it trades
>   against oscillator strength by an unmeasured amount. **This is the single watch-condition;
>   it needs no simulation — only two numbers from one material.**
> - **The nearest mechanism, and the actual remaining gap** (revised on review — see
>   *Corrections* below). Two facts are settled — but note they were demonstrated in *different*
>   systems and via a *different* mechanism than RT blockade, so they remove two objections
>   without composing into a working device: RT strong coupling exists, and a RT single-photon-
>   *triggered* nonlinearity exists (which is evidence against a *general* no-go for RT
>   single-photon nonlinearity — **not** an existence proof of RT blockade, which remains
>   unshown). With those two objections removed, the gap is:
>   - *Room-temperature strong coupling is solved.* RT cavity polariton condensation in
>     colloidal CsPbBr₃ QDs (Nat. Commun. 2025, s41467-025-60553-3, threshold ~160 µJ/cm²,
>     first for any QD platform); RT single-QD strong coupling reproduced across ~840
>     plasmonic nanocavities with **electrical** injection (Nat. Commun. 2024,
>     s41467-024-51170-7, Ω/ω₀≈0.2); RT topological CsPbCl₃ valley-Hall condensation
>     (Nat. Nanotechnol. 19, 1283 (2024)). **Strike "temperature" from the gap.**
>   - *A room-temperature single-photon-triggered nonlinearity exists* — Zasedatelev et al.,
>     *Nature* 597, 493 (2021), MeLPPP organic polariton, ~490 nm, ambient. But scored against
>     this spec it does **not** clear it: the single photon is the **control/seed** (~1–2.5
>     photons, sub-aJ), while each operation is powered by a **separate ~300 pJ optical pump**
>     (80 µJ/cm² ≈ 2·P_th) — so per-operation energy is **~10⁴–10⁷× the spec**, not sub-fJ.
>     Contrast at one photon is ~11% (single-shot); **switching time is not measured**
>     (150 fs pulse widths, 500 Hz laser rate; "sub-picosecond" is cited, not shown);
>     **insertion loss is not reported.** *(Parameters flagged as unreported, per rule 9.)*
>   - **So the remaining gap is three specific problems, not temperature or existence:**
>     1. **Cascadability.** The 2021 element is a single localized condensate whose output must
>        be re-pumped at ~80 µJ/cm² at every stage. That per-stage pump **is** the inter-layer
>        amplification the Task-1 gate forbids — so as a network element it trips the
>        pre-registered gate and the escape closes **unless the pump is shared/eliminated**.
>        Cascadability is asserted only by a cited companion (arXiv:2005.04802), not shown.
>     2. **Blockade vs stimulation.** The 2021 mechanism is **bosonic stimulation**
>        (pump-powered amplification: threshold + saturating gain, ~23,000× polariton gain),
>        not **blockade**. Stimulation gives a monotonic activation-like curve but needs the
>        pump and carries large shot-to-shot condensate-population noise; the clean, pump-free,
>        low-energy version is **blockade**, which every source calls "notoriously elusive"
>        and which is the same open problem for the QD route.
>     3. **Integration and loss.** Organic microcavities (~490 nm), colloidal-QD films, and
>        plasmonic nanocavities are not CMOS/photonic-process-compatible, operate in the
>        **visible** (not 1550 nm), and the plasmonic route carries **metal loss**; per-element
>        insertion loss is unreported for all of them and must be scored against the ≤1 dB max.
>   - *Not a candidate for this spec:* **free-carrier silicon modulators** (e.g. Timurdogan
>     et al., Nat. Commun. 5, 5008 (2014): 1 fJ/bit, 25 Gb/s, depletion-mode reverse-biased
>     p-n, 250 pm/V — fast because depletion sweep-out avoids the recombination cap of
>     forward-biased injection). These are **electrically driven modulators** — light switched
>     by *electronics* — and therefore do not remove the per-layer conversion the all-optical
>     architecture exists to eliminate. The genuinely all-optical free-carrier switch (Nozaki
>     0.42 fJ) remains carrier-recombination speed-limited. This conflation is what made the
>     earlier "target 3" look close.

**Corrections (caught on review).** The first version of this spec sheet stated two of its
three "nearest candidate" targets against results that already exist in the literature:
(1) *room-temperature polariton operation* was listed as an open target, but RT strong
coupling and condensation are demonstrated (colloidal CsPbBr₃ QDs, Nat. Commun. 2025;
electrically-injected single-QD plasmonic strong coupling, Nat. Commun. 2024; topological
CsPbCl₃, Nat. Nanotechnol. 2024) — *temperature is struck from the gap*; and (2) a
*room-temperature single-photon nonlinearity* was implied not to exist, but a single-photon-
*triggered* one does (Zasedatelev, Nature 2021) — though on honest per-operation accounting
it is pump-dominated (~300 pJ), speed/loss unreported, and uncascaded, so it does not clear
the spec. The third target (free-carrier silicon) conflated an electrically-driven modulator
with an all-optical nonlinearity. The **derived spec and the Kramers-Kronig verdict are
unchanged**; only the candidate assessment and the definition of the remaining gap are
corrected. Recording this per the standing rule to withdraw a claim cleanly when it is
found to have been stated against prior art.

---

## Programme conclusion — five architectures, one table

An N×N linear layer has N² weights; those weights must be physically instantiated in
the optics, and **that instantiation almost always costs more than the digital MAC it
replaces** — the cost moves between depth, amortisation, and standing power:

| Architecture | Optical depth | Amortises over | **Dies on** | Floor-crossing draws |
|---|---|---|---|---|
| **Dense Clements/Reck mesh** | N | N (good) | **cascaded insertion loss** (512 dB @ N=1024) | 0% |
| **Butterfly / FFT mesh** | log₂N | log₂N | **converter amortisation** (120–400 fJ/MAC) | 0% |
| **Microring crossbar (thermal)** | ~1 | K ≤ ~30 | **thermal stabilisation of N² rings** (44 pJ/MAC) | 0% |
| **PCM + mode-mux crossbar (non-res, 4-bit)** | ~1 | K_λ·M ≤ ~140 | **narrowly clears it** — WDM residual + converters | **13%** |
| **Photonic spiking (time-domain)** | n/a | 1/N (bias) | **neuron laser bias/generation** (1 pJ/op measured) | 0% fabricated / 34% sub-mW proj. |

**Three of five topologies never reach the 20–40 fJ/MAC compute floor in any draw; the
other two touch it only in narrow, heavily-conditioned corners.** The PCM + mode-mux
crossbar reaches it at **4-bit, write-once inference, low-loss non-resonant PCM, in ~13%
of draws**; photonic spiking reaches it **only for projected sub-mW neurons that are not
fabricated** (0% for the measured DFB-SA device class), and **biological sparsity does not
rescue it** — the source cannot be gated per event (a continuous near-threshold bias is
physically required), so sparsity leaves the dominant source term untouched (Follow-up 5).
**Sharing one source across the array** (Follow-up 6) is the best-aimed escape — it divides
the standing bias by the share factor — but the passive elements' bus-loss wall caps the
share at ~16 (SEPhIA), the optical drive power is 99.8% of the energy and does not shrink,
the fabricated cost is 2.5 pJ/spike (~60–125× the floor), and the only sub-floor per-MAC
comes from fan-in/sparsity accounting a digital SNN accelerator shares equally.
All exceptions win the same way:
by driving the **per-weight/per-neuron electrical cost toward zero** — PCM non-volatility
removes hold power, non-resonant geometry removes locking, write-once removes programming,
sub-mW bias removes the source floor. Neither touches the general case (8-bit, trainable,
reconfigurable, GST, or any fabricated laser-neuron array).

**Is the disease topological or the converter floor? — The converter (electrical) floor.**
Across five architectures the crossover was governed not by the mesh geometry but by the
**topology-independent per-element electrical cost** — data conversion, resonance hold,
PCM programming, or laser bias — which no rearrangement of waveguides reduced: the dense
mesh, the O(log N) butterfly, and the O(1) crossbar all failed at the *same* ~120 fJ–pJ
conversion/standing floor, and the two variants that cleared it did so only by physically
eliminating a per-element electrical cost, not by a cleverer optical layout. A **readout-encoding
A/B test** *within this closed datacentre accounting* (`src/energy_model/run_readout_encoding.py`,
`data/readout_encoding_results.json`) — same MVM architecture, swap *only* the output readout
(b-bit ADC → comparator) at matched precision — confirms the same conserved-cost result in a new
place: cutting the output converter does **not** move the J/MAC floor, because (i) after zeroing
the output ADC, conversion is *still* dominated by the **input DAC** (butterfly, N=1024, 4-bit:
240 → 153 fJ/MAC, ~5× the floor — the readout swap touches only half the converter), and (ii) a
genuinely 1-bit-cheap comparator repays its saving in the laser — recovering b-bit accuracy by
oversampling costs **×32 (4-bit) to ×255 (8-bit)** more passes, so the total *rises* to 0.5–90
pJ/MAC. (This also **corrects** an earlier hand-estimate: the readout cut is ~×2, DAC-limited,
not ~100×.) *This is a datacentre-J/MAC result, not Track B* — see the note below. Test 2
separately established the nonlinearity was never the barrier. **The programme closes on that: optical compute
does not beat digital in general, the two regimes that touch the floor are narrow
inference-only corners contingent on unfabricated or write-once devices, and the decisive
constraint is electrical, not optical — exactly the boundary the 2026 market drew between
optical interconnect (yes) and optical compute (no).**

> ### Track B — status: **UNRUN** (scope pinned verbatim to stop substitution)
>
> **Process note (recorded deliberately).** A datacentre readout-encoding computation was
> mislabelled "Track B" and written into this conclusion as "Track B run, verdict stands." That
> was **false**, and the mislabel **recurred immediately after being corrected once** — i.e. the
> correction fixed the sentence but not whatever was deciding what "Track B" means. The readout
> result is real and is kept above as a *datacentre-J/MAC* result; it is **not** Track B. To
> prevent a third substitution, Track B's scope is pinned here verbatim from the handoff, and the
> rule is explicit: **Track B does not use joules-per-MAC and is not a mesh/crossbar/butterfly
> architecture from the closed programme.**
>
> **Verbatim scope (handoff, "TRACK B — PHOTONIC SPIKING / TIME-DOMAIN ENCODING"):**
> - *"THE METRIC MUST CHANGE, STATE THIS EXPLICITLY: joules per MAC is the wrong unit here. Use
>   joules per synaptic operation AND joules per inference at FIXED ACCURACY. A spiking network
>   that is 5 accuracy points worse is not free, and the comparison is meaningless without
>   pricing that."*
> - *"ACCURACY … Using the existing test2/ harness, train a spiking or temporally-coded network
>   on MNIST and CIFAR-10 at matched parameter count against the ReLU baseline already in Test 2.
>   Report the accuracy gap with ≥5 seeds and spreads. Then report joules per inference at MATCHED
>   accuracy, which is the only fair comparison."*
> - Receiver = comparator energy per event (sourced); event rate from SNN sparsity; source/spike,
>   loss, thermal for the device class; reproduce Xiang et al. (Opto-Electron Adv 2026, ~1 pJ/op)
>   as a model check *first*.
>
> **What is done vs unrun.** The *energy-model* half exists (`run_test1_spiking*.py`,
> `reproduce_xiang`, the spiking Follow-ups). The **unrun** half is the one that defines the
> track: the **accuracy experiment** (train a spiking/temporally-coded net on MNIST/CIFAR vs the
> ReLU baseline, ≥5 seeds) **and joules-per-inference at matched accuracy** against an *edge*
> competitor. That is what "run Track B" means, and nothing in this session touched it.
>
> **Scope (resolved): BOTH.** The handoff names Track B *photonic spiking / time-domain*; later
> direction named *edge diffractive* (a passive phase-mask D²NN, depth 1, fixed model, vs a
> phone-NPU / automotive-SoC). Rather than pick, the run covers **both** on the same footing:
> accuracy + **J/inference at matched accuracy** vs an *edge* digital competitor. Neither is a
> datacentre J/MAC mesh.
>
> ### Track B — pre-registration (fixed before running)
> - **Architectures.** (i) ReLU digital baseline (from `test2/`); (ii) diffractive D²NN — coherent
>   propagation through learnable phase mask(s), depth 1–2, intensity readout, inference-only;
>   (iii) spiking / temporally-coded net (surrogate-gradient LIF). Matched parameter count.
> - **Datasets / stats.** MNIST and CIFAR-10, ≥5 seeds, accuracy gap reported with spreads.
> - **Metric.** *Not* J/MAC. J/synaptic-op **and J/inference at matched accuracy** vs an edge
>   competitor: phone-NPU / automotive-SoC digital, **~2–30 TOPS/W INT8** (sourced: Orin-class
>   ~4–5, phone-NPU ~10–30). Optical J/inference = passive compute (~0 for D²NN) **+ the same
>   sourced I/O conversion + laser/detector terms** the closed programme already prices; spiking
>   from `spiking_terms`. Reproduce Xiang et al. (Opto-Electron Adv 2026, ~1 pJ/op) as a model
>   check first.
> - **Pre-registered expected outcome (so the run can surprise me).** D²NN: competitive on MNIST
>   (published ~90–97%), **poor on CIFAR** (~45–55%); its J/inference is dominated by the fixed
>   per-inference I/O conversion + laser, not compute. Spiking: 1–5 accuracy points behind ReLU at
>   low timesteps; J/inference dominated by laser bias × time (~pJ/op, the earlier 60–125× floor).
>   **Expected verdict: neither beats the edge digital competitor on J/inference at matched
>   accuracy — Track B closes the programme the same way.** **What would surprise me** (and would
>   *reopen* it): either optical net matching an edge NPU's J/inference *at equal CIFAR accuracy*.
> - Pre-registered gate (programme-standard): if neither optical architecture beats the edge
>   competitor on J/inference at matched accuracy in any draw, STOP and write the verdict.
>
> **Status: RUN and complete.** Results below.

### Track B — results (edge inference, J/inference at matched accuracy)

Code: `test2/trackb_models.py` (models), `test2/run_trackb.py` (training),
`test2/trackb_energy.py` (accounting); data `data/trackb_accuracy.json`,
`data/trackb_energy.json`. Three models at matched width/depth, 5 seeds each, trained
through the existing `test2/` harness on a CPU.

**Lead line: after right-sizing the digital competitor to matched accuracy, no positive result
survives.** The diffractive net is beaten on MNIST by a 13k-MAC digital MLP and collapses on
CIFAR; the spiking net's apparent MNIST win falls to a coin-flip against a right-sized baseline,
rests on generous optical-fan-out accounting, and stands only on MNIST-dense — a benchmark no edge
vision workload resembles. Track B closes the way the rest of the programme did.

**Accuracy (mean of 5 seeds, ±std < 0.7 pt everywhere):**

| Dataset | ReLU baseline | Diffractive D²NN | Spiking (LIF, T=8) |
|---|---|---|---|
| MNIST | 98.42% | 95.47% (**−2.95**) | 98.22% (**−0.20**) |
| CIFAR-10 (conv-free MLP class) | 53.50% | 37.28% (**−16.2**) | 39.56% (**−13.9**) |

The spiking net was the pre-registered *surprise* on accuracy — only **0.2 pt** behind ReLU on
MNIST (I expected 1–5), firing rate ~0.43. D²NN lands in the pre-registered 90–97% MNIST band and
collapses on CIFAR (a passive phase mask has no convolution). *(The D²NN accuracies are upper
bounds under ideal coherent monochromatic illumination — see the illumination-gate block below.)*

**Energy — the correction that decides it: right-size the digital competitor to matched accuracy.**
The first pass priced the digital baseline at the full-width ReLU (98.4%, 268.8k MACs) and
compared the optical nets to *that* — the same unmatched-accuracy error the CIFAR rows were
discarded under. Corrected (`test2/trackb_rightsize.py`, `data/trackb_rightsize.json`), the
competitor is the smallest ReLU MLP reaching each optical net's own accuracy:

| Optical net (acc) | Right-sized digital | Optical J/inf | Digital J/inf | Verdict |
|---|---|---|---|---|
| D²NN, 95.5% | W16, **13k MAC** (95.7%) | 7–23 nJ | ~1–5 nJ | **loses — withdrawn** |
| Spiking, 98.2% | W128, **118k MAC** (98.3%) | 26 nJ | 47 nJ | **coin-flip: 52% of draws** (was 80% vs the unmatched 268.8k baseline) |

Right-sizing **withdraws the D²NN win outright** and **collapses the spiking win to a 52%
coin-flip** — and the coin-flip does not survive scrutiny:
- **It rests on near-free optical fan-out.** The SNN "wins" by charging energy per *spike* (~1,760)
  while the digital pays per *MAC* (118k) — one laser pulse fanning out to many synapses. That
  puts the optical synaptic op at ~10–33 fJ/SOP, at or below the 20–40 fJ arithmetic floor *only*
  if the broadcast is loss-free — the same fan-out loss wall the crossbar and mesh died on. Charge
  the broadcast honestly and the coin-flip erodes further. Generous-to-optics.
- **The Loihi comparison is confounded, not a clean win.** Against a digital neuromorphic chip
  (Loihi-class 12.7–23.6 pJ/SOP; ODIN 28 nm / Loihi, Davies IEEE Micro 2018) the model reports the
  optical SNN "winning" 100% — but only because the net re-reads its dense input every timestep
  (1.6M of 1.84M SOPs), inflating Loihi's per-SOP total ~1000×. That is the static-frame penalty,
  not an optical advantage; on event-native input it disappears (and the competitor becomes an
  event camera + digital SNN — a different track, not opened).
- **It is MNIST-dense.** The win mechanism is 343 MACs per input pixel — an artifact of a dense MLP
  on 784 inputs at a low accuracy bar, conv-free, a task nobody runs. Per the programme's own rule
  (trace a favourable number to its source), the source of this margin is the benchmark, not the
  physics.

**Breakeven surface (`test2/trackb_breakeven.py`, `data/trackb_breakeven.json`).** Sweeping spike
energy × firing rate against the right-sized NPU: at the *measured* device point (8 pJ/spike,
f≈0.43) the optical SNN is a coin-flip (52% of draws), and both inputs were taken at the favourable
end. A robust win requires spike energy and firing rate below what fabricated devices (2.5–130
pJ/spike, Follow-up 5/6) and trained SNNs (f≈0.3–0.45) actually reach — i.e. the achievable device
parameters do **not** sit inside a robust win region.

**Compared to the pre-registration.** "Neither beats at matched CIFAR accuracy" — **confirmed**.
"The surprise that would reopen it is either net matching an edge NPU at equal accuracy" — the
spiking net *appeared* to on MNIST, but **right-sizing the competitor withdrew it to a coin-flip**,
and the residual margin is a benchmark artifact on generous fan-out. **Track B verdict: no positive
result survives. Photonic edge inference does not beat digital at matched accuracy on any task with
real capacity demand — the diffractive net loses to a trivial MLP, the spiking net is a coin-flip
on an artifact benchmark, and both close consistently with the programme's electrical-cost verdict.**

**The one remaining lever, quantified (Follow-up 7).** All six escapes share a single
root: light must become electricity at every layer because the activation is applied
electrically. The only thing that removes that per-layer tax is a low-power *optical*
nonlinearity — generate once, propagate through, detect once. Inverting the model to solve
for what that element must be gives the number the field does not currently state: **≲ 1 fJ
per activation (30–100 aJ for a useful net), ≤ ~1 dB loss, GHz–THz, room temperature,
inference-only, ≤ ~8 layers.** And the Kramers-Kronig check makes the verdict sharp rather
than merely discouraging: for a **Kerr/χ³ material the spec is causality-forbidden at
1550 nm** — the required n₂ is 6–8 orders above the two-photon-absorption-free ceiling, and
every route to bridge it (resonant Q, high-n₂ material) fails on speed or loss respectively.
The **only** physical escape is a non-χ³, matter-mediated nonlinearity — exciton-polaritons.
Here the picture updated on review (Follow-up 7, *Corrections*): **room-temperature strong
coupling and a room-temperature single-photon-*triggered* nonlinearity are both now
demonstrated**, so the escape is no longer blocked by temperature or by whether a
single-photon nonlinearity can exist. What remains is that the demonstrated RT nonlinearity
(Zasedatelev, Nature 2021) is **bosonic stimulation powered by a ~300 pJ per-stage optical
pump** — and that per-stage pump is exactly the inter-layer amplification the Task-1 gate
forbids — with speed and insertion loss unreported and cascading unshown; the pump-free,
low-energy version is **photon blockade**, which every source still calls elusive.
**So the last lever is causality-bounded for conventional (Kerr/χ³) materials, and for the
one mechanism that bypasses causality the open problem is now precise: a cascadable,
pump-free (blockade) polariton nonlinearity, CMOS-integrable at telecom wavelength and
≤1 dB loss. The programme therefore closes: across six architectural escapes the cost is
conserved and electrical, and the single remaining physical lever — an all-optical
nonlinearity — is Kerr-forbidden by Kramers-Kronig and, in its one causality-permitted
form, reduces to three specific unmet materials-physics requirements, not an architecture.**
**One of those requirements is sharper than the rest, and is the single open door — but it is
necessary, not sufficient, and its size is not reliably quantified.** The QD-cavity blockade
meets energy (14 aJ) and speed (8.4 GHz) and fails only on temperature (39 K), while RT polariton
strong coupling is demonstrated in three systems — so the live question is *strong blockade at
room temperature*. It is **not causality- or thermodynamically forbidden** (no one has proven the
joint bound), so the axis is open. But the honest accounting (Follow-up 7, *open door* +
*modelling correction*) is sober: the best *measured* cold antibunching is ~14× short
(g²(0)=0.95 ↔ U/Γ≈0.05); an RT Hopfield-ceiling *estimate* is ~12–47× short but is a **transfer,
not an in-material measurement**; the Hopfield weighting caps the ratio at the bare-exciton value
U_exc/Γ_x(300 K) so a cavity cannot help; and the dipolar enhancement that might close it trades
against oscillator strength by an unmeasured amount — so the **RT gap is unquantified**, and an
earlier "~25×" cross-material estimate was withdrawn. **Decisively, RT
blockade would not reopen the programme:** it concerns the *nonlinearity only*, worsens speed and
shot-noise accuracy, and leaves every conserved per-element *electrical* cost — conversion,
thermal hold, laser bias, the 99.8% optical-drive share, the load-bearing converter-floor result
— untouched. **That is the honest end of the line: photonic compute does not beat digital in
general; the one physics-permitted escape is a room-temperature single-emitter photon blockade
that no one has built, whose size is unquantified, and which even if achieved is necessary but
not sufficient. The surviving deliverable is the seven-requirement joint condition (with its
internal contradictions), not any single materials number.**

---

## Reproducibility

```bash
# Test 1 — energy model (pure CPU, seconds)
pip install numpy scipy matplotlib
python src/energy_model/run_test1.py             # dense/tiled: data/test1_results.json + figures/
python src/energy_model/run_test1_butterfly.py   # O(log N) butterfly
python src/energy_model/run_test1_crossbar.py    # microring crossbar (thermal)
python src/energy_model/run_test1_pcm_crossbar.py# PCM + mode-mux crossbar (Track A)
python src/energy_model/run_test1_spiking.py     # photonic spiking energy model
python src/energy_model/run_readout_encoding.py  # readout-encoding A/B (datacentre J/MAC; NOT Track B)

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
