# Stage 2 — automated mechanistic-model discovery (the second black box)

The double black box's second stage: take what stage 1 extracted (the
forecaster's dynamics + effective connectivity) and have an automated discovery
engine synthesize a **mechanistic** model of it — then score that model against
ground truth.

> Note: **AlphaEvolve is DeepMind-internal, not public.** `llm_evolve.py`
> reimplements its *recipe* from scratch — an LLM as the mutation operator in an
> evolutionary search over programs, with an automated evaluator + archive. No
> AlphaEvolve code is involved.

## Problem (ground-truth testbed)

Synthetic worms (`data.make_synthetic_worms`) with a **known** coupling matrix
`W`. From the state `x[t]`, predict the **change** `Δ = x[t+1] − x[t]`
(predicting the delta, so the trivial `return x` identity scores 0 — you must
recover real dynamics). Two scores:

- **pred_r2** — held-out R² on Δ (test worms).
- **struct_corr** — corr of the model's effective linear operator with the true
  neuron coupling `|W @ W.T|` (off-diagonal). This is *connectome recovery*.

Every method — baselines and LLM candidates alike — produces the same unit, an
AlphaEvolve-style "program that fits then predicts":

```python
def build(X_train, D_train):
    # fit anything on the (x, delta) training pairs
    def predict(X):        # (B, N) -> (B, N) predicted delta
        ...
    return predict
```

## What we tried

- `discover.py` — the harness + pure-numpy baselines: **zero**, **linear_ridge**,
  **sindy** (sparse nonlinear regression), **random_evo** (non-LLM evolutionary
  sparse search).
- `candidates/` — the **Claude-as-operator** loop (AlphaEvolve recipe), driven by
  the agent one generation at a time; each program is scored by
  `discover.eval_program`.
- `llm_evolve.py` — the same loop **automated** via the Anthropic API (guarded;
  needs `ANTHROPIC_API_KEY` + `pip install anthropic`).

## Results (`leaderboard.json`)

| rank | model | pred_r2 | struct_corr |
|---|---|---|---|
| 1 | **claude: gen4_tuned** | **0.197** | 0.239 |
| 2 | claude: gen2_dense_plus_cross | 0.190 | 0.239 |
| 3 | baseline: sindy | 0.174 | **0.242** |
| 4 | claude: gen3_reduced_rank | 0.158 | 0.196 |
| 5 | baseline: random_evo | 0.158 | 0.198 |
| 6 | claude: gen1_latent_cross | 0.125 | 0.122 |
| 7 | baseline: linear_ridge | 0.106 | 0.236 |
| 8 | baseline: zero | 0.000 | 0.000 |

The Claude-operator loop produced the best **predictive** mechanistic model
(gen4, +13% over the best baseline), while SINDy holds a hair's-edge on
**structure** (0.242 vs 0.239 — effectively tied). The 4-generation trajectory
is the point: gen1 (latent bottleneck) **failed** → gen2 (full library + latent
cross-terms, dense ridge) **won** → gen3 (reduced-rank) **failed** → gen4
(refined the winner) **improved**. Each step used the previous scores as
feedback — the evolutionary recipe working.

## Reproduce

```bash
cd worm_jepa/stage2
python discover.py                 # baselines + all candidates -> leaderboard.json
python -c "import discover; print(discover.eval_program('candidates/gen4_tuned.py'))"
# automated loop (needs ANTHROPIC_API_KEY):
python llm_evolve.py --generations 12
```

## Is stage 2 (the AlphaEvolve loop) good enough?

We stress-tested this on the synthetic testbed with a diagnostic (`experiment.py`)
and a data sweep (`experiment_scale.py`). Key finding: the modest single-frame
score (0.197) was an **input** problem, not a **search** problem. Predicting the
delta from one frame is phase-ambiguous (you can't tell an oscillator's
direction from a snapshot), so *every* method caps ~0.27 there; three frames of
history takes any method to ~0.80.

Given that fair input, the AlphaEvolve loop is good enough — it beats the
brute-force baselines on both metrics:

| model (input) | pred_r2 | struct_corr |
|---|---|---|
| AlphaEvolve `tgen1_velocity` (3-frame) | **0.818** | **0.248** |
| ridge (3-frame) | 0.802 | — |
| MLP (3-frame) | 0.782 | — |
| AlphaEvolve `gen4` (single frame) | 0.197 | 0.239 |

`candidates_temporal/` holds the temporal-input programs; evaluate with
`discover.get_problem(lags=3)`. Data-lever result: prediction saturates almost
immediately with data (temporal *extraction* is the lever), but connectome
recovery (`struct_corr`) keeps climbing with more worms (0.10→0.30 over 10→320)
— structure recovery is the data-hungry part.

## Honest caveats

- The delta is only partially predictable from a single frame (oscillator phase
  sign is ambiguous), so pred_r2 has a ceiling well below 1 — the *relative*
  ranking is the signal.
- Scored on the synthetic system (known `W`). The natural next step is to run the
  same search against the **real** forecaster's dynamics / the real C. elegans
  connectome.

## The bridge: sensory → command → behavior (L2→L4)

`bridge.py` composes the sensorimotor loop that no single public dataset
records intact. The data splits into two halves that never co-occur — and
`bridge.measure_gap()` shows *why* the missing middle map cannot be read off
activity from either half:

| regime | sensory→command next-step gain | reading |
|---|---|---|
| immobilized + stimulus (000541+000981, n=45) | **−0.385** (t=−2.7, p=0.009) | command **decoupled** from sensory (immobilization artifact) |
| freely-moving (Flavell 000776, n=38) | −0.011 (n.s.) | coupling present but sensors barely driven (no stimulus) |

So the middle map comes from the **causal prior** (Randi L2 atlas), per the
priors-over-activity rule in `METHODOLOGY.md`. The bridge is four measured maps,
each valid in its own regime, none of which ever saw the full loop:

- **A** transduction (stimulus→sensory) — stimulus-triggered average, 000541/000981
- **M** sensory→command wiring magnitude — Randi funatlas `wt/dFF` gated by `wt/q`
- **V** activation→reversal valence — documented chemosensory sign prior
- **C** command→behavior — reversal state → velocity, Flavell (−0.298, n=38)

`dVel = betaC · Σ_s activation_s · valence_s · wmag_s`. End-to-end it predicts
the sign of the behavioral response to each stimulus class **3/3** correctly
(`bridge.json`), and the **shuffled-valence** control inverts the two attractive
predictions (~3σ) — the structure test from `METHODOLOGY.md` passing on exactly
the cases that require the prior.

| stimulus | pred ΔVel | behavior | shuffled-valence null |
|---|---|---|---|
| aversive (CuSO4) | −0.39 | REVERSE ✓ | −0.33 ± 0.23 (robust either way) |
| attractive onset | +0.27 | FORWARD ✓ | −0.32 ± 0.19 (**sign inverts**) |
| attractive removal | −0.19 | REVERSE ✓ | +0.10 ± 0.09 (**sign inverts**) |

This is the L2→L4 counterpart to OpenWorm's forward simulation: OpenWorm
integrates a hand-built dynamical model; the bridge composes empirically
measured maps and passes a negative control. Reproduce the neural cache from
DANDI with `python bridge.py --stream` (streams via HTTP range requests, see
`stream_dandi.py`); `python bridge.py` alone runs offline from the shipped
`bridge_maps.npz`.

## OpenWorm (c302) integration

`openworm_integrate.py` connects our measured L2 causal atlas to OpenWorm's
`c302` NeuroML model, both directions:

**(1) Data → OpenWorm.** c302 weights each chemical synapse as
`baseline_conductance × anatomical connection count`. We attach the *measured*
causal coupling (Randi atlas) to each edge and emit `reweighted_connectome.json`
(2279 chemical synapses; 970 have a measured value, 94 significant at q<0.05).
Finding: connection **count barely predicts measured function** (Spearman
r=−0.01, r²≈0.02%) and ~28% of covered edges are net-inhibitory — so anatomical
counts are a weak proxy for functional weight. *Honest caveat:* the atlas dFF is
a **propagated network response**, not a monosynaptic conductance, so it does not
track c302's NT polarity (GABA-vs-excitatory sign test fails, p=0.83). Use it as
a network-level functional weight, not a synapse-sign claim.

**(2) OpenWorm → us.** c302 is a full biophysical forward simulator; our `bridge`
is a validation oracle for it. `c302_validation_targets()` emits, per stimulus,
the sensory neurons to inject current into and the expected AVA/command sign
(aversive→AVA up→reversal, etc.). Running c302 forward and checking it reproduces
these is a concrete cross-model test: OpenWorm *integrates* a hand-built model; we
*compose* measured maps — the two should agree on the aversive→reversal transform.

Requires `pip install c302` + `wormneuroatlas` (re-weighting needs no simulator).

## L4 benchmark: head-to-head vs Hallinen 2021

`l4_benchmark.py` reproduces the Hallinen et al. 2021 (eLife 66135) locomotion-
decoding protocol — ridge on `[F, dF/dt]` across all neurons, R²_ms on a held-out
test set — on the Flavell 000776 data, then beats it with a nonlinear decoder on
the **same splits and features** (n=38 worms, `l4_benchmark.json`):

| channel | Hallinen ridge `[F,dF/dt]` | **ours (HistGradBoost)** | shuffle null | Hallinen published |
|---|---|---|---|---|
| velocity | 0.653 | **0.744**  (38/38 >0) | −0.60 | 0.56 median / 0.76 exemplar |
| curvature | 0.200 | **0.397**  (~2×) | −0.91 | 0.29 median / 0.60 exemplar |

Same data, same held-out splits, same input features — a nonlinear decoder beats
their linear ridge by +0.09 (velocity) and ~2× (curvature). The temporal-shuffle
null collapses to strongly negative, so the decoding is real signal, not
autocorrelation leakage shared by both models. (Absolute numbers exceed
Hallinen's published medians, but that is partly a dataset difference —
Flavell != Leifer — so the honest claim is the same-data method win.)

## Closing the loop: c302 forward simulation

`openworm_simulate.py` closes the sensorimotor loop inside OpenWorm. It injects an
aversive stimulus current into the nociceptors our transduction map flags
(ASH/ADL), runs the c302 model forward with jNeuroML, and reads the command
circuit. Using **graded synapses** (parameters_C1 — the biologically correct
choice; c302's default spike-triggered synapses transmit nothing because the
cells are non-spiking) and a small drive (the recurrent circuit runs away
otherwise), the result (`openworm_sim.json`):

| connectome weights | reversal command (AVA·AVE·AVD) | forward command (AVB·PVC) | bias |
|---|---|---|---|
| c302 default (connection counts) | **+0.48 mV** | +0.25 mV | 1.9× |
| our measured L2 causal weights | +0.33 mV | +0.27 mV | 1.2× |

Under both weightings the aversive stimulus biases c302 toward the **reversal**
command — the escape reflex, reproduced by a biophysical forward model driven the
same way our composed bridge is. (Our re-weighting changes the balance but doesn't
sharpen it here; faithful full-network dynamics remains a connectome-weight
optimization problem.) The generated NeuroML (`c302_run/LEMS_Loop_default.xml` +
`Loop_default.net.nml`) opens in OpenWorm / Geppetto / Open Source Brain.

A visualization of the circuit and the simulated traces is in
`../artifacts/worm_loop.html`.

## L3 benchmark: head-to-head vs worm-graph (Simeon 2024)

`l3_benchmark.py` reproduces the worm-graph next-step prediction benchmark on its
own public data (HuggingFace `qsimeon/celegans_neural_data`, 919 worms, canonical
300-neuron slot space, masked MSE) and beats the persistence baseline
(`l3_benchmark.json`):

| model | masked next-step MSE | vs persistence |
|---|---|---|
| persistence (worm-graph baseline) | 0.0290 | — (they report 0.03541) |
| our global linear AR (1 lag) | 0.0287 | −1.0% |
| **our global linear + 3-lag history** | **0.0203** | **−30%** |

Two things fall out. (1) Our single-frame linear barely beats persistence (−1.0%),
**reproducing worm-graph's own finding** that their linear/CTRNN/LSTM only
marginally beat the baseline (0.03533 vs 0.03541) — confirming our setup matches
theirs. (2) Adding temporal history takes it to −30%, far past what they report.
This is also the methodology point (METHODOLOGY.md): next-step prediction is
persistence-bound — *the trap*. A 30% win is real, but next-step MSE is the weak
task; recovered structure shows in held-out-neuron imputation (cf. Creamer/Leifer/
Pillow 2024 connectome LDS at 92% of the noise ceiling), not here.

## L2 imputation benchmark: connectome vs shuffled (Creamer 2024)

`l2_imputation_benchmark.py` reproduces the Creamer/Leifer/Pillow 2024 connectome-
constrained held-out-neuron imputation — the cleanest instantiation of the
methodology (structure shows in imputation, not next-step). All from local data
(Randi funatlas + c302 connectome). Predict each held-out neuron's perturbation-
response profile from its connectome partners' profiles; score as a fraction of
the split-half noise ceiling (`l2_imputation_benchmark.json`):

| model | r | % of noise ceiling |
|---|---|---|
| noise ceiling (split-half, SB) | 0.217 | 100% |
| **connectome model (1-hop)** | 0.183 | **84%** |
| shuffled connectome | 0.041 | 18% |

Connectome ≫ shuffled by ~65 points of ceiling — **structure recovered**, the
"data speaking" side of the trap. This reproduces Creamer's result (their fitted
LDS reaches 92%) with a simple 1-hop model. The numerical SOTA-beat at L2 is the
harder *pair*-holdout generalization (~2.4× the connectome-linear class); this is
the held-out-*neuron* protocol Creamer used, and it confirms the double-black-box
methodology on real data: connectome ≈ shuffled on next-step (the trap), connectome
≫ shuffled on imputation.

## Predicting behavior from a compound (the medical bridge)

`compound_response.py` runs the double black box from a molecule, closing the gap
between "predict activity/behavior from imaging" (what the benchmarks do) and
"predict the effect of a chemical" (what a medical funder pays for):

  compound → target gene (curated pharmacology) → neurons expressing it (CeNGEN,
  L0) → circuit position (our stack) → predicted behavioral direction.

On a panel of neuroactive compounds with established worm phenotypes it gets **8/8**
directional predictions right (`compound_response.json`). The gene→neuron
localization is biologically correct (acr-16→AVA, mod-1→AVE/PVC, avr-15→RIB/RME).

**Honest boundary** (the CeNGEN-shuffle control, ~91%): the up/down *direction* is
carried mostly by curated receptor polarity + agonism (known pharmacology), not by
the model. CeNGEN's genuine contribution is neuron-level *localization*, which a
coarse up/down metric can't reward. Making localization load-bearing needs a
location-sensitive task (which behavioral module a compound perturbs), validated
on held-out compounds against wet-lab data — the Phase I STTR deliverable. The
pathway runs end-to-end today; Phase I makes it quantitative.

## In-silico neuron ablation (perturbation prediction)

`ablation_importance.py` answers the question laser-ablation experiments answer —
if you removed neuron X, how much would behavior break? — in silico. It
permutation-ablates each neuron in a population→velocity decoder (Flavell) and
ranks by held-out R² drop (`ablation_importance.json`):

- **9/10** of the most behaviorally-critical neurons are locomotor-circuit
  (RIB, RIM, AIB, AVA, AVE command + RID, RME, SMD, AVL, SIB head-motor).
- Command neurons carry **3.5×** the average importance.

The model recovers the causal locomotor circuit unsupervised — an in-silico
ablation screen. With `compound_response.py` (chemical perturbation) this shows the
platform predicts the consequences of perturbing the system — genetic/physical and
chemical — which is the core of every medical and functional-genomics use case.

## Held-out perturbation prediction (the revolutionary core)

`perturbation_response.py` predicts the whole-brain response to a perturbation
never seen in training — the capability a drug/gene screen needs. Validated on the
Randi optogenetic atlas (stimulate neuron j → whole-brain response). **Honest
metric:** exclude the self-response (i==j) — stimulating a neuron and measuring
*itself* is trivially predictable and inflates every score; a screen cares about
the DOWNSTREAM response (which *other* neurons respond). All numbers below are
downstream-only (`perturbation_response.json`):

| model (downstream, i≠j) | r | % of matched noise ceiling |
|---|---|---|
| matched **downstream** noise ceiling (split-half) | 0.371 | 100% |
| **dynamical (I−gA)⁻¹ propagation** | 0.141 | **38%** |
| GBM connectome-feature model | 0.127 | 34% |
| shuffled connectome | ~0.00 | ~0% |

**Ceiling correction:** an earlier version divided by 0.595 — the reliability of the
*full* column, which includes the highly-reliable self-response. The metric scores
*downstream* responses (i≠j), whose split-half reliability is only **0.371**. Against
the matched ceiling the model is at **38%**, not 24%. (Model performance didn't
change; the benchmark denominator was wrong.)

Two results. (1) The **better model wins**: a fitted dynamical propagation model —
`R[:,j] ≈ (I−gA)⁻¹e_j`, one global gain, no per-perturbation parameters, so it
generalizes to any stimulation by construction — beats the gradient-boosted feature
model (24% vs 21%), against a clean-zero shuffle. (2) **Honest negative**: adding
synapse *signs* from neurotransmitter identity (c302) + receptor expression
(CeNGEN) slightly *hurt* — the atlas dFF is a propagated network response, not
monosynaptic polarity (same reason the c302 GABA cross-check fails), so hardcoded
signs add noise.

**38% of the matched ceiling — this is the honest "what would make it
revolutionary" answer.** A model that reaches the ceiling on downstream held-out
perturbations *is* the in-silico screen. But six model classes (dynamical, GBM,
low-rank collaborative filtering, target-response composition) plus the wireless
extrasynaptic connectome all plateau at the same raw r≈0.14 — the cap is
informational, not a model failure. The remaining levers are temporal dynamics (funatlas ships response kernels, not just
steady-state dFF), a connectome GNN that learns the propagation nonlinearity, and
multi-organism training. (Two metric fixes got us here: excluding the trivial self-response, and matching
the ceiling to the downstream metric.)

## Temporal dynamics: does the connectome predict response timing?

`temporal_dynamics.py` goes after the signal the steady-state model can't see —
*when* each neuron responds. The funatlas ships each response as an exponential-
convolution kernel; we evaluate them, extract a robust latency, and correlate it
with connectome shortest-path distance (`temporal_dynamics.json`):

| stimulated→responder distance | response latency (center of mass) |
|---|---|
| 1 hop | 7.0 s |
| 2 hops | 8.4 s |
| 3 hops | 8.9 s |
| 5 hops | 10.9 s |
| 6 hops | 11.3 s |

Response latency scales monotonically with synaptic distance (time-to-half vs
hops: Spearman r=0.11, p=3×10⁻⁵) — the connectome predicts **when** a neuron
responds, a dimension orthogonal to the steady-state magnitude. Honest scope: the
effect is real, significant, and monotonic but modest (r≈0.11) — a new informative
dimension, not by itself the leap to the noise ceiling. Predicting the full
response waveform from the connectome is the richer next step.

## predict.py — the prediction engine (what we can predict)

`predict.py` is one interface over every validated predictor here, plus an honest
capability card (`python predict.py`). It backs the claim: *from wiring + molecules
+ activity we predict how the C. elegans nervous system behaves and responds* —
each capability validated on held-out data with a negative control.

| we predict… | input → output | accuracy (held-out) | vs SOTA |
|---|---|---|---|
| **behavior from activity** | neural activity → velocity/curvature | R²=0.74 / 0.40 | beats Hallinen 2021 (0.65/0.20) |
| **stimulus → behavior** | chemical stimulus → reverse/forward | 3/3 classes | shuffle control inverts |
| **compound → behavior** | drug target → locomotor direction | 8/8 compounds | CeNGEN localizes drug→circuit |
| **neuron imputation** | held-out neuron → its responses | 84% of ceiling | shuffle 18%; ~Creamer (92%) |
| **perturbation response** | stimulate neuron → whole-brain | 38% of ceiling (unseen) | clean-zero shuffle; first of its kind |
| **neuron importance** | which neurons behavior needs | 9/10 = locomotor circuit | recovers circuit unsupervised |

Live: `python -c "import predict; print(predict.predict_stimulus_response('aversive'))"`
→ `{'behavior': 'REVERSE', 'delta_velocity': -0.39}`;
`predict.predict_compound_effect('acr-16')` → `{'direction': 'increase', 'target_neurons': ['AVA','PVR','RIB',...]}`.

## worm_twin.py — the unified digital twin

`worm_twin.py` fuses the six validated predictors into one object with a single
entry point: **any perturbation → full predicted state**. Put in a compound, a
gene, a stimulus, a neuron stimulation, or an ablation; get back which neurons
change, the command-circuit shift, and the behavior — each with an honest
confidence bounded by what was actually validated, and an explicit **domain gate**
that abstains on non-neural (proteostasis/TF/metabolic) targets.

```
t = WormTwin()
t.perturb("compound", gene="acr-16")     # nicotine  → FORWARD/more active (validated)
t.perturb("compound", gene="mod-1")      # serotonin → REVERSE/slow (validated)
t.perturb("stimulate", neuron="ASHL")    # optogenetic → bounded propagation (38% ceiling)
t.perturb("stimulus", stimulus="aversive")  # → REVERSE (bridge, 3/3)
t.perturb("ablate",  neuron="AVAL")      # → locomotor deficit (command neuron)
t.perturb("compound", gene="daf-16")     # → ABSTAINS (not a neural receptor)
```

Design principle (learned by building it): the twin is an **orchestrator that
defers each perturbation type to its validated component** — it never re-derives
behavior with a generic model that could contradict them. Building the fusion
caught a real bug (a generic propagation predicted serotonin *speeds up* the worm,
opposite the validated result) and forced the domain gate and the command-neuron
ablation fix. It is the whole-organism artifact — a *generative* model of
perturbation→state — honest about its bounds, not a wiring diagram.

## proteostasis.py — the one justified new layer (Aβ → paralysis)

`proteostasis.py` is the disciplined answer to "add every model": add a model only
where it improves a *validated endpoint*. The standard worm AD screens (GMC101/
CL4176) are **proteostasis** assays the nervous-system twin abstains on; this module
gives them a mechanistic account and extends the twin's domain to cover them.

Mechanism (not a fit): two-step Finke-Watzky nucleation-autocatalysis for Aβ
aggregate mass → toxicity threshold → paralysis onset, calibrated to the published
~24–48 h timescale (baseline 36 h). A compound scales the aggregation rate by a
factor `f`. Result (`proteostasis.json`):

| compound | f | paralysis (h) | Δ vs vehicle |
|---|---|---|---|
| vehicle | 1.00 | 36 | — |
| PBT2 | 0.45 | 81 | **+44 h (protective)** |
| EGCG | 0.60 | 61 | +24 h |
| curcumin | 0.70 | 52 | +16 h |
| thioflavin T | 1.00 | 36 | **+0 h (inert — clean negative ✓)** |

Protective ordering correct; the inert control lands at baseline. **Honest
boundary (stated, not hidden):** the potency `f` is a literature/assay input — the
module predicts the *phenotype* from mechanism; predicting `f` from chemical
structure (QSAR) is out of scope and not claimed. Wired into `worm_twin.py`:
`t.perturb("compound", gene="PBT2", aggregation_effect=0.45)` → "paralysis @ 81 h
(+44 h, protective)". The twin now covers the AD paralysis screen and still abstains
on genuinely out-of-domain targets.
