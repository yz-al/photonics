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
