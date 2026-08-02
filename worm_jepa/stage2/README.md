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
