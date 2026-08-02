# worm_jepa — a JEPA foundation model for *C. elegans* whole-brain activity

Stage 0 of the **double black box**: train a self-supervised foundation model on
worm neural activity, then open it up (SAE / probes / ablations) to extract a
mechanism that could feed an automated model-discovery engine.

Why *C. elegans*: it is the one animal whose full wiring (the 302-neuron
connectome) is known, so extracted features can be **validated against ground
truth** — something impossible in mouse or human.

## What's here

| file | role |
|------|------|
| `data.py` | Loads `qsimeon/celegans_neural_data` (HuggingFace) into per-worm `(time, neurons)` matrices over a canonical neuron axis; windows + I-JEPA block masks. Also a synthetic "toy-connectome" generator with known latents for offline validation. |
| `model.py` | Time-series JEPA: patch-embed → context encoder + EMA target encoder + predictor; latent-space prediction (no trace reconstruction). |
| `models_bench.py` | Two variants for the benchmark: a 2-level **hierarchical** JEPA (adds a coarse/slow level) and a **forecasting NN** (GRU next-step predictor with an effective-connectivity Jacobian). |
| `benchmark.py` | Head-to-head: flat JEPA vs hierarchical JEPA vs forecaster. Honest held-out ridge probe for latent recovery (overall / slow / fast band), forecast MSE, and `connectome_corr` (forecaster effective coupling vs the true `W @ W.T`) → `artifacts/benchmark.json`. |
| `train.py` | Device-agnostic training loop (CPU smoke or CUDA). Writes `artifacts/checkpoint.pt`, `history.json`, `meta.json`. |
| `extract.py` | The "see what we extract" step: SAE on the frozen embeddings, linear probes (known latents for synthetic / labeled-neuron activity for real), feature-ablation ranking → `artifacts/extraction.json`. |
| `modal_app.py` | Modal GPU launcher (mirrors `test2/modal_app.py`). |
| `.github/workflows/worm-jepa-modal.yml` | Runs the Modal GPU job from CI and commits `artifacts/` back. |

## Running it

**Modal GPU (real data) — via GitHub Actions.** Modal uses gRPC, which the
sandboxed dev container blocks, so the GPU run is launched from CI (same as
`test2`). Trigger **Actions → "C. elegans JEPA on Modal GPU" → Run workflow**
(inputs: `epochs`, `max_worms`, `synthetic`). It trains + extracts on an A10G and
commits the artifacts to the branch.

**Local CPU smoke test (offline synthetic data), no Modal, no download:**

```bash
pip install -r worm_jepa/requirements.txt
cd worm_jepa
WORM_JEPA_SYNTHETIC=1 WORM_JEPA_EPOCHS=5 WORM_JEPA_MAX_WORMS=24 \
  WORM_JEPA_DMODEL=64 WORM_JEPA_DEPTH=2 python train.py
WORM_JEPA_SYNTHETIC=1 WORM_JEPA_MAX_WORMS=24 python extract.py
```

## Baseline artifacts (checked in)

`artifacts/` holds a **synthetic-data smoke baseline** (CPU, 5 epochs) so the
pipeline is reproducible without a GPU. Key result: the frozen JEPA embeddings
linearly decode the *known* latent trajectories at **R² ≈ 0.90**, and the SAE
recovers a near-complete feature dictionary (1 dead feature / 512). The real
GPU run over `qsimeon/celegans_neural_data` overwrites these via CI.

## Config (environment variables)

`WORM_JEPA_SYNTHETIC` `WORM_JEPA_EPOCHS` `WORM_JEPA_MAX_WORMS` `WORM_JEPA_DEVICE`
`WORM_JEPA_DMODEL` `WORM_JEPA_DEPTH` `WORM_JEPA_WINDOW` `WORM_JEPA_PATCH`
`WORM_JEPA_BATCH` `WORM_JEPA_OUT` — see `train.py` for defaults.

## Data source

Simeon, Venâncio, Skuhersky, Nayebi, Boyden, Yang — *Homogenized C. elegans
Neural Activity and Connectivity Data* (arXiv:2411.12091); dataset
`qsimeon/celegans_neural_data`, framework `qsimeon/worm-graph`.
