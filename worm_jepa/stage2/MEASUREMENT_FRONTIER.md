# Measurement frontier — what data would actually break the 43% headroom

The perturbation-prediction arc is 38% → 49% → 57% of the reproducible noise ceiling
(`perturbation_response.py` → `perturbation_improved.py` → `perturbation_completion.py`).
The remaining ~43% is **reproducible** response variance (it is inside the ceiling) that no
model of the current data recovers. This document argues, and the evidence supports, that
**this is a measurement-dimensionality problem, not a model problem** — and specifies the
measurements that would actually add information, versus merely estimating the same matrix
more precisely.

## The claim, and the evidence for it

The current representation is a single averaged matrix:

    stimulate neuron i  →  AVERAGE response of neuron j     (R_ij)

Averaging is a projection that discards trial-to-trial variance, timing, interactions,
effective sign/weight, and neuromodulatory context. More observations of the **same**
averaged quantity estimate `R_ij` more precisely; they do not restore the discarded
dimensions. Two results establish that we are already near that estimation limit, so more N
of the same kind will not help:

1. **Six model classes plateau** at the same raw r (~0.14–0.18 before the few-shot lift):
   dynamical propagation, gradient-boosted features, low-rank collaborative filtering,
   target-response composition, the wireless extrasynaptic connectome, and the raw
   single-worm steady-state. Different function classes, same ceiling → the information,
   not the model, is the bind.
2. **The pseudo-stimulus result is the key evidence** (`state_conditioned_dynamics.py`):
   single-worm "state" looks highly informative (per-trial r 0.06 → 0.39) until you ask
   whether it is *stimulus-specific*. A pseudo-stimulus null (same pre/post windows at
   random non-stimulus timepoints) predicts equally well — it was mean-reversion. The true
   stimulus-specific, state-gated component is ~0.02. **This is direct evidence against
   throwing a bigger dynamics model at the existing measurements**: the state signal that
   looks useful is mostly not about the perturbation.

So the frontier is **more dimensions per experiment, not larger N**.

## The degeneracies averaging leaves, and the measurement that breaks each

| # | Measurement (new dimension) | Degeneracy it breaks | In public data now? | Priority |
|---|---|---|---|---|
| 1 | **Many perturbations in the SAME identified worm** | one pooled `R_ij` can't express worm-specific coupling, latent state, or interaction structure | **Partially** — 001075 (reconnected) has median **21** identified perturbations/worm across 45 worms, but **sequential & sparse** (~51 co-recorded neurons, not whole-brain) | **Highest** |
| 2 | **Voltage imaging** (vs calcium) | calcium is slow + nonlinear; steady-state amplitude conflates **direct transmission with polysynaptic** downstream effect | No (whole-brain identified worm voltage is not a public dataset yet) | High |
| 3 | **Simultaneous whole-brain + perturbation, high temporal res** | amplitudes `R_ij` instead of trajectories `X(t)`; can't fit a dynamical `(X_t, intervention) → X_{t+Δt}` | Partially (001075 is targeted-sequential, ~51 neurons; not dense simultaneous) | High |
| 4 | **Perturbation COMBINATIONS (A, B, A+B)** | single-neuron perturbations assume superposition; interactions `R(A+B) ≠ R(A)+R(B)` are **structurally unrecoverable** from singles | No (optogenetic atlases are single-target) | High (cleanest interaction test) |
| 5 | **Synaptic sign / effective strength** | the anatomical connectome says an edge EXISTS far better than its effective weight/sign; the inverse problem is under-constrained (hardcoded signs *hurt* — see perturbation_response) | Partial proxies only (NT identity, receptor expression, funatlas `q`/kernels); no direct effective-weight measurement | Medium–High |
| 6 | **Neuromodulatory state measured with activity** | serotonin/dopamine/neuropeptides re-route the functional network **without changing the structural connectome** | No (GRAB/dLight sensors exist; no whole-brain identified worm dataset) | Medium |

## The dream dataset and the model target

    one identified worm  ×  many perturbations (incl. combinations)  ×  whole-brain voltage  ×  behavior
    repeated across worms

The model then has to predict, for **held-out interventions in held-out worms**:

    (X_t, intervention, connectome)  →  X_{t+Δt}  →  behavior

This is strictly harder and more mechanistically informative than predicting an averaged
perturbation matrix: it forces a *dynamical, worm-specific, interaction-aware* account
rather than a static map. It is also the representation in which the connectome's role can
be cleanly tested (does structure constrain the trajectory operator?), which the averaged
matrix cannot do.

## What is testable now vs what needs generation

- **Now (weak dimension 1):** 001075 already has ~21 identified perturbations/worm. A
  limited test of worm-specific structure is possible, but it is sequential, sparse
  (~51 neurons), calcium, single-target — so it under-samples exactly the dimensions that
  matter, and the pseudo-stimulus result warns that its state signal is mostly not
  stimulus-specific. Expect small gains; it is a proof-of-concept, not the frontier.
- **Needs generation (the real value):** dense simultaneous whole-brain recording, voltage,
  and above all **perturbation combinations** and **many perturbations per worm** at high
  temporal resolution. These are wet-lab experiments (a CITP-style partner), and they are
  the honest content of a "better measurements" aim — the STTR/partnership deliverable.

## Prioritization (information gained per unit experimental effort)

1. **Many perturbations per identified worm** (dimension 1) — highest value, partially
   exists, directly enables worm-specific `R` + latent state.
2. **Perturbation combinations A/B/A+B** (dimension 4) — the single cleanest, cheapest test
   of whether interactions matter; a decisive yes/no on the superposition assumption.
3. **Voltage + dense simultaneous** (dimensions 2–3) — biggest information gain, hardest to
   generate.
4. **Effective sign/weight and neuromodulatory state** (dimensions 5–6) — constrain the
   inverse problem; medium effort, medium gain.

## The one-line honest framing

The 57% ceiling on the averaged perturbation atlas is not a failure of modeling; it is the
information limit of a representation that averaged away trial state, timing, interactions,
effective weights, and neuromodulation. The path past it is **more dimensions per
experiment** — worm-specific, dynamical, combinatorial, ideally voltage — not more rows of
the same matrix or a bigger model of it. Our own pseudo-stimulus negative is the cleanest
internal evidence for exactly that.
