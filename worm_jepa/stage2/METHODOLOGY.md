# Methodology: feature discovery, priors, and structure vs prediction

Working reference for the stage-2 program. Each principle is annotated with
**→ in our work:** showing where our C. elegans experiments instantiate it.

## Finding signal (feature discovery), roughly by hit rate

**1. Mine the residuals.** Fit a weak baseline, inspect where it fails. Sort
errors, inspect the tails. Cluster residuals and ask what the clusters share.
Near-duplicate rows with divergent outcomes: whatever distinguishes them is a
missing feature by construction. Regress the residual on every column you
excluded, including "junk."
→ *in our work:* not yet done. The obvious next step on the real forecaster:
cluster its worst-predicted (neuron, time) residuals and see if they concentrate
on specific neurons / behavioral states / labs.

**2. Mine metadata and provenance.** Row IDs, timestamps, filenames, batch,
source system, ordering, missingness. Missingness is often the strongest feature —
whatever caused a value to be absent is usually informative. Rounding, defaults,
and sentinel codes leak the collection process.
→ *in our work:* the activity dataset has `source_dataset`, `raw_data_file`, and a
per-neuron **presence mask** (which neurons each lab labeled). We used presence
only as a filter; the *pattern* of which neurons are labeled by which lab is
almost certainly informative and unused.

**3. Dimensional / structural reasoning.** Write down the data-generating process,
even crudely. Ratios/rates/differences beat raw levels. Counts → exposure-adjusted
rates; money → deflated + log. Conservation laws, monotonicity, asymptotes convert
directly to features or priors.
→ *in our work:* predicting the **delta** (x[t+1]−x[t]) not the level; using
**velocity/acceleration** from stacked frames; modeling the **softplus observation
nonlinearity** (exp-correction). Each was a structural-reasoning move, and each was
the thing that actually moved a metric.

**4. Find the quasi-experiments already in the data.** Policy changes, thresholds,
staggered rollouts, boundaries, waitlist positions — variation you didn't create.
Even without a clean causal estimate they tell you which variables move the outcome.
→ *in our work:* the multi-lab structure and any stimulation epochs in the dataset
are unexploited natural variation.

**5. Negative controls.** Pick a variable that cannot plausibly affect the outcome;
if the model says it does, you have confounding, and characterizing it describes the
missing variable.
→ *in our work:* the **shuffled connectome IS a negative control.** In next-step
prediction it "worked" as well as the real wiring (0.584 vs 0.585) → the prediction
rides on generic correlation/sparsity, a confound, not the circuit. In held-out-
neuron imputation it correctly underperformed (0.30 vs 0.47) → there the real
structure is load-bearing.

## Priors

- **Hierarchical structure is the highest-value prior available.** Groups → partial
  pooling gives a prior for free, defensible without domain expertise.
  → *in our work:* worms are grouped by lab/source and by individual — partial
  pooling across worms/labs is the obvious unused hierarchical prior.
- **Prior predictive simulation** (the step most skip): draw from the prior,
  simulate datasets, look at them. Impossible/absurd outcomes ⇒ the prior is wrong.
  → *in our work:* our synthetic generator *is* a prior-predictive simulation of the
  worm — and running it is precisely what revealed the sim-to-real gap.
- **Elicitation:** ask domain people about observables ("if this rose 10%, what
  happens?"), not coefficients; back out the parameter prior.
- **Meta-analytic effect sizes** from adjacent fields, shrunk toward zero for
  publication bias, are a legitimate prior scale.
  → *in our work:* the literature review (CC-LVM 0.3–0.5, eff-vs-struct 0.49) is
  exactly this — an external, shrinkable prior on achievable magnitudes.
- **Representation transfer:** pretrained encoders / adjacent-task models borrow
  features rather than derive them.
  → *in our work:* the JEPA / forecaster embeddings are this — borrowed structure
  from self-supervised pretraining.
- **Automated search, last:** symbolic regression, GBMs, interaction detection find
  things, but mostly sample artifacts. Use them to generate hypotheses you check
  against the structural story, not as a substitute for one.
  → *in our work:* the AlphaEvolve / MAP-Elites loop is the automated search; it
  *confirmed and tuned* the coupling idea we seeded rather than inventing it —
  exactly this caveat.

**General failure mode:** adding features **downstream** of the outcome instead of
upstream. Statistics can't distinguish them without experiments; rule them out by
reasoning about timing and mechanism.

## Do priors help with structure, or just prediction?

Both — but the structural help is **narrower than it looks and comes with a trap.**

**Where priors do genuine structural work**
- **Sparsity priors** (horseshoe, spike-and-slab) give posterior *inclusion
  probabilities* — "this edge is present with prob 0.8" is a structural claim.
- **Hierarchical variance:** if the between-group SD posterior piles up at zero, the
  grouping you assumed doesn't exist — a structural finding from a prior over structure.
- **GP kernel hyperparameters:** lengthscale posterior = the timescale the process
  varies on; periodic components = whether the periodicity is real.
- **Identification diagnostics (cheapest, most useful):** compare prior to posterior
  parameter by parameter. Where posterior = prior, the data carries no information
  and your statement is pure assumption. This tells you exactly where conclusions are
  load-bearing on belief vs evidence.
- **Partial identification:** for set-identified parameters, prior + data yields
  *bounds* — real information even when the point estimate inside isn't.

**Where they don't**
- A prior over parameters cannot certify the *structure containing them* is correct.
  Wrong DAG + well-calibrated prior = confident, well-behaved, wrong answer.
  Regularization **absorbs** misspecification, it doesn't detect it.
- **Markov equivalence** is the hard wall: some causal structures give identical
  observational distributions; no parameter prior distinguishes them. A prior over
  structures breaks the tie, but then the conclusion is your prior and won't update.
  Look at the *posterior over graphs*, not the MAP graph — it's flatter than people expect.

**The trap.** A strong prior improves prediction *while masking structural error*:
shrinkage buys out-of-sample performance by suppressing exactly the variance that
would have exposed a misspecified model. So **good held-out performance under strong
regularization is weak evidence for structure** — you've lost the diagnostic.

**The way around it.** Use the prior to make the model **fail loudly, not fit
quietly.** Fit both a tight structural prior and a vague one: where they agree, the
data is speaking; where they diverge, you're looking at your own assumptions. Prior
predictive checks do this before you see data — if simulated datasets don't resemble
anything the process could produce, the structure is already wrong.

### → Our stage-2 results ARE this argument, empirically

We fit, on real worm data, a **tight structural prior** (coupling constrained to the
real connectome), a **vague** one (dense coupling), and a **negative control**
(shuffled wiring):

| task | connectome (tight) | shuffled (control) | dense (vague) | reading |
|---|---|---|---|---|
| next-step Δ (observed neurons) | 0.585 | 0.584 | 0.604 | **all converge → the trap**: prediction is high but *non-diagnostic* of structure; the prior fit quietly |
| held-out-neuron imputation | 0.467 | 0.300 | 0.555 | **tight ≫ control → the data is speaking**: the real wiring is load-bearing |

- Next-step prediction is the **trap in action**: a strong connectome prior predicts
  as well as random wiring, so held-out accuracy there is *weak evidence for
  structure*. We nearly concluded the connectome was useless from it.
- The **way around** was to pick a task where a wrong prior fails loudly — held-out-
  neuron imputation — and there connectome vs shuffled **diverged** (+0.17, wins 84%).
  That divergence is the structural finding.
- Our earlier **struct_corr collapse** (0.68 synthetic → 0.11 real) is the same
  lesson from the other side: recovering structure *as a target* from activity fails,
  because observational correlation is Markov-equivalent-ish to many wirings; the
  connectome only pays off used *as a prior*, in the regime that makes it falsifiable.

**Operating rule going forward:** never read structure off held-out prediction under
a strong prior. Read it off (a) prior-vs-posterior / tight-vs-vague *divergence*, and
(b) negative-control (shuffled-structure) gaps on tasks designed so a wrong prior
must fail loudly.
