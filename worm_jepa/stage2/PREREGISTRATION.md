# Pre-registered analysis plan — C. elegans mechanistic-prioritization NAM

The point of this document is to remove analytic degrees of freedom. It fixes, *in
advance of any new data*, the hypotheses, metrics, decision thresholds, controls, and
falsification criteria for the platform's validation — so a reviewer can see the
results were not tuned to the conclusion. It has two parts, and the distinction is the
whole point:

- **Part I — Exploratory (already run).** The in-silico results in this repo. These
  are *hypothesis-generating and pipeline-defining*. You cannot pre-register an
  analysis you have already seen, so we do not pretend to; instead we **freeze** the
  analysis pipeline and its thresholds here, so they become the confirmatory rule.
- **Part II — Confirmatory (not yet run).** The forward-looking tests: (a) applying
  the **frozen** pipeline blind to *new* public datasets, and (b) the prospective,
  blinded **wet-lab concordance study**. These are the tests whose endpoints and
  pass/fail thresholds are genuinely fixed before the data exist.

Frozen reference: the committed artifacts at this commit — `rigor.json`,
`external_validation.json`, `external_replication.json`, `pipeline_evidence.json`,
`l3/l4/perturbation/imputation` JSONs — are the locked record. `test_platform.py`
enforces every threshold below in CI.

---

## Part I — Frozen analysis pipeline and thresholds (the confirmatory rule)

Every claim is scored at its natural experimental unit with a **paired one-sided test
+ bootstrap 95% CI + effect size**, corrected for multiple comparisons (Holm), and
must hold across a hyperparameter sweep. A claim passes **only if all four hold**:

| gate | unit | estimator | pre-specified PASS threshold (frozen) |
|---|---|---|---|
| L1 imputation | neuron | connectome − shuffled r | diff>0, CI excludes 0, Holm p<0.05, sign-stable over `min_cols∈{6,8,12}` |
| L2 perturbation | perturbation | connectome − shuffled r | diff>0, CI excludes 0, Holm p<0.05, sign-stable over `g∈{0.5..0.8}` |
| L2* external OOD | animal | connectome − shuffled r | diff>0, CI excludes 0, Holm p<0.05, sign-stable over `train_frac∈{0.5,0.6,0.7}` |
| L3 next-step | worm | persistence − model MSE | diff>0, CI excludes 0, Holm p<0.05, sign-stable over seed |
| L4 behavior | worm | ours − ridge R² (vel, curv) | diff>0, CI excludes 0, Holm p<0.05, sign-stable over `train_frac` |

**Directionality is pre-specified** (connectome > shuffled; model < persistence; ours >
ridge). Two-sided results in the wrong direction count as failures, not "trends."

**The negative control is mandatory, not optional.** Each gate is paired with a
control that must *fail*: shuffled-wiring (structural null), temporal-shuffle
(behavior), persistence (forecasting), shuffled-valence (bridge). A gate with a
significant positive result but a control that also "passes" is discarded.

**Domain of applicability (frozen).** In scope: neural-circuit and locomotor/chemotaxis
prediction for perturbations whose molecular target is a neural receptor / ion channel /
transporter expressed in the CeNGEN atlas (see `worm_twin.RECEPTOR_PREFIXES`). Out of
scope, the platform must **abstain**: proteostasis/aggregation, metabolic, and
transcription-factor targets (handled separately or not at all). Predicting *human*
efficacy is out of scope categorically.

---

## Part II — Confirmatory tests (endpoints fixed before the data)

### II.a  Blind application of the frozen pipeline to new public datasets

**Rule:** any new whole-brain dataset is run through the **unchanged** pipeline and
thresholds in Part I. No re-tuning, no per-dataset hyperparameters, no metric swaps.

**Pre-registered prediction (made before running):** connectome > shuffled at the frozen
thresholds on each new cohort. **Demonstrated once already:** the frozen imputation
pipeline, applied blind to the chemosensory **000981** cohort (`external_replication.py`),
passed — connectome gap 0.020, 95% CI [0.011, 0.029], p=3.8e-5 — replicating the Flavell
**000776** result (gap 0.097, CI [0.079, 0.115], p=5.1e-11). The gap is smaller on 000981
because dense recording raises the global-brain-state floor; this was predicted, not
explained after the fact. Future cohorts (e.g. Kato, Uzel) enter the same way.

### II.b  Prospective, blinded wet-lab concordance study (the qualification test)

This is the test no public dataset can substitute for, because it requires a
measurement that does not yet exist. Endpoints and thresholds fixed here, before any
compound is run.

- **Context of use:** rank-order neuroactive candidates for a C. elegans AD screen; a
  triage filter *in front of* the phenotypic assay (see `CONTEXT_OF_USE.md`). **Not** an
  efficacy claim.
- **Design:** the platform predicts each compound's neural-circuit/behavioral direction
  **and its rank** *before* the assay. A partner lab (CL2355 chemotaxis / GMC101
  paralysis; CITP as the multi-lab anchor) runs the assay **blind** to the predictions.
- **Reference set (frozen before unblinding):** actives PBT2, galantamine, reserpine,
  curcumin/EGCG/ferulic acid (antioxidant caveat), metformin, lithium; inactives
  thioflavin T, vehicle/DMSO. Required discrimination: PBT2 active AND thioflavin T
  inactive AND generic antioxidants down-weighted.
- **Primary endpoint (pre-specified):** binary concordance (predicted active/inactive vs
  assay hit/non-hit) reported as **sensitivity, specificity, and ROC-AUC** — *not* R².
- **Pre-specified success criterion:** ROC-AUC point estimate ≥ 0.75 **and** its lower
  95% CI bound > 0.5; sensitivity ≥ 0.70 at a pre-fixed operating point. Missing either
  is a negative result and will be reported as such.
- **Power / sample size (pre-specified):** to detect AUC 0.75 vs the 0.5 null at 80%
  power, two-sided α=0.05, balanced classes (Hanley–McNeil approximation) →
  **≈ 50 compounds (≈ 25 active / 25 inactive)** minimum; final n set with the partner
  lab before enrollment. Analysis is intention-to-screen: every enrolled compound is
  scored, none dropped post hoc.
- **Blinding & pre-registration of predictions:** the platform's per-compound predictions
  and ranks are hashed and time-stamped (committed) **before** the assay is run;
  unblinding happens only after assay readout. This is what makes the result
  prospective rather than a retrospective fit.

---

## Falsification criteria (what would make us say the model is wrong)

- Any Part I gate whose 95% CI includes zero, or whose effect flips sign under its
  hyperparameter sweep, on the frozen data → that gate is **not** validated.
- A new public cohort where connectome does **not** beat shuffled at the frozen
  thresholds → the OOD claim does **not** generalize; report it.
- Wet-lab AUC lower-CI ≤ 0.5, or thioflavin T ranked active, or generic antioxidants
  not separated from true anti-Aβ mechanism → the COU fails and is withdrawn.

## Amendment policy

Any change to a metric, threshold, direction, or the domain of applicability after data
are seen is logged as a dated **amendment** in this file with its rationale, and the
affected result is demoted to exploratory until re-confirmed on fresh data. Silent
changes are disallowed; `test_platform.py` is the enforcement record.

## One line

Thresholds, directions, controls, and endpoints are fixed here in advance;
`test_platform.py` enforces them; the only thing left that a computer cannot decide is
the blinded wet-lab AUC — which is exactly why that study, and not more public data, is
the qualification step.
