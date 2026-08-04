# Double black box with asymptotic + inverse-problem scaffolding

Method for a **two-way opacity**: a real system observed only indirectly, AND a
mechanistic model of it that is incomplete in ways we cannot see. Fit a flexible model
to the **gap** between mechanism and reality, extract structure from that fit, push the
structure back into the mechanistic framework as an explicit hypothesis about what was
missing. The scaffolding below makes each step constrained instead of arbitrary.

**Order is load-bearing.** Several stages produce uninterpretable output if run before
the stage above them. Do not reorder for convenience. Stage 5 (search) without Stages 1–4
is unconstrained search over an arbitrary basis — the original failure mode.

## Stage 0 — FEASIBILITY GATE (before anything)
- **0a Identifiability (profile likelihood).** Analytic pass: do params appear only in
  fixed combinations? Count independent observables vs free params. Profile pass: fix each
  param off optimum, **re-optimise all others**, sweep, plot loss vs param. Rises both
  sides = identifiable; flat = structurally non-identifiable (point estimate is arbitrary
  on a manifold); one side flat = practically non-identifiable (fix with data). Report
  co-moving compensators — that combination is identifiable even if its parts are not.
  **GATE: structurally non-identifiable → STOP, go design a new observable (Stage 6/7).**
- **0b Learnability ceiling.** Label ceiling (annotation agreement in the scoring metric);
  small-subset overfit (over-param model → train loss ~0 on a few hundred, else input
  lacks signal / labels inconsistent); label shuffle (must collapse to chance, else
  leakage). **GATE: either fails → downstream numbers uninterpretable.**

## Stage 1 — NONDIMENSIONALISE the mechanistic model
Governing equation with constants explicit → characteristic scales from the constants →
substitute, divide by the largest group → dimensionless groups remain. Report each group,
its **numerical value**, and its physical meaning as a ratio of competing effects. The
group list IS the ranked list of what the model neglects; a group near the residual size
is a candidate answer *before* any search. Collapses param count; improves conditioning.

## Stage 2 — DOMINANT BALANCE / regime structure
Set ε=0: degenerate? order drops / blow-up → **singular** perturbation. For every pairing
of terms, assume two balance, solve for the unknown's scale, and **check with numbers**
that dropped terms are really smaller. Discard contradictory balances. Report the regime
diagram / distinguished limits.

## Stage 3 — RESIDUAL SIGNATURE (not size)
- **3a exponent:** sweep a param that changes ε; slope of log(residual) vs log(ε) = p. A
  term entering at εᵖ gives residual εᵖ — an integer fingerprint that rules out candidates.
- **3b localisation:** smooth = regular; width ε^½/ε = **boundary layer**; linear-in-time =
  **secular** (multiple scales).
- **3c stability:** perturb data (fresh noise / bootstrap); if structure moves a lot we are
  ill-posed (continuity failing) — not an optimiser bug.
Report the PAIR (exponent, localisation) + stability.

## Stage 4 — CONSTRAINED CANDIDATE SET
Library of admissible missing terms: only the order in ε implied by p; only matching
localisation; dimensionally consistent; expressible in the model's own variables. ~10
candidates, not a function space. State the **regulariser explicitly** (usually sparsity)
as a modelling assumption; pick strength by discrepancy principle (known noise) or L-curve.

## Stage 5 — SEARCH (AlphaEvolve / PySR / SINDy) — ONLY here
Search the restricted set. Prefer program search (AlphaEvolve) when the missing physics is
regime-dependent (expression trees fit a smooth compromise across regimes). Encode Stage-3
constraints as **hard rejections in the evaluator** (prune, don't seed). Score on
**forward-simulated trajectory error**, not pointwise derivatives. Hold out a **parameter
regime**, not random points. Report separately: (i) prediction error; (ii)
**term-extractability** (single additive dimensionally-consistent term under a stated
complexity limit). Wins (i) but fails (ii) = a better black box = the failure we avoid.

## Stage 6 — ASYMPTOTIC CONSISTENCY (after search)
Insert the recovered term; **redo Stages 1–2 from scratch** with it present. Dimensionally
coherent? In its regime, is it actually order-(residual) or asymptotically negligible?
Repairs the failing region without breaking the outer solution? Predicts the scaling of
something we did **not** fit — go measure that (strongest test). Genuine physics passes;
a residual-fitter fails ≥1, loudly.

## Stage 7 — DESIGN NEXT MEASUREMENT, then LOOP
Don't fit harder; compute where surviving candidates disagree and measure there (commit in
advance to what each outcome falsifies). Formal: maximise expected information gain over
design d. Single model: E-optimality (max the smallest Fisher eigenvalue) = the flattest
Stage-0a profile. No experiment separates them → Stage 0a again: new observable needed.
Loop: updated model → new scales → return to Stage 1.

## Standing rules
Order is load-bearing. Never report a param on a flat profile. Never conclude "needs more
data" from a flat learning curve. Negatives are findings. **Never blend competing candidate
models** (destroys the discriminating structure; mixing weights become a new flat
direction) — combine only for prediction, never for mechanism discovery. Winner still
leaves structured residual → candidate set incomplete, return to Stage 4.

---

## Mapping to THIS project (honest applicability record)

The equation-based stages (1, 2, 3a-exponent, 4, 6) require a **mechanistic model expressed
as a governing equation with physical constants and a small parameter ε**. That gates the
target choice:

- **Worm connectome dynamics** — real system: measured neural activity; mechanistic model:
  connectome-coupled dynamics (a governing equation); gap: prediction residual; score:
  forward-simulated **trajectory** error (Stage 5's required objective). **Glove fit** for
  all seven stages.
- **EM segmentation affinity model** — a U-Net is **not** a governing equation: no physical
  constants, no dimensionless groups, no ε, no forward-simulated trajectory. Stages 1, 2,
  4, 6 do **not** apply as written. Only Stage 3 (residual signature) and Stage 0 map, and
  only by analogy (ε≈resolution/SNR; "boundary layer"≈membranes). Applying Stages 1–6 here
  is the misapplication the method warns against.

### Stage 0 status (already partly executed on EM segmentation)
- **0b learnability ceiling: PASSED.** Overfit gate: over-param net → train affinity acc
  0.99, train VOI 0.14 (signal present, fittable). Label shuffle: collapsed to VOI 4.0 /
  ERL 0.4 (no leakage). Label ceiling (corrected in-plane proxy): VOI floor ~0.2 vs
  achieved ~1.87 → not label-limited, large headroom. **Gate satisfied.**
- **0a identifiability: NOT YET RUNNABLE** — profile likelihood needs a *parametric*
  mechanistic model with few recoverable parameters to profile. A U-Net (millions of
  weights) is not that object. This is the open item, and it is why the target model must
  be named before Stage 0a / Stage 1 can run.

**OPEN DECISION (load-bearing): which mechanistic model does this scaffold onto?** Until
that is fixed, Stages 0a and 1+ cannot be run without violating the method's own order rule.
