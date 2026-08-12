# Context of Use — Alzheimer's neuroactive-compound prioritization (NAM)

The regulatory claim we commit to, matched honestly to what the platform actually
does and validated on held-out data (see `predict.py`).

## The context of use (COU)

> **Predict, for a small molecule with a known or predicted molecular target, its
> effect on C. elegans neural-circuit activity and locomotor/chemotaxis behavior,
> to rank-order and triage neuroactive candidates entering a C. elegans Alzheimer's
> efficacy screen — reducing the number of compounds that require wet-lab and
> animal testing.**

- **Decision supported:** internal preclinical **prioritization / go–no-go** on
  which compounds advance to the phenotypic worm assay and then to mouse testing.
- **What it is NOT:** it does **not** predict human Alzheimer's efficacy, and it
  does not replace the phenotypic assay. It is a *mechanistic filter in front of*
  the assay. Claiming worm→human efficacy is neither established nor qualifiable,
  and reviewers know it (worms have no BACE/β-secretase, no BBB; muscle-Aβ models
  are proteostasis assays). We state this limitation up front.

## Why this COU fits the platform (and the standard one does not)

The field's workhorse screen is **paralysis in GMC101 / CL4176** — full-length or
truncated Aβ1-42 expressed in **body-wall muscle**. That is a **proteostasis**
endpoint; our platform models the **nervous system**, not muscle protein
aggregation, so predicting paralysis directly is out of domain. The honest fit:

- **Neuronal-Aβ model, behavioral endpoint.** CL2355 (pan-neuronal Aβ, *snb-1*)
  produces **chemotaxis and associative-learning deficits** from *neuronal*
  dysfunction — our domain. Our validated `compound_response` pipeline predicts a
  compound's effect on the neural circuit and the resulting behavioral direction
  (8/8 on known neuroactive compounds; `compound_response.json`).
- **As a prioritization filter on ANY worm AD screen.** Given a compound's target
  (ChEMBL/WormBase) → the neurons that express it (CeNGEN) → the circuit-level
  effect (our stack) → predicted behavioral direction. That ranks candidates and
  flags predicted off-target neural liabilities before the wet assay — regardless
  of whether the downstream endpoint is paralysis or chemotaxis.

## Model + endpoints (established)

| model | Aβ tissue | endpoint we tie to | role |
|---|---|---|---|
| **CL2355** | pan-neuronal | chemotaxis index / associative learning | primary (neural fit) |
| GMC101 | muscle | time-to-paralysis | secondary (as a filter target only) |
| CL4176 | muscle, temp-inducible | % paralyzed @ 24–28 h | rapid confirmation |

## Reference compound set (self-assembled — no regulator-endorsed set exists)

Positive and negative controls with published worm-AD activity; the benchmark must
force the model to separate true anti-Aβ mechanism from **generic antioxidant/AChE
confounds** (the loud-negative-control rule from `METHODOLOGY.md`):

- **Actives:** PBT2 (GMC101, the same-strain gold positive), galantamine (CL4176),
  reserpine (CL2006/CL2355), curcumin / EGCG / ferulic acid (with the antioxidant
  caveat), metformin / lithium (GRU102 HTS hits).
- **Inactives (clean negatives):** thioflavin T (failed to rescue GRU102), vehicle/DMSO.
- **Required discrimination:** rank PBT2 active AND thioflavin T inactive AND
  down-weight generic antioxidants — if a wrong prior doesn't fail here, the
  benchmark is worthless.

## What "more" this specific COU needs (the gap to qualification)

1. **Assemble + freeze the reference chemical set** (actives/inactives above),
   with pre-registered ranking thresholds. *(computational — we can start now.)*
2. **Prospective, blinded concordance study:** predict, then a partner lab runs
   the CL2355 chemotaxis / GMC101 paralysis assay; report **sensitivity,
   specificity, ROC**, not R². *(needs the wet-lab partner.)*
3. **ASME V&V 40 credibility package:** model form, verification, validation,
   uncertainty, domain-of-applicability. *(documentation.)*
4. **Domain of applicability stated explicitly** — compound classes / targets in
   vs out of scope.
5. **Reproducibility + inter-lab transferability** (a ring trial; CITP is the
   natural multi-lab anchor).
6. **AOP linkage** — tie predictions to a proteostasis/neurodegeneration Adverse
   Outcome Pathway; our connectome-vs-shuffled controls are the "recovered
   mechanism, not fit" evidence a NAM reviewer demands.

## Regulatory + funding pathway

- **FDA door = ISTAND** (now a permanent DDT qualification program). No precedent
  for an organism/in-silico AD model — so open it only **after** concordance data,
  via a pre-submission / Letter of Intent for a *fit-for-purpose triage* COU (never
  an efficacy claim). FDA's Animal-Model qualification is Animal-Rule-only and does
  NOT fit; AD COU precedent to date is biomarkers, not preclinical screens.
- **Funding = NIA + Hevolution + STTR.** NIA Milestone 7.B explicitly wants
  computational prediction / repurposing; PAR-25-297 (ADDP U01), PAR-25-331/332.
  Hevolution HF-GRO for the geroscience framing. **Note:** aging is *not* an FDA
  indication — geroscience is a funding angle, not a qualification angle.
- **Validation partner = a CL2355/GMC101 lab or the Caenorhabditis Intervention
  Testing Program (CITP)** — NIA-funded, multi-lab, built for exactly this kind of
  standardized worm screening (supplies the STTR ≥30% wet-lab work).

## One-line summary

A **computational mechanistic-prioritization NAM** that ranks neuroactive compounds
for a C. elegans Alzheimer's screen by predicting their neural-circuit and
behavioral effects — honest triage, not efficacy; fundable now (NIA/Hevolution +
STTR), FDA-qualifiable later (ISTAND) once a blinded concordance study against a
reference chemical set is in hand.
