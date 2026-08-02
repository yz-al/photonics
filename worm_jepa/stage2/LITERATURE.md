# Stage 2 — literature review: does anyone else do this, and how do our numbers compare?

**Question:** who discovers mechanistic/dynamical models of neural activity that
are scored on *both* prediction and ground-truth connectivity recovery, and where
do our numbers (`pred_r2 ≈ 0.80` temporal / `0.20` single-frame; `struct_corr ≈
0.35`) sit relative to theirs?

**Big caveat up front:** our numbers are on a *synthetic toy connectome* with our
own metric definitions (held-out R² on the state-change Δ; `struct_corr` = Pearson
correlation of the model's effective operator off-diagonal vs `|W·Wᵀ|`). Published
work uses different organisms, real data, and different metrics. So this is
*contextual placement*, not a head-to-head leaderboard.

## The closest work — and it validates our whole approach

**"Discovering Mechanistic Models of Neural Activity: System Identification in an
in-silico Zebrafish"** (Lueckmann, Jain, Januszewski; arXiv 2602.04492) is almost
exactly our setup:
- **Method: LLM-based tree search** — the same AlphaEvolve-family LLM discovery we
  use, on an in-silico system with known ground truth.
- **Their two headline conclusions are our two findings, independently:**
  1. *"models exploit statistical shortcuts"* — i.e. good prediction without the
     right mechanism. **This is our pred_r2 ≠ struct_corr divergence.**
  2. *"structural priors prove essential for ... recovery of interpretable
     mechanistic models."* **This is exactly why our coupling term lifted
     structure** (0.248 → 0.358) while prediction stayed flat.

Same method family, different organism, same two lessons. Strong external
validation that (a) the LLM-evolutionary approach is a real research direction and
(b) our prediction-vs-structure story is the right frame.

## Prediction — held-out neural activity

| work | system / data | metric | number |
|---|---|---|---|
| **ours** (synthetic, temporal 3-frame) | toy connectome, 48 neurons | R² on Δ | **~0.80** |
| **ours** (synthetic, single frame) | " | R² on Δ | 0.20 |
| **ours** (real worm, GRU) | 247 neurons, next-step | R² (next-step) | 0.92 |
| CC-LVM (Janelia) | **real** C. elegans, held-out neurons *and* worms | R² | **0.3–0.5** |
| POCO | **real** C. elegans (Zimmer/Flavell) | skill score vs copy | 0.36–0.42 |
| POCO | real zebrafish / mice | skill score | 0.42–0.53 |

**Honest placement:** our synthetic 0.80 is on a much *easier* setting (controlled
generator, temporal input, all neurons observed) than CC-LVM's 0.3–0.5, which
predicts *withheld neurons and worms on real data*. Our real-data GRU (0.92) is a
different, easier task (next-step with all neurons observed), not held-out-neuron.
**We should not claim to beat them** — different difficulty. Adjusted for task,
our numbers are consistent with the field.

## Structure — connectivity recovery vs ground truth

| work | metric | number |
|---|---|---|
| **ours** (synthetic) | Pearson r, effective operator vs `\|W·Wᵀ\|` | **~0.35** |
| pairwise-correlation FC (generic) | accuracy vs true FC | ~0.30 |
| effective-vs-structural connectivity (Hopf/fMRI, canonical) | r | **0.49** |
| SynapsNet | edge-recovery accuracy, **sparse** synthetic FC | >0.80 |
| derivative/LCC methods | reliability on C. elegans | "most reliable" (no single r) |

**Honest placement:** our `struct_corr ≈ 0.35` sits **between** generic
pairwise-correlation FC (~0.30) and the canonical effective-vs-structural
correlation (~0.49), and **below** purpose-built edge-inference nets like SynapsNet
(>0.80). But SynapsNet's >0.80 is *edge-classification accuracy on a sparse,
direct-connectivity* ground truth — a different metric and a different kind of
connectome than ours. Our `precision/partial-correlation` test already showed our
synthetic ground truth is a *shared-latent (functional)* coupling with **no direct
edges**, so the sparse-edge-inference regime (where SynapsNet shines) doesn't apply
here. Against the most comparable figure — effective-vs-structural r ≈ 0.49 — our
0.35 is in the same ballpark but below, consistent with using a generic
discovered-operator rather than a purpose-built connectivity method.

## Takeaways

1. **We are not alone**, and the closest work (in-silico zebrafish, LLM tree
   search) independently reproduces our two core findings (shortcuts break
   mechanism; structural priors are needed). That's the strongest external
   signal here.
2. **Prediction**: numbers only compare after adjusting for task difficulty; our
   controlled synthetic scores are consistent with, not superior to, real-data
   work (CC-LVM 0.3–0.5, POCO 0.36–0.53 skill).
3. **Structure**: our ~0.35 is in the classic effective-vs-structural ballpark
   (0.49) and above generic correlation FC (~0.30); specialized edge-inference
   methods score higher but on sparse-direct-edge connectomes unlike ours.
4. **Gap we could close**: the field's best structure recovery uses *purpose-built*
   connectivity inference (SynapsNet) or *connectome-constrained* generative
   models (CC-LVM). Our next lever — feeding stage 2 the stage-1 forecaster's
   learned connectivity — moves us toward the "structural prior" that both the
   zebrafish paper and CC-LVM find essential.

## Sources
- [Discovering Mechanistic Models of Neural Activity: in-silico Zebrafish (arXiv 2602.04492)](https://arxiv.org/abs/2602.04492)
- [Connectome-constrained Latent Variable Model of Whole-Brain Neural Activity (Janelia / OpenReview)](https://openreview.net/forum?id=CJzi3dRlJE-)
- [Bridging the gap between the connectome and whole-brain activity in C. elegans (bioRxiv)](https://www.biorxiv.org/content/10.1101/2024.09.22.614271v1)
- [POCO: Scalable Neural Forecasting through Population Conditioning (arXiv 2506.14957)](https://arxiv.org/html/2506.14957)
- [SynapsNet: Enhancing Neuronal Population Dynamics Modeling via Learning Functional Connectivity (arXiv 2411.08221)](https://arxiv.org/pdf/2411.08221)
- [Comparison of derivative-based and correlation-based methods to estimate effective connectivity (Sci Reports 2025)](https://www.nature.com/articles/s41598-025-88596-y)
- [Ensemble learning and ground-truth validation of synaptic connectivity from spike trains (PLOS Comp Biol)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011964)
