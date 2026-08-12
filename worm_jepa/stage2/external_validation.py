"""
external_validation.py -- the "untrained data" test. Predict a whole-brain dataset
the platform was NEVER built on, with a model that has ZERO parameters fit to it.

Why this benchmark exists
-------------------------
Every other benchmark here is scored on the dataset whose atlas we were built from
(the Randi/funatlas response matrix). A fair reviewer -- and an FDA reviewer -- will
ask the obvious question the user asked: *"why do I need a wet lab; can't you just
take some data you never trained on and predict it?"* This module does exactly that,
in silico, and shows both what that proves and where it stops.

The frozen model has independent provenance from the test data
--------------------------------------------------------------
- MODEL: the C. elegans anatomical connectome (Cook/White/Witvliet, via c302
  SpreadsheetDataReader). Fixed wiring. No free parameters fit to the test set --
  the predictor for a held-out neuron is the ANATOMICAL-weight average of its
  synaptic partners' activity. One design choice (1-hop, leak-free), nothing fit.
- TEST DATA: the Flavell/Atanas 2023 whole-brain spontaneous-activity recordings
  (38 animals, ~85 identified neurons each) -- a DIFFERENT lab, different microscope,
  freely-moving vs immobilized, spontaneous vs optogenetic. The connectome was built
  from electron microscopy of entirely different animals. Nothing here was used to
  build, fit, or tune any part of the platform.

The task (connectome-constrained held-out-neuron imputation, spontaneous data)
------------------------------------------------------------------------------
For each animal, hold out one identified neuron and predict its activity time-series
as the connectome-weighted average of its synaptic partners' activity. Weights are
picked on a TRAIN window (first 60% of the recording); the score is the correlation
on a held-out TEST window (last 40%). Three predictors, same protocol:

  connectome   partners + weights = the real anatomical wiring
  shuffled     identities permuted -> same sparsity, wrong wiring (structural null)
  functional   the k best-correlated neurons, least-squares weights on train
               (a no-connectome SOFT CEILING: how predictable is the neuron at all)

Why the shuffled control is not optional: whole-brain activity is dominated by global
brain-state, so *any* set of neurons partially predicts *any* neuron. The shuffled
connectome inherits that global signal and still scores > 0. Structure is recovered
only if the REAL connectome beats it -- which is the whole "data speaking" test.

Result (see external_validation.json), pooled over 38 unseen animals:
  connectome   r ~ 0.21   -- anatomy predicts held-out neurons in data it never saw
  shuffled     r ~ 0.12   -- global brain-state floor (wrong wiring)
  functional   r ~ 0.30   -- best-case in-animal predictability (soft ceiling)
  -> connectome >> shuffled (structure transfers across datasets), reaching a large
     fraction of the functional ceiling, with zero parameters fit to these animals.

What this proves -- and the honest boundary (why a wet lab is still needed)
---------------------------------------------------------------------------
PROVES: the mechanism generalizes out of distribution. This is real external
validation and it needs no wet lab -- the strongest computational case we can make.
DOES NOT PROVE: prospective effect prediction for a NEW compound. This dataset is
untreated spontaneous activity; it contains no perturbation the platform is meant to
triage. A regulator qualifies a *decision*, and the decision (does compound X shift
the phenotype) requires a measurement that no existing public dataset contains --
i.e. a prospective, blinded assay. External data closes the "does the model
generalize" gap; only the wet lab closes the "is the prediction right on something
nobody has measured yet" gap. See CONTEXT_OF_USE.md step 2.

    python external_validation.py     # writes external_validation.json
Self-contained: reads external_flavell.npz (distilled from DANDI:000776); c302 only.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "external_flavell.npz")


def load_connectome():
    """Symmetric name-keyed synaptic weight (chemical 'Send'); the frozen anatomy."""
    import c302
    from collections import defaultdict
    _, conns = c302.get_cell_names_and_connection("SpreadsheetDataReader")
    W = defaultdict(float)
    for c in conns:
        if c.syntype == "Send":
            W[(c.pre_cell, c.post_cell)] += c.number
    def wsym(a, b):
        return W[(a, b)] + W[(b, a)]
    return wsym


def _impute_connectome(Xtr, Xte, names, wsym, min_partners=1):
    """Held-out-neuron imputation with FIXED anatomical weights. Weights come from the
    connectome (no fitting); score on the test window."""
    from scipy.stats import pearsonr
    N = len(names)
    cs = []
    for i in range(N):
        w = np.array([wsym(names[i], names[j]) if j != i else 0.0 for j in range(N)])
        if (w > 0).sum() < min_partners or w.sum() <= 0:
            continue
        w = w / w.sum()
        pred = Xte @ w
        if np.std(pred) < 1e-9:
            continue
        cs.append(float(pearsonr(pred, Xte[:, i])[0]))
    return cs


def _impute_functional(Xtr, Xte, k=8):
    """Soft ceiling: pick each neuron's k best-correlated partners on TRAIN, fit their
    linear weights on TRAIN, score on TEST. Uses no connectome."""
    from scipy.stats import pearsonr
    N = Xtr.shape[1]
    C = np.corrcoef(Xtr.T)
    cs = []
    for i in range(N):
        c = C[i].copy(); c[i] = -np.inf
        part = np.argsort(c)[::-1][:k]
        A = Xtr[:, part]
        coef, *_ = np.linalg.lstsq(A, Xtr[:, i], rcond=None)
        pred = Xte[:, part] @ coef
        if np.std(pred) < 1e-9:
            continue
        cs.append(float(pearsonr(pred, Xte[:, i])[0]))
    return cs


def run(seed=0, train_frac=0.6):
    d = np.load(DATA, allow_pickle=True)
    acts, names_all = d["acts"], d["names"]
    wsym = load_connectome()
    rng = np.random.RandomState(seed)
    con, sh, fun, nmap = [], [], [], []
    for X, names in zip(acts, names_all):
        names = [str(n) for n in names]
        X = np.asarray(X, float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)          # z-score each neuron
        cut = int(len(X) * train_frac)
        Xtr, Xte = X[:cut], X[cut:]
        if Xte.shape[0] < 30 or X.shape[1] < 15:
            continue
        c = _impute_connectome(Xtr, Xte, names, wsym)
        # shuffled: permute which anatomical identity each column is assigned
        perm = rng.permutation(len(names)); sn = [names[p] for p in perm]
        s = _impute_connectome(Xtr, Xte, sn, wsym)
        f = _impute_functional(Xtr, Xte)
        if c and s and f:
            con.append(np.mean(c)); sh.append(np.mean(s)); fun.append(np.mean(f))
            nmap.append(X.shape[1])
    con_m, sh_m, fun_m = float(np.mean(con)), float(np.mean(sh)), float(np.mean(fun))
    # paired significance of connectome > shuffled across animals
    from scipy.stats import wilcoxon
    try:
        p = float(wilcoxon(con, sh, alternative="greater")[1])
    except Exception:
        p = float("nan")
    return {
        "dataset": "Flavell/Atanas 2023 whole-brain (DANDI:000776) -- NEVER used to build the model",
        "model": "frozen anatomical connectome (c302 SpreadsheetDataReader), 0 params fit to this data",
        "task": "held-out-neuron imputation, spontaneous activity, train/test temporal split",
        "n_animals": len(con),
        "mean_mapped_neurons": round(float(np.mean(nmap)), 1),
        "connectome": {"r": round(con_m, 4), "pct_of_functional_ceiling": round(con_m / fun_m * 100, 1)},
        "shuffled": {"r": round(sh_m, 4), "pct_of_functional_ceiling": round(sh_m / fun_m * 100, 1)},
        "functional_ceiling": {"r": round(fun_m, 4)},
        "connectome_minus_shuffled": round(con_m - sh_m, 4),
        "wilcoxon_p_connectome_gt_shuffled": p,
        "structure_transfers_out_of_distribution": bool(con_m > sh_m + 0.02 and p < 0.01),
    }


if __name__ == "__main__":
    r = run()
    print("=== EXTERNAL VALIDATION -- predict a dataset the model NEVER saw ===")
    print("  test data : %s" % r["dataset"])
    print("  model     : %s" % r["model"])
    print("  %d unseen animals, ~%.0f mapped neurons each\n" % (r["n_animals"], r["mean_mapped_neurons"]))
    print("  connectome   r=%.3f  = %.0f%% of functional ceiling"
          % (r["connectome"]["r"], r["connectome"]["pct_of_functional_ceiling"]))
    print("  shuffled     r=%.3f  = %.0f%% of functional ceiling  (global brain-state floor)"
          % (r["shuffled"]["r"], r["shuffled"]["pct_of_functional_ceiling"]))
    print("  functional   r=%.3f  (soft ceiling: best in-animal predictability)"
          % r["functional_ceiling"]["r"])
    print("  connectome - shuffled = %.3f  (Wilcoxon p=%.1e)"
          % (r["connectome_minus_shuffled"], r["wilcoxon_p_connectome_gt_shuffled"]))
    print("  -> %s" % ("STRUCTURE TRANSFERS OUT OF DISTRIBUTION (untrained data predicted, zero fit params)"
                       if r["structure_transfers_out_of_distribution"] else "n.s."))
    with open(os.path.join(HERE, "external_validation.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("\nwrote external_validation.json")
