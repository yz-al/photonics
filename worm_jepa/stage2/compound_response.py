"""
compound_response.py -- predict a compound's behavioral effect FROM THE COMPOUND.

This closes the gap between what the stack currently does (predict activity/behavior
from imaging) and what a medical funder pays for (predict the effect of a chemical
perturbation). The chain runs the double black box from a molecule:

  compound
    -> molecular target gene            (curated pharmacology: ChEMBL / WormBase)
    -> neurons expressing that target   (CeNGEN single-cell transcriptome, L0)
    -> position in the command/motor circuit  (our L1-L4 stack)
    -> predicted behavioral direction
  validated against known C. elegans phenotypes.

What is CURATED vs PREDICTED (honesty matters for a grant)
----------------------------------------------------------
CURATED INPUTS (known, from pharmacology databases): the target gene, whether the
receptor is excitatory or inhibitory, and whether the compound is an agonist or
antagonist. These are established facts about each molecule.
PREDICTED (the platform's contribution): WHICH neurons the compound acts on
(CeNGEN expression), those neurons' role in the locomotor circuit, and the
resulting behavioral direction. That is a neuron-level mechanistic prediction, not
a black-box QSAR -- the differentiator for a New Approach Methodology.

Result (see compound_response.json): 8/8 directional predictions correct on a panel
of neuroactive compounds with established worm phenotypes.

HONEST BOUNDARY (the CeNGEN-shuffle control makes this explicit): shuffling the
gene->neuron map barely drops accuracy (~91%), because the up/down DIRECTION is
carried mostly by the curated receptor polarity + agonism -- known pharmacology,
not the model. What CeNGEN genuinely contributes is correct NEURON-LEVEL
LOCALIZATION (acr-16 -> AVA, mod-1 -> AVE/PVC, avr-15 -> RIB/RME are all
biologically right), which a coarse up/down metric cannot reward. Demonstrating
that localization improves prediction needs a LOCATION-SENSITIVE task -- e.g.
predicting which behavioral module (reversal vs forward vs feeding) a compound
perturbs, where two compounds hitting the same receptor in different neurons
diverge. That task, validated against partner wet-lab calcium/behavior data on
held-out compounds, is exactly the Phase I STTR deliverable this de-risks: the
pathway runs end to end today; Phase I makes the localization load-bearing and
quantitative.

    python compound_response.py     # writes compound_response.json
Requires wormneuroatlas (ships CeNGEN cengen.h5).
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

# Command/motor circuit roles (CeNGEN uses neuron CLASS names, e.g. AVA not AVAL/AVAR).
REVERSAL = {"AVA", "AVE", "AVD", "AIB"}
FORWARD = {"AVB", "PVC"}
MOTOR = {"VD_DD", "VA", "VB", "DA", "DB", "RME", "AS", "RIM"}

# Panel: (compound, target gene, receptor polarity, agonist(+1)/antagonist(-1), known locomotor effect).
# Target gene + polarity + agonism are curated pharmacology; the behavioral effect is the ground truth.
PANEL = [
    ("nicotine",   "acr-16", "exc", +1, "increase"),   # nAChR agonist -> stimulation
    ("levamisole", "unc-29", "exc", +1, "increase"),   # nAChR agonist -> hypercontraction
    ("serotonin",  "mod-1",  "inh", +1, "decrease"),   # 5-HT-gated Cl agonist -> slowing
    ("fluoxetine", "mod-1",  "inh", +1, "decrease"),   # potentiates 5-HT -> slowing
    ("muscimol",   "unc-49", "inh", +1, "decrease"),   # GABA-A agonist -> paralysis
    ("ivermectin", "avr-15", "inh", +1, "decrease"),   # GluCl agonist -> paralysis
    ("PTZ",        "unc-49", "inh", -1, "increase"),   # GABA-A ANTAGONIST -> convulsion
    ("dopamine",   "dop-3",  "inh", +1, "decrease"),   # D2-like Gi agonist -> basal slowing
]


def load_cengen():
    import h5py, wormneuroatlas
    p = os.path.join(os.path.dirname(wormneuroatlas.__file__), "data", "cengen.h5")
    with h5py.File(p, "r") as h:
        genes = [g.decode() if isinstance(g, bytes) else str(g) for g in h["gene_names_th2"][:]]
        neurons = [n.decode() if isinstance(n, bytes) else str(n) for n in h["neuron_ids"][:]]
        tpm = h["tpm_th2"][:]
    return genes, neurons, tpm, {g: i for i, g in enumerate(genes)}


def _is_locomotor(nm):
    return nm in REVERSAL or nm in FORWARD or any(nm.startswith(m) for m in MOTOR)


def predict(gene, polarity, agonism, neurons, tpm, g2i, thresh=50.0):
    """Predict locomotor direction. Returns ('increase'|'decrease'|'n.d.', target_neurons)."""
    if gene not in g2i:
        return "n.d.", {}
    col = tpm[:, g2i[gene]]
    tn = {neurons[i]: float(col[i]) for i in np.where(col > thresh)[0]}
    act = agonism * (+1 if polarity == "exc" else -1)   # +1 target neurons more active
    drive = sum(act * tp for nm, tp in tn.items() if _is_locomotor(nm))
    pred = "increase" if drive > 0 else ("decrease" if drive < 0 else "n.d.")
    return pred, tn


def run(seed=0):
    genes, neurons, tpm, g2i = load_cengen()
    rows, correct = [], 0
    for name, gene, pol, ag, known in PANEL:
        pred, tn = predict(gene, pol, ag, neurons, tpm, g2i)
        ok = pred == known
        correct += ok
        top = [n for n, _ in sorted(tn.items(), key=lambda kv: -kv[1])[:4]]
        rows.append(dict(compound=name, gene=gene, polarity=pol, agonism=ag,
                         target_neurons=top, predicted=pred, known=known, correct=bool(ok)))
    # CeNGEN-shuffle control: permute which neurons express each gene; does accuracy collapse?
    rng = np.random.RandomState(seed); sh_acc = []
    for _ in range(20):
        perm = rng.permutation(tpm.shape[0])
        tpm_sh = tpm[perm]
        c = sum(predict(g, p, a, neurons, tpm_sh, g2i)[0] == k for _, g, p, a, k in PANEL)
        sh_acc.append(c / len(PANEL))
    return dict(n=len(PANEL), correct=correct, accuracy=round(correct / len(PANEL), 3),
                shuffle_accuracy_mean=round(float(np.mean(sh_acc)), 3),
                shuffle_accuracy_std=round(float(np.std(sh_acc)), 3), rows=rows)


if __name__ == "__main__":
    r = run()
    print("=== PREDICT BEHAVIOR FROM COMPOUND (compound -> CeNGEN neurons -> circuit -> behavior) ===\n")
    print("%-11s %-7s %-4s  %-34s %-9s %-8s %s"
          % ("compound", "gene", "type", "target neurons (CeNGEN)", "predicted", "known", "ok"))
    for x in r["rows"]:
        print("%-11s %-7s %-4s  %-34s %-9s %-8s %s"
              % (x["compound"], x["gene"], x["polarity"], ", ".join(x["target_neurons"]),
                 x["predicted"], x["known"], "OK" if x["correct"] else "x"))
    print("\naccuracy: %d/%d = %.0f%%   (CeNGEN-shuffle control: %.0f%% +/- %.0f%%)"
          % (r["correct"], r["n"], 100 * r["accuracy"],
             100 * r["shuffle_accuracy_mean"], 100 * r["shuffle_accuracy_std"]))
    print("curated: target gene + receptor polarity + agonism (pharmacology).")
    print("predicted: target neurons via CeNGEN + circuit role -> behavioral direction.")
    with open(os.path.join(HERE, "compound_response.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote compound_response.json")
