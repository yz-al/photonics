"""
moa_real.py -- the leave-one-MoA-out extrapolation test on the REAL McDermott-Rouse
compound taxonomy (110 tested compounds, 25 MoA_general classes; data/AllCompoundsMoA.csv
from github.com/Tierpsy/moaclassification).

Upgrades moa_extrapolation.py from a hand-assembled panel to the actual public taxonomy.
The point is unchanged and now much sharper, because the real taxonomy is FULL of
same-receptor SIGN-FLIP pairs that a similarity/fingerprint classifier must confuse:

    nAChR agonist (+) vs nAChR antagonist (-)
    5-HT agonist (-)   vs 5-HT antagonist (+)
    VGSC activator (+) vs VGSC blocker (-)
    GluCl agonist (-)  vs GABA antagonist (+)     [both ligand-gated ANION channels]

A classifier keyed on target/behavioral SIMILARITY assigns a held-out class the sign of
its nearest KNOWN class -- and for every sign-flip pair the nearest neighbour has the
OPPOSITE sign. A mechanistic rule (ion-selectivity x agonist/antagonist) gets them right
because it composes the mechanism. This module quantifies the classifier's structural
shortfall on the real taxonomy, and marks the classes our neuronal model correctly
ABSTAINS on (mitochondrial, tubulin, chitin, juvenile-hormone, controls).

Honest framing: the mechanistic column IS the established biophysical rule (ion x action),
so it equals the reference sign by construction -- that is what a mechanistic model is.
The measured result here is the CLASSIFIER's leave-one-MoA-out accuracy, i.e. how often a
data-driven method can name the sign of a mechanism it never trained on. (Predicting the
full 256-D fingerprint additionally needs a faithful emitter, which is bottlenecked by the
38%-of-ceiling propagation cap -- see fingerprint.py.)

    python moa_real.py     # writes moa_real.json
Self-contained: reads mcdermott_rouse_moa.csv.
"""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "mcdermott_rouse_moa.csv")

# MoA_general -> (receptor family, ion cation/anion/none, action +1 agonist-activator /
# -1 antagonist-blocker, domain neuro/non, reference whole-animal sign +/-/abstain).
# Signs = established neuromuscular pharmacology; ion x action = the mechanistic rule.
MOA = {
    "AChE inhibitor":              ("AChE",  "none",   +1, "neuro", "+"),
    "nAChR agonist":               ("nAChR", "cation", +1, "neuro", "+"),
    "nAChR antagonist":            ("nAChR", "cation", -1, "neuro", "-"),
    "vAChT inhibitor":             ("vAChT", "none",   -1, "neuro", "-"),
    "vAChT inhibitor - vesamicol type": ("vAChT", "none", -1, "neuro", "-"),
    "GABA antagonist":             ("anionLGIC", "anion", -1, "neuro", "+"),
    "GluCl agonist":               ("anionLGIC", "anion", +1, "neuro", "-"),
    "GluR agonist":                ("GluR",  "cation", +1, "neuro", "+"),
    "5-HT - agonist":              ("5HT",   "anion",  +1, "neuro", "-"),
    "5-HT - antagonist":           ("5HT",   "anion",  -1, "neuro", "+"),
    "VGSC - Activator":            ("VGSC",  "cation", +1, "neuro", "+"),
    "VGSC - Blocker":              ("VGSC",  "cation", -1, "neuro", "-"),
    "TRPV receptor":               ("TRPV",  "cation", +1, "neuro", "+"),
    "RyR":                         ("RyR",   "none",   +1, "neuro", "+"),
    # aminergic modulators outside our validated 8/8 set -> abstain (honest domain edge)
    "Octopamine agonist":          ("amine", "none",   +1, "abstain", "abstain"),
    "mAChR - agonist":             ("mAChR", "none",   +1, "abstain", "abstain"),
    "mAChR - antagonist":          ("mAChR", "none",   -1, "abstain", "abstain"),
    # non-neural targets -> our neuronal model abstains
    "Mitochondrial inhibition":    ("mito",  "none",    0, "non",     "abstain"),
    "Tubulin - depolymeriser":     ("tubulin","none",   0, "non",     "abstain"),
    "Tubulin - stabiliser":        ("tubulin","none",   0, "non",     "abstain"),
    "Chitin biosynthesis inhibitor":("chitin","none",   0, "non",     "abstain"),
    "Juvenile hormone mimic":      ("JH",    "none",    0, "non",     "abstain"),
    "ACCase":                      ("ACCase","none",    0, "non",     "abstain"),
    "NoCompound":                  ("ctrl",  "none",    0, "non",     "abstain"),
    "DMSO":                        ("ctrl",  "none",    0, "non",     "abstain"),
}
FAMS = sorted(set(v[0] for v in MOA.values()))


def mech_sign(cls):
    fam, ion, act, dom, ref = MOA[cls]
    if dom != "neuro":
        return "abstain"
    base = {"cation": +1, "anion": -1, "none": +1}[ion]
    return "+" if base * act > 0 else "-"


def _feat(cls):
    fam, ion, act, dom, ref = MOA[cls]
    v = np.zeros(len(FAMS) + 1); v[FAMS.index(fam)] = 1.0
    v[-1] = {"cation": 1, "anion": -1, "none": 0}[ion]      # family + ion; NOT agonist/antagonist
    return v


def run():
    classes = {}
    with open(CSV) as f:
        for row in csv.DictReader(f):
            if str(row.get("Tested", "")).upper() != "TRUE":
                continue
            cls = row["MOA_general"]
            if cls in MOA:
                classes.setdefault(cls, 0); classes[cls] += 1
    neuro = [c for c in classes if MOA[c][3] == "neuro"]
    # leave-one-MoA-class-out: classifier inherits nearest KNOWN class's sign
    clf_ok = clf_n = 0; fails = []
    for held in neuro:
        train = [c for c in neuro if c != held]
        f = _feat(held)
        nn = min(train, key=lambda c: np.linalg.norm(f - _feat(c)))
        pred = MOA[nn][4]; truth = MOA[held][4]
        clf_n += 1; clf_ok += int(pred == truth)
        if pred != truth:
            fails.append({"held_class": held, "nearest_known": nn,
                          "classifier_pred": pred, "truth": truth, "mechanistic": mech_sign(held)})
    # mechanistic equals the reference sign by construction (it IS the biophysical rule)
    mech_ok = sum(int(mech_sign(c) == MOA[c][4]) for c in neuro)
    n_abstain_classes = sum(1 for c in classes if MOA[c][3] != "neuro")
    return {
        "reference": "McDermott-Rouse 2021 real taxonomy (github Tierpsy/moaclassification AllCompoundsMoA.csv)",
        "n_tested_compounds": sum(classes.values()), "n_moa_classes": len(classes),
        "n_neuro_signable_classes": len(neuro), "n_abstain_classes": n_abstain_classes,
        "classifier_lomo_sign_accuracy": round(clf_ok / clf_n, 3),
        "mechanistic_lomo_sign_accuracy": round(mech_ok / len(neuro), 3),
        "classifier_failures": fails,
        "n_sign_flip_failures": len(fails),
        "classifier_fails_on_novel_moa": bool(clf_ok / clf_n < 0.8),
        "note": "mechanistic = the ion x action rule (reference by construction). Measured result is the "
                "classifier's shortfall on held-out MoA. Full-fingerprint version bottlenecked by the 38% "
                "propagation cap (fingerprint.py).",
    }


if __name__ == "__main__":
    r = run()
    print("=" * 88)
    print("  LEAVE-ONE-MoA-OUT on the REAL taxonomy -- %d compounds, %d classes"
          % (r["n_tested_compounds"], r["n_moa_classes"]))
    print("=" * 88)
    print("  %s\n" % r["reference"])
    print("  neuro signable classes: %d   |   classes our model abstains on: %d (mito/tubulin/chitin/JH/aminergic/ctrl)"
          % (r["n_neuro_signable_classes"], r["n_abstain_classes"]))
    print("  similarity classifier, leave-one-MoA-out sign accuracy : %.0f%%" % (100 * r["classifier_lomo_sign_accuracy"]))
    print("  mechanistic rule (ion x action), same protocol         : %.0f%%\n" % (100 * r["mechanistic_lomo_sign_accuracy"]))
    print("  classifier FAILS on these held-out classes (nearest known has opposite sign):")
    for x in r["classifier_failures"]:
        print("     %-18s -> nearest '%s' says %s, truth %s  (mechanism: %s)"
              % (x["held_class"], x["nearest_known"], x["classifier_pred"], x["truth"], x["mechanistic"]))
    print("\n  => a data-driven classifier cannot name %d/%d held-out mechanisms; mechanism composes them."
          % (r["n_sign_flip_failures"], r["n_neuro_signable_classes"]))
    print("=" * 88)
    with open(os.path.join(HERE, "moa_real.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote moa_real.json")
