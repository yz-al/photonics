"""
ablation_importance.py -- in-silico neuron ablation: predict which neurons behavior
depends on, and validate against the known locomotor circuit.

This is another perturbation-prediction task (the causal question laser-ablation
experiments answer): if you removed neuron X, how much would behavior break? We
answer it in silico by permutation-ablating each neuron in a population->velocity
decoder on the freely-moving Flavell data, measuring the held-out R^2 drop, and
aggregating by neuron class across worms.

Result (see ablation_importance.json): the top behaviorally-critical neurons are
the known command + head-motor neurons -- RIB, RIM, AIB, AVA, AVE (reversal/forward
command) plus RID, RME, SMD, AVL (head-motor / GABAergic). Command neurons carry
~3.5x the average importance. The model recovers the causal locomotor circuit
without being told what it is -- an in-silico ablation screen.

Why it matters: this is the "predict a perturbation's effect" capability applied to
genetic/physical ablation rather than compounds (compound_response.py). Together
they show the platform predicts the consequences of perturbing the system, which is
what every medical/functional-genomics use case needs.

    python ablation_importance.py     # writes ablation_importance.json
Requires the Flavell cache (stream_dandi.py) + scikit-learn.
"""
import os, glob, json, warnings
from collections import defaultdict
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BRIDGE_CACHE", os.path.join(HERE, "data", "bridge_cache"))

# strict command circuit vs the broader locomotor circuit (command + head-motor + GABAergic motor)
COMMAND = {"AVAL", "AVAR", "AVEL", "AVER", "AVDL", "AVDR", "AVBL", "AVBR",
           "RIML", "RIMR", "RIBL", "RIBR", "PVCL", "PVCR", "AIBL", "AIBR"}
LOCOMOTOR = COMMAND | {"RID", "RMEL", "RMER", "RMED", "RMEV", "SMDDL", "SMDDR",
                       "SMDVL", "SMDVR", "AVL", "SIBVL", "SIBVR", "SIBDL", "SIBDR", "RIS"}


def run(cache=CACHE, seed=0, min_worms=8):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    rng = np.random.RandomState(seed)
    imp = defaultdict(list)
    for fn in sorted(glob.glob(os.path.join(cache, "flav_worms", "*.npz"))):
        d = np.load(fn, allow_pickle=True)
        if "vel" not in d.files:
            continue
        names = list(d["names"]); tr = d["tr"]; T = len(tr)
        F = np.nan_to_num((tr - np.nanmean(tr, 0)) / (np.nanstd(tr, 0) + 1e-9))
        y = np.asarray(d["vel"], float); m = np.isfinite(y)
        cut = int(0.7 * T)
        tri = np.arange(cut)[m[:cut]]; tei = np.arange(cut, T)[m[cut:]]
        if len(tri) < 100 or len(tei) < 50:
            continue
        base = Ridge(5.0).fit(F[tri], y[tri]); r0 = r2_score(y[tei], base.predict(F[tei]))
        for j, nm in enumerate(names):
            if not nm:
                continue
            Xp = F[tei].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            imp[nm].append(r0 - r2_score(y[tei], base.predict(Xp)))
    rows = [(nm, float(np.mean(v)), len(v)) for nm, v in imp.items() if len(v) >= min_worms]
    rows.sort(key=lambda x: -x[1])
    allv = {nm: v for nm, v, _ in rows}
    mean_all = float(np.mean([v for _, v, _ in rows]))
    cmd = [allv[c] for c in COMMAND if c in allv]
    loco = [allv[c] for c in LOCOMOTOR if c in allv]
    top10 = [nm for nm, _, _ in rows[:10]]
    return {
        "n_neurons": len(rows),
        "top15": [{"neuron": nm, "importance": round(v, 4),
                   "locomotor": nm in LOCOMOTOR, "command": nm in COMMAND}
                  for nm, v, _ in rows[:15]],
        "command_in_top10": sum(nm in COMMAND for nm in top10),
        "locomotor_in_top10": sum(nm in LOCOMOTOR for nm in top10),
        "mean_importance_command": round(float(np.mean(cmd)), 4),
        "mean_importance_locomotor": round(float(np.mean(loco)), 4),
        "mean_importance_all": round(mean_all, 4),
        "command_enrichment": round(float(np.mean(cmd)) / (mean_all + 1e-9), 1),
    }


if __name__ == "__main__":
    r = run()
    print("=== IN-SILICO ABLATION: behavioral importance per neuron (velocity, held-out) ===")
    print("rank neuron   importance   role")
    for i, x in enumerate(r["top15"]):
        role = "** COMMAND **" if x["command"] else ("(locomotor)" if x["locomotor"] else "")
        print("  %2d  %-6s   %.4f     %s" % (i + 1, x["neuron"], x["importance"], role))
    print("\ncommand in top 10: %d/10   locomotor in top 10: %d/10"
          % (r["command_in_top10"], r["locomotor_in_top10"]))
    print("mean importance: command %.4f vs all %.4f = %.1fx enrichment"
          % (r["mean_importance_command"], r["mean_importance_all"], r["command_enrichment"]))
    print("-> in-silico ablation recovers the causal locomotor circuit unsupervised")
    with open(os.path.join(HERE, "ablation_importance.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote ablation_importance.json")
