"""Track B follow-up: RIGHT-SIZE the digital competitor to matched accuracy.

The first pass priced the digital baseline at the full-width ReLU (98.4%, 268.8k MACs)
and compared the SNN at 98.2% to it — the same unmatched-accuracy error the CIFAR rows
were discarded under. Correct competitor = the SMALLEST digital MLP that reaches the SNN's
accuracy. Sweep width down, record best test acc + MAC count, so the matched-accuracy MAC
count (hence J/inference) can be read off. Writes data/trackb_rightsize.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import data as D
from models import MLP
from experiment import train_eval, set_seed

DEPTH = 2
WIDTHS = [16, 24, 32, 48, 64, 96, 128, 192, 256]
SEEDS = [0, 1]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "trackb_rightsize.json")

def macs(W, depth=DEPTH, in_dim=784, nc=10):
    return in_dim * W + (depth - 1) * W * W + W * nc

loaders = D.mnist(workers=0)
tr, te, meta = loaders
rows = []
print("=== MNIST digital right-size sweep (depth-2 ReLU MLP) ===")
for W in WIDTHS:
    accs = []
    for s in SEEDS:
        set_seed(s)
        m = MLP(meta["in_dim"], meta["n_classes"], width=W, depth=DEPTH, act="relu", norm="batch")
        r = train_eval(m, tr, te, epochs=12, device="cpu")
        accs.append(r["best_acc"])
    mean = float(np.mean(accs))
    rows.append({"width": W, "macs": macs(W), "mean_acc": mean,
                 "per_seed": [float(a) for a in accs]})
    print(f"  W={W:4d}  MACs={macs(W):7d}  acc={mean:.4f}  (seeds {[f'{a:.4f}' for a in accs]})")

# smallest width reaching each target accuracy
targets = {}
for tgt in (0.982, 0.9822, 0.980, 0.975):
    ok = [r for r in rows if r["mean_acc"] >= tgt]
    targets[f"acc_{tgt}"] = ({"width": ok[0]["width"], "macs": ok[0]["macs"]} if ok else None)
    if ok:
        print(f"  -> matched {tgt:.4f}: smallest W={ok[0]['width']} ({ok[0]['macs']} MACs)")
    else:
        print(f"  -> matched {tgt:.4f}: NOT reached in sweep (need W>{WIDTHS[-1]})")

json.dump({"depth": DEPTH, "sweep": rows, "matched_targets": targets}, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
