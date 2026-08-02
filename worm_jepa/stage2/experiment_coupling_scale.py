"""
Compound the two structure levers: the coupling term x more data. Earlier we saw
(a) the coupling term lifts struct_corr 0.248 -> 0.336 at 60 worms, and (b) more
data lifts a plain model's struct_corr 0.10 -> 0.30 over 10 -> 320 worms. Do they
stack? Sweep n_worms for the coupling model vs the no-coupling temporal model.

W is fixed across sizes (same seed) so struct_corr is comparable. Also reports a
better coupling estimate benefits from more rows (M = corr of activity).
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discover

HERE = os.path.dirname(os.path.abspath(__file__))
COUP = os.path.join(HERE, "candidates_temporal", "tgen2_mapelites_coupling.py")
NONE = os.path.join(HERE, "candidates_temporal", "tgen1_velocity.py")


def main():
    sizes = [20, 40, 80, 160, 320, 480]
    rows = []
    for n in sizes:
        prob = discover.get_problem(lags=3, n_worms=n)
        c = discover.eval_program(COUP, prob)
        b = discover.eval_program(NONE, prob)
        row = {"n_worms": n, "n_pairs_train": int(len(prob["Xtr"])),
               "coupling_struct": round(c["struct_corr"], 4), "coupling_pred": round(c["pred_r2"], 4),
               "nocoup_struct": round(b["struct_corr"], 4), "nocoup_pred": round(b["pred_r2"], 4)}
        rows.append(row)
        print(f"  n={n:4d} pairs={row['n_pairs_train']:6d}  "
              f"COUPLING struct={row['coupling_struct']:+.3f} pred={row['coupling_pred']:+.3f}  |  "
              f"none struct={row['nocoup_struct']:+.3f} pred={row['nocoup_pred']:+.3f}", flush=True)
    out = os.path.join(HERE, "experiment_coupling_scale.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[stage2] wrote {out}")


if __name__ == "__main__":
    main()
