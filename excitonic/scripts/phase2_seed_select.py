#!/usr/bin/env python3
"""
Phase 2, step 1 — select the GW-BSE + EPW seed set.

Ranks the C2DB BSE materials (all of which lack Γ and U) by room-temperature
blockade promise, so first-principles jobs target the best candidates first.

    python excitonic/scripts/phase2_seed_select.py [N]

Writes excitonic/data/manifests/phase2_seed_set.json.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.seed_select import select_seed_set  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "processed", "c2db_excitonic.csv")
OUT = os.path.join(ROOT, "data", "manifests", "phase2_seed_set.json")

NUMERIC = ["E_B", "alphax_el", "alphay_el", "alphaz_el", "plasmafrequency_x",
           "gap", "gap_gw", "gap_hse", "emass_cbm", "emass_vbm", "alphax_lat",
           "thickness", "area", "natoms", "nspecies", "dipz", "ehull", "hform"]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    df = pd.read_csv(CSV)
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    rows = df.to_dict(orient="records")
    # convert NaN -> None
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, float) and v != v:
                r[k] = None

    seed = select_seed_set(rows, n=n, per_family_cap=None)

    fam_counts: dict[str, int] = {}
    for d in seed:
        fam_counts[d["family"]] = fam_counts.get(d["family"], 0) + 1

    manifest = {
        "purpose": "GW-BSE + DFPT/EPW label-generation seed set (targets: Gamma(300K), U)",
        "note": ("All listed materials carry a C2DB BSE E_b but NO Gamma or U "
                 "label; they are the candidates to run first-principles on. "
                 "Ranking is a static priority score; the active-learning loop "
                 "(Phase 3) replaces it with surrogate uncertainty."),
        "n_selected": len(seed),
        "family_counts": fam_counts,
        "score_weights": "binding .34 / low_frohlich .24 / gap_window .16 / oscillator .12 / stability .14",
        "seed_set": seed,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[phase2] selected {len(seed)} seed materials -> {OUT}")
    print(f"[phase2] family breakdown: {fam_counts}")
    print("[phase2] top 10:")
    for d in seed[:10]:
        print(f"    {d['formula']:<12s} score={d['score']:.3f} fam={d['family']:<22s} "
              f"E_b={d['E_b_eV']} {'; '.join(d['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
