#!/usr/bin/env python3
"""
Phase 2 — Rydberg escape screen: is the n≥2 exciton route even populated at RT?

The Rydberg mechanism (Cu₂O-style, interaction ∝ n^high) raises U through the
principal quantum number, NOT through spatial extent — so it sidesteps BOTH bound
exponents (the moiré q_sat and the interlayer λ_f). Its obstacle is that excited
excitons are more weakly bound (2D hydrogenic: E_n = Ry/(n−½)², so E_2 = E_1/9,
E_3 = E_1/25) and ionise first. This checks whether any screened C2DB material has an
n=2 exciton that clears the room-temperature threshold.

RIGOR / caveat: the 2D-hydrogenic (n−½)² series assumes a Wannier-Mott exciton.
Many high-E_b C2DB hits are wide-gap MAGNETIC d-electron halides/oxides whose lowest
exciton may be Frenkel / charge-transfer, NOT hydrogenic-Rydberg — so the count is an
UPPER BOUND on the populated Rydberg route, a flag for follow-up, not a confirmation.

    python excitonic/scripts/phase2_rydberg_screen.py
Writes excitonic/data/manifests/phase2_rydberg_screen.json.
"""
from __future__ import annotations

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "c2db_excitonic.csv"))
OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests", "phase2_rydberg_screen.json"))

RT_THRESHOLD_eV = 0.10       # n=2 must bind this hard to survive 300 K
E2_OVER_E1 = 1.0 / 9.0       # 2D hydrogenic (n−½)²: E_2/E_1


def main() -> int:
    rows = list(csv.DictReader(open(CSV)))
    mats = []
    for r in rows:
        try:
            e1 = float(r["E_B"])
        except (TypeError, ValueError):
            continue
        mag = str(r.get("is_magnetic", "")).strip().lower() in ("true", "1", "yes")
        mats.append({"formula": r.get("formula"), "E1_eV": e1,
                     "E_n2_eV": e1 * E2_OVER_E1, "is_magnetic": mag,
                     "gap_eV": r.get("gap")})
    survivors = [m for m in mats if m["E_n2_eV"] >= RT_THRESHOLD_eV]
    survivors.sort(key=lambda m: -m["E1_eV"])
    nonmag = [m for m in survivors if not m["is_magnetic"]]

    manifest = {
        "question": "Is the Rydberg (n≥2) route — which raises U with no spatial-extent "
                    "penalty, sidestepping both bound exponents — populated at RT?",
        "series": "2D hydrogenic E_n=Ry/(n−½)²; E_2=E_1/9",
        "rt_threshold_eV": RT_THRESHOLD_eV,
        "n_total": len(mats),
        "n_with_n2_above_RT": len(survivors),
        "n_with_n2_above_RT_nonmagnetic": len(nonmag),
        "top_candidates": survivors[:12],
        "top_nonmagnetic": nonmag[:12],
        "finding": (f"{len(survivors)}/{len(mats)} C2DB materials have an n=2 binding above "
                    f"{RT_THRESHOLD_eV} eV — the Rydberg escape is NOT empty. But only "
                    f"{len(nonmag)} are non-magnetic; most high-E_b hits are magnetic "
                    "d-electron halides/oxides whose lowest exciton may be Frenkel/CT, not "
                    "hydrogenic-Rydberg. So this is an UPPER BOUND on the route, a follow-up "
                    "flag (verify Rydberg character + measure U(n) via BSE), not a result."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[rydberg] {len(survivors)}/{len(mats)} have n=2 binding > {RT_THRESHOLD_eV} eV "
          f"({len(nonmag)} non-magnetic)")
    for m in survivors[:6]:
        print(f"    {m['formula']:16s} E1={m['E1_eV']:.2f} eV  E_n2~{m['E_n2_eV']:.3f} eV  "
              f"{'(magnetic)' if m['is_magnetic'] else ''}")
    print(f"[rydberg] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
