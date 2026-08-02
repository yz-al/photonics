#!/usr/bin/env python3
"""
Phase 2 — dipolar candidate screen (the actionable seed list).

Integrates the pieces that survived review:
  - U-definition decision: the independent U is the dipolar/interlayer channel.
  - HetDB (heterostructures) as the host of spatially-indirect (type-II) excitons.
  - a GEOMETRIC dipolar-strength prior g_dip ∝ d²/ε_eff from the interlayer
    distance d (100% coverage), used only to *rank* which candidates deserve the
    expensive GW-BSE later.

It produces a ranked list of type-II heterostructures with a model-tier dipolar
prior, plus the honest flags: (a) the FOM is NOT computed — d is only a geometric
upper bound on the CT-weighted exciton dipole d_exc, and Γ needs EPW; (b) these
heterostructures are OUT OF DISTRIBUTION for the C2DB-monolayer surrogate, so the
surrogate cannot predict their excited-state properties — which is exactly why
they are Phase-2 label-generation targets, not fit-from-existing-data targets.

    python excitonic/scripts/phase2_dipolar_screen.py
Writes excitonic/data/manifests/phase2_dipolar_candidates.json + a ranked table.
Live queries to HetDB + C2DB; no fabrication.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

# Reuse the inventory module's live helpers (Anderson's-rule classifier + C2DB
# band-edge join) without re-implementing them.
_spec = importlib.util.spec_from_file_location(
    "p2inv", os.path.join(HERE, "phase2_dipolar_inventory.py"))
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)

from exciton_fm.bidb_client import hetdb_client          # noqa: E402
from exciton_fm.exciton_u import u_dipolar                # noqa: E402
from exciton_fm.provenance import not_run                 # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                   "phase2_dipolar_candidates.json"))

# Model assumptions for the geometric prior (flagged, not fitted).
EPS_EFF = 5.0            # effective screening (dimensionless) — a placeholder
REF_MODE_AREA_nm2 = 100.0 * 100.0
GAP_LO, GAP_HI = 0.6, 2.6  # semiconducting, optically useful window [eV]


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    h = hetdb_client()
    print("[dipolar] pulling live HetDB rows ...", flush=True)
    rows = h.fetch_rows("", max_pages=40)
    print(f"[dipolar] {len(rows)} heterostructures", flush=True)

    # Cross-join to C2DB monolayer band edges for Anderson's-rule alignment.
    uids = set()
    for r in rows:
        for k in ("uid_a", "uid_b"):
            if r.get(k):
                uids.add(r[k])
    edges = inv.c2db_band_edges(uids)

    cands = []
    for r in rows:
        ea, eb = edges.get(r.get("uid_a")), edges.get(r.get("uid_b"))
        if not (ea and eb):
            continue
        align = inv.classify_alignment(ea["vbm"], ea["cbm"], eb["vbm"], eb["cbm"])
        d = _f(r.get("interlayer_distance"))       # Å
        gap = _f(r.get("pbe_gap_soc"))
        area = _f(r.get("area"))                   # Å²
        strain = _f(r.get("maxstrain"))
        if align != "II" or d is None or area is None:  # classify returns "II"
            continue
        # geometric dipolar prior: g_dip = e^2 d^2 /(eps0 eps_eff)  [eV·nm²];
        # U at a reference mode area. d,area Å -> nm.
        d_nm, area_nm2 = d / 10.0, area / 100.0
        U = u_dipolar(d_nm, REF_MODE_AREA_nm2, eps=EPS_EFF)
        g_dip = None if U.value is None else U.value * REF_MODE_AREA_nm2  # eV·nm²
        gap_ok = gap is not None and GAP_LO <= gap <= GAP_HI
        # priority: large geometric dipole strength, semiconducting-in-window,
        # low strain (commensurate/stable). Purely a prioritization score.
        score = 0.0
        if g_dip is not None:
            score += min(g_dip / 2.0, 1.0) * 0.6
        if gap_ok:
            score += 0.25
        if strain is not None:
            score += max(0.0, 1.0 - min(strain / 0.05, 1.0)) * 0.15
        cands.append({
            "hetero": f"{r.get('uid_a')} / {r.get('uid_b')}",
            "interlayer_d_A": d, "gap_pbe_soc_eV": gap, "area_A2": area,
            "maxstrain": strain, "alignment": align,
            "g_dip_model_eV_nm2": (None if g_dip is None else round(g_dip, 4)),
            "U_dip_at_ref_area": U.to_dict(),
            "score": round(score, 4),
        })

    cands.sort(key=lambda c: c["score"], reverse=True)

    manifest = {
        "purpose": "Ranked dipolar/interlayer-exciton candidates (Phase-2 BSE seed).",
        "u_channel": "dipolar (independent of a_B); see reports/u_definition_decision.md",
        "model_assumptions": {"eps_eff": EPS_EFF, "ref_mode_area_nm2": REF_MODE_AREA_nm2,
                              "gap_window_eV": [GAP_LO, GAP_HI]},
        "rigor": (
            "RANKING ONLY. g_dip/U_dip are MODEL-tier geometric priors: d is the "
            "interlayer distance, an UPPER BOUND on the CT-weighted exciton dipole "
            "d_exc (a BSE observable, not yet computed). Γ(300 K) is NOT included — "
            "no EPW. The blockade FOM is therefore NOT computed here; these are the "
            "materials worth spending GW-BSE + EPW on, ranked."),
        "surrogate_note": (
            "These heterostructures are OUT OF DISTRIBUTION for the C2DB-monolayer "
            "surrogate (all would score high OOD), so the surrogate cannot predict "
            "their excited-state properties — confirming they are label-GENERATION "
            "targets, not fit-from-existing-data targets."),
        "n_type_ii_candidates": len(cands),
        "candidates": cands,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[dipolar] {len(cands)} type-II candidates ranked -> {OUT}")
    print("[dipolar] top 10 by geometric dipolar prior (MODEL tier; FOM not computed):")
    for c in cands[:10]:
        print(f"    {c['hetero']:<34s} d={c['interlayer_d_A']:.2f} Å  "
              f"gap={c['gap_pbe_soc_eV']}  g_dip≈{c['g_dip_model_eV_nm2']} eV·nm²  "
              f"score={c['score']:.3f}")
    print("[dipolar] NOTE: ranking prior only — d is an upper bound on d_exc; "
          "no Γ, no FOM. GW-BSE + EPW on the top candidates is the next real step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
