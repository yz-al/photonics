#!/usr/bin/env python3
"""
Phase 2, step 2 — generate first-principles input decks + model-tier estimates.

For the TMD validation anchors this:
  1. builds the 2D structure (ASE),
  2. writes the full QE electron-phonon deck (scf -> ph -> epw) and the
     BerkeleyGW GW-BSE deck (pw2bgw -> epsilon -> sigma -> kernel -> absorption),
  3. computes MODEL-tier Γ(300 K) (Fröhlich) and U (saturation) estimates and the
     resulting U/Γ — all explicitly flagged as estimates, NOT first-principles.

This proves the whole pipeline is runnable end to end without any GW/EPW binary,
and gives an honest prioritization floor. The real labels come from the Modal
CPU jobs (modal_phase2.py); until those run, Γ/U carry tier 'frohlich_model' /
'saturation_model' or 'not_run' — never 'epw'/'gw_bse'.

    python excitonic/scripts/phase2_gen_inputs.py

Writes decks under excitonic/data/phase2_inputs/<name>/ and a summary manifest.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.structures import anchor_structures, structure_summary   # noqa: E402
from exciton_fm.qe_inputs import write_eph_chain                          # noqa: E402
from exciton_fm.bgw_inputs import write_bse_chain                         # noqa: E402
from exciton_fm.frohlich import estimate_gamma_300K                       # noqa: E402
from exciton_fm.exciton_u import bohr_radius_2D, u_saturation             # noqa: E402
from exciton_fm.fom import blockade_fom                                   # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
INP_DIR = os.path.join(ROOT, "data", "phase2_inputs")
OUT = os.path.join(ROOT, "data", "manifests", "phase2_anchor_estimates.json")

# Literature-ish anchor parameters for the MODEL estimates (not fitted labels):
#   hw_LO   : LO/A1' phonon energy [meV]
#   alpha   : representative Fröhlich coupling (dimensionless)
#   E_b     : C2DB BSE exciton binding [eV]
#   mu      : reduced effective mass [m_e]
#   eps     : effective screening for the 2D Bohr radius
ANCHOR_PARAMS = {
    "MoS2":  dict(hw_LO=48.0, alpha=0.40, E_b=0.547, mu=0.27, eps=5.0),
    "MoSe2": dict(hw_LO=36.0, alpha=0.42, E_b=0.500, mu=0.29, eps=5.5),
    "WS2":   dict(hw_LO=44.0, alpha=0.38, E_b=0.516, mu=0.22, eps=5.0),
    "WSe2":  dict(hw_LO=31.0, alpha=0.40, E_b=0.480, mu=0.24, eps=5.5),
}
# Reference mode area for expressing U (a device parameter, not a material one).
MODE_AREA_nm2 = 100.0 * 100.0  # 100 nm × 100 nm reference cavity mode


def main() -> int:
    os.makedirs(INP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    structs = anchor_structures()
    results = {}
    for name, atoms in structs.items():
        d = os.path.join(INP_DIR, name)
        eph = write_eph_chain(atoms, d, prefix=name)
        bse = write_bse_chain(d, prefix=name)
        p = ANCHOR_PARAMS.get(name)
        entry = {"structure": structure_summary(atoms),
                 "qe_eph_deck": {k: os.path.relpath(v, ROOT) for k, v in eph.items()},
                 "bgw_bse_deck": {k: os.path.relpath(v, ROOT) for k, v in bse.items()},
                 "labels": {}}
        if p:
            gamma = estimate_gamma_300K(p["hw_LO"], alpha=p["alpha"])
            a_B = bohr_radius_2D(p["mu"], p["eps"])
            U = u_saturation(p["E_b"], a_B, MODE_AREA_nm2)
            fom = blockade_fom(U, gamma)
            entry["labels"] = {
                "a_B_nm_model": round(a_B, 3),
                "Gamma_300K": gamma.to_dict(),
                "U": U.to_dict(),
                "blockade_fom": fom,
            }
        results[name] = entry

    manifest = {
        "purpose": "Phase-2 input decks + MODEL-tier Γ/U estimates for the TMD anchors",
        "rigor": ("Γ/U here are analytic MODEL estimates (tiers 'frohlich_model' / "
                  "'saturation_model'), NOT GW-BSE/EPW results. U is quoted at a "
                  f"{int(MODE_AREA_nm2)} nm² reference mode area. Real labels come "
                  "from modal_phase2.py and will carry tier 'epw' / 'gw_bse'."),
        "mode_area_nm2": MODE_AREA_nm2,
        "anchors": results,
    }
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[phase2] wrote decks under {os.path.relpath(INP_DIR, ROOT)}/<anchor>/ "
          f"and estimates -> {os.path.relpath(OUT, ROOT)}")
    for name, e in results.items():
        L = e.get("labels", {})
        if L:
            g = L["Gamma_300K"]["value"]; u = L["U"]["value"]
            fom = L["blockade_fom"]["U_over_Gamma"]
            print(f"[phase2] {name:6s} Γ(300K)≈{g:5.1f} meV [model]  "
                  f"a_B≈{L['a_B_nm_model']:.2f} nm  U≈{u*1e3:.3f} meV [model]  "
                  f"U/Γ≈{fom} (model, mode-area-dependent)")
    print("[phase2] NOTE: all Γ/U above are MODEL estimates — no GW-BSE/EPW has run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
