#!/usr/bin/env python3
"""
Phase 2 — the FEASIBILITY BOUND (theory, not compute): try to close the question
with three inequalities before Phase 3 runs.

If max(U_sat_max, U_dip_max) < 0.71·Γ_floor across thermally-stable materials, the
feasible set is empty and no search is needed (the Kramers-Kronig-for-Kerr move).
This script populates the two legs it can from the DFPT screen (which emits Z*, ω_LO,
ε∞ at cents/material) and marks the third — the dipolar U ceiling — as needing λ_f
from the flagship GW-BSE. It does NOT declare the bound closed before its inputs exist.

    python excitonic/scripts/phase2_bound.py
Writes excitonic/data/manifests/phase2_bound.json.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.bound import (gamma_LO_floor_300K, u_saturation_ceiling,       # noqa: E402
                              u_dipolar_ceiling, feasibility)
from exciton_fm.frohlich import bose, C_CAL                                    # noqa: E402

GAMMA = os.path.abspath(os.path.join(HERE, "..", "data", "manifests", "phase2_tmd_gamma.json"))
WINDOW = os.path.abspath(os.path.join(HERE, "..", "data", "manifests", "phase2_dipolar_window.json"))
OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests", "phase2_bound.json"))

# Thermal-stability threshold and the reduced mass it forces (the small-a_B regime).
E_B_STABLE_eV = 0.10        # ~4 kT: an exciton must bind this hard to survive 300 K
MU_MIN = 0.20               # thermal stability forbids much lighter μ (E_b ∝ μ/ε²)
EPS_EFF = 5.0
# Anchor Fröhlich coupling: TMDs sit at α≈0.4; the floor argument is that a
# thermally-stable (hence ionic) exciton material cannot have α far below this.
ALPHA_MIN = 0.3


def main() -> int:
    if not os.path.exists(GAMMA):
        print(f"[bound] need {GAMMA} (run the DFPT gate first)"); return 1
    g = json.load(open(GAMMA))["results"]

    # --- (Γ floor) from the DFPT TMDs: real ω_LO, floor coupling α_min ---
    floors = {}
    for name, r in g.items():
        wlo = (r.get("omega_LO_meV") or {}).get("value")
        z = (r.get("Z_born") or {}).get("value")
        eps = (r.get("eps_inf") or {}).get("value")
        if not wlo:
            continue
        gamma_floor = gamma_LO_floor_300K(ALPHA_MIN, wlo)   # α_min ⇒ LOWER bound
        floors[name] = {"omega_LO_meV": wlo, "Z_born": z, "eps_inf": eps,
                        "n_LO_300K": round(bose(wlo, 300.0), 5),
                        "gamma_floor_meV": round(gamma_floor, 3)}
    if not floors:
        print("[bound] no physical ω_LO in the gate manifest"); return 1
    gamma_floor_min = min(v["gamma_floor_meV"] for v in floors.values())

    # --- (U_sat ceiling): 6·E_b·a_B². The per-polariton value is this SUPPRESSED by
    # the mode area / a_B² = N. So the saturation leg is NOT closed by 1/μ alone: it is
    # bounded by a confinement↔inhomogeneous-broadening tradeoff, structurally parallel
    # to the dipolar f-penalty (both raise U by localizing, both pay in readout). We
    # report the single-exciton max (g_xx) and a realistic delocalized value, and leave
    # the leg's status confinement-dependent rather than declaring it clear.
    g_xx_single = u_saturation_ceiling(E_B_STABLE_eV, MU_MIN, EPS_EFF, 1.0)      # ~a_B² area
    u_sat_delocalized = u_saturation_ceiling(E_B_STABLE_eV, MU_MIN, EPS_EFF, 100.0)  # 10x10nm mode
    u_sat_max = None            # the USABLE saturation U is confinement-set → not a single number

    # --- (U_dip ceiling): needs λ_f from the flagship. Try to read it; else None ---
    lambda_f = None
    if os.path.exists(WINDOW):
        try:
            fs = json.load(open(WINDOW)).get("flagship_gwbse_dsweep")
            if isinstance(fs, dict):
                lambda_f = fs.get("f_decay_length_lambda")
        except Exception:
            lambda_f = None
    # dipolar-U model U(d)=U0·d (capacitor, linear); U0 at d0 from the screen prior
    u_dip_max = u_dipolar_ceiling(lambda_f, f_min_frac=0.1,
                                  U_dip_of_d=lambda d: 0.10 * (d / 0.6)) if lambda_f else None

    verdict = feasibility(gamma_floor_min, u_sat_max, u_dip_max)

    manifest = {
        "structure": "close the question with a bound, not a search: "
                     "max(U_sat_max, U_dip_max) < 0.71·Γ_floor ⇒ feasible set EMPTY.",
        "analogy": "Kramers-Kronig closed the Kerr route by bounding the ratio, not by "
                   "failing to find a material. Same shape here.",
        "assumptions": {"E_b_thermal_stable_eV": E_B_STABLE_eV, "mu_min": MU_MIN,
                        "eps_eff": EPS_EFF, "alpha_floor": ALPHA_MIN, "f_min_frac": 0.1},
        "gamma_floor": {
            "per_material": floors,
            "gamma_floor_min_meV": gamma_floor_min,
            "note": "Γ_LO(300 K)=C·α_min·ħω_LO·n_LO. Real LOWER bound GIVEN α≥α_min; "
                    "proving stability⟹α≥α_min tightly (via Z*/LO-TO ionicity) is the "
                    "theory crux — the DFPT screen supplies Z*, ω_LO to test it densely."},
        "u_saturation_leg": {
            "g_xx_single_exciton_meV": (None if g_xx_single is None else round(g_xx_single * 1000, 3)),
            "u_sat_delocalized_10nm_mode_meV": (None if u_sat_delocalized is None else round(u_sat_delocalized * 1000, 3)),
            "status": "confinement-dependent — usable U_sat = g_xx·(a_B²/A_mode); rises "
                      "toward g_xx as you localize (moiré traps) but pays in inhomogeneous "
                      "broadening. NOT a single number, NOT closed by 1/μ alone; the "
                      "confinement↔broadening slope is a second theory input."},
        "u_dipolar_ceiling_eV": (None if u_dip_max is None else round(u_dip_max, 5)),
        "lambda_f_nm": lambda_f,
        "feasibility": verdict,
        "empirical_link_evidence": (
            "Across regimes the floor argument's monotonicity holds: GaAs (E_b~4 meV, "
            "weakly ionic) has a tiny RT Γ; TMDs (E_b~0.5 eV, ionic, Z*~0.4-1.2) sit at "
            "Γ~10-15 meV. Binding↑ tracks ionicity↑ tracks Γ↑ — the DFPT screen can fill "
            "this curve at cents/material to pin α_min."),
        "gap_to_close": (
            "The DIPOLAR U ceiling is the one open leg: it needs the oscillator-strength "
            "decay length λ_f from GW-BSE at ≥2 interlayer separations (the flagship). "
            "Until then the verdict is `undetermined` — NOT closed. This is precisely why "
            "the flagship run matters: it supplies the exponent that closes inequality #3."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[bound] Γ floor (min over DFPT TMDs, α≥{ALPHA_MIN}): {gamma_floor_min} meV")
    print(f"[bound]   per-material Γ floor: "
          + ", ".join(f"{k}={v['gamma_floor_meV']}" for k, v in floors.items()))
    print(f"[bound] need U > 0.71·Γ_floor = {round(0.71*gamma_floor_min,3)} meV to blockade")
    print(f"[bound] U_sat leg: g_xx(single)={round((g_xx_single or 0)*1000,1)} meV, "
          f"delocalized(10nm)={round((u_sat_delocalized or 0)*1000,3)} meV "
          f"→ confinement-dependent (not closed by 1/μ alone)")
    print(f"[bound] U_dip ceiling: {'λ_f unknown → PENDING flagship' if u_dip_max is None else round(u_dip_max*1000,3)}")
    print(f"[bound] FEASIBILITY: {verdict['verdict'].upper()} — {verdict.get('reason', verdict.get('interpretation',''))}")
    print(f"[bound] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
