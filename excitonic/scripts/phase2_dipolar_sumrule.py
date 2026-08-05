#!/usr/bin/env python3
"""
Phase 2 — close the dipolar leg analytically (the KK / oscillator-strength bound).

Evaluates the two-site sum-rule feasibility inequality (★) over the literature-
grounded parameter box, so the dipolar leg gets a verdict WITHOUT waiting on the
BSE. All parameters are measured/literature (see phase2_lambda_f_proxy's
literature_check block); the trade-off f∝x, U∝(1-x)² is the model.

    python excitonic/scripts/phase2_dipolar_sumrule.py
Writes excitonic/data/manifests/phase2_dipolar_sumrule.json.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.dipolar_sumrule import feasible_window, scan, BLOCKADE_THRESHOLD  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                   "phase2_dipolar_sumrule.json"))

# Literature-grounded ranges (all meV):
GAMMA_RANGE = (6.0, 15.0)      # RT phonon linewidth floor (DFPT TMDs)
U_MAX_RANGE = (8.0, 20.0)      # measured on-site dipolar interaction (npj 2D Mater 2020)
OMEGA0_RANGE = (20.0, 50.0)    # intralayer TMD RT Rabi splitting (demonstrated strong coupling)


def main() -> int:
    # Two strong-coupling criteria: onset (Ω>Γ) vs resolved doublet (Ω>2Γ, device-real).
    result = scan(GAMMA_RANGE, U_MAX_RANGE, OMEGA0_RANGE, BLOCKADE_THRESHOLD, sc_margin=1.0)
    result_resolved = scan(GAMMA_RANGE, U_MAX_RANGE, OMEGA0_RANGE, BLOCKADE_THRESHOLD,
                           sc_margin=2.0)

    manifest = {
        "question": "Does the KK / oscillator-strength sum rule CLOSE the dipolar leg "
                    "analytically (no BSE)?",
        "model": "Two-site: f(x)=x·f0 (brightness borrowed via layer-B weight x), "
                 "U(x)=(1-x)²·U_max (dipole ∝ 1-x). One parameter x∈[0,1].",
        "master_inequality": "OPEN iff (Γ/Ω0)² < 1 − √(0.71·Γ/U_max)   ... (★)",
        "necessary_condition": "Even the max-dipole (x→0) exciton needs U_max > 0.71·Γ; "
                               "else CLOSED at every x with no strong-coupling analysis.",
        "parameter_box_meV": {"gamma_floor": GAMMA_RANGE, "U_max_dipolar": U_MAX_RANGE,
                              "omega0_intralayer_Rabi": OMEGA0_RANGE},
        "provenance": "model (two-site trade-off) + measured (Γ, U_max, Ω0). NOT GW-BSE.",
        "scan_onset_sc_OmegaGtGamma": result,
        "scan_resolved_sc_OmegaGt2Gamma": result_resolved,
        "verdict": (
            f"The bound does NOT cleanly close the leg, but it is CRITERION-SENSITIVE. "
            f"Under bare strong-coupling onset (Ω>Γ) it is open in "
            f"{result['open_fraction']*100:.0f}% of the literature box (nominal "
            f"{result['nominal_midpoint']['verdict'].upper()}); under a resolved doublet "
            f"(Ω>2Γ, device-realistic) it collapses to "
            f"{result_resolved['open_fraction']*100:.0f}% (nominal "
            f"{result_resolved['nominal_midpoint']['verdict'].upper()}). So the dipolar "
            "route lives ONLY on the favorable corner (low Γ≈6, high U_max≈20, high Ω0≈50 "
            "meV) AND only if a bare Rabi onset counts as usable. The pessimistic corner is "
            "CLOSED outright (U_max<0.71·Γ — blockade fails at every brightness). Honest "
            "state: not a clean analytic 'no', but 'closed except on a thin corner that a "
            "resolved-polariter criterion erases'."),
        "what_the_bse_or_experiment_must_show": [
            "U_max — the x→0 (dark, max-dipole) on-site interaction. If U_max ≲ 0.71·Γ_floor "
            "the leg is CLOSED with no further work (necessary condition fails).",
            "Whether an interlayer exciton in the thin open window (x≈0.1–0.3) reaches "
            "Ω>Γ at 300 K — i.e. is bright enough to strong-couple while still dipolar "
            "enough to blockade. Ω0·√x vs Γ.",
        ],
        "relation_to_KK": "The two-site trade-off is the discrete realization of the "
                          "Kramers-Kronig linkage: layer-B weight x raises the imaginary "
                          "part (absorption/f) and lowers the static real part (dipole/U). "
                          "The single-photon-χ³ causality bound (Shapiro 2006; Gea-Banacloche "
                          "2010) is the continuum analog — precedent that this closure shape "
                          "is real.",
        "rigor": "Leading-order two-site model; a stricter strong-coupling criterion (Ω>2Γ "
                 "for a well-resolved doublet) shrinks the window further, tightening toward "
                 "CLOSED. This bound REPLACES 'undetermined' with 'closed except in a thin, "
                 "named corner' — the honest analytic state of the dipolar leg.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("[sumrule] Master inequality (★): OPEN iff (sc·Γ/Ω0)² < 1 − √(0.71·Γ/U_max)")
    print(f"[sumrule] onset  Ω>Γ  : open in {result['open_fraction']*100:.0f}% of box; "
          f"nominal {result['nominal_midpoint']['verdict'].upper()}")
    print(f"[sumrule] resolved Ω>2Γ: open in {result_resolved['open_fraction']*100:.0f}% of box; "
          f"nominal {result_resolved['nominal_midpoint']['verdict'].upper()}")
    print(f"[sumrule]   optimistic corner (Γ6,U20,Ω50): "
          f"{result['optimistic_corner']['verdict'].upper()}")
    print(f"[sumrule]   pessimistic corner (Γ15,U8,Ω20): "
          f"{result['pessimistic_corner']['verdict'].upper()} — U_max<0.71Γ, closed at every x")
    print(f"[sumrule] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
