#!/usr/bin/env python3
"""
Phase 2 — feasibility-bound SYNTHESIS: the whole question in Γ_floor + two exponents.

After Track C, the blockade feasibility bound collapses to a small, decision-shaped
set of quantities:

  1. Γ_floor(E_b)  — the RT phonon-linewidth floor. DATA-DERIVED (phase2_bound_alpha):
     the stability⟹coupling chain is real (E_b↔Fröhlich ρ=0.62) but binding-dependent,
     so the floor is a curve, not a constant. Robustly-bound materials floor at ~6-15 meV.
  2. q_sat  — the SATURATION channel's confinement-broadening exponent. Localizing to
     raise U (moiré) costs inhomogeneous broadening ∝ L^-q; U ∝ L^-2. Leg CLOSES iff
     q > 2. UNMEASURED.
  3. λ_f / U-slope — the DIPOLAR channel's oscillator-strength decay vs U-rise. Leg
     CLOSES iff f falls faster than U rises. UNMEASURED (needs interlayer BSE).

So: **the entire room-temperature blockade question reduces to two exponents** — the
moiré confinement-broadening slope q_sat and the interlayer oscillator-strength decay
λ_f — measured against the data-derived Γ floor. Both are single, targeted measurements.
Neither is known yet ⇒ the bound is UNDETERMINED, but it is no longer vague: it is two
numbers away from closed-or-open.

    python excitonic/scripts/phase2_bound_synthesis.py
Writes excitonic/data/manifests/phase2_bound_synthesis.json.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.bound import saturation_window                       # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                   "phase2_bound_synthesis.json"))

# saturation-channel inputs (MoS2-like): g_xx≈6·E_b (single-site), a_B≈1 nm, Γ_ph≈6 meV.
G_XX_meV = 6 * 500.0        # 6·E_b(=0.5 eV)  at the single-exciton scale
A_B_nm = 1.0
GAMMA_PH_meV = 6.0          # the data-derived Γ floor (robustly-bound regime)
GAMMA_INH0_meV = 5.0        # inhomogeneous broadening at L=a_B (moiré-trap scale; literature ~meV)


def main() -> int:
    # demonstrate the critical-exponent behavior: q below vs above 2.
    q_open = saturation_window(G_XX_meV, A_B_nm, GAMMA_PH_meV, GAMMA_INH0_meV, q=1.5)
    q_crit = saturation_window(G_XX_meV, A_B_nm, GAMMA_PH_meV, GAMMA_INH0_meV, q=2.0)
    q_closed = saturation_window(G_XX_meV, A_B_nm, GAMMA_PH_meV, GAMMA_INH0_meV, q=3.0)

    manifest = {
        "headline": "Room-temperature blockade feasibility reduces to Γ_floor(E_b) + TWO "
                    "decisive exponents: q_sat (moiré confinement-broadening) and λ_f "
                    "(interlayer oscillator-strength decay). Both unmeasured ⇒ undetermined, "
                    "but the question is now two targeted measurements from resolved.",
        "leg_1_gamma_floor": {
            "status": "DATA-DERIVED (binding-dependent)",
            "robustly_bound_floor_meV": "≈6-15 (the 4 DFPT TMDs)",
            "source": "phase2_bound_alpha.json; E_b↔Fröhlich ρ=0.62",
            "target_U_to_blockade_meV": "0.71·Γ_floor ≈ 4-11 (robustly bound)",
        },
        "leg_2_saturation_exponent_q_sat": {
            "critical": "q_crit = 2 (since U_sat ∝ L^-2)",
            "closes_iff": "q_sat > 2",
            "status": "UNMEASURED — needs moiré-exciton linewidth vs confinement slope",
            "demo_q1.5_open": {"verdict": q_open["verdict"], "F_tightest": q_open["F_tightest"]},
            "demo_q2.0_crit": {"verdict": q_crit["verdict"], "F_tightest": q_crit["F_tightest"]},
            "demo_q3.0_closed": {"verdict": q_closed["verdict"], "F_tightest": q_closed["F_tightest"]},
        },
        "leg_3_dipolar_exponent_lambda_f": {
            "closes_iff": "f falls faster than U_dip rises with separation d",
            "status": "UNMEASURED — needs interlayer GW-BSE at ≥2 separations (BSE build "
                      "currently broken; see feasibility_bound.md fork)",
            "machinery": "exciton_fm.dipolar_window (built, self-tested)",
        },
        "overall_verdict": "UNDETERMINED — but fully localized to two exponents.",
        "what_closes_it": [
            "q_sat > 2  (saturation/moiré leg closed)  — measurable: moiré-exciton "
            "linewidth vs trap confinement (published data may already bound it).",
            "f-decay faster than U-rise (dipolar leg closed) — measurable: the interlayer "
            "BSE flagship (once the BSE toolchain is fixed).",
            "If BOTH close against the data-derived Γ floor ⇒ feasible set empty ⇒ "
            "Phase 3 never runs. If either stays open, that corner IS the search space.",
        ],
        "rigor": "The saturation inputs (g_xx, Γ_inh0) are order-of-magnitude/literature; "
                 "the CRITICAL-EXPONENT structure (q_crit=2) is exact given U_sat ∝ L^-2. "
                 "This reframes, it does not yet resolve — but it names the two numbers.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("[synth] Blockade feasibility = Γ_floor(E_b) + two exponents:")
    print(f"[synth]   leg1 Γ_floor: DATA-DERIVED, ~6-15 meV (robustly bound) → need U>4-11 meV")
    print(f"[synth]   leg2 q_sat (crit=2): q=1.5 → {q_open['verdict']}; "
          f"q=3.0 → {q_closed['verdict']}  [UNMEASURED]")
    print(f"[synth]   leg3 λ_f (dipolar): UNMEASURED (interlayer BSE)")
    print(f"[synth] verdict: UNDETERMINED, but reduced to TWO measurable exponents.")
    print(f"[synth] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
