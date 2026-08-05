#!/usr/bin/env python3
"""
Phase 2 — blockade CROSSOVER-TEMPERATURE criterion (demonstration + self-test).

Locks the decision: the pass condition stays 300 K, and the per-material headline
is the crossover temperature T* — the temperature at which U = 0.71·Γ(T) — reported
alongside the full Γ(T) curve, rather than a bare 300 K pass/fail. This
  (a) keeps the ranking on U (the interaction), not on the phonon linewidth we are
      trying to hold fixed — at a relaxed target the search would sort on quiet
      phonons instead of strong interaction;
  (b) makes the negative result quantitative: "clears only at T*=X K, short of
      300 K by factor Y", and flags any material that crosses near RT.

This script does NOT compute a physics verdict. It (1) self-tests that the
inversion is correct (Γ(T*) = U/0.71), (2) demonstrates the criterion on the
best-measured cryogenic anchor to show the literature negative is reproduced, and
(3) shows the criterion would surface a near-RT crossover if one existed. Real
per-material T* comes from GW-BSE (U) + EPW (Γ(T)); inputs here are model/measured
and flagged as such.

    python excitonic/scripts/phase2_crossover_demo.py
Writes excitonic/data/manifests/phase2_crossover_demo.json.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.fom import crossover_temperature, blockade_vs_temperature  # noqa: E402
from exciton_fm.frohlich import gamma_of_T                                 # noqa: E402
from exciton_fm.provenance import Label                                     # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                   "phase2_crossover_demo.json"))

# A MoS2-like Γ(T) model (γ0≈2 meV, γ_LO≈15 meV, ħω_LO≈48 meV) — used only to
# exercise the criterion. Real γ0/γ_LO come from EPW; ħω_LO from DFPT.
G0, GLO, HW = 2.0, 15.0, 48.0


def _selftest():
    """Γ(T*) must equal U/0.71 (the definition of the crossover)."""
    checks = []
    for U in (0.006, 0.010, 0.020, 0.050):
        Ts, crosses, _ = crossover_temperature(U, G0, GLO, HW)
        if not crosses:
            checks.append({"U_eV": U, "crosses": False})
            continue
        g_at = gamma_of_T(Ts, HW, GLO, G0)
        target = U * 1000.0 / 0.71
        checks.append({"U_eV": U, "T_star_K": Ts, "gamma_at_Tstar_meV": round(g_at, 3),
                       "target_meV": round(target, 3),
                       "ok": bool(abs(g_at - target) < 0.05)})
    return checks


def main() -> int:
    selftest = _selftest()
    ok = all(c.get("ok", True) for c in selftest)

    # (2) best-measured cryogenic anchor: U/Γ ≈ 0.05 at ~4 K (best reported).
    g4 = gamma_of_T(4.0, HW, GLO, G0)
    U_lit_eV = round(0.05 * g4 / 1000.0, 6)
    U_lit = Label("U", U_lit_eV, "eV", "measured",
                  source="back-out of best reported U/Γ≈0.05 at ~4 K (cryogenic)")
    lit = blockade_vs_temperature(U_lit, G0, GLO, HW)

    # (3) an ILLUSTRATIVE material engineered to cross near RT, to show the
    # criterion surfaces it (NOT a prediction — a demonstration of sensitivity).
    #   choose U so that Γ(250 K) = U/0.71.
    U_hi_eV = round(0.71 * gamma_of_T(250.0, HW, GLO, G0) / 1000.0, 6)
    U_hi = Label("U", U_hi_eV, "eV", "saturation_model",
                 source="illustrative: U set so T*≈250 K (demonstrates surfacing)")
    hi = blockade_vs_temperature(U_hi, G0, GLO, HW)

    manifest = {
        "criterion": "300 K pass condition (LOCKED); per-material headline = crossover "
                     "temperature T* where U=0.71·Γ(T); full Γ(T) curve reported.",
        "why_not_relax_T": [
            "A 77 K yes fails the program's pump-free/CMOS energy accounting — a "
            "cryostat is standing power (thermal hold) at scale, the same argument we "
            "used to kill 44 pJ/MAC architectures, and it does not amortize.",
            "At a relaxed target the ranking sorts on the phonon linewidth (quiet "
            "phonons) not on U — capturing the search with the variable we hold fixed; "
            "we would learn about dephasing (already characterized), not interaction.",
            "The cryogenic corner is not empty: best measured U/Γ≈0.05 at a few K is "
            "~14× short at that temperature and never crosses — relaxing T past 77 K "
            "re-asks a question the literature already answered negatively.",
        ],
        "gamma_T_model_used": {"gamma0_meV": G0, "gamma_LO_meV": GLO, "hw_LO_meV": HW,
                               "note": "illustrative; real γ0/γ_LO from EPW, ħω_LO from DFPT"},
        "selftest_inversion_ok": ok,
        "selftest": selftest,
        "best_measured_cryo_anchor": lit,
        "illustrative_near_RT_crosser": hi,
        "rigor": "No physics verdict. Inputs are model/measured and flagged; real "
                 "T* requires GW-BSE U + EPW Γ(T). This fixes the CRITERION and shows "
                 "it reproduces the known cryogenic negative and would surface a near-RT "
                 "crosser if one existed.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[crossover] inversion self-test ok = {ok}")
    print(f"[crossover] best cryo anchor (U/Γ≈0.05 @4K): U≈{U_lit_eV*1000:.3f} meV -> "
          f"T*={lit['T_star_K']} K (crosses={lit['crosses_at_any_T']}), "
          f"300 K ratio={lit['U_over_gamma_at_300K']}, shortfall={lit['shortfall_factor_at_300K']}×")
    print(f"    reason: {lit['crossover_reason']}")
    print(f"[crossover] illustrative near-RT crosser: U≈{U_hi_eV*1000:.1f} meV -> "
          f"T*={hi['T_star_K']} K, 300 K ratio={hi['U_over_gamma_at_300K']}, "
          f"meets@300K={hi['meets_at_300K']}")
    print(f"[crossover] wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
