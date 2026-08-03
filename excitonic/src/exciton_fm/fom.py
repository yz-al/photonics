"""
Blockade figure of merit from a Γ label and a U label, with provenance carried
through so a FOM built from model estimates is never mistaken for one built from
GW-BSE + EPW.

The pass condition is 300 K (locked — a cryogenic "yes" fails the program's own
pump-free / CMOS-integrable energy accounting, and relaxing T lets the search
rank on quiet phonons instead of on the interaction it exists to measure). But the
headline per material is NOT a 300 K pass/fail — it is the *crossover temperature*
T*: the temperature at which U = 0.71·Γ(T). Because U is ~T-independent and
Γ(T)=Γ0+γ_LO·n_LO(T) rises monotonically, blockade holds for T<T* and fails above.
Reporting T* (rather than a point) makes the negative result quantitative — "clears
only at T*=X K, short of 300 K by factor Y" — without letting the denominator
capture the ranking.
"""
from __future__ import annotations

import math

from .provenance import Label, FIRST_PRINCIPLES, not_run
from .frohlich import KB_meV, bose, gamma_of_T
from .targets import BLOCKADE_THRESHOLD


def blockade_fom(U: Label, Gamma: Label) -> dict:
    """Compute U/Γ and the blockade verdict from two Labels.

    Units: U in eV, Γ in meV -> convert. Returns a dict with the ratio, the
    verdict, and the *weakest* provenance tier among the inputs (a chain is only
    as trustworthy as its least-trusted link).
    """
    if U.value is None or Gamma.value is None:
        return {"U_over_Gamma": None, "meets_blockade": None,
                "provenance": "not_run",
                "reason": "one or both inputs not computed"}
    U_meV = U.value * 1000.0
    ratio = U_meV / Gamma.value if Gamma.value > 0 else None
    both_fp = (U.tier in FIRST_PRINCIPLES) and (Gamma.tier in FIRST_PRINCIPLES)
    weakest = _weakest_tier(U.tier, Gamma.tier)
    return {
        "U_over_Gamma": (None if ratio is None else round(ratio, 4)),
        "meets_blockade": (None if ratio is None else bool(ratio > BLOCKADE_THRESHOLD)),
        "threshold": BLOCKADE_THRESHOLD,
        "provenance": weakest,
        "first_principles": both_fp,
        "U": U.to_dict(), "Gamma": Gamma.to_dict(),
        "caveat": ("" if both_fp else
                   "FOM built from MODEL estimates — not a GW-BSE/EPW result; "
                   "for prioritization only."),
    }


def _weakest_tier(*tiers: str) -> str:
    from .provenance import TIERS
    order = {t: i for i, t in enumerate(TIERS)}
    return min(tiers, key=lambda t: order.get(t, 0))


def crossover_temperature(U_eV: float, gamma0_meV: float, gamma_LO_meV: float,
                          hw_LO_meV: float, threshold: float = BLOCKADE_THRESHOLD):
    """Temperature T* [K] at which U = threshold·Γ(T*), i.e. blockade crosses.

    Γ(T) = γ0 + γ_LO·n_LO(T), n_LO(T)=1/(exp(ħω_LO/kT)−1), monotonic in T. Blockade
    (U > thr·Γ) holds for T < T* and fails above. The FOM crosses at Γ(T*)=U/thr.

    Returns (T_star_K, crosses_at_any_T, reason). If U/thr ≤ γ0 (the T→0 floor),
    the interaction cannot beat the zero-temperature linewidth and blockade never
    opens at ANY temperature — the strongest possible negative for that material.
    """
    if None in (U_eV, gamma0_meV, gamma_LO_meV, hw_LO_meV):
        return None, None, "missing input"
    if hw_LO_meV <= 0 or gamma_LO_meV <= 0:
        return None, None, "non-physical Γ(T) parameters"
    target_meV = (U_eV * 1000.0) / threshold          # Γ value at which FOM crosses
    if target_meV <= gamma0_meV:
        return 0.0, False, (f"U/{threshold}={target_meV:.2f} meV ≤ Γ0={gamma0_meV:.2f} "
                            "meV: below the T→0 linewidth floor — never blockades")
    n_star = (target_meV - gamma0_meV) / gamma_LO_meV  # required LO occupation
    T_star = hw_LO_meV / (KB_meV * math.log1p(1.0 / n_star))
    return round(T_star, 1), True, "crosses at finite T*"


def blockade_vs_temperature(U: Label, gamma0_meV: float, gamma_LO_meV: float,
                            hw_LO_meV: float, T_report: float = 300.0,
                            gamma_tier: str = "frohlich_model") -> dict:
    """Full Γ(T)-curve blockade report for one material (the reporting standard).

    Instead of a bare 300 K verdict, returns the crossover temperature T*, the
    300 K headroom, the shortfall factor (how many× U must grow to clear at 300 K),
    and a sampled Γ(T)/ratio curve. U is a Label (provenance carried); the Γ(T)
    parameters come from EPW (γ0, γ_LO) + DFPT (ħω_LO) — model-tier until then.
    """
    if U.value is None:
        return {"T_star_K": None, "meets_at_300K": None, "provenance": "not_run",
                "reason": "U not computed"}
    T_star, crosses, reason = crossover_temperature(
        U.value, gamma0_meV, gamma_LO_meV, hw_LO_meV)
    g_report = gamma_of_T(T_report, hw_LO_meV, gamma_LO_meV, gamma0_meV)
    U_meV = U.value * 1000.0
    ratio_report = U_meV / g_report if g_report > 0 else None
    # shortfall: factor by which U must grow to clear at T_report (>1 means short)
    shortfall = (BLOCKADE_THRESHOLD * g_report / U_meV) if U_meV > 0 else None
    weakest = _weakest_tier(U.tier, gamma_tier)
    both_fp = (U.tier in FIRST_PRINCIPLES) and (gamma_tier in FIRST_PRINCIPLES)
    curve = []
    for T in (4, 77, 150, 200, 250, 300):
        g = gamma_of_T(float(T), hw_LO_meV, gamma_LO_meV, gamma0_meV)
        r = U_meV / g if g > 0 else None
        curve.append({"T_K": T, "gamma_meV": round(g, 3),
                      "U_over_gamma": (None if r is None else round(r, 4)),
                      "meets": (None if r is None else bool(r > BLOCKADE_THRESHOLD))})
    return {
        "T_star_K": T_star,
        "crosses_at_any_T": crosses,
        "crossover_reason": reason,
        "meets_at_300K": (None if ratio_report is None
                          else bool(ratio_report > BLOCKADE_THRESHOLD)),
        "U_over_gamma_at_300K": (None if ratio_report is None else round(ratio_report, 4)),
        "gamma_at_300K_meV": round(g_report, 3),
        "shortfall_factor_at_300K": (None if shortfall is None else round(shortfall, 2)),
        "threshold": BLOCKADE_THRESHOLD,
        "provenance": weakest,
        "first_principles": both_fp,
        "U": U.to_dict(),
        "gamma_T_model": {"gamma0_meV": gamma0_meV, "gamma_LO_meV": gamma_LO_meV,
                          "hw_LO_meV": hw_LO_meV},
        "gamma_T_curve": curve,
        "caveat": ("" if both_fp else
                   "Γ(T) params and/or U are MODEL-tier — not GW-BSE/EPW. T* is a "
                   "demonstration of the criterion, not a first-principles result."),
    }
