"""
Blockade figure of merit from a Γ label and a U label, with provenance carried
through so a FOM built from model estimates is never mistaken for one built from
GW-BSE + EPW.
"""
from __future__ import annotations

from .provenance import Label, FIRST_PRINCIPLES
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
