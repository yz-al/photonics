"""
Feasibility bound — close the question with inequalities, not a search.

A search that finds nothing is vulnerable to "you looked in the wrong place." A
bound is not. If the largest U any thermally-stable material can have is smaller
than 0.71× the smallest Γ(300 K) it can have, the feasible set is EMPTY and no
search is required — the same shape as the Kramers-Kronig result that closed the
Kerr route (causality caps nonlinear-phase-shift / loss).

Three linked inequalities, each in DFPT-computable parameters, each DERIVED:

  (Γ floor)   thermal-stable binding ⟹ ionicity ≥ ι_min ⟹ Fröhlich α ≥ α_min
              ⟹ Γ(300 K) ≥ Γ_floor.  [Born charges Z*, LO-TO — from the DFPT screen]
  (U_sat cap) saturation U ∝ 1/μ, and E_b>threshold with bounded ε forces μ ≥ μ_min
              ⟹ U_sat ≤ U_sat_max.
  (U_dip cap) dipolar U rises with layer separation d, but f falls ~exp(-d/λ_f);
              strong coupling needs f ≥ f_min ⟹ d ≤ d_max ⟹ U_dip ≤ U_dip(d_max).

Feasibility: if max(U_sat_max, U_dip_max) < 0.71·Γ_floor  → CLOSED.

RIGOR / tiers. The Γ floor's *phonon* half (α → Γ) is a real lower bound. The
*stability → ionicity* link is the crux and is semi-empirical until proven; it is
returned flagged. The U_dip cap needs λ_f from GW-BSE at ≥2 separations (the
flagship) — until then the dipolar ceiling, and hence the feasibility verdict, is
`undetermined`. Nothing here is presented as closed before its inputs exist.
"""
from __future__ import annotations

import math

from .frohlich import bose, C_CAL

KB_meV = 0.0861733
RY_meV = 13605.693


# --- (Γ floor) --------------------------------------------------------------
def frohlich_alpha(eps0: float, eps_inf: float, hw_LO_meV: float, mu: float) -> float:
    """Dimensionless Fröhlich coupling α (Fröhlich 1954)."""
    if min(eps0, eps_inf, hw_LO_meV, mu) <= 0:
        return float("nan")
    inv = (1.0 / eps_inf) - (1.0 / eps0)
    return inv * math.sqrt(RY_meV / hw_LO_meV) * math.sqrt(mu) if inv > 0 else 0.0


def gamma_LO_floor_300K(alpha: float, hw_LO_meV: float, gamma0_meV: float = 0.0) -> float:
    """RT LO-phonon linewidth Γ_LO(300 K) = γ_LO·n_LO(300 K) for coupling α.

    This is a genuine LOWER bound on Γ(300 K): the phonon dephasing is unavoidable
    at finite T. gamma0 (radiative/disorder) only ADDS. γ_LO = C·α·ħω_LO calibrated
    to the TMD anchor (frohlich.C_CAL).
    """
    if not math.isfinite(alpha) or hw_LO_meV <= 0:
        return float("nan")
    return gamma0_meV + C_CAL * alpha * hw_LO_meV * bose(hw_LO_meV, 300.0)


def eps0_from_born(eps_inf: float, Z_born: float, mu_amu: float,
                   hw_TO_meV: float, cell_volume_A3: float, n_pairs: int = 1) -> float:
    """Static ε0 = ε∞ + ionic part, from Born charge Z* and the TO frequency.

    Ionic dielectric response (one polar mode, SI→atomic collapsed to a prefactor):
    Δε ≈ (e²/ε0_vac) · Z*² / (V · μ_ion · ω_TO²). Uses the reduced ionic mass and
    the cell volume. A standard estimate — flagged, used only to size the Fröhlich
    contrast (1/ε∞ − 1/ε0); the SIGN and monotonicity (more ionic ⇒ larger Δε) are
    what the floor argument needs, and those are robust.
    """
    if min(eps_inf, abs(Z_born), mu_amu, hw_TO_meV, cell_volume_A3) <= 0:
        return eps_inf
    # prefactor C_ion [Å³·amu·meV²] calibrated so a Z*≈1, μ≈30 amu, ω≈50 meV,
    # V≈100 Å³ polar mode gives Δε≈1 (order-of-magnitude; overridable per study).
    C_ion = 3.0e5
    d_eps = C_ion * (Z_born ** 2) / (cell_volume_A3 * mu_amu * hw_TO_meV ** 2)
    return eps_inf + d_eps


# --- (U_sat ceiling) --------------------------------------------------------
def u_saturation_ceiling(E_b_eV: float, mu_min: float, eps_eff: float,
                         area_nm2: float) -> float:
    """Saturation U ceiling in mode area A: g_xx = 6·E_b·a_B², a_B = (ε/μ)a0.

    g_xx ∝ 1/μ, so the SMALLEST allowed μ (set by thermal stability) gives the
    LARGEST U_sat. Returns U_sat_max = g_xx(μ_min)/A [eV].
    """
    A0_nm = 0.052917721
    if min(E_b_eV, mu_min, eps_eff, area_nm2) <= 0:
        return float("nan")
    a_B = (eps_eff / mu_min) * A0_nm            # nm
    g_xx = 6.0 * E_b_eV * a_B ** 2              # eV·nm²
    return g_xx / area_nm2                       # eV


# --- (U_dip ceiling) — needs λ_f from the flagship --------------------------
def max_usable_separation(lambda_f_nm: float, f_min_frac: float) -> float:
    """d_max where f(d)=f0·exp(-(d-d0)/λ_f) drops to f_min_frac·f0.

    d_max − d0 = λ_f · ln(1/f_min_frac). Beyond d_max the exciton is too dark to
    strong-couple / read out, so U_dip cannot be used past it.
    """
    if lambda_f_nm is None or lambda_f_nm <= 0 or not (0 < f_min_frac < 1):
        return None
    return lambda_f_nm * math.log(1.0 / f_min_frac)


def u_dipolar_ceiling(lambda_f_nm, f_min_frac, U_dip_of_d, d0_nm=0.6):
    """U_dip ceiling = U_dip evaluated at the max usable separation.

    U_dip_of_d(d) is the dipolar-U model (rises with d). The cap is U_dip(d0+d_max_gain)
    where the gain is bounded by the oscillator-strength decay. Returns None if λ_f is
    unknown — the verdict is then `undetermined`, NOT closed (this is the flagship's job).
    """
    dd = max_usable_separation(lambda_f_nm, f_min_frac)
    if dd is None:
        return None
    return U_dip_of_d(d0_nm + dd)


# --- feasibility ------------------------------------------------------------
def feasibility(gamma_floor_meV: float, u_sat_max_eV: float,
                u_dip_max_eV, threshold: float = 0.71) -> dict:
    """Is the feasible set empty? max usable U vs threshold·Γ_floor.

    U in eV → meV for the ratio (Γ in meV). If u_dip_max is None the dipolar cap is
    unknown ⇒ verdict `undetermined` (the bound is NOT closed without λ_f).
    """
    if gamma_floor_meV is None or not math.isfinite(gamma_floor_meV):
        return {"verdict": "undetermined", "reason": "Γ floor not available"}
    need_meV = threshold * gamma_floor_meV
    sat_meV = None if u_sat_max_eV is None else u_sat_max_eV * 1000.0
    dip_meV = None if u_dip_max_eV is None else u_dip_max_eV * 1000.0
    if dip_meV is None:
        return {"verdict": "undetermined",
                "reason": "dipolar U ceiling needs λ_f from GW-BSE at ≥2 separations "
                          "(the flagship). Saturation cap alone cannot close the bound.",
                "gamma_floor_meV": round(gamma_floor_meV, 3),
                "need_U_gt_meV": round(need_meV, 3),
                "u_sat_max_meV": (None if sat_meV is None else round(sat_meV, 4))}
    max_U_meV = max(v for v in (sat_meV, dip_meV) if v is not None)
    closed = max_U_meV < need_meV
    return {"verdict": "closed" if closed else "open",
            "gamma_floor_meV": round(gamma_floor_meV, 3),
            "threshold": threshold,
            "need_U_gt_meV": round(need_meV, 3),
            "max_usable_U_meV": round(max_U_meV, 4),
            "u_sat_max_meV": (None if sat_meV is None else round(sat_meV, 4)),
            "u_dip_max_meV": (None if dip_meV is None else round(dip_meV, 4)),
            "interpretation": ("max achievable U is below 0.71·Γ_floor → feasible set "
                               "EMPTY, no search needed" if closed else
                               "a material could in principle clear the bound → search/"
                               "flagship decides")}
