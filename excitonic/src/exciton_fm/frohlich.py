"""
Γ(300 K) — the room-temperature homogeneous exciton linewidth.

Two ways to get it, sharply distinguished by provenance:

  (A) `estimate_gamma_300K(...)`  — an analytic Fröhlich/LO-phonon model.
      Tier = 'frohlich_model'. This is an ESTIMATE for prioritization and a
      sanity floor; it is NOT an EPW result and must never be reported as one.

  (B) `parse_epw_linewidth(...)`  — the real value from a completed EPW run.
      Tier = 'epw'. If the file is missing/unfinished it returns a `not_run`
      Label (value=None) — we never fabricate an EPW number.

Model physics (standard, e.g. Rudin-Reinecke 1990; Selig et al. Nat.Commun. 2016):
    Γ(T) = Γ0 + γ_LO · n_LO(T),   n_LO(T) = 1 / (exp(ħω_LO / kB T) − 1)
The LO coupling γ_LO scales with the dimensionless Fröhlich coupling α
(Fröhlich 1954):  α = (e²/ħ) · sqrt(m*/(2ħω_LO)) · (1/ε∞ − 1/ε0).
The α→γ_LO proportionality is calibrated to the monolayer-TMD anchor
(Γ(300 K) ≈ 5–15 meV; Selig/Moody) and is only a model-level mapping.
"""
from __future__ import annotations

import math

from .provenance import Label, not_run

KB_meV = 0.0861733       # Boltzmann constant [meV/K]
HBAR_eVs = 6.582119e-16  # ħ [eV·s]

# Calibration: γ_LO = C_CAL · α · ħω_LO, chosen so a MoS2-like anchor
# (α≈0.4, ħω_LO≈48 meV, Γ0≈2 meV) gives Γ(300 K)≈10 meV. See tests/anchors.
C_CAL = 2.9


def bose(hw_meV: float, T: float) -> float:
    """LO-phonon Bose occupation n_LO(T)."""
    if T <= 0 or hw_meV <= 0:
        return 0.0
    x = hw_meV / (KB_meV * T)
    return 1.0 / (math.expm1(x))


def frohlich_alpha(eps0: float, eps_inf: float, hw_LO_meV: float, m_eff: float) -> float:
    """Dimensionless Fröhlich coupling α (Fröhlich 1954).

    m_eff in units of the electron mass; energies in meV. Uses the standard
    α = (1/ε∞ − 1/ε0) · sqrt(Ry / ħω_LO) · sqrt(m*)  with Ry = 13605.7 meV.
    """
    RY_meV = 13605.693
    if hw_LO_meV <= 0 or eps0 <= 0 or eps_inf <= 0:
        return float("nan")
    inv = (1.0 / eps_inf) - (1.0 / eps0)
    if inv <= 0:
        return 0.0
    return inv * math.sqrt(RY_meV / hw_LO_meV) * math.sqrt(m_eff)


def gamma_of_T(T: float, hw_LO_meV: float, gamma_LO_meV: float,
               gamma0_meV: float = 2.0) -> float:
    """Γ(T) = Γ0 + γ_LO · n_LO(T)  [meV]."""
    return gamma0_meV + gamma_LO_meV * bose(hw_LO_meV, T)


def estimate_gamma_300K(hw_LO_meV: float, *, alpha: float | None = None,
                        eps0: float | None = None, eps_inf: float | None = None,
                        m_eff: float | None = None, gamma0_meV: float = 2.0,
                        T: float = 300.0) -> Label:
    """Model estimate of Γ(300 K). Provide α directly, or (eps0, eps_inf, m_eff).

    Returns a Label with tier='frohlich_model' (an estimate, not EPW).
    """
    if alpha is None:
        if None in (eps0, eps_inf, m_eff):
            return not_run("Gamma_300K", "meV",
                           "insufficient inputs for the Fröhlich model estimate")
        alpha = frohlich_alpha(eps0, eps_inf, hw_LO_meV, m_eff)
    if not math.isfinite(alpha) or hw_LO_meV <= 0:
        return not_run("Gamma_300K", "meV", "non-finite Fröhlich inputs")
    gamma_LO = C_CAL * alpha * hw_LO_meV
    g = gamma_of_T(T, hw_LO_meV, gamma_LO, gamma0_meV)
    return Label(
        name="Gamma_300K", value=round(g, 3), unit="meV", tier="frohlich_model",
        source=f"Fröhlich model (α={alpha:.3f}, ħω_LO={hw_LO_meV:.1f} meV, "
               f"γ_LO={gamma_LO:.1f} meV, C_cal={C_CAL})",
        uncertainty=round(0.4 * g, 3),  # ~40% model spread; calibrated to anchors
        notes="MODEL estimate for prioritization; replace with EPW (tier 'epw').",
    )


def parse_epw_linewidth(epw_out_path: str, T: float = 300.0) -> Label:
    """Extract Γ(T) from a completed EPW electron-phonon self-energy calculation.

    Looks for EPW's imaginary self-energy / linewidth output. If the expected
    output is absent or the run did not finish, returns a `not_run` Label —
    NEVER a fabricated value.
    """
    import os
    if not os.path.exists(epw_out_path):
        return not_run("Gamma_300K", "meV", f"EPW output not found: {epw_out_path}")
    try:
        with open(epw_out_path) as fh:
            text = fh.read()
    except OSError as e:
        return not_run("Gamma_300K", "meV", f"could not read EPW output: {e}")
    if "JOB DONE" not in text and "EPW.bib" not in text:
        return not_run("Gamma_300K", "meV",
                       "EPW run did not complete (no 'JOB DONE'); refusing to guess")
    # Real parsing of the linewidth (2*Im Σ) at the exciton/band-edge k-point and
    # temperature T is code-version specific and is implemented against actual
    # EPW output files (linewidth.elself / *.lambda). Until such a file is present
    # and parsed, do not emit a number.
    return not_run("Gamma_300K", "meV",
                   "EPW completed but linewidth parser not yet wired to this "
                   "output format; no value emitted (no fabrication)")
