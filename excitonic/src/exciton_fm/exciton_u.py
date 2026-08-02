"""
U — the exciton–exciton interaction (per-particle optical nonlinearity).

Two channels, two provenances:

  (A) model estimates (tier 'saturation_model'):
      - `u_saturation(E_b, a_B, area)` — the saturation / exchange interaction that
        dominates for Wannier excitons (Ciuti et al. PRB 1998; Tassone-Yamamoto):
        g_xx ≈ 6 · E_b · a_B²   (energy·area); U per exciton in mode area A = g_xx/A.
        This U is tied to the Bohr radius a_B — the exact coupling the brief wants
        to *break* with a dipolar source.
      - `u_dipolar(dipole_d, area, eps)` — interlayer/dipolar excitons interact via
        an oriented-dipole (capacitor) term U_dip ≈ e² d / (ε0 ε A), which
        decouples U from a_B and thus from the thermal-stability tradeoff.

  (B) `parse_bse_biexciton(...)` — the real U from a completed BSE biexciton /
      exciton-saturation calculation (tier 'gw_bse'). Missing/unfinished -> a
      `not_run` Label; never a fabricated first-principles number.

All model helpers return provenance-tagged Labels, so a model U is never mistaken
for a GW-BSE U.
"""
from __future__ import annotations

import math

from .provenance import Label, not_run

# physical constants
A0_nm = 0.052917721       # Bohr radius [nm]
RY_eV = 13.605693         # Rydberg [eV]
E2_OVER_4PIEPS0_eV_nm = 1.439964  # e²/(4πε0) [eV·nm]


def bohr_radius_2D(m_eff_reduced: float, eps: float) -> float:
    """Hydrogenic exciton Bohr radius a_B [nm] = (ε/μ)·a0.

    A first-order estimate; true 2D excitons feel non-local Rytova-Keldysh
    screening that shrinks a_B and raises E_b — this is a model input, flagged.
    """
    if m_eff_reduced <= 0 or eps <= 0:
        return float("nan")
    return (eps / m_eff_reduced) * A0_nm


def u_saturation(E_b_eV: float, a_B_nm: float, area_nm2: float,
                 c_prefactor: float = 6.0) -> Label:
    """Saturation/exchange U for one exciton in mode area `area_nm2`.

    g_xx = c · E_b · a_B²  (eV·nm²);  U = g_xx / area.
    """
    if E_b_eV is None or a_B_nm is None or area_nm2 is None or area_nm2 <= 0:
        return not_run("U", "eV", "insufficient inputs for the saturation-U estimate")
    if not (math.isfinite(E_b_eV) and math.isfinite(a_B_nm)):
        return not_run("U", "eV", "non-finite inputs for saturation-U")
    g_xx = c_prefactor * E_b_eV * a_B_nm ** 2      # eV·nm²
    U = g_xx / area_nm2                            # eV
    return Label(
        name="U", value=round(U, 5), unit="eV", tier="saturation_model",
        source=f"saturation/exchange model g_xx={c_prefactor}·E_b·a_B² "
               f"(E_b={E_b_eV:.3f} eV, a_B={a_B_nm:.2f} nm, A={area_nm2:.1f} nm²)",
        uncertainty=round(0.5 * U, 5),
        notes="MODEL estimate (Ciuti/Tassone-Yamamoto); a_B-tied. Replace with BSE "
              "biexciton/saturation (tier 'gw_bse').",
    )


def u_dipolar(dipole_d_nm: float, area_nm2: float, eps: float = 1.0) -> Label:
    """Oriented-dipole (interlayer) U per exciton: U_dip ≈ (e²/4πε0) · d / (ε·A)·4π.

    Capacitor form for co-oriented dipoles of moment e·d in area A. Decouples U
    from a_B — the route the brief flags for keeping U large while binding stays
    thermally stable.
    """
    if dipole_d_nm is None or area_nm2 is None or area_nm2 <= 0 or eps <= 0:
        return not_run("U", "eV", "insufficient inputs for the dipolar-U estimate")
    # U_dip = e^2 d /(ε0 ε A) = 4π · [e²/4πε0] · d /(ε A)
    U = 4.0 * math.pi * E2_OVER_4PIEPS0_eV_nm * dipole_d_nm / (eps * area_nm2)
    return Label(
        name="U", value=round(U, 5), unit="eV", tier="saturation_model",
        source=f"dipolar (interlayer) capacitor model d={dipole_d_nm:.2f} nm, "
               f"ε={eps:.1f}, A={area_nm2:.1f} nm²",
        uncertainty=round(0.5 * U, 5),
        notes="MODEL estimate; a_B-DEcoupled dipolar channel. Replace with a "
              "many-body/BSE interlayer treatment (tier 'gw_bse').",
    )


def parse_bse_biexciton(bse_out_path: str) -> Label:
    """Extract U from a completed BSE biexciton / exciton-saturation calculation.

    Returns a `not_run` Label if the output is missing or the run is unfinished —
    never a fabricated value.
    """
    import os
    if not os.path.exists(bse_out_path):
        return not_run("U", "eV", f"BSE output not found: {bse_out_path}")
    try:
        text = open(bse_out_path).read()
    except OSError as e:
        return not_run("U", "eV", f"could not read BSE output: {e}")
    if "JOB DONE" not in text and "TIMING" not in text:
        return not_run("U", "eV", "BSE run did not complete; refusing to guess")
    # The biexciton binding / saturation shift extraction is code-specific and is
    # implemented against real BerkeleyGW absorption/biexciton output. Until wired
    # to an actual finished run, emit no number.
    return not_run("U", "eV",
                   "BSE completed but biexciton/U parser not yet wired to this "
                   "output format; no value emitted (no fabrication)")
