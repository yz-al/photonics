"""
Physics acceptance tests — the checks that decide whether a pipeline number is
trustworthy. Each returns a (passed, detail) result; several are GATES that must
pass before any GW-BSE/EPW output is accepted (no "accept whatever finished" on a
single expensive run).

Grouped by what they need:
  - MODEL/DATA (runnable now): Γ(T) shape, chemical trends, reduced-mass direction.
  - DFPT (needs a phonon run): Born-charge polarity (Si≈0 vs GaAs≈2.2).
  - GW-BSE (needs the BSE pipeline): dimensionality trend, vacuum-truncation
    plateau, BSE↔GW energy-reference consistency, interlayer-exciton signature.

These are deliberately trend/consistency checks the code cannot accidentally pass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Check:
    name: str
    passed: bool | None          # None = could not run (missing inputs)
    detail: str
    gate: str                    # "model" | "dft" | "dfpt" | "gw_bse"

    def to_dict(self):
        return {"name": self.name, "passed": self.passed, "gate": self.gate,
                "detail": self.detail}


# --- MODEL / DATA (runnable now) -------------------------------------------
def gamma_T_shape(gamma_of_T, hw_LO_meV: float, gamma_LO_meV: float) -> Check:
    """Γ(T) must rise ~linearly at high T (kT >> ħω_LO) and saturate/flatten below
    the phonon energy. Checks monotonic increase + high-T near-linearity."""
    Ts = [10, 50, 100, 150, 200, 250, 300, 400, 600]
    g = [gamma_of_T(T, hw_LO_meV, gamma_LO_meV, gamma0_meV=0.0) for T in Ts]
    monotonic = all(g[i + 1] >= g[i] - 1e-9 for i in range(len(g) - 1))
    # high-T slope (200->400 vs 400->600) should be ~constant (linear) within 25%
    s1 = (g[7] - g[4]) / (400 - 200)
    s2 = (g[8] - g[7]) / (600 - 400)
    near_linear = abs(s1 - s2) <= 0.25 * max(abs(s1), 1e-9)
    # low-T (below ħω_LO ~ hw_LO/kB) should be strongly suppressed vs high-T
    suppressed = g[0] < 0.15 * g[6]
    ok = monotonic and near_linear and suppressed
    return Check("gamma_T_shape", ok, gate="model",
                 detail=f"monotonic={monotonic}, high-T linear (slopes {s1:.3f},{s2:.3f})="
                        f"{near_linear}, low-T suppressed (Γ(10)/Γ(300)={g[0]/g[6]:.3f})="
                        f"{suppressed}")


def chemical_trend(binding: dict[str, float]) -> Check:
    """Within a TMD family, binding must fall down the chalcogen:
    MoS2 > MoSe2 > MoTe2 (increasing ε and reduced mass). MoS2 vs WS2 should be
    close. `binding` maps formula->E_b (eV)."""
    order_ok = None
    detail = []
    have = all(k in binding for k in ("MoS2", "MoSe2", "MoTe2"))
    if have:
        a, b, c = binding["MoS2"], binding["MoSe2"], binding["MoTe2"]
        order_ok = a > b > c
        detail.append(f"MoS2({a:.3f})>MoSe2({b:.3f})>MoTe2({c:.3f}) = {order_ok}")
    if "WS2" in binding and "MoS2" in binding:
        d = abs(binding["MoS2"] - binding["WS2"])
        detail.append(f"|MoS2-WS2|={d:.3f} eV (expect small)")
    return Check("chemical_trend_TMD", order_ok, gate="dft", detail="; ".join(detail))


def reduced_mass_direction(E_b, mu) -> Check:
    """E_b (Wannier-Mott ∝ μ/ε²) must correlate POSITIVELY with reduced mass μ
    across the set (heavier → more bound, all else equal). If uncorrelated,
    something upstream is broken. `E_b`, `mu` are equal-length arrays."""
    import numpy as np
    E_b = np.asarray(E_b, float); mu = np.asarray(mu, float)
    m = np.isfinite(E_b) & np.isfinite(mu) & (mu > 0)
    if m.sum() < 30:
        return Check("reduced_mass_direction", None, gate="dft",
                     detail=f"too few paired points ({int(m.sum())})")
    from scipy.stats import spearmanr
    rho, p = spearmanr(mu[m], E_b[m])
    ok = rho > 0.1 and p < 0.05
    return Check("reduced_mass_direction", bool(ok), gate="dft",
                 detail=f"Spearman(mu,E_b)={rho:.3f} (p={p:.1e}), n={int(m.sum())}; "
                        f"expect >0")


# --- DFPT (needs a phonon run) ---------------------------------------------
def born_charge_polarity(Z_nonpolar, Z_polar) -> Check:
    """Non-polar (Si) Born charge ≈ 0; polar (GaAs) clearly nonzero. A large Z*
    for Si means the Fröhlich pipeline is not computing what we think."""
    if Z_nonpolar is None or Z_polar is None:
        return Check("born_charge_polarity", None, gate="dfpt",
                     detail="DFPT Born charges not available yet")
    ok = Z_nonpolar < 0.3 and Z_polar > 1.0
    return Check("born_charge_polarity", bool(ok), gate="dfpt",
                 detail=f"Z*(non-polar)={Z_nonpolar:.3f} (<0.3?), "
                        f"Z*(polar)={Z_polar:.3f} (>1.0?)")


# --- GW-BSE (needs the BSE pipeline) — GATES on accepting BSE output --------
def dimensionality_trend(E_mono, E_bi, E_bulk) -> Check:
    """Binding must fall monotonically monolayer → bilayer → bulk (added screening
    is real physics). Non-monotonic ⇒ screening treatment wrong."""
    vals = [E_mono, E_bi, E_bulk]
    if any(v is None for v in vals):
        return Check("dimensionality_trend", None, gate="gw_bse",
                     detail="needs mono/bilayer/bulk BSE of the same material")
    ok = E_mono > E_bi > E_bulk
    return Check("dimensionality_trend", bool(ok), gate="gw_bse",
                 detail=f"E_mono({E_mono})>E_bi({E_bi})>E_bulk({E_bulk}) = {ok}")


def vacuum_truncation_plateau(bindings_vs_vacuum) -> Check:
    """Without Coulomb truncation the 2D binding diverges (log) with vacuum; WITH
    truncation it plateaus. Pass iff the last two vacuum steps agree within 2%."""
    if not bindings_vs_vacuum or len(bindings_vs_vacuum) < 3:
        return Check("vacuum_truncation_plateau", None, gate="gw_bse",
                     detail="needs E_b at ≥3 increasing vacuum spacings")
    a, b = bindings_vs_vacuum[-2], bindings_vs_vacuum[-1]
    ok = abs(b - a) <= 0.02 * abs(a)
    return Check("vacuum_truncation_plateau", bool(ok), gate="gw_bse",
                 detail=f"last two vacuum steps {a:.3f}->{b:.3f} (Δ<2%?)={ok}; "
                        f"climbing ⇒ truncation off")


def bse_energy_reference(gw_gap, lowest_bse, e_b, tol=0.02) -> Check:
    """The lowest BSE eigenvalue must lie below the GW gap by exactly E_b. Catches
    misaligned energy references."""
    if None in (gw_gap, lowest_bse, e_b):
        return Check("bse_energy_reference", None, gate="gw_bse",
                     detail="needs GW gap, lowest BSE eigenvalue, and E_b")
    resid = abs((gw_gap - lowest_bse) - e_b)
    ok = resid <= tol
    return Check("bse_energy_reference", bool(ok), gate="gw_bse",
                 detail=f"|(GW_gap - E_BSE) - E_b|={resid:.4f} eV (<{tol})={ok}")


def interlayer_exciton_signature(E_intra_a, E_intra_b, E_inter,
                                 f_intra, f_inter) -> Check:
    """THE project-critical test. A type-II interlayer exciton must be (i) lower in
    energy than both intralayer excitons and (ii) far weaker in oscillator strength
    (electron and hole are spatially separated). If f_inter ~ f_intra, the charge
    separation — the entire source of the dipolar U — is not being captured."""
    if None in (E_intra_a, E_intra_b, E_inter, f_intra, f_inter):
        return Check("interlayer_exciton_signature", None, gate="gw_bse",
                     detail="needs intra/inter exciton energies + oscillator strengths")
    lower = E_inter < min(E_intra_a, E_intra_b)
    weaker = f_inter < 0.2 * f_intra
    ok = lower and weaker
    return Check("interlayer_exciton_signature", bool(ok), gate="gw_bse",
                 detail=f"E_inter<min(intra)={lower}; f_inter/f_intra="
                        f"{f_inter / f_intra:.3f}(<0.2?)={weaker}")
