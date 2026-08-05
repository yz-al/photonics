"""
The four excitonic targets, the blockade figure-of-merit, the validation
anchors, and the mapping from each target onto C2DB label keys (or the
explicit absence thereof).

Everything a downstream model needs to know about *what* is being predicted
and *what quality* of label backs it lives here, so the rigor rule -- "flag
DFT-level proxies vs GW-quality labels" -- is enforced from one place.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- label provenance tiers -------------------------------------------------
# Ordered worst -> best. A target's usable label quality is the best tier that
# actually has coverage in the dataset.
QUALITY = ("none", "dft_proxy", "gw_bse", "epw", "measured")


@dataclass(frozen=True)
class Target:
    key: str                 # short internal name
    name: str                # human description
    unit: str
    want: str                # the design direction we are optimizing toward
    c2db_keys: tuple[str, ...]   # C2DB column(s) that (partly) label this; () = none
    label_quality: str       # best available provenance tier in C2DB today
    note: str = ""


# The C2DB column that *directly* labels each target, with its true provenance.
#   E_B          -> "Exciton binding energy (BSE)"   [GW-BSE quality]
#   gap_gw       -> "Band gap (G0W0)"                 [GW quality, auxiliary]
#   alpha*_el    -> "Interband polarizability"        [PBE/RPA electronic -> DFT proxy]
#   plasmafreq*  -> "Plasma frequency"                [PBE -> DFT proxy]
#   emass_*      -> "DOS effective mass (PBE)"        [DFT; feeds Bohr radius / sat. U]
#   alpha*_lat   -> "Static polarizability (phonons)" [DFPT; a Fröhlich *ingredient*,
#                                                       NOT the linewidth]
TARGETS: tuple[Target, ...] = (
    Target(
        key="E_b",
        name="Exciton binding energy",
        unit="eV",
        want="large (>~0.2 eV for room-temperature stability)",
        c2db_keys=("E_B",),
        label_quality="gw_bse",
        note="Directly labeled in C2DB by the BSE 'E_B' column. Our anchor set.",
    ),
    Target(
        key="Gamma_300K",
        name="RT homogeneous exciton linewidth (Fröhlich/LO-phonon dephasing)",
        unit="meV",
        want="small (~10 meV)",
        c2db_keys=(),  # <-- NO C2DB column. This is the gap.
        label_quality="none",
        note=(
            "Requires exciton-phonon coupling (DFPT + EPW) at 300 K. C2DB carries "
            "lattice polarizability (alpha*_lat) as a partial Fröhlich *ingredient* "
            "but not the temperature-dependent linewidth itself."
        ),
    ),
    Target(
        key="U_xx",
        name="Exciton–exciton interaction (per-particle nonlinearity)",
        unit="eV (or meV·µm^2 areal)",
        want="large; separate saturation- vs dipolar/Rydberg-sourced",
        c2db_keys=(),  # <-- NO C2DB column. This is the deepest gap.
        label_quality="none",
        note=(
            "Requires biexciton binding / exciton saturation from BSE or "
            "many-body (or a dipolar/interlayer treatment). Not tabulated anywhere "
            "at scale. Saturation-U scales ~ as area / a_B^2, so emass_* + E_B give "
            "only a crude proxy."
        ),
    ),
    Target(
        key="f_osc",
        name="Oscillator strength / transition dipole",
        unit="dimensionless / Å (interband polarizability)",
        want="large enough to strong-couple and read out (|X| tradeoff)",
        c2db_keys=("alphax_el", "alphay_el", "alphaz_el", "plasmafrequency_x", "plasmafrequency_y"),
        label_quality="dft_proxy",
        note=(
            "C2DB gives interband (electronic) polarizability and plasma frequency "
            "at PBE/RPA level -> a DFT-quality proxy for oscillator strength, NOT a "
            "GW-BSE transition dipole. Flag as proxy."
        ),
    ),
)


# --- derived blockade figure of merit --------------------------------------
# Single-photon/polariton blockade opens when U/Γ exceeds ~0.7.
BLOCKADE_THRESHOLD = 0.71   # U_exc > 0.71 · Γ_x(300 K)  (consistent units)
BLOCKADE_RATIO_MIN = 0.7    # U/Γ target
BEST_MEASURED_RATIO = 0.05  # best reported to date (cryogenic) — the RT gap is ~1–2 orders


# --- auxiliary C2DB features worth carrying (not targets, but predictive) ---
AUX_KEYS = {
    "gap_gw": "Band gap G0W0 [eV] (GW quality)",
    "gap_dir_gw": "Direct band gap G0W0 [eV] (GW quality)",
    "gap": "Band gap PBE [eV] (DFT)",
    "gap_hse": "Band gap HSE06 [eV] (hybrid)",
    "emass_cbm": "CBM DOS effective mass PBE (-> exciton reduced mass, Bohr radius)",
    "emass_vbm": "VBM DOS effective mass PBE",
    "alphax_lat": "Static polarizability (phonons) x [Å] (DFPT; Fröhlich ingredient)",
    "alphay_lat": "Static polarizability (phonons) y [Å]",
    "alphaz_lat": "Static polarizability (phonons) z [Å]",
    "dipz": "Out-of-plane dipole [e Å/cell]",
    "thickness": "Layer thickness [Å]",
    "ehull": "Energy above hull [eV/atom] (thermodynamic stability filter)",
    "dyn_stab": "Dynamically stable (phonon check)",
}


# --- validation anchors (known physics the surrogate must reproduce) --------
# Values are literature reference points, NOT predictions. Sources named inline.
@dataclass(frozen=True)
class Anchor:
    name: str
    family: str
    E_b_eV: float | None
    a_B_nm: float | None
    Gamma_300K_meV: tuple[float, float] | float | None
    source: str


ANCHORS: tuple[Anchor, ...] = (
    Anchor("GaAs (bulk QW)", "III–V", 0.004, 10.0, None,
           "textbook; cold-only, a_B~10 nm, E_b~4 meV"),
    Anchor("Monolayer MoS2/WSe2 (TMD)", "TMD", 0.5, 1.0, (5.0, 15.0),
           "Selig Nat.Commun. 2016; Moody Nat.Commun. 2015 (Γ(300K)~5–15 meV)"),
    Anchor("MAPbI3 / halide perovskite", "halide perovskite", None, None, None,
           "Wright Nat.Commun. 2016: Fröhlich LO phonon ~40–70 meV"),
)
