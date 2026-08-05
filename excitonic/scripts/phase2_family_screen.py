#!/usr/bin/env python3
"""
Phase 2 — cross-channel FAMILY screen (widen the search where the U–Γ
cancellation is weakest).

The interlayer/dipolar screen (`phase2_dipolar_screen.py`) covers ONE escape
route from the U↔Γ near-cancellation that drives the likely bounded-no. This
script places that route alongside two non-TMD families that attack DIFFERENT
axes of the blockade trilemma (U up, Γ(300 K) down, oscillator strength f up):

  * high-ħω_LO lattices (hBN, III-nitrides) — attack the DENOMINATOR: a stiff,
    light-element polar lattice has a high LO-phonon energy, so the 300 K Bose
    occupation n_LO(300 K) — the thermal driver of the Fröhlich linewidth — is
    strongly suppressed.
  * Rydberg excitons (Cu₂O) — attack the NUMERATOR: the yellow-series exciton
    interaction / blockade scales ~n⁷ in the principal quantum number, the
    strongest known excitonic-blockade mechanism (Rydberg-atom analog).

RIGOR. The blockade FOM U/Γ is NOT computed for any family here — it needs
GW-BSE (U, f) + EPW (Γ). What IS computed, from MEASURED phonon energies, is the
one term measured inputs legitimately give us: the LO-phonon occupation
n_LO(300 K) = 1/(exp(ħω_LO/kT)−1), the temperature-driven part of Γ. Everything
else (the coupling γ_LO, the actual U, f) is documented as mechanism/scaling with
the numeric value left `not_run`. Literature phonon/binding values are tagged
tier='measured' WITH a source; nothing is fabricated, and no model U/Γ is
presented as a first-principles number.

    python excitonic/scripts/phase2_family_screen.py
Writes excitonic/data/manifests/phase2_family_screen.json + a ranked table.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.frohlich import bose            # noqa: E402  (measured-input, real calc)
from exciton_fm.provenance import Label, not_run  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                   "phase2_family_screen.json"))
DIP = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                   "phase2_dipolar_candidates.json"))
T_RT = 300.0

# ---------------------------------------------------------------------------
# Curated family table. Every phonon/binding number is a MEASURED literature
# value with a source; it enters only the comparative n_LO(300 K) calc and the
# qualitative axis placement. The FOM-deciding U, f, Γ are left not_run.
# ---------------------------------------------------------------------------
FAMILIES = [
    dict(
        key="tmd_intralayer",
        label="TMD intralayer (MoS₂) — BASELINE",
        hw_LO_meV=48.0,                       # A1'/LO ~390 cm⁻¹ (Raman)
        hw_src="MoS₂ monolayer A₁′/LO ≈48 meV (Raman; Molina-Sánchez & Wirtz PRB 2011)",
        E_b_eV=0.50,                          # 2D Wannier
        E_b_src="E_b≈0.5 eV (Chernikov PRL 2014; Ugeda Nat.Mater 2014)",
        u_channel="saturation/exchange g_xx∝E_b·a_B² (a_B-TIED, small)",
        u_axis="low", f_axis="high",
        optical_window="visible (~1.9 eV) — good",
        regime="the cancellation regime; best measured U/Γ ~0.05 lives here",
    ),
    dict(
        key="interlayer_dipolar",
        label="Interlayer dipolar (type-II heterobilayer)",
        hw_LO_meV=48.0,                       # similar TMD phonons
        hw_src="TMD LO ≈30–48 meV (inherits monolayer phonons)",
        E_b_eV=0.30,                          # interlayer exciton binding (smaller)
        E_b_src="interlayer exciton E_b≈0.2–0.3 eV (Wilson Sci.Adv 2017; Merkl Nat.Mater 2019)",
        u_channel="DIPOLAR capacitor U_dip∝d²/(εA) (a_B-DEcoupled, gate-tunable)",
        u_axis="mid-high", f_axis="low",
        optical_window="near-IR interlayer PL (~1.4 eV) — good",
        regime="the escape route: U decoupled from Γ; PAYS in oscillator strength f",
    ),
    dict(
        key="hbn_highphonon",
        label="High-ħω_LO: hBN",
        hw_LO_meV=180.0,                      # E2g/LO ~1366–1610 cm⁻¹
        hw_src="hBN E₂g/LO ≈170–200 meV (IR/Raman; Geick PR 1966; Cai SSC 2007)",
        E_b_eV=2.0,                           # Frenkel-like, huge binding
        E_b_src="E_b≈2 eV, Frenkel-like (Cassabois Nat.Photon 2016; Elias Nat.Commun 2019)",
        u_channel="Frenkel; tiny a_B → weak biexciton/saturation nonlinearity",
        u_axis="low", f_axis="mid",
        optical_window="deep-UV (~6 eV) — OUTSIDE useful window",
        regime="denominator WIN (n_LO crushed) but numerator+optical window LOSE",
    ),
    dict(
        key="gan_nitride",
        label="High-ħω_LO: GaN (III-nitride)",
        hw_LO_meV=92.0,                       # ħω_LO ~735 cm⁻¹
        hw_src="GaN ħω_LO ≈92 meV (Raman; Harima J.Phys.Condens.Matter 2002)",
        E_b_eV=0.025,                         # 3D Wannier
        E_b_src="E_b≈25 meV bulk (Wannier); larger in QWs",
        u_channel="Wannier saturation; strongly polar (large Fröhlich α competes)",
        u_axis="low", f_axis="high",
        optical_window="UV (~3.4 eV) — edge of useful",
        regime="mid denominator win; strong polarity keeps γ_LO high — partial",
    ),
    dict(
        key="cu2o_rydberg",
        label="Rydberg excitons: Cu₂O yellow series",
        hw_LO_meV=79.0,                       # Γ⁻₄ LO ~640 cm⁻¹
        hw_src="Cu₂O Γ⁻₄ LO ≈79 meV (Ito & Hoshina JPSJ; Kavoulakis PRB 1997)",
        E_b_eV=0.097,                         # Rydberg Ry* (1s deeper ~0.15)
        E_b_src="Rydberg Ry*≈97 meV, n≤25 observed (Kazimierczuk Nature 2014)",
        u_channel="RYDBERG BLOCKADE ~n⁷ (giant); strongest known excitonic mechanism",
        u_axis="very-high(cryo)", f_axis="low(quadrupole)",
        optical_window="visible (~2.0 eV yellow) — good",
        regime="numerator WIN (n⁷) but RT Γ + thermal ionization of high-n LOSE",
    ),
]


def _rel(n, n_ref):
    return None if not n_ref else round(n / n_ref, 4)


def main() -> int:
    # comparative axis: LO-phonon occupation at 300 K (measured ħω_LO -> real calc)
    base = next(f for f in FAMILIES if f["key"] == "tmd_intralayer")
    n_ref = bose(base["hw_LO_meV"], T_RT)

    rows = []
    for f in FAMILIES:
        n_LO = bose(f["hw_LO_meV"], T_RT)
        # the thermal driver of Γ, RELATIVE to the TMD baseline (dimensionless).
        # <1 means the 300 K phonon population — hence the Fröhlich Γ at fixed
        # coupling — is SUPPRESSED vs MoS₂.
        gamma_thermal_rel = _rel(n_LO, n_ref)
        rows.append({
            "family": f["label"],
            "key": f["key"],
            "hw_LO_meV": Label("hw_LO", f["hw_LO_meV"], "meV", "measured",
                               source=f["hw_src"]).to_dict(),
            "n_LO_300K": round(n_LO, 5),
            "gamma_thermal_rel_to_MoS2": gamma_thermal_rel,
            "E_b_eV": Label("E_b", f["E_b_eV"], "eV", "measured",
                            source=f["E_b_src"]).to_dict(),
            "U_channel": f["u_channel"],
            "U_axis": f["u_axis"],
            "f_axis": f["f_axis"],
            "optical_window": f["optical_window"],
            "regime": f["regime"],
            # the deciding quantities — explicitly NOT computed here.
            "U_eV": not_run("U", "eV", "needs GW-BSE biexciton/dipolar (per family)").to_dict(),
            "Gamma_300K_meV": not_run("Gamma_300K", "meV",
                                      "needs EPW (γ_LO coupling); only n_LO(300K) computed").to_dict(),
            "FOM_U_over_Gamma": not_run("U_over_Gamma", "1",
                                        "FOM NOT computed — no family has both U and Γ yet").to_dict(),
        })

    # integrate lever-1 (interlayer screen) headcount if present
    dip_n = None
    if os.path.exists(DIP):
        try:
            dip_n = json.load(open(DIP)).get("n_type_ii_candidates")
        except Exception:
            dip_n = None

    manifest = {
        "purpose": "Cross-channel family map: where to spend GW-BSE/EPW, by which "
                   "axis of the U↑/Γ↓/f↑ trilemma each family attacks.",
        "computed_axis": "n_LO(300 K) from MEASURED ħω_LO — the thermal driver of Γ. "
                         "Suppression vs MoS₂ quantifies the denominator lever.",
        "not_computed": "U, f, Γ(300 K) and the FOM U/Γ are not_run for every family "
                        "(need GW-BSE + EPW). This is a PRIORITIZATION, not a verdict.",
        "interlayer_screen_candidates": dip_n,
        "families": rows,
        "synthesis": (
            "Each family wins on exactly ONE axis and loses another: TMD-intra is "
            "balanced-but-capped (the cancellation); hBN/GaN crush n_LO(300 K) "
            "(denominator) but lose U and/or the optical window; Cu₂O wins the "
            "numerator (n⁷ Rydberg) but loses at RT (thermal ionization, quadrupole f). "
            "The ONLY family that decouples U from Γ without a fatal third-axis loss is "
            "the interlayer-dipolar channel — it trades oscillator strength f, which a "
            "high-Q / small-mode-volume cavity can recover. That ranks the spend."),
        "spend_ranking": [
            "1. interlayer-dipolar heterobilayer BSE (U decoupled; f recoverable by cavity)",
            "2. Cu₂O Rydberg blockade — measure U/Γ vs n and vs T (is the RT loss fatal?)",
            "3. high-ħω_LO (hBN/nitride) — denominator anchor; confirms γ_LO vs occupation split",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[family] wrote {OUT}")
    print(f"[family] interlayer-screen candidates on file: {dip_n}")
    print("\n  family                                  ħω_LO   n_LO(300K)  Γ_therm/MoS2  U-axis        f-axis")
    print("  " + "-" * 98)
    for r in rows:
        print(f"  {r['family']:<38s}  {r['hw_LO_meV']['value']:>5.0f}   "
              f"{r['n_LO_300K']:>8.4f}   {str(r['gamma_thermal_rel_to_MoS2']):>10s}   "
              f"{r['U_axis']:<12s}  {r['f_axis']}")
    print("\n  spend ranking (FOM NOT computed — prioritization only):")
    for s in manifest["spend_ranking"]:
        print("    " + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
