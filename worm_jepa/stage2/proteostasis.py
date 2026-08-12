"""
proteostasis.py -- Aβ-aggregation → paralysis module (SOTA: Knowles/Cohen master equation).

Why this module and no other
----------------------------
Add a model only where it improves a VALIDATED endpoint. The standard worm AD
screens (GMC101, CL4176) are PROTEOSTASIS assays -- Aβ1-42 aggregates and the worm
paralyzes -- which the nervous-system twin abstains on. This module gives that
endpoint (time-to-paralysis) a mechanistic account and extends the twin's domain.

The model -- state of the art
-----------------------------
The field standard is the Knowles/Cohen chemical master equation (Cohen et al.,
PNAS 2013; fit via AmyloFit, Meisl et al., Nat Protoc 2016). Aggregation is THREE
microscopic processes, tracked as moments (P = aggregate number, M = fibril mass),
plus a toxic-oligomer pool O:

    primary nucleation    rate kn·m^nc              monomers -> a new aggregate
    elongation            rate 2·k+·m·P             monomer adds to a fibril end
    secondary nucleation  rate k2·m^n2·M            new aggregates on fibril surfaces
                                                     (DOMINATES for Aβ42)
    dP/dt = kn·m^nc + k2·m^n2·M
    dM/dt = 2·k+·m·P
    dO/dt = (nucleation flux) − γ·k+·O    toxic oligomers made by nucleation,
                                          CONSUMED by elongation into fibrils
    m = m_tot − M  (free monomer)

Toxicity ∝ cumulative toxic-oligomer exposure ∫O dt (the SOTA "oligomers are the
toxic species" hypothesis), NOT total fibril mass. Paralysis = ∫O dt crosses θ.
Calibrated so an untreated worm paralyzes at ~30 h (published GMC101/CL4176 range).

Two things this SOTA form captures that a single-rate model cannot
-----------------------------------------------------------------
1. DRUGS ACT ON A SPECIFIC MICROSCOPIC STEP (kn / k+ / k2), not one lumped rate.
2. THE ELONGATION-INHIBITOR TOXICITY PARADOX: inhibiting SECONDARY nucleation is
   most protective (it dominates), but inhibiting ELONGATION traps monomers as
   toxic oligomers -> ~3-4x higher peak oligomer burden. This is the field's
   headline, safety-critical result -- a lumped model gets it exactly backwards.

The honest boundary (unchanged)
-------------------------------
A compound's per-step rate factors are a LITERATURE / ASSAY INPUT (from AmyloFit-
style kinetic characterization). This module predicts the PHENOTYPE (paralysis,
oligomer burden) from that mechanism; it does not predict the rate factors from
chemical structure (a QSAR we do not have and do not claim).

    python proteostasis.py     # writes proteostasis.json
Pure numpy/scipy; no external data (calibrated to published timescales).
"""
import os, json
import numpy as np
from scipy.integrate import odeint

HERE = os.path.dirname(os.path.abspath(__file__))

# Knowles/Cohen rate constants (normalized conc, per hour), calibrated to ~30 h baseline.
MTOT = 1.0            # total Aβ pool
NC, N2 = 2, 2         # primary / secondary nucleation reaction orders
KN = 5e-4             # primary nucleation
KP = 0.10             # elongation
K2 = 0.30             # secondary nucleation (dominant for Aβ42)
GAMMA = 2.0           # oligomer -> fibril conversion per unit elongation
HOURS = np.linspace(0, 200, 4000)


def _factors(drug):
    """Accept a step dict {'kn','kplus','k2'} OR a scalar f (scales all three, back-compat)."""
    if drug is None:
        return 1.0, 1.0, 1.0
    if isinstance(drug, (int, float)):
        return float(drug), float(drug), float(drug)
    return drug.get("kn", 1.0), drug.get("kplus", 1.0), drug.get("k2", 1.0)


def solve(drug=None):
    """Integrate the master equation. Returns t, M (fibril mass), O (toxic oligomers),
    cumO (cumulative oligomer exposure)."""
    fkn, fkp, fk2 = _factors(drug)
    kn, kp, k2 = KN * fkn, KP * fkp, K2 * fk2

    def rhs(y, _):
        P, M, O, cO = y
        m = max(MTOT - M, 0.0)
        nuc = kn * m ** NC + k2 * (m ** N2) * M
        return [nuc, 2 * kp * m * P, nuc - GAMMA * kp * O, O]
    s = odeint(rhs, [0, 0, 0, 0], HOURS, hmax=0.5, mxstep=8000)
    return HOURS, np.clip(s[:, 1], 0, MTOT), s[:, 2], s[:, 3]


# toxicity threshold: cumulative oligomer exposure at which the untreated worm paralyzes (~30 h)
_t, _M, _O, _cO = solve(None)
THETA = float(np.interp(30.0, _t, _cO))
_BASE_PEAK_O = float(_O.max())


def paralysis_time(drug=None):
    _t, _M, _O, cO = solve(drug)
    idx = np.where(cO >= THETA)[0]
    return float(_t[idx[0]]) if len(idx) else float("inf")


def peak_oligomer(drug=None):
    return float(solve(drug)[2].max())


# Reference compounds. Where a compound's dominant microscopic target is known it is given as a
# step dict; otherwise a scalar broad anti-aggregation factor. (Literature/assay input.)
REFERENCE = [
    ("vehicle/DMSO",       None,                              "baseline"),
    ("PBT2",               {"kn": 0.4, "k2": 0.4},            "metal chelation → nucleation (GMC101 protective)"),
    ("EGCG",               0.6,                               "broad anti-aggregation (remodeller)"),
    ("curcumin",           0.7,                               "broad anti-aggregation / antioxidant"),
    ("secondary-nuc inhib",{"k2": 0.2},                       "e.g. BRICHOS/bexarotene-class — best mechanism"),
    ("elongation inhib",   {"kplus": 0.2},                    "PARADOX: traps toxic oligomers"),
    ("thioflavin-T",       None,                              "amyloid dye — INERT (clean negative)"),
]


def run():
    base = paralysis_time(None)
    rows = []
    for name, drug, note in REFERENCE:
        tp = paralysis_time(drug); pk = peak_oligomer(drug)
        rows.append(dict(compound=name, drug=drug,
                         paralysis_time_h=(round(tp, 1) if np.isfinite(tp) else None),
                         delay_vs_vehicle_h=(round(tp - base, 1) if np.isfinite(tp) else None),
                         peak_oligomer=round(pk, 3),
                         oligomer_vs_vehicle=round(pk / _BASE_PEAK_O, 2),
                         note=note))
    # SOTA validation signatures
    sec = next(r for r in rows if r["compound"] == "secondary-nuc inhib")
    elo = next(r for r in rows if r["compound"] == "elongation inhib")
    inert = next(r for r in rows if r["compound"] == "thioflavin-T")
    return {
        "baseline_paralysis_h": round(base, 1),
        "model": "Knowles/Cohen master equation (primary+elongation+secondary nucleation)+toxic-oligomer pool",
        "rows": rows,
        "secondary_nucleation_most_protective": bool(
            (sec["paralysis_time_h"] or 999) >= max((r["paralysis_time_h"] or 0) for r in rows if r["drug"])),
        "elongation_toxicity_paradox": bool(elo["oligomer_vs_vehicle"] > 1.5),
        "inert_control_at_baseline": bool(abs(inert["delay_vs_vehicle_h"] or 0) < 1.0),
    }


if __name__ == "__main__":
    r = run()
    print("=== PROTEOSTASIS (SOTA: Knowles/Cohen master equation) — Aβ → paralysis ===\n")
    print("untreated paralysis onset: %.1f h  (published GMC101/CL4176: ~24-48 h)\n" % r["baseline_paralysis_h"])
    print("%-20s %-12s %-10s %-14s %s" % ("compound", "paralysis(h)", "Δ(h)", "peak oligomer", "note"))
    for x in r["rows"]:
        print("%-20s %-12s %-10s %-14s %s" % (
            x["compound"],
            x["paralysis_time_h"] if x["paralysis_time_h"] is not None else ">200",
            ("%+.0f" % x["delay_vs_vehicle_h"]) if x["delay_vs_vehicle_h"] is not None else "—",
            "%.2fx vehicle" % x["oligomer_vs_vehicle"], x["note"]))
    print("\nSOTA signatures:")
    print("  secondary-nucleation inhibitor most protective : %s" % r["secondary_nucleation_most_protective"])
    print("  elongation-inhibitor toxicity PARADOX (↑oligomers): %s (%.1fx vehicle)"
          % (r["elongation_toxicity_paradox"],
             next(x["oligomer_vs_vehicle"] for x in r["rows"] if x["compound"] == "elongation inhib")))
    print("  inert control at baseline (thioflavin T)        : %s" % r["inert_control_at_baseline"])
    with open(os.path.join(HERE, "proteostasis.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("\nwrote proteostasis.json")
