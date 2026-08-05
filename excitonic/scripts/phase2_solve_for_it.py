#!/usr/bin/env python3
"""
Phase 2 — SOLVE FOR IT: invert the feasibility bound into a spec, screen for the
one or two materials that sit in the surviving corner.

The bound analysis (phase2_dipolar_sumrule, phase2_bound_alpha) does not say "no
material" — it says the survivor must occupy a thin, specific corner. Inverting it
gives a concrete spec a candidate must meet AT ONCE:

  1. LOW Γ floor      — weak Fröhlich coupling. Γ_LO ∝ α·ħω_LO·n(300K), and the
                        polar coupling α ∝ (1/ε∞ − 1/ε0)·√μ. So favor LOW ionic
                        polarizability (alphax_lat) — a direct Fröhlich proxy in C2DB.
  2. THERMAL STABILITY — E_B ≫ kT and synthesizable: E_B≥0.3 eV, non-magnetic,
                        ehull≤0.1 eV/atom, hform<0, real gap.
  3. HIGH U           — tight exciton (small a_B ∝ ε/μ ⇒ large on-site g_xx∝E_B/a_B²)
                        AND, for the dipolar route, a permanent dipole (dipz≠0, the
                        Janus/broken-inversion families) OR a moiré-trap construction.
  4. ENOUGH BRIGHTNESS — Ω∝√f must clear Γ. f falls with μ, so not-too-heavy mass.

Two routes fall out (the "one or two"):
  A. SATURATION / MOIRÉ single-emitter — a tightly-bound, weakly-polar monolayer,
     moiré- or defect-localized to one trap so Γ→phonon floor and confinement→U.
  B. INTRINSIC DIPOLAR — a Janus/polar monolayer (dipz≠0) whose built-in dipole gives
     a dipolar exciton in ONE layer (no interlayer darkness penalty of a heterobilayer).

This screens all viable C2DB materials on the inverted spec and names the top
candidates per route. TIER: proxy (C2DB features + the bound), NOT a GW-BSE verdict —
each hit is a hypothesis to confirm, with the single deciding measurement named.

    python excitonic/scripts/phase2_solve_for_it.py
Writes excitonic/data/manifests/phase2_solve_for_it.json.
"""
from __future__ import annotations

import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "c2db_excitonic.csv"))
OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests", "phase2_solve_for_it.json"))


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def reduced_mass(r):
    me, mh = num(r, "emass_cbm"), num(r, "emass_vbm")
    if me is None or mh is None:
        return None
    me, mh = abs(me), abs(mh)
    if me <= 0 or mh <= 0:
        return None
    return me * mh / (me + mh)


def frohlich_contrast(r):
    """Proxy for (1/ε∞ − 1/ε0): ionic polarizability over the two dielectric factors.

    ε∞ ~ 1+α_el, ε0 ~ 1+α_el+α_lat ⇒ 1/ε∞−1/ε0 = α_lat/[(1+α_el)(1+α_el+α_lat)].
    Lower = weaker polar coupling = lower Γ floor. (In-plane x component used.)
    """
    a_el, a_lat = num(r, "alphax_el"), num(r, "alphax_lat")
    if a_el is None or a_lat is None or a_lat < 0:
        return None
    return a_lat / ((1.0 + a_el) * (1.0 + a_el + a_lat))


def viable(r):
    eh, hf = num(r, "ehull"), num(r, "hform")
    return (
        (num(r, "E_B") or 0) >= 0.3
        and str(r.get("is_magnetic", "")).strip().lower() == "no"
        and (num(r, "gap") or 0) > 0.3
        and eh is not None and eh <= 0.1
        and (hf is None or hf < 0)
    )


def main() -> int:
    rows = list(csv.DictReader(open(CSV)))
    pool = []
    for r in rows:
        if not viable(r):
            continue
        mu = reduced_mass(r)
        cf = frohlich_contrast(r)
        if mu is None or cf is None:
            continue
        E_B = num(r, "E_B")
        dipz = abs(num(r, "dipz") or 0.0)
        # Γ-floor proxy: α ∝ contrast·√μ  (higher ⇒ higher Γ). Add small floor to avoid /0.
        gamma_proxy = cf * math.sqrt(mu) + 1e-4
        # U-over-Γ favorability: binding (interaction scale) beats the phonon floor.
        u_over_gamma = E_B / gamma_proxy
        # brightness: f ∝ 1/μ (lighter = brighter); penalize very heavy excitons.
        brightness = 1.0 / mu
        rec = {
            "formula": r.get("formula"), "E_B": round(E_B, 3),
            "mu": round(mu, 3), "frohlich_contrast": round(cf, 4),
            "gamma_proxy": round(gamma_proxy, 4),
            "U_over_gamma_proxy": round(u_over_gamma, 2),
            "brightness_1_over_mu": round(brightness, 2),
            "dipz": round(dipz, 3), "gap_eV": num(r, "gap"),
            "ehull": num(r, "ehull"), "layergroup": r.get("layergroup"),
        }
        pool.append(rec)

    # Route A — saturation/moiré: maximize U/Γ (bound-favorability) with usable brightness.
    routeA = sorted([p for p in pool if p["brightness_1_over_mu"] >= 1.0],
                    key=lambda p: -p["U_over_gamma_proxy"])[:8]
    # Route B — intrinsic dipolar: built-in dipole × bound-favorability.
    dip_pool = [dict(p, dipolar_score=round(p["dipz"] * p["U_over_gamma_proxy"], 3))
                for p in pool if p["dipz"] > 0.01]
    routeB = sorted(dip_pool, key=lambda p: -p["dipolar_score"])[:8]

    manifest = {
        "framing": "Inverting the feasibility bound: assume a rare solution EXISTS and "
                   "solve for the corner it must occupy, then screen C2DB for who lands there.",
        "spec": {
            "low_gamma_floor": "weak Fröhlich (low ionic polarizability alphax_lat)",
            "thermal_stability": "E_B≥0.3 eV, non-magnetic, ehull≤0.1, hform<0, gap>0.3 eV",
            "high_U": "tight exciton (small a_B) and/or intrinsic dipole (dipz≠0) / moiré trap",
            "brightness": "not-too-heavy μ so Ω∝√f can clear Γ",
        },
        "n_viable_pool": len(pool),
        "route_A_saturation_moire": {
            "idea": "tightly-bound, weakly-polar monolayer, moiré/defect-localized to ONE "
                    "trap → Γ reverts to phonon floor, confinement raises U. Single-emitter.",
            "top": routeA,
            "operating_point": "single moiré/defect trap in a strong-coupling cavity, 300 K",
            "confirming_measurement": "moiré-trap exciton linewidth vs confinement (the q_sat "
                                      "exponent) + on-site U at one trap.",
        },
        "route_B_intrinsic_dipolar": {
            "idea": "Janus / broken-inversion polar monolayer (dipz≠0): a permanent out-of-"
                    "plane dipole gives a dipolar exciton in ONE layer — avoids the 10–100× "
                    "darkness penalty of an interlayer (spatially-indirect) exciton.",
            "top": routeB,
            "operating_point": "monolayer (or homobilayer) in a cavity, 300 K",
            "confirming_measurement": "GW-BSE oscillator strength f AND dipolar U for the top "
                                      "Janus candidate — does U>0.71·Γ with f bright enough?",
        },
        "the_answer_solved_for": (
            "Not a long list — one or two archetypes. (A) a moiré-trapped single interlayer "
            "exciton in the Γ-lowest TMD heterobilayer (MoS₂-based, from the DFPT gate), run as "
            "a single emitter in a strong-coupling cavity; and (B) a Janus/polar monolayer "
            "(dipz≠0, e.g. the top hit below) whose built-in dipole gives an intrinsically "
            "dipolar exciton without the interlayer darkness tax. Both live in the bound's "
            "surviving corner; each is decided by ONE named measurement."),
        "tier": "proxy (C2DB features + inverted bound). Hypotheses to confirm, not a verdict.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[solve] viable pool (stable, non-mag, E_B≥0.3, gap>0.3): {len(pool)}")
    print("[solve] ROUTE A — saturation/moiré (max U/Γ, usable brightness):")
    for p in routeA[:5]:
        print(f"    {p['formula']:12s} U/Γ={p['U_over_gamma_proxy']:6.1f}  E_B={p['E_B']:.2f}  "
              f"contrast={p['frohlich_contrast']:.3f}  1/μ={p['brightness_1_over_mu']:.1f}")
    print("[solve] ROUTE B — intrinsic dipolar (dipz≠0, Janus/polar):")
    for p in routeB[:5]:
        print(f"    {p['formula']:12s} dipolar_score={p['dipolar_score']:6.2f}  "
              f"dipz={p['dipz']:.3f}  U/Γ={p['U_over_gamma_proxy']:.1f}  E_B={p['E_B']:.2f}")
    print(f"[solve] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
