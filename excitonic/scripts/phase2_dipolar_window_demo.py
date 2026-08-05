#!/usr/bin/env python3
"""
Phase 2 — dipolar-branch WINDOW analysis (the one exponent that decides it).

The dipolar shortlist (209 heterostructures) is ranked by a geometric prior
g_dip ∝ d²/ε_eff that ASSUMES separating the layers helps — U rises with the
static dipole. But separation also collapses the oscillator strength f (overlap
of the wavefunctions that make the transition dipole), and f sets the light-matter
coupling g ∝ √f. So the branch lives or dies on how fast f falls versus how fast
U rises — one exponent, which nobody has measured for these systems.

This script (1) self-tests the window logic on two illustrative regimes — f
falling SLOWLY (window open) vs FAST (window closed) — and (2) consumes the real
flagship f(d)/U(d) points from phase2_gwbse_dsweep.json when they exist. The
illustrative inputs are tier 'saturation_model' and flagged; the REAL verdict
needs GW-BSE U and f at ≥2 interlayer separations.

    python excitonic/scripts/phase2_dipolar_window_demo.py
Writes excitonic/data/manifests/phase2_dipolar_window.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.dipolar_window import window_verdict, fit_f_decay, fit_U_rise  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                   "phase2_dipolar_window.json"))
FLAGSHIP = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                        "phase2_gwbse_dsweep.json"))

# Illustrative separations [nm] and a U(d) ∝ d dipolar rise (capacitor form).
DS = [0.6, 0.9, 1.2]
US = [0.10, 0.15, 0.20]                       # eV — rises linearly with d


def _f_curve(lam):
    """f(d) = exp(-d/lam), normalized to d=DS[0]."""
    return [math.exp(-(d - DS[0]) / lam) for d in DS]


def main() -> int:
    # (1) self-test: f falls SLOWLY (lam large) vs FAST (lam small)
    slow = window_verdict(DS, US, _f_curve(2.0), tier="saturation_model")   # gentle f loss
    fast = window_verdict(DS, US, _f_curve(0.25), tier="saturation_model")  # steep f loss
    selftest_ok = (slow["verdict"] == "window_open" and fast["verdict"] == "window_closed")

    # (2) one-point guard: a single separation cannot decide the branch
    one_point = window_verdict([0.6], [0.1], [1.0], tier="gw_bse")

    # (3) consume the REAL flagship sweep if present
    flagship = None
    if os.path.exists(FLAGSHIP):
        try:
            data = json.load(open(FLAGSHIP))
            pts = data.get("points", [])
            ds = [p.get("d_nm") for p in pts]
            us = [p.get("U_eV") for p in pts]
            fs = [p.get("f_osc") for p in pts]
            flagship = window_verdict(ds, us, fs, tier="gw_bse")
        except Exception as e:
            flagship = {"verdict": "undetermined", "reason": f"could not read flagship: {e}"}

    manifest = {
        "purpose": "Decide the dipolar branch: does a usable separation window exist "
                   "(U gain outruns f loss) or not (f loss dominates)?",
        "the_one_exponent": "f decay length λ_f vs U rise exponent p; net figure "
                            "N(d)=U·f^α (α=0.5 for g∝√f) rising ⇒ open, falling ⇒ closed.",
        "selftest_ok": selftest_ok,
        "selftest": {"f_falls_slowly (λ=2.0nm)": slow, "f_falls_fast (λ=0.25nm)": fast},
        "one_point_guard": one_point,
        "flagship_gwbse_dsweep": flagship if flagship is not None else
            "not yet computed — run the heterobilayer GW-BSE at ≥2 interlayer "
            "separations (phase2_gwbse_dsweep.json). This is the branch-deciding run.",
        "rigor": "Self-test inputs are illustrative (tier saturation_model). The real "
                 "verdict requires GW-BSE U and oscillator strength f at ≥2 separations; "
                 "until then the 209-shortlist ranking's favorable-tradeoff ASSUMPTION "
                 "is untested.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[window] self-test ok = {selftest_ok}")
    print(f"[window]   f falls slowly (λ=2.0nm): {slow['verdict']} "
          f"(p={slow['U_rise_exponent_p']}, λ_f={slow['f_decay_length_lambda']}, "
          f"net_slope={slow['net_loglog_slope']})")
    print(f"[window]   f falls fast   (λ=0.25nm): {fast['verdict']} "
          f"(net_slope={fast['net_loglog_slope']})")
    print(f"[window]   one-point guard: {one_point['verdict']} ({one_point['reason'][:60]}…)")
    print(f"[window] flagship dsweep: "
          f"{(flagship or {}).get('verdict', 'not yet computed')}")
    print(f"[window] wrote {OUT}")
    return 0 if selftest_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
