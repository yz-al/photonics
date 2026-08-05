#!/usr/bin/env python3
"""
Physics validation suite — the checks that decide whether pipeline numbers are
trustworthy, run BEFORE any expensive GW-BSE spend.

Runs the tests that need no new compute (Γ(T) shape; TMD chemical trend and
reduced-mass direction from the committed C2DB labels), pulls the DFPT Born-charge
polarity result if the Modal DFPT run has produced it, and lists the GW-BSE-gated
acceptance tests (dimensionality, vacuum truncation, BSE↔GW reference, interlayer-
exciton signature) as PENDING gates that must pass before BSE output is accepted.

    python excitonic/scripts/physics_validation.py
Writes excitonic/data/manifests/physics_validation.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm import acceptance as A                # noqa: E402
from exciton_fm.frohlich import gamma_of_T            # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "processed", "c2db_excitonic.csv")
DFPT = os.path.join(ROOT, "data", "manifests", "phase2_dfpt_validation.json")
OUT = os.path.join(ROOT, "data", "manifests", "physics_validation.json")


def _reduced_mass(df):
    me = pd.to_numeric(df["emass_cbm"], errors="coerce")
    mh = pd.to_numeric(df["emass_vbm"], errors="coerce")
    me = me.where(np.isfinite(me)); mh = mh.where(np.isfinite(mh))
    return (me * mh) / (me + mh)   # reduced mass μ


def main() -> int:
    df = pd.read_csv(CSV)
    E_b = pd.to_numeric(df["E_B"], errors="coerce").to_numpy()
    mu = _reduced_mass(df).to_numpy()
    binding = {}
    for f in ("MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2"):
        row = df[df["formula"].astype(str) == f]
        if len(row):
            binding[f] = float(pd.to_numeric(row["E_B"], errors="coerce").iloc[0])

    checks = []
    # --- runnable now ---
    checks.append(A.gamma_T_shape(gamma_of_T, hw_LO_meV=48.0, gamma_LO_meV=55.0))
    checks.append(A.chemical_trend(binding))
    checks.append(A.reduced_mass_direction(E_b, mu))

    # --- DFPT-gated: use the Modal DFPT validation result if present ---
    zsi = zga = None
    if os.path.exists(DFPT):
        d = json.load(open(DFPT))
        pol = d.get("polarity_test") or {}
        zsi = pol.get("Z_Si"); zga = pol.get("Z_GaAs")
        if zsi is None:
            r = d.get("results", {})
            zsi = (r.get("Si", {}).get("Z_born", {}) or {}).get("value")
            zga = (r.get("GaAs", {}).get("Z_born", {}) or {}).get("value")
    checks.append(A.born_charge_polarity(zsi, zga))

    # --- GW-BSE-gated: pending until the BSE pipeline runs ---
    checks.append(A.dimensionality_trend(None, None, None))
    checks.append(A.vacuum_truncation_plateau(None))
    checks.append(A.bse_energy_reference(None, None, None))
    checks.append(A.interlayer_exciton_signature(None, None, None, None, None))

    manifest = {
        "purpose": "Physics validation gates — trustworthiness of pipeline numbers.",
        "anchor_spread_required": {
            "GaAs": "E_b~4 meV, a_B~10 nm (3D, cold)",
            "MoS2": "E_b~0.5 eV, a_B~1 nm (2D)",
            "halide_perovskite": "Fröhlich LO ~40-70 meV",
            "note": "4 orders of magnitude in binding + full polarity range; a "
                    "pipeline that only gets TMDs is tuned to TMDs (the Phase-1 failure).",
        },
        "checks": [c.to_dict() for c in checks],
        "summary": {
            "passed": [c.name for c in checks if c.passed is True],
            "failed": [c.name for c in checks if c.passed is False],
            "pending": [c.name for c in checks if c.passed is None],
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("[physics] validation checks:")
    for c in checks:
        tag = {True: "PASS", False: "FAIL", None: "PENDING"}[c.passed]
        print(f"  [{tag:7s}] ({c.gate:6s}) {c.name}: {c.detail}")
    s = manifest["summary"]
    print(f"[physics] passed={len(s['passed'])} failed={len(s['failed'])} "
          f"pending={len(s['pending'])} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
