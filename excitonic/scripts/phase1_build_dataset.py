#!/usr/bin/env python3
"""
Phase 1, step 1 — pull the row-level C2DB training frame.

Pulls every material that carries a BSE exciton-binding label (`E_B`), with the
feature columns C2DB exposes, and writes a tidy dataset + a manifest recording
exactly what was pulled and the per-column missingness.

    python excitonic/scripts/phase1_build_dataset.py

Writes:
    excitonic/data/processed/c2db_excitonic.csv
    excitonic/data/manifests/phase1_dataset_manifest.json

Plain HTTPS to the C2DB web front-end; no login, no external ML deps here.
"""
from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from exciton_fm.c2db_client import C2DBClient, FEATURE_TOGGLE_KEYS  # noqa: E402

PROC_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "processed"))
MAN_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "manifests"))

# Column order for the CSV (id/label/feature grouping is documented in manifest).
COLUMNS = [
    "formula",                       # composition (id + featurization source)
    "E_B",                           # TARGET 1 label  (GW-BSE)
    "alphax_el", "alphay_el", "alphaz_el",   # TARGET 4 label/feature (DFT proxy)
    "plasmafrequency_x",             # TARGET 4 proxy
    "gap", "gap_gw", "gap_hse",       # electronic-structure features (PBE/GW/HSE)
    "emass_cbm", "emass_vbm",        # effective mass -> Bohr radius
    "alphax_lat",                    # Fröhlich ingredient (DFPT)
    "thickness", "area", "natoms", "nspecies", "dipz",   # geometry
    "ehull", "hform",                # stability (default columns)
    "is_magnetic", "layergroup",     # symmetry/magnetism
]


def main() -> int:
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(MAN_DIR, exist_ok=True)
    client = C2DBClient()

    print("[phase1] pulling E_B-labeled rows from C2DB ...", flush=True)
    rows = client.fetch_rows("E_B", toggle_keys=FEATURE_TOGGLE_KEYS)
    print(f"[phase1] pulled {len(rows)} rows", flush=True)

    # Normalize: ensure every column key exists per row.
    for r in rows:
        for c in COLUMNS:
            r.setdefault(c, "")

    out_csv = os.path.join(PROC_DIR, "c2db_excitonic.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    print(f"[phase1] wrote {out_csv}")

    # Missingness manifest.
    miss = {}
    for c in COLUMNS:
        nonempty = sum(1 for r in rows if str(r.get(c, "")).strip() != "")
        miss[c] = {"present": nonempty, "missing": len(rows) - nonempty}
    manifest = {
        "source": "C2DB via ASE-db web front-end",
        "filter": "E_B (has BSE exciton binding label)",
        "n_rows": len(rows),
        "columns": COLUMNS,
        "column_presence": miss,
        "targets": {
            "E_b": {"col": "E_B", "quality": "gw_bse"},
            "f_osc": {"cols": ["alphax_el", "alphay_el", "alphaz_el"], "quality": "dft_proxy"},
        },
    }
    out_man = os.path.join(MAN_DIR, "phase1_dataset_manifest.json")
    with open(out_man, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[phase1] wrote {out_man}")
    print("[phase1] column presence:")
    for c in COLUMNS:
        print(f"    {c:<20s} present={miss[c]['present']:>4d}  missing={miss[c]['missing']:>4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
