#!/usr/bin/env python3
"""
Phase 2 driver — inventory the DIPOLAR-exciton datasets (BiDB + HetDB) the same
way Phase 0 inventoried C2DB, and quantify coverage of the fields that a
dipolar exciton-exciton interaction U_dip is built from.

    python excitonic/scripts/phase2_dipolar_inventory.py

Writes:
    excitonic/data/manifests/dipolar_dataset_inventory.json

Every number is a live query against the public ASE-db front-ends
(bidb.fysik.dtu.dk, hetdb.fysik.dtu.dk, c2db.fysik.dtu.dk). Nothing is
fabricated: if a database or key is unreachable it is recorded as
"unreachable"/null, not guessed.

Why these two DBs (see reports/u_definition_decision.md):
  The search objective's U is the DIPOLAR / interlayer U, set by a permanent
  excited-state dipole d (electron and hole on different layers). That requires
  a spatially-INDIRECT exciton, which lives in vdW BILAYERS (BiDB) and
  HETEROSTRUCTURES (HetDB) — not in monolayer C2DB. type-II (staggered) band
  alignment is the fingerprint of a charge-transfer / interlayer exciton.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from exciton_fm.bidb_client import bidb_client, hetdb_client   # noqa: E402
from exciton_fm.c2db_client import C2DBClient                  # noqa: E402

MANIFEST_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "manifests"))

# ---------------------------------------------------------------------------
# Keys that matter for a dipolar-U estimate, per database.
# ---------------------------------------------------------------------------
BIDB_KEYS = {
    "interlayer_distance": "Interlayer distance d [Å] — the geometric lever for U_dip",
    "ebind": "Interlayer binding energy [meV/Å2] — stacking cohesion",
    "has_inversion_symmetry": "Inversion symmetry — broken => out-of-plane dipole allowed",
    "gap_pbe": "PBE band gap [eV]",
    "number_of_layers": "Number of layers (2 for a bilayer)",
    "slide_stability": "Slide (stacking) stability",
    "thickness": "Slab thickness [Å]",
    "area": "Unit-cell area [Å2] (=> areal density n=1/A for U_dip)",
    "monolayer_uid": "Parent monolayer id",
    "c2db_uid": "Parent monolayer C2DB uid (join key)",
    "magnetic": "Magnetic",
}

HETDB_KEYS = {
    "interlayer_distance": "Interlayer distance d [Å] — geometric lever for U_dip",
    "twist_angle": "Twist angle [deg]",
    "maxstrain": "Max lattice strain to commensurate cell [%]",
    "uid_a": "C2DB uid of layer 1 (join key)",
    "uid_b": "C2DB uid of layer 2 (join key)",
    "pbe_gap_soc": "Heterostructure PBE gap w/ SOC [eV]",
    "scs_gap_soc": "Heterostructure LAPS gap w/ SOC [eV]",
    "pbe_cbm_soc": "CBM wrt vacuum, PBE+SOC [eV] (band-alignment ingredient)",
    "pbe_vbm_soc": "VBM wrt vacuum, PBE+SOC [eV] (band-alignment ingredient)",
    "scs_cbm_soc": "CBM wrt vacuum, LAPS+SOC [eV]",
    "scs_vbm_soc": "VBM wrt vacuum, LAPS+SOC [eV]",
    "pbe_evac": "Vacuum level, PBE [eV]",
    "area": "Unit-cell area [Å2]",
}


def _safe_count(client, expr):
    """count(expr)[0] or None if the query fails (key/db unreachable)."""
    try:
        return client.count(expr)[0]
    except Exception as e:                       # noqa: BLE001
        return {"unreachable": str(e)}


# ---------------------------------------------------------------------------
# C2DB monolayer band-edge lookup (for Anderson's-rule type-II derivation).
# ---------------------------------------------------------------------------
_TAG = re.compile(r"<[^>]+>")
_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.S | re.I)
_C2DB_EDGE_HEADERS = {
    "Formula": "formula",
    "Energy above hull [eV/atom]": "ehull",
    "Heat of formation [eV/atom]": "hform",
    "Band gap (PBE) [eV]": "gap",
    "Magnetic": "is_magnetic",
    "Layer group (not Space group)": "layergroup",
    "VBM wrt. vacuum (PBE) [eV]": "vbm",
    "CBM wrt. vacuum (PBE) [eV]": "cbm",
    "Vacuum level [eV]": "evac",
}


def _clean(cell_html: str) -> str:
    import html as _html
    return _html.unescape(_TAG.sub("", cell_html)).strip()


def c2db_band_edges(uids):
    """Return {uid: {'vbm': float|None, 'cbm': float|None}} from live C2DB.

    Toggles VBM/CBM-wrt-vacuum columns on, then filters one uid at a time
    (each uid returns exactly one row). Missing edge -> None. On any network
    failure the uid maps to {'unreachable': msg}.
    """
    c = C2DBClient()
    sid = c.sid(refresh=True)
    for k in ("vbm", "cbm", "evac"):
        c._toggle(k)
    out: dict[str, dict] = {}
    for u in sorted(uids):
        try:
            txt = c._table(f"uid={u}")
            cells = [_clean(x) for x in _CELL.findall(txt)]
            if "Formula" not in cells:
                out[u] = {"vbm": None, "cbm": None}
                continue
            cells = cells[cells.index("Formula"):]
            header = []
            for cell in cells:
                if cell in _C2DB_EDGE_HEADERS:
                    header.append(_C2DB_EDGE_HEADERS[cell])
                else:
                    break
            ncol = len(header)
            row = dict(zip(header, cells[ncol:2 * ncol]))

            def _f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None
            out[u] = {"vbm": _f(row.get("vbm")), "cbm": _f(row.get("cbm"))}
            time.sleep(0.1)
        except Exception as e:                   # noqa: BLE001
            out[u] = {"unreachable": str(e)}
    return out


def classify_alignment(vbm_a, cbm_a, vbm_b, cbm_b):
    """Anderson's-rule alignment type from isolated-monolayer vacuum-referenced
    band edges. Returns 'I', 'II', 'III', or None if data missing.

    type-II (staggered) => VBM and CBM localise on DIFFERENT layers =>
    spatially-indirect / charge-transfer exciton => permanent out-of-plane
    dipole. This is a DERIVED estimate (isolated-monolayer PBE edges, no
    interfacial charge-transfer or relaxation correction), NOT a stored label.
    """
    if None in (vbm_a, cbm_a, vbm_b, cbm_b):
        return None
    vbm_layer = "a" if vbm_a >= vbm_b else "b"     # highest occupied
    cbm_layer = "a" if cbm_a <= cbm_b else "b"     # lowest unoccupied
    het_vbm = max(vbm_a, vbm_b)
    het_cbm = min(cbm_a, cbm_b)
    if het_cbm < het_vbm:
        return "III"                               # broken-gap overlap
    return "II" if vbm_layer != cbm_layer else "I"


# ---------------------------------------------------------------------------
# Per-database inventories.
# ---------------------------------------------------------------------------
def inventory_bidb():
    try:
        b = bidb_client()
        total = b.total()
    except Exception as e:                         # noqa: BLE001
        return {"reachable": False, "endpoint": "https://bidb.fysik.dtu.dk",
                "error": str(e)}

    cov = {k: _safe_count(b, k) for k in BIDB_KEYS}
    derived = {
        "broken_inversion_symmetry (dipolar-capable)":
            _safe_count(b, "has_inversion_symmetry=False"),
        "preserved_inversion_symmetry (dipole-forbidden)":
            _safe_count(b, "has_inversion_symmetry=True"),
        "d AND ebind (geometry+cohesion)":
            _safe_count(b, "interlayer_distance, ebind"),
        "d AND broken_inversion (dipolar geometry present)":
            _safe_count(b, "interlayer_distance, has_inversion_symmetry=False"),
        "gap_pbe>0.1 AND broken_inversion (semiconducting + dipolar)":
            _safe_count(b, "gap_pbe>0.1, has_inversion_symmetry=False"),
        "slide_stability=Stable":
            _safe_count(b, "slide_stability=Stable"),
    }
    return {
        "reachable": True,
        "endpoint": "https://bidb.fysik.dtu.dk",
        "description": "van der Waals Bilayer Database (BiDB) — homobilayers "
                       "(two copies of the same monolayer, all stackings).",
        "total_bilayers": total,
        "key_coverage": cov,
        "key_descriptions": BIDB_KEYS,
        "derived_counts": derived,
        "dipolar_note": (
            "BiDB is HOMObilayers: both layers are the same material, so there "
            "is no intrinsic type-II band offset. A permanent out-of-plane "
            "dipole is only symmetry-allowed when the stacking BREAKS inversion "
            "symmetry (has_inversion_symmetry=False). That subset is the "
            "dipolar-capable pool; U_dip there comes from stacking-induced "
            "charge transfer across interlayer_distance d, NOT a band offset."
        ),
    }


def inventory_hetdb():
    try:
        h = hetdb_client()
        total = h.total()
    except Exception as e:                         # noqa: BLE001
        return {"reachable": False, "endpoint": "https://hetdb.fysik.dtu.dk",
                "error": str(e)}

    cov = {k: _safe_count(h, k) for k in HETDB_KEYS}

    # Confirm (live) that NO explicit alignment-type label exists in the schema.
    stored_type_probe = {
        k: _safe_count(h, k)
        for k in ("type_of_alignment", "alignment", "band_alignment", "type")
    }

    # Derive type-II via Anderson's rule cross-joined to C2DB monolayer edges.
    type_ii = None
    derivation = {"method": "Anderson's rule on isolated-monolayer PBE band "
                            "edges (C2DB vbm/cbm wrt vacuum), joined by uid_a/"
                            "uid_b. Estimate, not a stored label."}
    try:
        rows = h.fetch_rows("", max_pages=40)
        uids = set()
        for r in rows:
            uids.add(r.get("uid_a", ""))
            uids.add(r.get("uid_b", ""))
        uids.discard("")
        edges = c2db_band_edges(uids)
        edge_ok = {u: v for u, v in edges.items()
                   if isinstance(v, dict) and "unreachable" not in v}

        counts = {"I": 0, "II": 0, "III": 0, "unclassifiable": 0}
        for r in rows:
            ea = edge_ok.get(r.get("uid_a"))
            eb = edge_ok.get(r.get("uid_b"))
            if not ea or not eb:
                counts["unclassifiable"] += 1
                continue
            t = classify_alignment(ea["vbm"], ea["cbm"], eb["vbm"], eb["cbm"])
            if t is None:
                counts["unclassifiable"] += 1
            else:
                counts[t] += 1
        type_ii = counts
        derivation.update({
            "rows_pulled": len(rows),
            "unique_constituent_monolayers": len(uids),
            "monolayers_with_c2db_edges": len(edge_ok),
        })
    except Exception as e:                         # noqa: BLE001
        derivation["unreachable"] = str(e)

    return {
        "reachable": True,
        "endpoint": "https://hetdb.fysik.dtu.dk",
        "description": "vdW 2D Heterostructure Database (HetDB) — bilayers of "
                       "TWO different C2DB monolayers, with band edges wrt "
                       "vacuum for each side.",
        "total_heterostructures": total,
        "key_coverage": cov,
        "key_descriptions": HETDB_KEYS,
        "stored_alignment_type_label_probe": stored_type_probe,
        "stored_alignment_type_label_present": any(
            isinstance(v, int) and v > 0 for v in stored_type_probe.values()
        ),
        "derived_alignment_type_counts": type_ii,
        "derivation": derivation,
        "dipolar_note": (
            "HetDB is HETERObilayers, so a genuine type-II staggered offset can "
            "exist. type-II => electron and hole sit on different layers => "
            "spatially-indirect exciton => permanent dipole d ~ interlayer_"
            "distance. HetDB carries the band-alignment INGREDIENTS (CBM/VBM "
            "wrt vacuum) but NO explicit alignment-type label; type-II is "
            "derived here by Anderson's rule."
        ),
    }


def main() -> int:
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    print("[phase2] querying live BiDB / HetDB / C2DB ...", flush=True)

    bidb = inventory_bidb()
    hetdb = inventory_hetdb()

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Dipolar / interlayer-exciton dataset inventory (Phase 2 "
                   "seed). Quantifies coverage of the fields a dipolar U_dip is "
                   "built from, across BiDB (bilayers) and HetDB "
                   "(heterostructures). All counts are live queries.",
        "u_channel": "dipolar/interlayer (per reports/u_definition_decision.md)",
        "u_dip_formula": "U_dip = e^2 d^2 / (eps0 * eps_eff * A)  — set by the "
                         "interlayer charge-transfer separation d and the mode "
                         "area A; independent of a_B and mu (decouples U from "
                         "E_b and Gamma).",
        "databases": {"BiDB": bidb, "HetDB": hetdb},
    }

    out = os.path.join(MANIFEST_DIR, "dipolar_dataset_inventory.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)

    # Console summary.
    if bidb.get("reachable"):
        print(f"[phase2] BiDB  total bilayers        : {bidb['total_bilayers']}")
        print(f"[phase2]   interlayer_distance covered: {bidb['key_coverage']['interlayer_distance']}")
        print(f"[phase2]   ebind covered             : {bidb['key_coverage']['ebind']}")
        print(f"[phase2]   broken inversion (dipolar): "
              f"{bidb['derived_counts']['broken_inversion_symmetry (dipolar-capable)']}")
    else:
        print(f"[phase2] BiDB UNREACHABLE: {bidb.get('error')}")

    if hetdb.get("reachable"):
        print(f"[phase2] HetDB total heterostructures : {hetdb['total_heterostructures']}")
        print(f"[phase2]   interlayer_distance covered: {hetdb['key_coverage']['interlayer_distance']}")
        print(f"[phase2]   band edges (pbe_cbm+pbe_vbm): "
              f"{hetdb['key_coverage']['pbe_cbm_soc']} / {hetdb['key_coverage']['pbe_vbm_soc']}")
        print(f"[phase2]   stored alignment-type label: "
              f"{hetdb['stored_alignment_type_label_present']}")
        print(f"[phase2]   derived alignment types    : {hetdb['derived_alignment_type_counts']}")
    else:
        print(f"[phase2] HetDB UNREACHABLE: {hetdb.get('error')}")

    print(f"[phase2] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
