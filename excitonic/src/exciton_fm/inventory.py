"""
Phase 0 inventory: quantify, from live C2DB, exactly which of the four
excitonic targets already carry labels and — the whole point — the exact
size of the Fröhlich-linewidth + exciton–exciton-U data gap.

Produces a machine-readable manifest (dict, JSON-dumpable) with real counts.
No fabrication: every number here comes from a C2DB coverage query.
"""
from __future__ import annotations

from typing import Any

from .c2db_client import C2DBClient
from .targets import AUX_KEYS, TARGETS, Target


def _target_row(client: C2DBClient, t: Target, total: int) -> dict[str, Any]:
    per_key = {k: client.count(k)[0] for k in t.c2db_keys}
    covered = max(per_key.values()) if per_key else 0
    return {
        "target": t.key,
        "name": t.name,
        "unit": t.unit,
        "want": t.want,
        "c2db_keys": list(t.c2db_keys),
        "per_key_coverage": per_key,
        "covered_materials": covered,
        "coverage_frac": round(covered / total, 4) if total else 0.0,
        "label_quality": t.label_quality,
        "is_gap": covered == 0,
        "note": t.note,
    }


def build_inventory(client: C2DBClient | None = None) -> dict[str, Any]:
    client = client or C2DBClient()
    total = client.total()

    targets = [_target_row(client, t, total) for t in TARGETS]

    aux = {}
    for k, desc in AUX_KEYS.items():
        n = client.count(k)[0]
        aux[k] = {"coverage": n, "coverage_frac": round(n / total, 4), "desc": desc}

    # Intersections that define the usable GW-BSE training core.
    intersections = {
        "E_B": client.count("E_B")[0],
        "E_B AND gap_gw": client.count("E_B, gap_gw")[0],
        "E_B AND emass_cbm": client.count("E_B, emass_cbm")[0],
        "E_B AND alphax_el": client.count("E_B, alphax_el")[0],
        "E_B AND alphax_lat": client.count("E_B, alphax_lat")[0],
        "E_B > 0.2 eV (RT-stable)": client.count("E_B>0.2")[0],
        "E_B AND alphax_el AND emass_cbm": client.count("E_B, alphax_el, emass_cbm")[0],
    }

    gaps = [t["target"] for t in targets if t["is_gap"]]

    return {
        "source": "C2DB via ASE-db web front-end (c2db.fysik.dtu.dk)",
        "total_materials": total,
        "targets": targets,
        "auxiliary_features": aux,
        "training_core_intersections": intersections,
        "targets_with_labels": [t["target"] for t in targets if not t["is_gap"]],
        "targets_that_are_the_gap": gaps,
        "gap_summary": {
            "n_targets": len(targets),
            "n_labeled": sum(1 for t in targets if not t["is_gap"]),
            "n_gap": len(gaps),
            "gap_targets": gaps,
            "labels_to_generate": {
                g: total for g in gaps  # 0 exist -> all `total` are unlabeled
            },
        },
    }
