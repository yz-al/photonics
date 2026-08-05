#!/usr/bin/env python3
"""
Phase 0 driver — run the C2DB inventory and write the data manifests.

    python excitonic/scripts/phase0_inventory.py

Writes:
    excitonic/data/manifests/c2db_key_inventory.json   (full schema key list)
    excitonic/data/manifests/phase0_coverage.json      (target coverage + gap)

Runs against the live C2DB web front-end over plain HTTPS (works locally and
on a GitHub Actions runner). Also invoked on Modal by excitonic/modal_app.py
to demonstrate the GHA -> Modal compute path end to end.
"""
from __future__ import annotations

import json
import os
import sys

# Allow running as a plain script without installing the package.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from exciton_fm.c2db_client import C2DBClient          # noqa: E402
from exciton_fm.inventory import build_inventory        # noqa: E402

MANIFEST_DIR = os.path.abspath(os.path.join(HERE, "..", "data", "manifests"))


def main() -> int:
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    client = C2DBClient()

    print(f"[phase0] querying C2DB at {client.base} ...", flush=True)
    inv = build_inventory(client)

    total = inv["total_materials"]
    print(f"[phase0] total materials in C2DB: {total}", flush=True)
    print("[phase0] target coverage:")
    for t in inv["targets"]:
        tag = "GAP (0 labels)" if t["is_gap"] else f"{t['covered_materials']} labels [{t['label_quality']}]"
        print(f"    - {t['target']:<11s} {t['name'][:44]:<44s} : {tag}")

    cov_path = os.path.join(MANIFEST_DIR, "phase0_coverage.json")
    with open(cov_path, "w") as fh:
        json.dump(inv, fh, indent=2)
    print(f"[phase0] wrote {cov_path}", flush=True)

    # Return value is also handy for the Modal caller.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
