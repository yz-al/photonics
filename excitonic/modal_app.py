"""
Modal launcher for the excitonic-property project.

Phase 0 job: run the C2DB inventory on a Modal container and return the
coverage manifest to the caller, which writes it under
excitonic/data/manifests/. This proves the GitHub Actions -> Modal compute
path end to end (the same path Phases 1–2 use for GPU training and the
CPU/MPI GW-BSE + EPW label-generation jobs).

Launched from CI:  modal run excitonic/modal_app.py
(Requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment; the CI
workflow falls back to the committed excitonic/modal_tokens.secret.)

Design note: GW-BSE (BerkeleyGW/Yambo) and EPW are CPU/MPI-bound, so Phase 2
jobs will be added here as many-core `@app.function(cpu=...)` entries, NOT GPU.
The GNN surrogate training (Phase 1) will be a `gpu=` entry.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

# Phase 0 needs only requests + stdlib. Ship the project source into the image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests==2.33.1")
    .add_local_dir(HERE, remote_path="/root/excitonic", copy=True)
)

app = modal.App("exciton-fm-phase0")


@app.function(image=image, cpu=1.0, timeout=1800)
def inventory() -> dict:
    """Run the live C2DB inventory inside the Modal container."""
    import sys
    sys.path.insert(0, "/root/excitonic/src")
    from exciton_fm.c2db_client import C2DBClient
    from exciton_fm.inventory import build_inventory

    client = C2DBClient()
    inv = build_inventory(client)
    print(f"[modal] C2DB total={inv['total_materials']} "
          f"labeled={inv['targets_with_labels']} gap={inv['targets_that_are_the_gap']}",
          flush=True)
    return inv


@app.local_entrypoint()
def main():
    """Runs on the CI runner; calls the remote function and writes the manifest."""
    inv = inventory.remote()
    out_dir = os.path.join(HERE, "data", "manifests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase0_coverage.json")
    with open(out_path, "w") as fh:
        json.dump(inv, fh, indent=2)
    print(f"[modal] wrote {out_path}")
    print(f"[modal] total_materials={inv['total_materials']}")
    for t in inv["targets"]:
        tag = "GAP" if t["is_gap"] else f"{t['covered_materials']} [{t['label_quality']}]"
        print(f"[modal]   {t['target']:<11s}: {tag}")
