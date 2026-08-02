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


# --- Phase 1: multi-task baseline surrogate (CPU; the GNN upgrade will be GPU) ---
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests==2.33.1", "numpy", "pandas", "scikit-learn", "scipy",
        "mendeleev", "joblib",
    )
    .add_local_dir(HERE, remote_path="/root/excitonic", copy=True)
)


@app.function(image=train_image, cpu=4.0, timeout=3600)
def train_baseline() -> dict:
    """Pull C2DB, featurize, train the multi-task ensemble, return metrics."""
    import runpy
    import sys
    sys.path.insert(0, "/root/excitonic/src")
    os.chdir("/root/excitonic")
    # Build dataset then train (both scripts write under /root/excitonic).
    runpy.run_path("/root/excitonic/scripts/phase1_build_dataset.py", run_name="__main__")
    runpy.run_path("/root/excitonic/scripts/phase1_train_baseline.py", run_name="__main__")
    with open("/root/excitonic/data/manifests/phase1_baseline_metrics.json") as fh:
        return json.load(fh)


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


@app.local_entrypoint()
def train():
    """Phase 1: run the baseline training on Modal; write metrics locally for CI."""
    metrics = train_baseline.remote()
    out_dir = os.path.join(HERE, "data", "manifests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase1_baseline_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[modal] wrote {out_path}")
    for t, d in metrics.get("targets", {}).items():
        m = d["cv_metrics"]
        print(f"[modal]   {t:6s} MAE={m['mae']:.4f} R2={m['r2']:.3f} n={m['n']}")
