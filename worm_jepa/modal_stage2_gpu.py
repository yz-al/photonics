"""
Modal GPU launcher for the 3D segmentation super run -- PARALLELIZED across seeds.

The seeds are independent, so instead of one A10G running them sequentially
(~2 h), we fan out one A10G container PER SEED via .map() (~one seed's wall-clock,
~40 min) and merge the per-seed results into mean +- std. Same GPU-dollars, ~3x
faster wall-clock. (Further money optimisation -- moving the CPU-only mutex
watershed off the billed GPU -- is a TODO noted below.)

Launched from CI: modal run worm_jepa/modal_stage2_gpu.py
"""
import os
import json
import statistics
import modal

HERE = os.path.dirname(os.path.abspath(__file__))

_PASS = {
    "WORM_CREMI_SAMPLES": os.environ.get("WORM_CREMI_SAMPLES", "A,B,C"),
    "WORM_VOL_CROP": os.environ.get("WORM_VOL_CROP", "160"),
    "WORM_VOL_ZC": os.environ.get("WORM_VOL_ZC", "8"),
    "WORM_VOL_DIM": os.environ.get("WORM_VOL_DIM", "256"),
    "WORM_VOL_DEPTH": os.environ.get("WORM_VOL_DEPTH", "6"),
    "WORM_VOL_BATCH": os.environ.get("WORM_VOL_BATCH", "8"),
    "WORM_SEG_JEPA_STEPS": os.environ.get("WORM_SEG_JEPA_STEPS", "5000"),
    "WORM_SEG_DEC_STEPS": os.environ.get("WORM_SEG_DEC_STEPS", "600"),
    "WORM_S3_NEVAL": os.environ.get("WORM_S3_NEVAL", "3"),
    "WORM_S3_LABEL_POOL": os.environ.get("WORM_S3_LABEL_POOL", "1,16"),
}
SEEDS = [int(x) for x in os.environ.get("WORM_S3_SEEDS", "0,1,2").split(",")]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy<2.3", "scikit-learn", "scikit-image",
                 "scipy", "tifffile", "h5py")
    .env(_PASS)
    .add_local_dir(HERE, remote_path="/root/worm_jepa", copy=True)
)
app = modal.App("worm-stage2-gpu")


@app.function(gpu="A10G", image=image, timeout=10800)
def run_seed(seed: int) -> dict:
    """One A10G container runs segment3d for a SINGLE seed and returns its result
    dict (each metric's 'mean' is that seed's value, 'std' 0)."""
    import sys
    import runpy
    os.chdir("/root/worm_jepa")
    for p in ("/root/worm_jepa", "/root/worm_jepa/vjepa"):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["WORM_EM_DEVICE"] = "cuda"
    os.environ["WORM_S3_SEEDS"] = str(seed)          # this container = one seed
    import torch
    print(f"[modal] seed {seed} torch {torch.__version__} cuda {torch.cuda.is_available()}", flush=True)
    runpy.run_path("/root/worm_jepa/vjepa/segment3d.py", run_name="__main__")
    with open("/root/worm_jepa/vjepa/segment3d.json") as f:
        return json.load(f)


def _merge(dicts):
    """Merge per-seed result dicts into mean +- std across seeds."""
    def stats(vals):
        return {"mean": round(statistics.mean(vals), 4),
                "std": round(statistics.pstdev(vals), 4)}
    base = dict(dicts[0])
    base["seeds"] = [d.get("seeds", ["?"])[0] if d.get("seeds") else i for i, d in enumerate(dicts)]
    # label_efficiency[source][pool][metric] -> aggregate the per-seed means
    le = {}
    for src in dicts[0]["label_efficiency"]:
        le[src] = {}
        for pool in dicts[0]["label_efficiency"][src]:
            le[src][pool] = {}
            for metric in dicts[0]["label_efficiency"][src][pool]:
                le[src][pool][metric] = stats([d["label_efficiency"][src][pool][metric]["mean"] for d in dicts])
    base["label_efficiency"] = le
    es = {}
    for grp in dicts[0]["error_set_analysis"]:
        es[grp] = {m: stats([d["error_set_analysis"][grp][m]["mean"] for d in dicts])
                   for m in dicts[0]["error_set_analysis"][grp]}
    base["error_set_analysis"] = es
    base["anticollapse"]["embedding_std_final_per_seed"] = [
        (d["anticollapse"]["embedding_std_final_per_seed"] or [None])[0] for d in dicts]
    return base


@app.local_entrypoint()
def main():
    print(f"[modal] fanning out {len(SEEDS)} seeds in parallel: {SEEDS}", flush=True)
    results = list(run_seed.map(SEEDS))              # PARALLEL: one A10G per seed
    merged = _merge(results)
    art = os.path.join(HERE, "artifacts")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "segment3d.json"), "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[modal] merged {len(results)} seeds -> {os.path.join(art, 'segment3d.json')}", flush=True)
    print(json.dumps(merged.get("error_set_analysis", {}), indent=2))
