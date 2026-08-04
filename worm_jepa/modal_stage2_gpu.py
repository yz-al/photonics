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
    # HEAVY-SOTA RUN: the error debug said the residual errors are diffuse/low-confidence
    # and driven by AFFINITY QUALITY (not a targetable head), so the honest lever is heavier
    # affinity training. Train ONLY the sota affinity net (WORM_S3_SOURCES=sota) at ~30k
    # steps (base) + a warm-started continuation, then agglomerate + re-run the analysis.
    "WORM_SEG_JEPA_STEPS": os.environ.get("WORM_SEG_JEPA_STEPS", "5000"),  # JEPA feeds agglo edge features
    "WORM_SEG_DEC_STEPS": os.environ.get("WORM_SEG_DEC_STEPS", "30000"),  # heavy SOTA affinity training
    "WORM_S3_SOURCES": os.environ.get("WORM_S3_SOURCES", "sota"),         # focus: only train sota heavily
    "WORM_S3_CTX_STEPS": os.environ.get("WORM_S3_CTX_STEPS", "800"),      # LSD head fine-tune (agglo feature)
    "WORM_S3_NEVAL": os.environ.get("WORM_S3_NEVAL", "4"),
    "WORM_S3_LABEL_POOL": os.environ.get("WORM_S3_LABEL_POOL", "16"),
    "WORM_S3_SPARSE": os.environ.get("WORM_S3_SPARSE", ""),
    "WORM_S3_ACTIVE": os.environ.get("WORM_S3_ACTIVE", ""),
    "WORM_S3_DBB": os.environ.get("WORM_S3_DBB", "0"),
    "WORM_S3_EDGE": os.environ.get("WORM_S3_EDGE", "0"),
    "WORM_S3_BEAT": os.environ.get("WORM_S3_BEAT", "0"),
    "WORM_S3_AGGLO": os.environ.get("WORM_S3_AGGLO", "1"),          # two-specialist multicut vs MWS
    "WORM_S3_CTX": os.environ.get("WORM_S3_CTX", ""),
    "WORM_S3_LSD": os.environ.get("WORM_S3_LSD", ""),
    "WORM_S3_REFINER": os.environ.get("WORM_S3_REFINER", ""),
    "WORM_VOL_DENSE": os.environ.get("WORM_VOL_DENSE", "1"),         # V-JEPA-2.1 dense features
    "WORM_VOL_EMA": os.environ.get("WORM_VOL_EMA", "0.998"),        # collapse fix: slower EMA target
    "WORM_EM_VAR": os.environ.get("WORM_EM_VAR", "0.6"),           # collapse fix: stronger variance hinge (0.4 left 1/3 seeds collapsing)
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

# Optimization: cache the ~525MB CREMI download on a persistent Volume so it is
# fetched ONCE (by prepare) and reused by every seed-container and every future
# run, instead of re-downloading per container per run.
cremi_vol = modal.Volume.from_name("worm-cremi-cache", create_if_missing=True)
CACHE = "/cache/cremi"
_CREMI_URL = "https://cremi.org/static/data/sample_{}_20160501.hdf"


@app.function(image=image, volumes={"/cache": cremi_vol}, timeout=3600)
def prepare():
    """Download CREMI A/B/C into the shared Volume once (race-free, before fan-out)."""
    import urllib.request
    os.makedirs(CACHE, exist_ok=True)
    for s in os.environ.get("WORM_CREMI_SAMPLES", "A,B,C").split(","):
        p = os.path.join(CACHE, f"sample_{s}.hdf")
        if not os.path.exists(p):
            print(f"[modal] caching CREMI sample {s} -> Volume", flush=True)
            urllib.request.urlretrieve(_CREMI_URL.format(s), p)
    cremi_vol.commit()
    return sorted(os.listdir(CACHE))


@app.function(gpu="A10G", image=image, volumes={"/cache": cremi_vol}, timeout=18000)
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
    os.environ["WORM_EM_DIR"] = CACHE                # read CREMI from the cached Volume
    os.environ["WORM_S3_SEEDS"] = str(seed)          # this container = one seed
    # checkpoint agglomeration inputs to the Volume so agglo/error-analysis re-runs are FREE (CPU)
    os.environ["WORM_S3_AGGLO_SAVE"] = os.path.join(CACHE, f"agglo_seed{seed}.npz")
    import torch
    print(f"[modal] seed {seed} torch {torch.__version__} cuda {torch.cuda.is_available()}", flush=True)
    runpy.run_path("/root/worm_jepa/vjepa/segment3d.py", run_name="__main__")
    cremi_vol.commit()                               # persist the saved agglo inputs
    with open("/root/worm_jepa/vjepa/segment3d.json") as f:
        return json.load(f)


@app.function(image=image, volumes={"/cache": cremi_vol}, timeout=3600)
def rerun_agglo(seed: int = 0) -> dict:
    """FREE (CPU-only) re-run of agglomeration + error analysis from saved inputs -- no
    GPU, no retraining. Use to tune the multicut / lifted weights / features / error
    analysis at ~$0.10 instead of a full ~$6-8 run."""
    import sys
    import numpy as np
    os.chdir("/root/worm_jepa")
    for p in ("/root/worm_jepa", "/root/worm_jepa/vjepa"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import segment3d as S
    import agglomerate as AG
    d = np.load(os.path.join(CACHE, f"agglo_seed{seed}.npz"))

    def unstack(k):
        if k not in d:
            return None
        dt = np.int32 if k.endswith("seg") else np.float32   # segs stay integer
        return [d[k][i].astype(dt) for i in range(len(d[k]))]
    ctx = {"SHORT": S.SHORT, "OFFS": S.OFFS, "mutex_watershed": S.mutex_watershed,
           "seg_metrics": S.seg_metrics, "erl_proxy": S.erl_proxy}
    args = [unstack("tr_aff"), unstack("tr_seg"), unstack("ev_aff"), unstack("ev_seg")]
    kw = dict(train_lsds=unstack("tr_lsd"), eval_lsds=unstack("ev_lsd"),
              train_jepas=unstack("tr_jepa"), eval_jepas=unstack("ev_jepa"))
    # The winning multicut is SPLIT-dominated (over-segments), so we attack over-seg on
    # two free levers: merge_bias (GAEC merges more) and seed_q (the interior-core quantile
    # = fragment granularity; higher -> coarser -> fewer splits to stitch). NOTE thr_over
    # is inert here -- the adaptive quantile floor dominates it -- so we sweep seed_q, the
    # knob that actually moves mean_fragments. Both re-derive from the saved affinities.
    biases = [float(x) for x in os.environ.get("WORM_S3_BIAS_SWEEP", "1.0,1.5").split(",")]
    seedqs = [float(x) for x in os.environ.get("WORM_S3_SEEDQ_SWEEP", "0.6,0.7,0.8").split(",")]
    sweep = {}
    best = None
    for q in seedqs:
        for b in biases:
            r = AG.run(*args, ctx, seed_q=q, merge_bias=b, **kw)
            bv = r["best_by_voi"]
            cell = {"best": bv, "seed_q": q, "merge_bias": b, **r[bv],
                    "cremi": r["cremi_score_proxy"][bv],
                    "mean_fragments": r.get("mean_fragments"),
                    "error_types": r.get("error_types_mws_vs_best", {})}
            sweep[f"seedq{q}_bias{b}"] = cell
            if best is None or cell["VOI"] < best[1]["VOI"]:
                best = (f"seedq{q}_bias{b}", cell)
    print(json.dumps({"seed": seed, "best_cell": best[0], "grid": sweep}, indent=2))
    return {"seed": seed, "best_cell": best[0], "best": best[1], "grid": sweep}


@app.function(image=image, volumes={"/cache": cremi_vol}, timeout=3600)
def debug_errors(seed: int = 0) -> dict:
    """FREE (CPU-only) error debug on the saved agglomeration checkpoints -- confidence,
    structure (is-error AUC), and clustering of the residual edge errors. Tells us whether
    to spend GPU on heavier SOTA training (diffuse errors) or build a targeted fix
    (structured errors) BEFORE kicking off a big run."""
    import sys
    import numpy as np
    os.chdir("/root/worm_jepa")
    for p in ("/root/worm_jepa", "/root/worm_jepa/vjepa"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import segment3d as S
    import debug_errors as DBG
    d = np.load(os.path.join(CACHE, f"agglo_seed{seed}.npz"))

    def unstack(k):
        if k not in d:
            return None
        dt = np.int32 if k.endswith("seg") else np.float32
        return [d[k][i].astype(dt) for i in range(len(d[k]))]
    ctx = {"SHORT": S.SHORT, "OFFS": S.OFFS, "mutex_watershed": S.mutex_watershed,
           "seg_metrics": S.seg_metrics, "erl_proxy": S.erl_proxy}
    args = [unstack("tr_aff"), unstack("tr_seg"), unstack("ev_aff"), unstack("ev_seg")]
    kw = dict(train_lsds=unstack("tr_lsd"), eval_lsds=unstack("ev_lsd"),
              train_jepas=unstack("tr_jepa"), eval_jepas=unstack("ev_jepa"))
    mb = float(os.environ.get("WORM_S3_MERGE_BIAS", "1.25"))
    r = DBG.run(*args, ctx, merge_bias=mb, **kw)
    print(json.dumps({"seed": seed, **r}, indent=2))
    return {"seed": seed, **r}


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
    stds = [(d["anticollapse"]["embedding_std_final_per_seed"] or [None])[0] for d in dicts]
    base["anticollapse"]["embedding_std_final_per_seed"] = stds
    # JEPA-dependent analyses (double black box, edge detector/correction) are only
    # meaningful on a NON-collapsed encoder. Occasionally one seed collapses; pull
    # these sections from the HEALTHIEST seed (max embedding std) instead of dicts[0],
    # and record which seed that was.
    healthy = max(range(len(dicts)), key=lambda i: (stds[i] if stds[i] is not None else -1))
    for key in ("double_black_box", "sota_edge_cases", "beat_sota"):
        if key in dicts[healthy]:
            base[key] = dicts[healthy][key]
    base["jepa_analysis_seed"] = {"index": healthy, "std_final": stds[healthy],
                                  "seed": base["seeds"][healthy]}
    return base


@app.local_entrypoint()
def main():
    if os.environ.get("WORM_S3_MODE") == "rerun":              # FREE agglo sweep from saved checkpoint (CPU)
        print("[modal] rerun_agglo merge_bias x seed_q sweep (no training) ...", flush=True)
        results = list(rerun_agglo.map(SEEDS))
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "agglo_sweep.json"), "w") as f:
            json.dump({"seeds": SEEDS, "results": results}, f, indent=2)
        print(f"[modal] wrote {os.path.join(art, 'agglo_sweep.json')}", flush=True)
        print(json.dumps(results, indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "debug":              # FREE error debug from saved checkpoint (CPU)
        print("[modal] debug_errors: confidence + structure + clustering (no training) ...", flush=True)
        results = list(debug_errors.map(SEEDS))
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "debug_errors.json"), "w") as f:
            json.dump({"seeds": SEEDS, "results": results}, f, indent=2)
        print(f"[modal] wrote {os.path.join(art, 'debug_errors.json')}", flush=True)
        print(json.dumps(results, indent=2))
        return
    print("[modal] preparing CREMI cache (once) ...", flush=True)
    print("[modal] cache:", prepare.remote(), flush=True)      # populate Volume before fan-out
    print(f"[modal] fanning out {len(SEEDS)} seeds in parallel: {SEEDS}", flush=True)
    results = list(run_seed.map(SEEDS))              # PARALLEL: one A10G per seed
    merged = _merge(results)
    art = os.path.join(HERE, "artifacts")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "segment3d.json"), "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[modal] merged {len(results)} seeds -> {os.path.join(art, 'segment3d.json')}", flush=True)
    print(json.dumps(merged.get("error_set_analysis", {}), indent=2))
