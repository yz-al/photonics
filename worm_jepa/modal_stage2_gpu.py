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
    # FAIR-GATE RUN: the credibility gate for the double-black-box plan. Head-to-head on the
    # SAME held-out data: BASELINE = SOTA-recipe affinities + standard agglomeration (MWS);
    # OURS = the SAME SOTA-recipe affinities + our learned multicut+JEPA proofreading. The
    # SOTA-recipe affinity net = autocontext (acrlsd) + merge-averse MALIS + EM augmentation.
    # Scored on VOI AND merge-capped ERL. Both nets trained at equal budget for fairness.
    "WORM_SEG_JEPA_STEPS": os.environ.get("WORM_SEG_JEPA_STEPS", "5000"),  # JEPA feeds agglo edge features
    "WORM_SEG_DEC_STEPS": os.environ.get("WORM_SEG_DEC_STEPS", "8000"),   # real affinity training, both nets equal
    "WORM_S3_SOURCES": os.environ.get("WORM_S3_SOURCES", "sota"),         # plain-sota baseline affinity net
    "WORM_S3_ACRLSD": os.environ.get("WORM_S3_ACRLSD", "1"),              # SOTA-recipe net: autocontext+MALIS+aug
    "WORM_S3_AUGMENT": os.environ.get("WORM_S3_AUGMENT", "1"),            # EM augmentation (elastic/intensity/etc.)
    "WORM_S3_MALIS_M": os.environ.get("WORM_S3_MALIS_M", "5.0"),          # merge-averse structured loss weight
    "WORM_S3_MERGE_BIAS": os.environ.get("WORM_S3_MERGE_BIAS", "0.0"),    # neutral; safe_best_by_erl selects
    "WORM_S3_CTX_STEPS": os.environ.get("WORM_S3_CTX_STEPS", "800"),      # LSD head fine-tune (agglo feature)
    "WORM_S3_NEVAL": os.environ.get("WORM_S3_NEVAL", "4"),
    "WORM_S3_LABEL_POOL": os.environ.get("WORM_S3_LABEL_POOL", "16"),
    "WORM_S3_SPARSE": os.environ.get("WORM_S3_SPARSE", ""),
    "WORM_S3_ACTIVE": os.environ.get("WORM_S3_ACTIVE", ""),
    "WORM_S3_DBB": os.environ.get("WORM_S3_DBB", "0"),
    "WORM_S3_EDGE": os.environ.get("WORM_S3_EDGE", "0"),
    "WORM_S3_BEAT": os.environ.get("WORM_S3_BEAT", "0"),
    "WORM_S3_AUDIT": "1" if os.environ.get("WORM_S3_MODE") == "audit" else os.environ.get("WORM_S3_AUDIT", "0"),
    "WORM_S3_CURVE": "1" if os.environ.get("WORM_S3_MODE") == "curve" else os.environ.get("WORM_S3_CURVE", "0"),
    "WORM_S3_GROW": "1" if os.environ.get("WORM_S3_MODE") == "grow" else os.environ.get("WORM_S3_GROW", "0"),
    "WORM_S3_RECIPE": "1" if os.environ.get("WORM_S3_MODE") == "recipe" else os.environ.get("WORM_S3_RECIPE", "0"),
    "WORM_S3_MECHINTERP": "1" if os.environ.get("WORM_S3_MODE") == "mechinterp" else os.environ.get("WORM_S3_MECHINTERP", "0"),
    "WORM_S3_STAGE0A": "1" if os.environ.get("WORM_S3_MODE") == "stage0a" else os.environ.get("WORM_S3_STAGE0A", "0"),
    "WORM_S3_DBB_STAGES": "1" if os.environ.get("WORM_S3_MODE") == "dbb" else os.environ.get("WORM_S3_DBB_STAGES", "0"),
    "WORM_S3_THEORY": "1" if os.environ.get("WORM_S3_MODE") == "theory" else os.environ.get("WORM_S3_THEORY", "0"),
    "WORM_S3_HARDMINE": "1" if os.environ.get("WORM_S3_MODE") == "hardmine" else os.environ.get("WORM_S3_HARDMINE", "0"),
    "WORM_S3_AUG_BANK": os.environ.get("WORM_S3_AUG_BANK", "2"),     # autocontext aug-bank size (GPU mem)
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
    # ERL-FIRST (merge-averse) operating-point search. A MERGE is catastrophic in
    # connectomics (poisons a whole traced run) while a split is cheaply proofread, so we
    # optimize ERL, not VOI, and sweep merge_bias into the CONSERVATIVE (negative) range --
    # negative bias makes GAEC merge LESS (fewer catastrophic merges, more cheap splits).
    # seed_q tunes fragment granularity. Per cell we record the ERL-best variant + its
    # merge-error count, and pick the ERL-max cell. All re-derives from saved affinities.
    biases = [float(x) for x in os.environ.get("WORM_S3_BIAS_SWEEP", "-1.5,-1.0,-0.5,0,0.5").split(",")]
    seedqs = [float(x) for x in os.environ.get("WORM_S3_SEEDQ_SWEEP", "0.6,0.8").split(",")]
    sweep = {}
    best = None
    for q in seedqs:
        for b in biases:
            r = AG.run(*args, ctx, seed_q=q, merge_bias=b, **kw)
            be = r["best_by_erl"]; bv = r["best_by_voi"]
            me = r["merge_errors_by_method"]
            cell = {"seed_q": q, "merge_bias": b,
                    "best_by_erl": be, "ERL": r[be]["ERL"], "VOI_at_erlbest": r[be]["VOI"],
                    "merge_errors_at_erlbest": me.get(be),
                    "best_by_voi": bv, "VOI": r[bv]["VOI"], "ERL_at_voibest": r[bv]["ERL"],
                    "cremi_voibest": r["cremi_score_proxy"][bv],
                    "mean_fragments": r.get("mean_fragments"), "merge_errors_by_method": me}
            sweep[f"seedq{q}_bias{b}"] = cell
            if best is None or cell["ERL"] > best[1]["ERL"]:   # ERL-FIRST selection
                best = (f"seedq{q}_bias{b}", cell)
    print(json.dumps({"seed": seed, "objective": "ERL (merge-averse)",
                      "best_cell": best[0], "grid": sweep}, indent=2))
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
    if os.environ.get("WORM_S3_MODE") == "recipe":             # autocontext vs plain affinities (both MWS), per seed
        print(f"[modal] affinity-recipe head-to-head (autocontext vs plain, MWS) seeds {SEEDS} ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        results = list(run_seed.map(SEEDS))
        def agg(key, metric):
            vs = [r[key][metric] for r in results]
            return {"mean": round(statistics.mean(vs), 4), "std": round(statistics.pstdev(vs), 4) if len(vs) > 1 else 0.0}
        dv = [r["delta_VOI_autoctx_minus_plain"] for r in results]
        de = [r["delta_ERL_autoctx_minus_plain"] for r in results]
        merged = {"seeds": SEEDS,
                  "plain": {"VOI": agg("plain", "VOI"), "ERL": agg("plain", "ERL")},
                  "autocontext": {"VOI": agg("autocontext", "VOI"), "ERL": agg("autocontext", "ERL")},
                  "delta_VOI": {"mean": round(statistics.mean(dv), 4), "std": round(statistics.pstdev(dv), 4) if len(dv) > 1 else 0.0, "per_seed": dv},
                  "delta_ERL": {"mean": round(statistics.mean(de), 4), "std": round(statistics.pstdev(de), 4) if len(de) > 1 else 0.0, "per_seed": de},
                  "per_seed": results}
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        for fn in ("recipe.json", "segment3d.json"):
            with open(os.path.join(art, fn), "w") as f:
                json.dump(merged, f, indent=2)
        print(json.dumps(merged, indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "grow":               # progressive train->freeze->measure->run, per seed
        print(f"[modal] progressive grow (freeze+measure until ERL plateaus) seeds {SEEDS} ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        results = list(run_seed.map(SEEDS))
        from collections import defaultdict
        bys = defaultdict(lambda: defaultdict(list))          # aggregate trajectory by checkpoint step
        for r in results:
            for p in r.get("trajectory", []):
                for k in ("VOI", "ERL", "affinity_acc"):
                    bys[p["steps"]][k].append(p[k])
        traj = []
        for s in sorted(bys):
            row = {"steps": s, "n_seeds": len(bys[s]["ERL"])}
            for k in ("VOI", "ERL", "affinity_acc"):
                vs = bys[s][k]
                row[k] = {"mean": round(statistics.mean(vs), 4), "std": round(statistics.pstdev(vs), 4) if len(vs) > 1 else 0.0}
            traj.append(row)
        merged = {"seeds": SEEDS, "trajectory_agg": traj,
                  "converged_final": [{"seed": r.get("grow_seed"), "converged": r.get("converged"),
                                       "final_step": r.get("final_step"), "learned": r.get("learned"),
                                       "converged_metrics": r.get("converged_metrics"),
                                       "agglo_run": r.get("agglo_run")} for r in results]}
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        for fn in ("grow.json", "segment3d.json"):
            with open(os.path.join(art, fn), "w") as f:
                json.dump(merged, f, indent=2)
        print(json.dumps(merged, indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "curve":              # clean learning curve: 1 container/seed, aggregate
        print(f"[modal] clean learning curve (steps-scaled, multi-seed {SEEDS}) ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        results = list(run_seed.map(SEEDS))                    # one A10G per seed
        from collections import defaultdict
        byn = defaultdict(lambda: defaultdict(list))
        for r in results:
            for p in r.get("points", []):
                for k in ("VOI", "ERL", "affinity_acc"):
                    byn[p["n_labels"]][k].append(p["test"][k])
                byn[p["n_labels"]]["steps"] = p["steps"]
        agg = []
        for n in sorted(byn):
            row = {"n_labels": n, "steps": byn[n]["steps"]}
            for k in ("VOI", "ERL", "affinity_acc"):
                vs = byn[n][k]
                row[k] = {"mean": round(statistics.mean(vs), 4),
                          "std": round(statistics.pstdev(vs), 4) if len(vs) > 1 else 0.0}
            agg.append(row)
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        merged = {"seeds": SEEDS, "aggregate": agg, "per_seed": results}
        for fn in ("curve.json", "segment3d.json"):
            with open(os.path.join(art, fn), "w") as f:
                json.dump(merged, f, indent=2)
        print(json.dumps(agg, indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "hardmine":          # membrane hard-region mining vs uniform, per seed
        print(f"[modal] membrane hard-region mining vs uniform control, seeds {SEEDS} ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        results = list(run_seed.map(SEEDS))

        def agg(key, metric):
            vs = [r[key][metric] for r in results]
            return {"mean": round(statistics.mean(vs), 4), "std": round(statistics.pstdev(vs), 4) if len(vs) > 1 else 0.0}
        de = [r["dERL_hm_minus_control"] for r in results]; dv = [r["dVOI_hm_minus_control"] for r in results]
        de_mean = statistics.mean(de); de_std = statistics.pstdev(de) if len(de) > 1 else 0.0
        merged = {"seeds": SEEDS, "ERL_noise_floor_ref": 15.0,
                  "base": {"VOI": agg("base", "VOI"), "ERL": agg("base", "ERL")},
                  "control_uniform": {"VOI": agg("control_uniform", "VOI"), "ERL": agg("control_uniform", "ERL")},
                  "membrane_hardmine": {"VOI": agg("membrane_hardmine", "VOI"), "ERL": agg("membrane_hardmine", "ERL")},
                  "dERL_hm_minus_control": {"mean": round(de_mean, 4), "std": round(de_std, 4), "per_seed": de},
                  "dVOI_hm_minus_control": {"mean": round(statistics.mean(dv), 4), "per_seed": dv},
                  "verdict": ("REAL: hard-mining beats uniform on ERL beyond the ~15 noise floor"
                              if de_mean > 15 and de_mean - de_std > 0 else
                              "WITHIN NOISE: ERL gain does not clear the resample noise floor -> not a real effect")}
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        for fn in ("hardmine.json", "segment3d.json"):
            with open(os.path.join(art, fn), "w") as f:
                json.dump(merged, f, indent=2)
        print(json.dumps(merged, indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "theory":            # non-ODE pipeline theory (5 toolkits), single GPU
        print("[modal] pipeline theory: percolation / EVT / spectral / scale-space / discrete ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        result = run_seed.remote(0)
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        for fn in ("theory.json", "segment3d.json"):
            with open(os.path.join(art, fn), "w") as f:
                json.dump(result, f, indent=2)
        print(json.dumps(result.get("pipeline_theory", result), indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "dbb":                # double-black-box stages 3-7, single GPU
        print("[modal] DBB stages 3-7 on mechinterp-pulled compact model ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        result = run_seed.remote(0)
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "segment3d.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result.get("stages_3_7", result), indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "stage0a":            # feasibility gate: identifiability, single GPU
        print("[modal] STAGE 0a: identifiability (profile likelihood) on mechinterp-pulled model ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        result = run_seed.remote(0)
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "segment3d.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result.get("stage0a", result), indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "mechinterp":         # first black box only, single GPU
        print("[modal] FIRST BLACK BOX: mechinterp best model (no synthesis) ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        result = run_seed.remote(0)
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "segment3d.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result.get("mechinterp", result), indent=2))
        return
    if os.environ.get("WORM_S3_MODE") == "audit":              # learnability-ceiling audit on ONE GPU (no _merge)
        print("[modal] learnability ceiling audit (single seed, GPU) ...", flush=True)
        print("[modal] cache:", prepare.remote(), flush=True)
        result = run_seed.remote(0)
        art = os.path.join(HERE, "artifacts"); os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "segment3d.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result.get("audit", result), indent=2))
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
