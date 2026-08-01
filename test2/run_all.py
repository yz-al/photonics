"""
Test 2 orchestration: Stage A (clean activation comparison) + Stage B (optics
constraints). Writes test2/results/test2_results.json.

Config via env:
  TEST2_MODE = fast | full   (fast = local CPU smoke test; full = Modal GPU run)
  TEST2_DEVICE = cuda | cpu  (auto if unset)
"""
from __future__ import annotations
import os, sys, json, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from models import MLP, SmallCNN, TinyTransformer, OpticsMLP, count_params
from experiment import train_eval, summarize, set_seed

MODE = os.environ.get("TEST2_MODE", "fast").lower()
DEVICE = os.environ.get("TEST2_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

SKIP = set(os.environ.get("TEST2_SKIP", "").split(","))  # e.g. "cnn"

if MODE == "full":
    CFG = dict(seeds=[0, 1, 2, 3, 4],
               mlp_epochs=25, mlp_subset=60000,
               cnn_epochs=20, cnn_subset=None,
               tr_epochs=8, tr_subset=20000,
               sb_epochs=25, mnist_subset=None,
               depths=[2, 4, 8, 16], noise_seeds=[0, 1, 2])
elif MODE == "local":  # real statistics for CPU-feasible parts (skip CIFAR CNN)
    CFG = dict(seeds=[0, 1, 2, 3, 4],
               mlp_epochs=12, mlp_subset=20000,
               cnn_epochs=12, cnn_subset=8000,
               tr_epochs=5, tr_subset=10000,
               sb_epochs=12, mnist_subset=12000,
               depths=[2, 4, 8, 16], noise_seeds=[0, 1, 2])
else:  # fast local smoke test — validates code paths, not statistics
    CFG = dict(seeds=[0, 1],
               mlp_epochs=2, mlp_subset=6000,
               cnn_epochs=1, cnn_subset=2000,
               tr_epochs=1, tr_subset=2000,
               sb_epochs=2, mnist_subset=4000,
               depths=[2, 4, 8], noise_seeds=[0])

ACTIVATIONS = ["relu", "gelu", "square", "modsq", "abs", "scaled_quad"]
results = {"mode": MODE, "device": DEVICE, "config": CFG, "stageA": {}, "stageB": {}}
print(f"[test2] MODE={MODE} DEVICE={DEVICE}")


def run_seeds(build_model, loaders, epochs, lr=1e-3, seeds=None, sched="onecycle", **kw):
    tr, te, meta = loaders
    runs = []
    for s in (seeds or CFG["seeds"]):
        set_seed(s)
        model = build_model(meta)
        r = train_eval(model, tr, te, epochs=epochs, lr=lr, device=DEVICE, sched=sched, **kw)
        r["params"] = count_params(model)
        runs.append(r)
    summ = summarize(runs)
    summ["params"] = runs[0]["params"]
    summ["mean_wall_s"] = float(np.mean([r["wall_s"] for r in runs]))
    return summ


# ===========================================================================
# STAGE A — clean activation comparison, no optical constraints
# ===========================================================================
def stage_a():
    print("\n=== STAGE A: activation comparison ===")

    # --- MLP on covtype (tabular)
    print("[A] MLP / covtype")
    loaders = D.covtype(subset=CFG["mlp_subset"])
    tabA = {}
    for act in ACTIVATIONS:
        tabA[act] = run_seeds(
            lambda m, act=act: MLP(m["in_dim"], m["n_classes"], width=256, depth=3,
                                   act=act, norm="batch"),
            loaders, CFG["mlp_epochs"])
        print(f"   {act:12s} acc={tabA[act]['mean_best_acc']:.4f}±{tabA[act]['std_best_acc']:.4f} "
              f"div={tabA[act]['n_diverged']}")
    results["stageA"]["mlp_covtype"] = tabA

    # --- CNN on CIFAR-10 (image)
    if "cnn" in SKIP:
        print("[A] SmallCNN / CIFAR-10  -- SKIPPED (TEST2_SKIP=cnn)")
    else:
        print("[A] SmallCNN / CIFAR-10")
        loaders = D.cifar10(subset=CFG["cnn_subset"])
        cnnA = {}
        for act in ACTIVATIONS:
            cnnA[act] = run_seeds(
                lambda m, act=act: SmallCNN(n_classes=10, act=act, norm="batch", width=48),
                loaders, CFG["cnn_epochs"])
            print(f"   {act:12s} acc={cnnA[act]['mean_best_acc']:.4f}±{cnnA[act]['std_best_acc']:.4f} "
                  f"div={cnnA[act]['n_diverged']}")
        results["stageA"]["cnn_cifar10"] = cnnA

    # --- Transformer on AG News / assoc-recall (language/sequence)
    # TEST2_TEXT=assoc forces the synthetic task (no download); default tries AG News.
    text_task = os.environ.get("TEST2_TEXT", "agnews")
    print(f"[A] TinyTransformer / {text_task}")
    loaders = D.assoc_recall(subset=CFG["tr_subset"]) if text_task == "assoc" \
        else D.agnews(subset=CFG["tr_subset"])
    meta = loaders[2]
    trA = {}
    for act in ACTIVATIONS:
        trA[act] = run_seeds(
            lambda m, act=act: TinyTransformer(m["vocab"], m["seq_len"], m["n_classes"],
                                               d_model=128, layers=3, act=act),
            loaders, CFG["tr_epochs"], lr=5e-4)
        print(f"   {act:12s} acc={trA[act]['mean_best_acc']:.4f}±{trA[act]['std_best_acc']:.4f} "
              f"div={trA[act]['n_diverged']}")
    results["stageA"]["transformer_" + meta["task"]] = trA

    # --- Depth vs stability probe: square activation, norm in {none,batch,layer}
    print("[A] depth/stability probe (square activation)")
    loaders = D.covtype(subset=CFG["mlp_subset"])
    probe = {}
    for norm in ["none", "batch", "layer"]:
        for depth in CFG["depths"]:
            key = f"{norm}_d{depth}"
            probe[key] = run_seeds(
                lambda m, d=depth, nm=norm: MLP(m["in_dim"], m["n_classes"], width=256,
                                                depth=d, act="square", norm=nm),
                loaders, CFG["mlp_epochs"], seeds=CFG["noise_seeds"])
            print(f"   {key:12s} acc={probe[key]['mean_best_acc']:.4f} "
                  f"div={probe[key]['n_diverged']}/{probe[key]['n_seeds']}")
    results["stageA"]["depth_stability_square"] = probe


# ===========================================================================
# STAGE B — add the constraints optics actually imposes
# ===========================================================================
def stage_b():
    print("\n=== STAGE B: optical constraints (MNIST, clean testbed) ===")
    loaders = D.mnist(subset=CFG["mnist_subset"])
    meta = loaders[2]
    DEPTH = 2   # optics |.|^2 is competitive at depth 1-2; depth sweep below shows the break
    OPT = dict(sched="cosine")

    # optics-depth sweep: where does the |.|^2 cascade break? (full precision)
    print("[B] optics-native depth sweep (|.|^2 stability)")
    depthsweep = {}
    for d in [1, 2, 3, 4]:
        w = {1: 256, 2: 180, 3: 150, 4: 130}[d]
        depthsweep[f"depth{d}"] = run_seeds(
            lambda m, d=d, w=w: OpticsMLP(m["in_dim"], m["n_classes"], width=w, depth=d),
            loaders, CFG["sb_epochs"], **OPT)
        e = depthsweep[f"depth{d}"]
        print(f"   depth{d}: acc={e['mean_best_acc']:.4f}±{e['std_best_acc']:.4f} div={e['n_diverged']}")
    results["stageB"]["optics_depth_sweep"] = depthsweep

    # baselines + quantisation (ReLU and square, real MLP)
    print("[B] baselines + quantisation")
    quant = {}
    for act in ["relu", "square"]:
        for bits in [None, 8, 4]:
            key = f"{act}_{bits}b"
            quant[key] = run_seeds(
                lambda m, act=act, b=bits: MLP(m["in_dim"], m["n_classes"], width=256,
                                               depth=DEPTH, act=act, norm="batch", act_bits=b),
                loaders, CFG["sb_epochs"], **OPT)
            print(f"   {key:12s} acc={quant[key]['mean_best_acc']:.4f}±{quant[key]['std_best_acc']:.4f}")
    results["stageB"]["quantization_mlp"] = quant

    # complex-linear + |.|^2 optics-native MLP, matched param count to ReLU ref
    print("[B] optics-native complex + |.|^2 (matched params)")
    ref_p = quant["relu_None"]["params"]
    ow = 256
    for w in range(64, 512, 4):
        if count_params(OpticsMLP(meta["in_dim"], meta["n_classes"], width=w, depth=DEPTH)) >= ref_p:
            ow = w; break
    print(f"   ref relu params={ref_p}, optics width={ow} "
          f"params={count_params(OpticsMLP(meta['in_dim'], meta['n_classes'], width=ow, depth=DEPTH))}")
    optics = {}
    variants = {
        "complex_modsq_fp": dict(width=ow),
        "complex_modsq_8b": dict(width=ow, act_bits=8),
        "complex_modsq_4b": dict(width=ow, act_bits=4),
        "complex_modsq_4b_nonneg": dict(width=ow, act_bits=4, nonneg_input=True),
    }
    for name, kw in variants.items():
        optics[name] = run_seeds(
            lambda m, kw=kw: OpticsMLP(m["in_dim"], m["n_classes"], depth=DEPTH, **kw),
            loaders, CFG["sb_epochs"], **OPT)
        print(f"   {name:24s} acc={optics[name]['mean_best_acc']:.4f}±{optics[name]['std_best_acc']:.4f} "
              f"params={optics[name]['params']}")
    results["stageB"]["optics_native_mlp"] = optics

    # noise (SNR) sweep on the optics-native 4-bit model
    print("[B] shot-noise SNR sweep")
    noise = {}
    for snr in [None, 40, 30, 20, 15, 10]:
        key = f"snr_{snr}"
        noise[key] = run_seeds(
            lambda m, s=snr: OpticsMLP(m["in_dim"], m["n_classes"], width=ow, depth=DEPTH,
                                       act_bits=4, snr_db=s),
            loaders, CFG["sb_epochs"], seeds=CFG["noise_seeds"], **OPT)
        print(f"   {key:10s} acc={noise[key]['mean_best_acc']:.4f}±{noise[key]['std_best_acc']:.4f}")
    results["stageB"]["noise_sweep_mlp"] = noise

    # non-negativity cost: differential encoding param overhead vs signed
    results["stageB"]["nonneg_cost"] = {
        "signed_input_params": optics["complex_modsq_4b"]["params"],
        "nonneg_diff_params": optics["complex_modsq_4b_nonneg"]["params"],
        "relu_ref_params": ref_p,
    }

    # headline gap: optics-native (complex+|.|^2+4b+nonneg+noise@20dB) vs relu-8b
    print("[B] headline optics-native vs ReLU gap")
    head = {}
    head["relu_8b"] = quant["relu_8b"]
    head["optics_native"] = run_seeds(
        lambda m: OpticsMLP(m["in_dim"], m["n_classes"], width=ow, depth=DEPTH,
                            act_bits=4, nonneg_input=True, snr_db=20),
        loaders, CFG["sb_epochs"], **OPT)
    gap = head["relu_8b"]["mean_best_acc"] - head["optics_native"]["mean_best_acc"]
    head["gap_relu8b_minus_optics"] = float(gap)
    print(f"   relu_8b={head['relu_8b']['mean_best_acc']:.4f}  "
          f"optics_native={head['optics_native']['mean_best_acc']:.4f}  gap={gap:+.4f}")
    results["stageB"]["headline_gap_mlp"] = head


if __name__ == "__main__":
    stage_a()
    stage_b()
    path = os.path.join(OUT, f"test2_results_{MODE}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[test2] wrote {path}")
