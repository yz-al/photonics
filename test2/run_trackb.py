"""Track B accuracy experiment — three matched models via the test2/ harness.

Trains, on MNIST (full) and CIFAR-10 (flattened), with >=5 seeds:
  (a) ReLU MLP baseline        (models.MLP, act="relu")           -- the edge digital net
  (b) Diffractive D2NN         (trackb_models.DiffractiveD2NN)    -- passive phase-mask optic
  (c) Spiking LIF MLP          (trackb_models.SpikingMLP)         -- surrogate-grad, low T

Reports test-accuracy mean/std over seeds, the accuracy GAP vs ReLU, and the SNN's
measured firing sparsity (used by the energy model). Writes data/trackb_accuracy.json.

MNIST runs first and is flushed to disk before CIFAR starts, so a slow/failed CIFAR
never loses the (CPU-complete) MNIST result. Bounded by env:
  TRACKB_MNIST_SUBSET (default None=full), TRACKB_CIFAR_SUBSET (default 20000)
  TRACKB_SEEDS (default 0,1,2,3,4), TRACKB_SKIP (e.g. "cifar")
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from models import MLP
from trackb_models import DiffractiveD2NN, SpikingMLP, count_params
from experiment import train_eval, set_seed

DEVICE = os.environ.get("TEST2_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "trackb_accuracy.json")
SEEDS = [int(s) for s in os.environ.get("TRACKB_SEEDS", "0,1,2,3,4").split(",")]
SKIP = set(os.environ.get("TRACKB_SKIP", "").split(","))
MNIST_SUBSET = os.environ.get("TRACKB_MNIST_SUBSET")
MNIST_SUBSET = int(MNIST_SUBSET) if MNIST_SUBSET else None
CIFAR_SUBSET = int(os.environ.get("TRACKB_CIFAR_SUBSET", "20000"))

WIDTH, DEPTH, T = 256, 2, 8
results = {"device": DEVICE, "seeds": SEEDS, "config": {
    "width": WIDTH, "depth": DEPTH, "T": T,
    "mnist_subset": MNIST_SUBSET, "cifar_subset": CIFAR_SUBSET}, "datasets": {}}


def cifar_gray(subset, batch=256):
    """CIFAR-10 as a grayscale 32x32 field for the D2NN (color discarded — a real
    single-wavelength limitation, flagged)."""
    import torchvision
    import torchvision.transforms as T
    tr = T.Compose([T.Grayscale(), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize((0.5,), (0.25,))])
    te = T.Compose([T.Grayscale(), T.ToTensor(), T.Normalize((0.5,), (0.25,))])
    root = os.path.join(ROOT, "data", "cifar")
    trn = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr)
    tst = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te)
    if subset:
        trn = torch.utils.data.Subset(trn, range(subset))
    from torch.utils.data import DataLoader
    return (DataLoader(trn, batch, shuffle=True, num_workers=0, drop_last=True),
            DataLoader(tst, 512, shuffle=False, num_workers=0),
            {"in_dim": 32 * 32, "n_classes": 10, "in_side": 32})


def macs_mlp(in_dim, width, depth, n_classes):
    """MAC/inference for the ReLU / SNN dense MLP (per-timestep for SNN; multiply by T)."""
    m = in_dim * width + (depth - 1) * width * width + width * n_classes
    return m


def train_model(name, build, loaders, epochs, extra=None, **kw):
    tr, te, meta = loaders
    runs = []
    fires = []
    for s in SEEDS:
        set_seed(s)
        model = build(meta)
        r = train_eval(model, tr, te, epochs=epochs, device=DEVICE, sched="cosine", **kw)
        r["params"] = count_params(model)
        if hasattr(model, "firing_rate"):
            fr = model.firing_rate()
            r["firing_rate"] = fr
            if fr == fr:
                fires.append(fr)
        runs.append(r)
        print(f"    {name} seed{s}: best={r['best_acc']:.4f} "
              f"({r['wall_s']}s)" + (f" fire={r.get('firing_rate'):.3f}" if fires else ""))
    accs = np.array([r["best_acc"] for r in runs], float)
    summ = {
        "n_seeds": len(runs),
        "mean_acc": float(np.nanmean(accs)),
        "std_acc": float(np.nanstd(accs)),
        "min_acc": float(np.nanmin(accs)),
        "max_acc": float(np.nanmax(accs)),
        "per_seed_acc": [float(a) for a in accs],
        "params": runs[0]["params"],
        "mean_wall_s": float(np.mean([r["wall_s"] for r in runs])),
    }
    if fires:
        summ["mean_firing_rate"] = float(np.mean(fires))
        summ["std_firing_rate"] = float(np.std(fires))
    if extra:
        summ.update(extra)
    return summ


def flush():
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[trackb] flushed -> {OUT}")


# ============================ MNIST ============================
def run_mnist():
    print("\n=== MNIST (full-set clean testbed) ===")
    loaders = D.mnist(subset=MNIST_SUBSET, workers=0)
    meta = loaders[2]
    d2_grid = 56
    ds = {}
    ds["relu"] = train_model(
        "relu", lambda m: MLP(m["in_dim"], m["n_classes"], width=WIDTH, depth=DEPTH,
                              act="relu", norm="batch"), loaders, epochs=12,
        extra={"macs_per_inference": macs_mlp(meta["in_dim"], WIDTH, DEPTH, meta["n_classes"])})
    flush_ds("mnist", ds)
    ds["d2nn"] = train_model(
        "d2nn", lambda m: DiffractiveD2NN(m["in_dim"], m["n_classes"], M=d2_grid, depth=DEPTH),
        loaders, epochs=15, extra={"grid_M": d2_grid, "phase_planes": DEPTH})
    flush_ds("mnist", ds)
    ds["spiking"] = train_model(
        "spiking", lambda m: SpikingMLP(m["in_dim"], m["n_classes"], width=WIDTH,
                                        depth=DEPTH, T=T), loaders, epochs=12,
        extra={"T": T, "macs_per_timestep": macs_mlp(meta["in_dim"], WIDTH, DEPTH, meta["n_classes"])})
    ds["gap_vs_relu"] = {
        "d2nn": ds["relu"]["mean_acc"] - ds["d2nn"]["mean_acc"],
        "spiking": ds["relu"]["mean_acc"] - ds["spiking"]["mean_acc"]}
    flush_ds("mnist", ds)


def flush_ds(name, ds):
    results["datasets"][name] = ds
    flush()


# ============================ CIFAR-10 ============================
def run_cifar():
    print("\n=== CIFAR-10 (flattened MLP-class; expected poor, no conv) ===")
    rgb = D.cifar10(subset=CIFAR_SUBSET, workers=0)
    meta = rgb[2]
    ds = {}
    ds["relu"] = train_model(
        "relu", lambda m: MLP(m["in_dim"], m["n_classes"], width=WIDTH, depth=DEPTH,
                              act="relu", norm="batch"), rgb, epochs=20,
        extra={"macs_per_inference": macs_mlp(meta["in_dim"], WIDTH, DEPTH, meta["n_classes"])})
    flush_ds("cifar10", ds)
    ds["spiking"] = train_model(
        "spiking", lambda m: SpikingMLP(m["in_dim"], m["n_classes"], width=WIDTH,
                                        depth=DEPTH, T=T), rgb, epochs=20,
        extra={"T": T, "macs_per_timestep": macs_mlp(meta["in_dim"], WIDTH, DEPTH, meta["n_classes"])})
    flush_ds("cifar10", ds)
    # D2NN sees a grayscale 32x32 field (color discarded — flagged)
    gray = cifar_gray(CIFAR_SUBSET)
    gmeta = gray[2]
    ds["d2nn"] = train_model(
        "d2nn", lambda m: DiffractiveD2NN(gmeta["in_dim"], gmeta["n_classes"], M=48,
                                          depth=DEPTH, in_side=32), gray, epochs=25,
        extra={"grid_M": 48, "phase_planes": DEPTH, "input": "grayscale32 (color discarded)"})
    ds["gap_vs_relu"] = {
        "d2nn": ds["relu"]["mean_acc"] - ds["d2nn"]["mean_acc"],
        "spiking": ds["relu"]["mean_acc"] - ds["spiking"]["mean_acc"]}
    flush_ds("cifar10", ds)


if __name__ == "__main__":
    t0 = time.time()
    if "mnist" not in SKIP:
        run_mnist()
    if "cifar" not in SKIP:
        try:
            run_cifar()
        except Exception as e:
            print(f"[trackb] CIFAR failed/partial: {e}")
            results["datasets"].setdefault("cifar10", {})["error"] = str(e)
            flush()
    results["total_wall_s"] = round(time.time() - t0, 1)
    flush()
    print(f"[trackb] done in {results['total_wall_s']}s")
