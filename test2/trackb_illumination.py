"""Track B — DIFFRACTIVE illumination gate + no-mask controls.

Answers two reviewer questions about the D2NN (95.5% MNIST / 37.3% CIFAR were obtained
under IDEAL coherent monochromatic illumination — upper bounds, not deployable numbers):

  (1) NO-MASK CONTROLS: is the trained phase mask doing real work, or is the accuracy
      just blur + a trained linear readout?
        (a) mask_mode="zero"   -- free-space propagation, NO phase mask, train only readout.
        (b) mask_mode="random" -- phase mask frozen at random init, train only readout.
      If these approach the trained-D2NN accuracy, the mask is not the classifier.

  (2) ILLUMINATION GATE: take the TRAINED D2NN and measure test-time accuracy under
      realistic non-ideal illumination:
        (a) wavelength / chromatic  (wl_scale, mask phase ~1/lambda + kernel at lambda)
        (b) coherence               (incoherent_K: average intensity over K speckle draws)
        (c) defocus / object distance (z_scale)
        (d) pose                    (translate / rotate / scale test images)

Real runs only. Writes data/trackb_illumination.json. Bounded by env
  ILLUM_SEEDS (default "0,1,2"), ILLUM_CIFAR_SUBSET (default 20000).
"""
from __future__ import annotations
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from trackb_models import DiffractiveD2NN, count_params
from experiment import train_eval, set_seed

DEVICE = os.environ.get("TEST2_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "trackb_illumination.json")
SEEDS = [int(s) for s in os.environ.get("ILLUM_SEEDS", "0,1,2").split(",")]
CIFAR_SUBSET = int(os.environ.get("ILLUM_CIFAR_SUBSET", "20000"))
DEPTH = 2


def mnist_loaders():
    tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    root = os.path.join(ROOT, "data", "mnist")
    trn = torchvision.datasets.MNIST(root, train=True, download=True, transform=tf)
    tst = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf)
    return (DataLoader(trn, 256, shuffle=True, num_workers=0, drop_last=True),
            DataLoader(tst, 512, shuffle=False, num_workers=0),
            {"in_dim": 784, "n_classes": 10, "in_side": 28, "M": 56})


def cifar_gray_loaders(subset):
    tr = T.Compose([T.Grayscale(), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize((0.5,), (0.25,))])
    te = T.Compose([T.Grayscale(), T.ToTensor(), T.Normalize((0.5,), (0.25,))])
    root = os.path.join(ROOT, "data", "cifar")
    trn = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tr)
    tst = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=te)
    if subset:
        trn = torch.utils.data.Subset(trn, range(subset))
    return (DataLoader(trn, 256, shuffle=True, num_workers=0, drop_last=True),
            DataLoader(tst, 512, shuffle=False, num_workers=0),
            {"in_dim": 1024, "n_classes": 10, "in_side": 32, "M": 48})


@torch.no_grad()
def evaluate(model, loader, fwd_kwargs=None, pose=None):
    """Accuracy with optional model forward kwargs (wl_scale/z_scale/incoherent_K) and
    optional pose transform applied to each image batch (dict: angle, translate, scale)."""
    model.eval()
    fwd_kwargs = fwd_kwargs or {}
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if pose:
            xb = TF.affine(xb, angle=pose.get("angle", 0.0),
                           translate=list(pose.get("translate", (0, 0))),
                           scale=pose.get("scale", 1.0), shear=[0.0, 0.0])
        out = model(xb, **fwd_kwargs)
        correct += (out.argmax(1) == yb).sum().item(); total += yb.numel()
    return correct / max(total, 1)


def train_variant(loaders, meta, mask_mode, epochs, seeds):
    """Train `len(seeds)` D2NNs of a given mask_mode; return (models, accs)."""
    tr, te, _ = loaders
    models, accs = [], []
    for s in seeds:
        set_seed(s)
        m = DiffractiveD2NN(meta["in_dim"], meta["n_classes"], M=meta["M"], depth=DEPTH,
                            in_side=meta["in_side"], mask_mode=mask_mode)
        r = train_eval(m, tr, te, epochs=epochs, device=DEVICE, sched="cosine")
        models.append(m); accs.append(r["best_acc"])
    return models, accs


def stat(a):
    a = np.array(a, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "per_seed": [float(x) for x in a]}


def illumination_sweeps(models, loader):
    """Run each perturbation axis on the TRAINED D2NN models; mean+/-std over models."""
    def sweep(points, kw_fn=None, pose_fn=None):
        rows = []
        for pt in points:
            accs = [evaluate(m, loader,
                             fwd_kwargs=(kw_fn(pt) if kw_fn else None),
                             pose=(pose_fn(pt) if pose_fn else None)) for m in models]
            rows.append({"pt": pt, **stat(accs)})
        return rows

    res = {}
    # (a) wavelength / chromatic: lambda = lambda0 * (1+delta)
    res["wavelength_delta"] = sweep(
        [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10],
        kw_fn=lambda d: {"wl_scale": 1.0 + d})
    # (b) coherence: K speckle realisations averaged in intensity (K=0 == coherent)
    res["coherence_K"] = sweep(
        [0, 1, 2, 4, 8, 16],
        kw_fn=lambda K: {"incoherent_K": int(K)})
    # (c) defocus / object distance: z = z0 * (1+delta)
    res["defocus_delta"] = sweep(
        [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
        kw_fn=lambda d: {"z_scale": 1.0 + d})
    # (d) pose: translation (px), rotation (deg), scale
    res["pose_translate_px"] = sweep(
        [0, 1, 2, 3, 4], pose_fn=lambda t: {"translate": (t, t)})
    res["pose_rotate_deg"] = sweep(
        [0, 5, 10, 15, 20], pose_fn=lambda a: {"angle": float(a)})
    res["pose_scale"] = sweep(
        [1.0, 0.9, 1.1, 0.8, 1.2], pose_fn=lambda s: {"scale": float(s)})
    return res


def run_dataset(name, loaders, epochs_trained, epochs_control):
    tr, te, meta = loaders
    print(f"\n===== {name} =====")
    entry = {"M": meta["M"], "in_side": meta["in_side"], "seeds": SEEDS}

    print("[control] no-mask (zero) — free-space propagation + linear readout")
    _, zero_acc = train_variant(loaders, meta, "zero", epochs_control, SEEDS)
    print("[control] random frozen mask — fixed diffuser + linear readout")
    _, rand_acc = train_variant(loaders, meta, "random", epochs_control, SEEDS)
    print("[trained] learnable phase masks (the D2NN)")
    trained_models, trained_acc = train_variant(loaders, meta, "trained", epochs_trained, SEEDS)

    entry["controls"] = {
        "no_mask_zero": stat(zero_acc),
        "random_frozen_mask": stat(rand_acc),
        "trained_d2nn": stat(trained_acc),
        "mask_uplift_over_nomask": float(np.mean(trained_acc) - np.mean(zero_acc)),
        "mask_uplift_over_random": float(np.mean(trained_acc) - np.mean(rand_acc)),
    }
    print(f"   no-mask={np.mean(zero_acc):.4f}  random={np.mean(rand_acc):.4f}  "
          f"trained={np.mean(trained_acc):.4f}  "
          f"uplift(vs no-mask)={np.mean(trained_acc)-np.mean(zero_acc):+.4f}")

    print("[illumination gate] sweeping wavelength / coherence / defocus / pose ...")
    t0 = time.time()
    entry["illumination"] = illumination_sweeps(trained_models, te)
    entry["illumination_baseline_coherent"] = float(np.mean(trained_acc))
    print(f"   illumination sweeps done in {time.time()-t0:.0f}s")
    # concise console summary
    for axis in ("wavelength_delta", "coherence_K", "defocus_delta"):
        pts = entry["illumination"][axis]
        s = "  ".join(f"{r['pt']}:{r['mean']:.3f}" for r in pts)
        print(f"   {axis:18s} {s}")
    return entry


if __name__ == "__main__":
    results = {"device": DEVICE, "seeds": SEEDS, "depth": DEPTH, "datasets": {}}
    t0 = time.time()
    results["datasets"]["mnist"] = run_dataset(
        "MNIST", mnist_loaders(), epochs_trained=10, epochs_control=7)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[illum] flushed MNIST -> {OUT}")
    try:
        results["datasets"]["cifar10"] = run_dataset(
            "CIFAR-10 (grayscale)", cifar_gray_loaders(CIFAR_SUBSET),
            epochs_trained=14, epochs_control=10)
    except Exception as e:
        print(f"[illum] CIFAR failed/partial: {e}")
        results["datasets"].setdefault("cifar10", {})["error"] = str(e)
    results["total_wall_s"] = round(time.time() - t0, 1)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[illum] done in {results['total_wall_s']}s -> {OUT}")
