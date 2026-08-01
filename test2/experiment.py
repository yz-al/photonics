"""Training loop and metrics for Test 2 (CPU or GPU)."""
from __future__ import annotations
import math, time
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_eval(model, train_loader, test_loader, epochs=15, lr=1e-3, wd=5e-4,
               device="cuda", label_smoothing=0.0, log=False, sched="onecycle"):
    """Train, return dict with best test acc, final acc, convergence, stability."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = len(train_loader)
    if sched == "cosine":
        warmup = max(1, epochs * steps // 20)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps - warmup)
        lin = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=warmup)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [lin, cos], milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    accs, losses = [], []
    diverged = False
    epochs_to_90 = None  # epoch reaching 90% of own best
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run = 0.0; nb = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run += loss.item(); nb += 1
        losses.append(run / max(nb, 1))
        if diverged:
            accs.append(float("nan"))
            break
        acc = evaluate(model, test_loader, device)
        accs.append(acc)
        if log:
            print(f"    ep{ep:02d} loss={losses[-1]:.3f} acc={acc:.4f}")
    best = float(np.nanmax(accs)) if accs else float("nan")
    if not diverged and best == best:
        thr = 0.9 * best
        for i, a in enumerate(accs):
            if a >= thr:
                epochs_to_90 = i + 1
                break
    return {
        "best_acc": best,
        "final_acc": float(accs[-1]) if accs and accs[-1] == accs[-1] else float("nan"),
        "acc_curve": [float(a) for a in accs],
        "loss_curve": [float(l) for l in losses],
        "diverged": bool(diverged),
        "epochs_to_90pct": epochs_to_90,
        "wall_s": round(time.time() - t0, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item(); total += yb.numel()
    return correct / max(total, 1)


def summarize(runs):
    """Aggregate a list of train_eval dicts (over seeds) -> mean/std/spread."""
    best = np.array([r["best_acc"] for r in runs], dtype=float)
    conv = [r["epochs_to_90pct"] for r in runs if r["epochs_to_90pct"]]
    ndiv = sum(r["diverged"] for r in runs)
    valid = best[~np.isnan(best)]
    return {
        "n_seeds": len(runs),
        "mean_best_acc": float(np.mean(valid)) if len(valid) else float("nan"),
        "std_best_acc": float(np.std(valid)) if len(valid) else float("nan"),
        "min_best_acc": float(np.min(valid)) if len(valid) else float("nan"),
        "max_best_acc": float(np.max(valid)) if len(valid) else float("nan"),
        "median_epochs_to_90pct": float(np.median(conv)) if conv else None,
        "n_diverged": int(ndiv),
        "per_seed_best": [float(b) for b in best],
    }
