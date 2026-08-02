"""
Train the C. elegans JEPA.

Device-agnostic: runs on CPU (in-container smoke test) or CUDA (Modal A10G via
the GitHub Actions workflow). Configuration is by environment variable so the
same entrypoint serves both the smoke test and the real GPU run:

  WORM_JEPA_SYNTHETIC=1   use the offline toy-connectome data (no HF download)
  WORM_JEPA_EPOCHS        training epochs           (default 30; smoke uses ~5)
  WORM_JEPA_MAX_WORMS     cap number of worms loaded (default: all)
  WORM_JEPA_DEVICE        cpu | cuda                (default: auto)
  WORM_JEPA_OUT           output dir                (default: worm_jepa/artifacts)

Writes to WORM_JEPA_OUT:
  checkpoint.pt   context/target/predictor weights + config + neuron names
  history.json    per-epoch loss (for the training-curve figure)
  meta.json       run metadata (data mode, #worms, #windows, N, device)
"""
from __future__ import annotations

import os
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DataConfig, load_worms, WormWindows, make_block_masks
from model import JEPAConfig, WormJEPA


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _device() -> str:
    d = os.environ.get("WORM_JEPA_DEVICE", "").strip()
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def cosine_momentum(step: int, total: int, base: float) -> float:
    return 1.0 - (1.0 - base) * 0.5 * (1 + np.cos(np.pi * step / max(1, total)))


def train() -> dict:
    device = _device()
    out_dir = os.environ.get("WORM_JEPA_OUT", os.path.join(os.path.dirname(__file__), "artifacts"))
    os.makedirs(out_dir, exist_ok=True)

    dcfg = DataConfig(
        max_worms=_env_int("WORM_JEPA_MAX_WORMS", 0) or None,
        window=_env_int("WORM_JEPA_WINDOW", 256),
        stride=_env_int("WORM_JEPA_STRIDE", 128),
        patch=_env_int("WORM_JEPA_PATCH", 16),
    )
    worms, names, gt = load_worms(dcfg)
    N = len(names)
    ds = WormWindows(worms, N=N, window=dcfg.window, stride=dcfg.stride)
    if len(ds) == 0:
        raise RuntimeError("No training windows produced -- check window/stride vs worm length.")

    batch_size = _env_int("WORM_JEPA_BATCH", 32)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    mcfg = JEPAConfig(
        n_neurons=N, window=dcfg.window, patch=dcfg.patch,
        d_model=_env_int("WORM_JEPA_DMODEL", 128),
        depth=_env_int("WORM_JEPA_DEPTH", 4),
    )
    model = WormJEPA(mcfg).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.05
    )
    loss_fn = torch.nn.SmoothL1Loss()
    rng = np.random.default_rng(dcfg.seed)

    epochs = _env_int("WORM_JEPA_EPOCHS", 30)
    total_steps = epochs * max(1, len(loader))
    history: list[dict] = []
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for x, _present in loader:
            x = x.to(device)
            ctx_idx, tgt_idx = make_block_masks(mcfg.n_patches, rng=rng)
            if not tgt_idx:
                continue
            pred, tgt = model(x, ctx_idx, tgt_idx)
            loss = loss_fn(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target(cosine_momentum(step, total_steps, mcfg.ema_base))
            running += float(loss.item())
            n_batches += 1
            step += 1
        avg = running / max(1, n_batches)
        history.append({"epoch": epoch, "loss": avg})
        print(f"[worm-jepa] epoch {epoch:3d}/{epochs}  loss {avg:.5f}  "
              f"({time.time() - t0:.0f}s)")

    ckpt = {
        "context_encoder": model.context_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "config": mcfg.__dict__,
        "neuron_names": names,
    }
    torch.save(ckpt, os.path.join(out_dir, "checkpoint.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    meta = {
        "data_mode": "synthetic" if dcfg.synthetic else "real",
        "n_worms": len(worms), "n_windows": len(ds), "n_neurons": N,
        "device": device, "epochs": epochs, "final_loss": history[-1]["loss"],
        "d_model": mcfg.d_model, "depth": mcfg.depth,
        "window": dcfg.window, "patch": dcfg.patch,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if gt is not None:
        np.savez(os.path.join(out_dir, "synthetic_gt.npz"), **gt)
    print(f"[worm-jepa] wrote checkpoint + history + meta to {out_dir}")
    return meta


if __name__ == "__main__":
    train()
