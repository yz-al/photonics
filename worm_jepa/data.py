"""
Data loading for the C. elegans JEPA.

Source: the homogenized C. elegans neural-activity dataset
`qsimeon/celegans_neural_data` on the HuggingFace Hub (Simeon et al., 2024),
a single parquet `worm_data_short.parquet` in *long* format: one row per
(source_dataset, worm, neuron), with the neuron's calcium trace stored as a
list under `calcium_data` and named identity under `neuron`.

This module turns that into per-worm activity matrices of shape (T, N) over a
*canonical* neuron axis (fixed ordering of named neurons), windows them into
fixed-length training samples, and produces the context/target block masks the
JEPA consumes.

Two modes:
  * real (default on Modal/CI): downloads the parquet and reconstructs worms.
  * synthetic (WORM_JEPA_SYNTHETIC=1): generates worms from a *known* sparse
    linear-nonlinear dynamical system ("toy connectome"), so the extraction
    step has ground truth to be validated against. This is what the free
    in-container smoke test uses -- no 669 MB download, no GPU.
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

HF_REPO = "qsimeon/celegans_neural_data"
HF_FILE = "worm_data_short.parquet"


# --------------------------------------------------------------------------- #
# Worm container
# --------------------------------------------------------------------------- #
@dataclass
class Worm:
    source: str
    worm_id: str
    activity: np.ndarray          # (T, N) float32 over canonical neuron axis
    present: np.ndarray           # (N,) bool -- which canonical neurons were recorded
    dt: float = 0.5               # seconds per timestep (post-resampling)
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.activity.shape[0]


# --------------------------------------------------------------------------- #
# Real data: HuggingFace parquet -> per-worm (T, N) matrices
# --------------------------------------------------------------------------- #
def _resample_trace(trace: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a 1-D trace to target_len (handles ragged worm lengths)."""
    trace = np.asarray(trace, dtype=np.float32)
    if trace.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if trace.size == target_len:
        return trace
    xs = np.linspace(0.0, 1.0, num=trace.size, dtype=np.float32)
    xt = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(xt, xs, trace).astype(np.float32)


def load_real_worms(
    max_worms: Optional[int] = None,
    resample_T: int = 512,
    labeled_only: bool = True,
    cache_dir: Optional[str] = None,
) -> tuple[list[Worm], list[str]]:
    """
    Download the parquet and reconstruct per-worm activity matrices.

    Returns (worms, canonical_neuron_names). The canonical axis is the sorted
    union of labeled neuron names seen across the selected worms, so every worm
    shares one neuron ordering (missing neurons are zero-filled + flagged in
    `present`).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset", cache_dir=cache_dir
    )
    # Only the columns we need -- keeps the 669 MB file's memory footprint down.
    cols = ["source_dataset", "worm", "neuron", "is_labeled_neuron", "calcium_data"]
    df = pd.read_parquet(path, columns=cols)
    if labeled_only:
        df = df[df["is_labeled_neuron"]]

    df["key"] = df["source_dataset"].astype(str) + "::" + df["worm"].astype(str)
    keys = list(dict.fromkeys(df["key"].tolist()))
    if max_worms is not None:
        keys = keys[:max_worms]
    df = df[df["key"].isin(set(keys))]

    canon = sorted(df["neuron"].astype(str).unique().tolist())
    idx = {name: i for i, name in enumerate(canon)}
    N = len(canon)

    worms: list[Worm] = []
    for key, g in df.groupby("key", sort=False):
        src, wid = key.split("::", 1)
        act = np.zeros((resample_T, N), dtype=np.float32)
        present = np.zeros(N, dtype=bool)
        for neuron, trace in zip(g["neuron"].astype(str), g["calcium_data"]):
            j = idx[neuron]
            act[:, j] = _resample_trace(np.asarray(trace, dtype=np.float32), resample_T)
            present[j] = True
        worms.append(Worm(source=src, worm_id=wid, activity=act, present=present))
    return worms, canon


# --------------------------------------------------------------------------- #
# Synthetic data: known "toy connectome" -> activity (for offline smoke test)
# --------------------------------------------------------------------------- #
def make_synthetic_worms(
    n_worms: int = 24,
    N: int = 48,
    T: int = 512,
    n_latents: int = 6,
    seed: int = 0,
) -> tuple[list[Worm], list[str], dict]:
    """
    Generate worms from a shared latent dynamical system driving N neurons via a
    sparse mixing matrix W (the "toy connectome"). Each latent is a smooth
    oscillator; neuron activity = softplus(W @ latents) + noise, then calcium-
    smoothed. Returns (worms, names, ground_truth) where ground_truth carries W
    and the latent trajectories so extraction can be scored against them.
    """
    rng = np.random.default_rng(seed)
    # sparse latent->neuron mixing (each neuron reads from 1-2 latents)
    W = np.zeros((N, n_latents), dtype=np.float32)
    for i in range(N):
        for latent in rng.choice(n_latents, size=rng.integers(1, 3), replace=False):
            W[i, latent] = rng.normal(0.0, 1.0)
    names = [f"SYN{i:03d}" for i in range(N)]

    worms: list[Worm] = []
    latents_all = []
    for w in range(n_worms):
        t = np.arange(T, dtype=np.float32)
        latents = np.zeros((T, n_latents), dtype=np.float32)
        for latent in range(n_latents):
            freq = 0.01 * (latent + 1) * (1.0 + 0.1 * rng.standard_normal())
            phase = rng.uniform(0, 2 * math.pi)
            latents[:, latent] = np.sin(2 * math.pi * freq * t + phase)
        drive = latents @ W.T                                    # (T, N)
        act = np.log1p(np.exp(drive)) + 0.05 * rng.standard_normal((T, N)).astype(np.float32)
        # simple calcium smoothing (exponential kernel)
        k = np.exp(-np.arange(12) / 4.0).astype(np.float32)
        k /= k.sum()
        act = np.stack([np.convolve(act[:, j], k, mode="same") for j in range(N)], axis=1)
        act = (act - act.mean(0)) / (act.std(0) + 1e-6)
        worms.append(Worm(source="synthetic", worm_id=f"w{w:03d}",
                          activity=act.astype(np.float32),
                          present=np.ones(N, dtype=bool)))
        latents_all.append(latents)
    gt = {"W": W, "latents": np.stack(latents_all), "n_latents": n_latents}
    return worms, names, gt


def load_worms(cfg: "DataConfig") -> tuple[list[Worm], list[str], Optional[dict]]:
    """Dispatch on WORM_JEPA_SYNTHETIC / cfg.synthetic."""
    if cfg.synthetic:
        worms, names, gt = make_synthetic_worms(
            n_worms=cfg.max_worms or 24, N=cfg.synth_N, T=cfg.window * 2, seed=cfg.seed
        )
        return worms, names, gt
    worms, names = load_real_worms(
        max_worms=cfg.max_worms, resample_T=cfg.resample_T, cache_dir=cfg.cache_dir
    )
    return worms, names, None


# --------------------------------------------------------------------------- #
# Windowing + Dataset
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    synthetic: bool = field(default_factory=lambda: os.environ.get("WORM_JEPA_SYNTHETIC", "") == "1")
    max_worms: Optional[int] = None
    resample_T: int = 512
    synth_N: int = 48
    window: int = 256
    stride: int = 128
    patch: int = 16
    seed: int = 0
    cache_dir: Optional[str] = None


class WormWindows(Dataset):
    """Fixed-length windows over all worms; each item is (window, neuron_mask)."""

    def __init__(self, worms: list[Worm], N: int, window: int, stride: int):
        self.N = N
        self.window = window
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for worm in worms:
            act, present = worm.activity, worm.present
            for start in range(0, max(1, worm.T - window + 1), stride):
                w = act[start:start + window]
                if w.shape[0] < window:
                    continue
                self.samples.append((w.astype(np.float32), present.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        w, present = self.samples[i]
        return torch.from_numpy(w), torch.from_numpy(present)


def make_block_masks(
    n_patches: int,
    n_targets: int = 4,
    target_span: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[int], list[int]]:
    """
    I-JEPA style: choose `n_targets` contiguous target blocks (each `target_span`
    patches); context = all patches not covered by any target block.
    Returns (context_idx, target_idx).
    """
    rng = rng or np.random.default_rng()
    target = set()
    for _ in range(n_targets):
        if n_patches - target_span <= 0:
            break
        s = int(rng.integers(0, n_patches - target_span + 1))
        target.update(range(s, s + target_span))
    context = [p for p in range(n_patches) if p not in target]
    if not context:                         # degenerate guard
        context = [0]
        target.discard(0)
    return context, sorted(target)
