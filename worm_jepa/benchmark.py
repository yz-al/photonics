"""
Head-to-head benchmark: flat JEPA vs hierarchical JEPA vs forecasting NN.

Metrics (synthetic data, known ground truth):
  * latent_r2 / slow_r2 / fast_r2 -- ridge-probe each model's representation for
    the known latent trajectories, overall and per timescale band. Tests the
    hierarchy hypothesis: does the hier JEPA's *slow* level recover the *slow*
    latents better than a flat model?
  * forecast_mse -- next-step prediction error (forecaster's native task).
  * connectome_corr -- correlation between the forecaster's effective
    neuron-neuron coupling (input Jacobian) and the true Gram W @ W.T. This is
    the ground-truth-wiring recovery number -- the whole thesis in one metric.

Runs on CPU (smoke) or CUDA (Modal). Writes artifacts/benchmark.json.

Env: WORM_JEPA_BENCH_EPOCHS, WORM_JEPA_SYNTHETIC, WORM_JEPA_REAL (1 to also run
the real dataset for forecast_mse / activity probe), WORM_JEPA_MAX_WORMS.
"""
from __future__ import annotations

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import DataConfig, load_worms, WormWindows, make_block_masks
from model import JEPAConfig, WormJEPA, Encoder
from models_bench import HierWormJEPA, Forecaster


def probe_r2(X, Y, ridge=1.0, test_frac=0.3, seed=0):
    """
    Honest linear probe: standardize features on train, closed-form ridge fit on
    train, report mean per-target R^2 on a held-out test split. Avoids the
    underdetermined train==eval trap (d_features >= n_samples -> trivial R^2=1).
    """
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    te, tr = perm[:n_test], perm[n_test:]
    if len(tr) < 2:
        return float("nan")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr = (X[tr] - mu) / sd; Xte = (X[te] - mu) / sd
    ym = Y[tr].mean(0)
    Ytr = Y[tr] - ym
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(d), Xtr.T @ Ytr)
    pred = Xte @ W + ym
    ss_res = ((Y[te] - pred) ** 2).sum(0)
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0) + 1e-8
    return float((1 - ss_res / ss_tot).mean())


def _device() -> str:
    return os.environ.get("WORM_JEPA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def _int(name, default):
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def per_window_latents(gt, cfg: JEPAConfig, stride: int):
    """Per-window mean latent, aligned to WormWindows ordering."""
    latents = gt["latents"]                                # (n_worms, T, n_latents)
    out = []
    for w in range(latents.shape[0]):
        T = latents.shape[1]
        for start in range(0, max(1, T - cfg.window + 1), stride):
            seg = latents[w, start:start + cfg.window]
            if seg.shape[0] < cfg.window:
                continue
            out.append(seg.mean(0))
    return np.asarray(out, dtype=np.float32)


def windows_tensor(ds: WormWindows) -> torch.Tensor:
    return torch.stack([ds[i][0] for i in range(len(ds))])


# --------------------------------------------------------------------------- #
# Trainers (compact, self-contained)
# --------------------------------------------------------------------------- #
def train_flat(ds, cfg, epochs, device):
    model = WormJEPA(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.05)
    lossf = torch.nn.SmoothL1Loss()
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            ci, ti = make_block_masks(cfg.n_patches, rng=rng)
            if not ti:
                continue
            pred, tgt = model(x, ci, ti)
            loss = lossf(pred, tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.update_target(0.996)
    return model


def train_hier(ds, cfg, epochs, device):
    model = HierWormJEPA(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.05)
    lossf = torch.nn.SmoothL1Loss()
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    rng = np.random.default_rng(1)
    n_coarse = cfg.n_patches // 2
    for _ in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            ci, ti = make_block_masks(cfg.n_patches, rng=rng)
            c2, t2 = make_block_masks(n_coarse, n_targets=2, target_span=2, rng=rng)
            if not ti or not t2:
                continue
            (p1, g1), (p2, g2) = model(x, ci, ti, c2, t2)
            loss = lossf(p1, g1) + lossf(p2, g2)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.update_target(0.996)
    return model


def train_forecaster(ds, cfg, epochs, device):
    model = Forecaster(cfg.n_neurons).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    last = 0.0
    for _ in range(epochs):
        tot = nb = 0.0
        for x, _ in loader:
            x = x.to(device)
            pred = model(x)
            loss = lossf(pred, x[:, 1:, :])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        last = tot / max(1, nb)
    return model, last


# --------------------------------------------------------------------------- #
# Representations for probing
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rep_flat(model, X, device):
    enc: Encoder = model.context_encoder
    return enc.embed_full(X.to(device)).mean(1).cpu().numpy()


@torch.no_grad()
def rep_hier(model, X, device):
    return model.slow_representation(X.to(device)).cpu().numpy()


@torch.no_grad()
def rep_forecaster(model, X, device):
    h, _ = model.gru(X.to(device)[:, :-1, :])
    return h.mean(1).cpu().numpy()


def probe_bands(rep, Y, slow_idx, fast_idx):
    return {
        "latent_r2": probe_r2(rep, Y),
        "slow_r2": probe_r2(rep, Y[:, slow_idx]),
        "fast_r2": probe_r2(rep, Y[:, fast_idx]),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_synthetic(epochs, device):
    dcfg = DataConfig(synthetic=True, max_worms=_int("WORM_JEPA_MAX_WORMS", 0) or 40,
                      window=256, stride=128, patch=16, synth_N=48)
    worms, names, gt = load_worms(dcfg)
    ds = WormWindows(worms, N=len(names), window=dcfg.window, stride=dcfg.stride)
    cfg = JEPAConfig(n_neurons=len(names), window=dcfg.window, patch=dcfg.patch,
                     d_model=_int("WORM_JEPA_DMODEL", 128), depth=_int("WORM_JEPA_DEPTH", 4))
    X = windows_tensor(ds)
    Y = per_window_latents(gt, cfg, dcfg.stride)
    m = min(len(X), len(Y)); X, Y = X[:m], Y[:m]
    slow_idx, fast_idx = list(gt["slow_idx"]), list(gt["fast_idx"])
    coupling = gt["neuron_coupling"]

    results = {}

    flat = train_flat(ds, cfg, epochs, device)
    results["flat_jepa"] = probe_bands(rep_flat(flat, X, device), Y, slow_idx, fast_idx)

    hier = train_hier(ds, cfg, epochs, device)
    results["hier_jepa"] = probe_bands(rep_hier(hier, X, device), Y, slow_idx, fast_idx)

    fore, fmse = train_forecaster(ds, cfg, epochs, device)
    results["forecaster"] = probe_bands(rep_forecaster(fore, X, device), Y, slow_idx, fast_idx)
    results["forecaster"]["forecast_mse"] = fmse

    # connectome recovery: forecaster effective coupling vs true Gram
    J = torch.zeros(cfg.n_neurons, cfg.n_neurons, device=device)
    n_probe = min(8, len(X))
    for i in range(n_probe):
        J += fore.effective_connectivity(X[i:i + 1].to(device))
    J = (J / n_probe).cpu().numpy()
    off = ~np.eye(cfg.n_neurons, dtype=bool)
    corr = float(np.corrcoef(J[off], np.abs(coupling)[off])[0, 1])
    results["forecaster"]["connectome_corr"] = corr

    return {"data": "synthetic", "n_windows": m, "n_neurons": len(names),
            "n_latents": int(gt["n_latents"]), "epochs": epochs, "results": results}


def run_real(epochs, device):
    dcfg = DataConfig(synthetic=False, max_worms=_int("WORM_JEPA_MAX_WORMS", 0) or None,
                      window=256, stride=128, patch=16)
    worms, names, _ = load_worms(dcfg)
    ds = WormWindows(worms, N=len(names), window=dcfg.window, stride=dcfg.stride)
    cfg = JEPAConfig(n_neurons=len(names), window=dcfg.window, patch=dcfg.patch,
                     d_model=_int("WORM_JEPA_DMODEL", 128), depth=_int("WORM_JEPA_DEPTH", 4))
    X = windows_tensor(ds)
    act = X.mean(1).numpy()                                # per-window mean activity target

    results = {}
    flat = train_flat(ds, cfg, epochs, device)
    results["flat_jepa"] = {"activity_r2": probe_r2(rep_flat(flat, X, device), act)}
    hier = train_hier(ds, cfg, epochs, device)
    results["hier_jepa"] = {"activity_r2": probe_r2(rep_hier(hier, X, device), act)}
    fore, fmse = train_forecaster(ds, cfg, epochs, device)
    results["forecaster"] = {"activity_r2": probe_r2(rep_forecaster(fore, X, device), act),
                             "forecast_mse": fmse}
    return {"data": "real", "n_windows": len(ds), "n_neurons": len(names),
            "epochs": epochs, "results": results}


def main():
    device = _device()
    out_dir = os.environ.get("WORM_JEPA_OUT", os.path.join(os.path.dirname(__file__), "artifacts"))
    os.makedirs(out_dir, exist_ok=True)
    epochs = _int("WORM_JEPA_BENCH_EPOCHS", 30)

    report = {"device": device, "runs": []}
    report["runs"].append(run_synthetic(epochs, device))
    if os.environ.get("WORM_JEPA_REAL", "") == "1":
        report["runs"].append(run_real(epochs, device))

    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("[worm-jepa] benchmark:")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
