"""
"See what we extract" -- stage 1 of the double black box.

Opens the trained JEPA context encoder with interpretability tooling:

  1. Collect token embeddings for every window (the frozen encoder's output).
  2. Train a Sparse Autoencoder (SAE) on those embeddings -> a sparse,
     over-complete feature dictionary (candidate monosemantic features).
  3. Probe: linear-decode structure from the embeddings.
       - synthetic data: decode the *known* latent trajectories -> R^2 tells us
         how much of the generating mechanism the JEPA captured (ground-truth
         validation, the worm's unique advantage).
       - real data: decode per-window mean activity of labeled command neurons.
  4. Ablation: zero each SAE feature, measure the hit to reconstruction ->
     a crude feature-importance ranking.

Writes worm_jepa/artifacts/extraction.json (+ sae.pt). This is the signal that
would feed the *second* black box (program/model discovery) later.
"""
from __future__ import annotations

import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DataConfig, load_worms, WormWindows
from model import JEPAConfig, Encoder


# --------------------------------------------------------------------------- #
# Sparse autoencoder
# --------------------------------------------------------------------------- #
class SAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in)
        self.b_pre = nn.Parameter(torch.zeros(d_in))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.enc(x - self.b_pre))
        recon = self.dec(z) + self.b_pre
        return recon, z


def train_sae(feats: torch.Tensor, d_hidden: int, l1: float = 1e-3,
              epochs: int = 40, device: str = "cpu") -> tuple[SAE, dict]:
    d_in = feats.shape[1]
    sae = SAE(d_in, d_hidden).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    loader = DataLoader(torch.utils.data.TensorDataset(feats), batch_size=256, shuffle=True)
    hist = []
    for ep in range(epochs):
        tot_r = tot_s = nb = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, z = sae(batch)
            r = ((recon - batch) ** 2).mean()
            s = z.abs().mean()
            loss = r + l1 * s
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_r += float(r); tot_s += float(s); nb += 1
        hist.append({"epoch": ep, "recon": tot_r / nb, "l1": tot_s / nb})
    with torch.no_grad():
        _, z = sae(feats.to(device))
        active = (z > 1e-4).float().mean(0)          # per-feature activation freq
    stats = {
        "final_recon": hist[-1]["recon"],
        "n_features": d_hidden,
        "dead_features": int((active < 1e-3).sum()),
        "mean_active_frac": float(active.mean()),
    }
    return sae, {"history": hist, "stats": stats, "activation_freq": active.cpu().numpy()}


# --------------------------------------------------------------------------- #
# Embedding collection + probes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_embeddings(encoder: Encoder, ds: WormWindows, device: str
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (token_feats (M, d_model), window_mean_activity (n_windows, N))."""
    encoder.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    feats, acts = [], []
    for x, _present in loader:
        x = x.to(device)
        emb = encoder.embed_full(x)                  # (B, n_patches, d_model)
        feats.append(emb.reshape(-1, emb.shape[-1]).cpu())
        acts.append(x.mean(1).cpu())                 # mean over time -> (B, N)
    return torch.cat(feats), torch.cat(acts)


def linear_probe_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """Ridge-decode Y from X, report mean R^2 across targets (train==eval, diagnostic)."""
    from numpy.linalg import lstsq
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    W, *_ = lstsq(Xb + 1e-6 * np.random.standard_normal(Xb.shape).astype(X.dtype), Y, rcond=None)
    pred = Xb @ W
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-8
    return float((1 - ss_res / ss_tot).mean())


def extract() -> dict:
    device = os.environ.get("WORM_JEPA_DEVICE", "cpu")
    out_dir = os.environ.get("WORM_JEPA_OUT", os.path.join(os.path.dirname(__file__), "artifacts"))
    ckpt = torch.load(os.path.join(out_dir, "checkpoint.pt"), map_location=device, weights_only=False)
    cfg = JEPAConfig(**ckpt["config"])
    encoder = Encoder(cfg).to(device)
    encoder.load_state_dict(ckpt["context_encoder"])

    _mw = os.environ.get("WORM_JEPA_MAX_WORMS", "").strip()   # empty on push events
    dcfg = DataConfig(
        max_worms=int(_mw) if _mw else None,
        window=cfg.window, stride=cfg.window // 2, patch=cfg.patch,
    )
    worms, names, gt = load_worms(dcfg)
    ds = WormWindows(worms, N=len(names), window=cfg.window, stride=cfg.window // 2)

    token_feats, win_acts = collect_embeddings(encoder, ds, device)

    # --- SAE ---
    d_hidden = 8 * cfg.d_model
    sae, sae_report = train_sae(token_feats, d_hidden=d_hidden, device=device)
    torch.save(sae.state_dict(), os.path.join(out_dir, "sae.pt"))

    result = {
        "n_token_feats": int(token_feats.shape[0]),
        "d_model": cfg.d_model,
        "sae": sae_report["stats"],
    }

    # --- probe ---
    with torch.no_grad():
        # window-level embedding = mean over its patches
        n_patches = cfg.n_patches
        win_emb = token_feats.reshape(-1, n_patches, cfg.d_model).mean(1).numpy()

    if gt is not None:
        # decode known latents: align latent trajectory to each window's mean latent
        latents = gt["latents"]                       # (n_worms, T, n_latents)
        # windows were generated in worm order with stride cfg.window//2 over T=2*window
        # reconstruct per-window mean latent to match ds ordering
        per_win_lat = []
        stride = cfg.window // 2
        for w in range(latents.shape[0]):
            T = latents.shape[1]
            for start in range(0, max(1, T - cfg.window + 1), stride):
                seg = latents[w, start:start + cfg.window]
                if seg.shape[0] < cfg.window:
                    continue
                per_win_lat.append(seg.mean(0))
        Y = np.asarray(per_win_lat, dtype=np.float32)
        m = min(len(Y), len(win_emb))
        result["probe_latent_r2"] = linear_probe_r2(win_emb[:m], Y[:m])
        result["probe_target"] = "synthetic_latents"
    else:
        Y = win_acts.numpy().astype(np.float32)
        m = min(len(Y), len(win_emb))
        result["probe_activity_r2"] = linear_probe_r2(win_emb[:m], Y[:m])
        result["probe_target"] = "labeled_neuron_mean_activity"

    # --- ablation: importance of top-activating SAE features ---
    with torch.no_grad():
        feats = token_feats.to(device)
        base_recon, z = sae(feats)
        base_err = float(((base_recon - feats) ** 2).mean())
        freq = torch.from_numpy(sae_report["activation_freq"]).to(device)
        top = torch.argsort(freq, descending=True)[:10]
        ablation = []
        for f in top:
            z2 = z.clone()
            z2[:, f] = 0.0
            recon2 = sae.dec(z2) + sae.b_pre
            err = float(((recon2 - feats) ** 2).mean())
            ablation.append({"feature": int(f), "recon_err_increase": err - base_err,
                             "activation_freq": float(freq[f])})
    result["ablation_top_features"] = ablation
    result["sae_base_recon_err"] = base_err

    with open(os.path.join(out_dir, "extraction.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("[worm-jepa] extraction summary:")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    extract()
