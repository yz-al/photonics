"""
A time-series JEPA (Joint-Embedding Predictive Architecture) for C. elegans
whole-brain calcium activity.

Same family as I-JEPA / Brain-JEPA, adapted to multivariate neural time series:

  * A worm window (window, N) is split along time into patches of length `patch`,
    each flattened over neurons and linearly embedded -> token sequence.
  * A *context encoder* embeds the visible (context) patches.
  * A *predictor* takes the context tokens plus positional queries for the masked
    (target) patches and predicts their latent representations.
  * A *target encoder* (EMA of the context encoder) embeds the full sequence; its
    representations at the target patches are the (stop-grad) prediction targets.
  * Loss = smooth-L1 in latent space. No trace reconstruction -- prediction lives
    entirely in representation space, which is the whole point of a JEPA.

The context encoder is the artifact we later open up (SAE / probes / ablations).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class JEPAConfig:
    n_neurons: int          # canonical neuron-axis width N
    window: int = 256
    patch: int = 16
    d_model: int = 128
    depth: int = 4
    pred_depth: int = 2
    n_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    ema_base: float = 0.996

    @property
    def n_patches(self) -> int:
        return self.window // self.patch

    @property
    def patch_dim(self) -> int:
        return self.patch * self.n_neurons


def _sinusoidal(n_pos: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n_pos).unsqueeze(1).float()
    i = torch.arange(dim).unsqueeze(0).float()
    angle = pos / torch.pow(10000, (2 * (i // 2)) / dim)
    pe = torch.zeros(n_pos, dim)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


class TransformerTrunk(nn.Module):
    def __init__(self, d_model: int, depth: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(x))


class PatchEmbed(nn.Module):
    """(B, window, N) -> (B, n_patches, d_model)."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.patch_dim, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, N = x.shape
        p = self.cfg.patch
        x = x.reshape(B, W // p, p * N)         # patchify along time, flatten neurons
        return self.proj(x)


class Encoder(nn.Module):
    """Patch embed + positional embedding + transformer trunk."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PatchEmbed(cfg)
        self.register_buffer("pos", _sinusoidal(cfg.n_patches, cfg.d_model), persistent=False)
        self.trunk = TransformerTrunk(cfg.d_model, cfg.depth, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.embed(x) + self.pos.unsqueeze(0)
        if keep is not None:
            tokens = tokens[:, keep, :]
        return self.trunk(tokens)

    @torch.no_grad()
    def embed_full(self, x: torch.Tensor) -> torch.Tensor:
        """All-patch representations (B, n_patches, d_model) -- used by probes/SAE."""
        return self.forward(x, keep=None)


class Predictor(nn.Module):
    """Predict target-patch representations from context tokens + target queries."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", _sinusoidal(cfg.n_patches, cfg.d_model), persistent=False)
        self.trunk = TransformerTrunk(cfg.d_model, cfg.pred_depth, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)

    def forward(self, ctx: torch.Tensor, ctx_idx: list[int], tgt_idx: list[int]) -> torch.Tensor:
        B = ctx.shape[0]
        ctx = ctx + self.pos[ctx_idx].unsqueeze(0)
        q = self.mask_token.expand(B, len(tgt_idx), -1) + self.pos[tgt_idx].unsqueeze(0)
        seq = torch.cat([ctx, q], dim=1)
        out = self.trunk(seq)
        return out[:, ctx.shape[1]:, :]         # predictions at target positions


class WormJEPA(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = Encoder(cfg)
        self.predictor = Predictor(cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)

    def forward(self, x: torch.Tensor, ctx_idx: list[int], tgt_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context_encoder(x, keep=torch.tensor(ctx_idx, device=x.device))
        pred = self.predictor(ctx, ctx_idx, tgt_idx)
        with torch.no_grad():
            full = self.target_encoder(x, keep=None)
            tgt = full[:, tgt_idx, :]
        return pred, tgt
