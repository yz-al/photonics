"""
Benchmark model variants: a 2-level hierarchical JEPA and a plain forecasting NN,
alongside the flat WormJEPA in model.py.

  * HierWormJEPA -- adds a coarse (slow) level on top of the flat JEPA: level-1
    tokens are mean-pooled 2x into a slow sequence, a second encoder/predictor
    operate there, and the loss is the sum of both levels. The level-2 encoder is
    where the slow behavioural-state manifold should concentrate.

  * Forecaster -- a GRU next-step predictor in *data* space (the worm-graph
    baseline). It yields an explicit dynamics function whose input Jacobian gives
    an effective neuron-neuron coupling we can score against the true connectome.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from model import JEPAConfig, Encoder, Predictor, TransformerTrunk, _sinusoidal


# --------------------------------------------------------------------------- #
# Hierarchical (2-level temporal) JEPA
# --------------------------------------------------------------------------- #
class Level2(nn.Module):
    """Pool level-1 tokens 2x, encode the coarse sequence, predict masked coarse tokens."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.n_coarse = cfg.n_patches // 2
        self.register_buffer("pos", _sinusoidal(self.n_coarse, cfg.d_model), persistent=False)
        self.encoder = TransformerTrunk(cfg.d_model, max(1, cfg.pred_depth), cfg.n_heads,
                                        cfg.mlp_ratio, cfg.dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = TransformerTrunk(cfg.d_model, max(1, cfg.pred_depth), cfg.n_heads,
                                          cfg.mlp_ratio, cfg.dropout)

    @staticmethod
    def pool(tokens: torch.Tensor) -> torch.Tensor:
        B, L, D = tokens.shape
        return tokens[:, : (L // 2) * 2, :].reshape(B, L // 2, 2, D).mean(2)

    def encode(self, l1_tokens: torch.Tensor) -> torch.Tensor:
        c = self.pool(l1_tokens) + self.pos.unsqueeze(0)
        return self.encoder(c)

    def predict(self, coarse: torch.Tensor, ctx_idx: list[int], tgt_idx: list[int]) -> torch.Tensor:
        B = coarse.shape[0]
        ctx = coarse[:, ctx_idx, :] + self.pos[ctx_idx].unsqueeze(0)
        q = self.mask_token.expand(B, len(tgt_idx), -1) + self.pos[tgt_idx].unsqueeze(0)
        out = self.predictor(torch.cat([ctx, q], dim=1))
        return out[:, len(ctx_idx):, :]


class HierWormJEPA(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = Encoder(cfg)
        self.predictor = Predictor(cfg)
        self.level2 = Level2(cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)

    def forward(self, x, ctx_idx, tgt_idx, ctx2_idx, tgt2_idx):
        # level 1 (fine) -- same as flat JEPA
        ctx = self.context_encoder(x, keep=torch.tensor(ctx_idx, device=x.device))
        pred1 = self.predictor(ctx, ctx_idx, tgt_idx)
        with torch.no_grad():
            full = self.target_encoder(x, keep=None)
            tgt1 = full[:, tgt_idx, :]
            coarse_tgt = self.level2.encode(full)          # coarse targets from EMA tokens
        # level 2 (slow)
        coarse_ctx = self.level2.encode(self.context_encoder(x, keep=None))
        pred2 = self.level2.predict(coarse_ctx, ctx2_idx, tgt2_idx)
        tgt2 = coarse_tgt[:, tgt2_idx, :]
        return (pred1, tgt1), (pred2, tgt2)

    @torch.no_grad()
    def slow_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Mean over level-2 (coarse) tokens -- the slow rep used for probing."""
        coarse = self.level2.encode(self.context_encoder(x, keep=None))
        return coarse.mean(1)


# --------------------------------------------------------------------------- #
# Forecasting NN (data-space next-step predictor)
# --------------------------------------------------------------------------- #
class Forecaster(nn.Module):
    """GRU over the window; predicts the next timestep's activity for all N neurons."""

    def __init__(self, n_neurons: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.n = n_neurons
        self.gru = nn.GRU(n_neurons, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N) -> predict x[:, 1:, :] from x[:, :-1, :]
        h, _ = self.gru(x[:, :-1, :])
        return self.head(h)                                # (B, T-1, N)

    def effective_connectivity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean |d pred_i / d input_j| at the last step -> (N, N) effective coupling.
        Compared against the true neuron-neuron Gram (W @ W.T) in the benchmark.

        Backprops through the GRU. cuDNN only supports RNN backward in *training*
        mode, so we run this block in train mode (the GRU has no dropout, so this
        doesn't change the forward) with cuDNN disabled for safety, then restore.
        """
        was_training = self.training
        self.train()
        B, T, N = x.shape
        x0 = x[:1, :-1, :].clone().requires_grad_(True)    # single sample for jacobian
        J = torch.zeros(N, N, device=x.device)
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            h, _ = self.gru(x0)
            pred = self.head(h)[:, -1, :]                  # (1, N)
            for i in range(N):
                g = torch.autograd.grad(pred[0, i], x0, retain_graph=(i < N - 1))[0]
                J[i] = g[0, -1, :].abs()                    # sensitivity to last input step
        if not was_training:
            self.eval()
        return J.detach()
