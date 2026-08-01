"""Models for Test 2, each with a pluggable activation and optional normalisation.

  * MLP           -- tabular; configurable depth/width/norm (also the depth-vs-
                     stability probe for quadratic activations)
  * SmallCNN      -- CIFAR-10; VGG-style
  * TinyTransformer -- char-level sequence classifier (language task)
  * OpticsMLP     -- coherent optics-native: ComplexLinear + |.|^2 blocks, with
                     optional activation quantisation, shot noise, non-neg input
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_optics import (make_activation, ActQuant, ComplexLinear, ModulusSquared,
                       ShotNoise, NonNegSplit)


def norm_layer(kind, num_features, is_conv=False):
    if kind in (None, "none"):
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm2d(num_features) if is_conv else nn.BatchNorm1d(num_features)
    if kind == "layer":
        return nn.GroupNorm(1, num_features) if is_conv else nn.LayerNorm(num_features)
    raise ValueError(kind)


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, width=256, depth=3, act="relu",
                 norm="batch", act_bits=None):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(norm_layer(norm, width))
            layers.append(make_activation(act, width))
            if act_bits is not None:
                layers.append(ActQuant(act_bits, signed=False))
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.head(self.body(x))


class SmallCNN(nn.Module):
    """VGG-style: (conv-norm-act) x2 per stage, 3 stages, GAP, linear head."""
    def __init__(self, n_classes=10, act="relu", norm="batch", width=64, act_bits=None):
        super().__init__()
        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                      norm_layer(norm, cout, is_conv=True),
                      make_activation(act, cout)]
            if act_bits is not None:
                layers.append(ActQuant(act_bits, signed=False))
            return layers
        w = width
        self.features = nn.Sequential(
            *block(3, w), *block(w, w), nn.MaxPool2d(2),
            *block(w, 2 * w), *block(2 * w, 2 * w), nn.MaxPool2d(2),
            *block(2 * w, 4 * w), *block(4 * w, 4 * w), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4 * w, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class TinyTransformer(nn.Module):
    """Small char-level transformer classifier. Activation used in the FFN."""
    def __init__(self, vocab, seq_len, n_classes, d_model=128, nhead=4,
                 layers=3, act="relu", ffn_mult=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            _EncoderBlock(d_model, nhead, act, ffn_mult) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        for b in self.blocks:
            h = b(h)
        h = self.norm(h).mean(dim=1)
        return self.head(h)


class _EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, act, ffn_mult):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        hidden = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), make_activation(act, hidden),
            nn.Linear(hidden, d_model))

    def forward(self, x):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
        x = x + a
        x = x + self.ffn(self.n2(x))
        return x


# ---------------------------------------------------------------------------
# Optics-native MLP: the physically honest model of the hardware.
# ---------------------------------------------------------------------------
class OpticsMLP(nn.Module):
    """Coherent optical MLP: each layer is ComplexLinear -> |.|^2 (detection).

    Physical constraints (toggle each):
      * complex weights + |.|^2 activation  (always on; this is the hardware)
      * act_bits: quantise the detected intensity to b bits
      * snr_db: detector shot/thermal noise on the intensity
      * nonneg_input: inputs are intensities (>=0); if signed data, we differential-
        encode -> doubles input width (cost reported by param count)

    Between layers the real intensity is re-embedded as a real field (phase 0) for
    the next complex layer; this is the honest 'intensity carries to next layer'
    picture. Width halved vs real MLP so parameter count matches (complex = 2x)."""
    def __init__(self, in_dim, n_classes, width=128, depth=3, act_bits=None,
                 snr_db=None, nonneg_input=False):
        super().__init__()
        self.nonneg_input = nonneg_input
        d_in = in_dim * 2 if nonneg_input else in_dim
        self.in_norm = nn.BatchNorm1d(d_in)   # calibrate field scale -> intensity O(1)
        self.clin = nn.ModuleList()
        self.quant = nn.ModuleList()
        self.noise = nn.ModuleList()
        self.bn = nn.ModuleList()
        d = d_in
        for i in range(depth):
            self.clin.append(ComplexLinear(d, width))
            self.bn.append(nn.BatchNorm1d(width))
            self.quant.append(ActQuant(act_bits, signed=False) if act_bits else nn.Identity())
            self.noise.append(ShotNoise(snr_db))
            d = width
        self.detect = ModulusSquared()
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        if self.nonneg_input:
            x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)  # intensity-only encoding
        x = self.in_norm(x)
        zr, zi = x, torch.zeros_like(x)
        for clin, bn, q, ns in zip(self.clin, self.bn, self.quant, self.noise):
            yr, yi = clin(zr, zi)
            inten = self.detect(yr, yi)         # |.|^2 photodetection, >=0
            inten = bn(inten)                   # normalisation tames quadratic growth
            inten = torch.relu(inten)           # keep non-negative after norm (intensity)
            inten = ns(inten)                   # detector noise
            inten = q(inten)                    # b-bit ADC
            zr, zi = inten, torch.zeros_like(inten)   # re-embed intensity as real field
        return self.head(zr)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
