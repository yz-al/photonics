"""
Optics-native neural-network primitives for Test 2.

The physically honest model of a coherent optical layer is:

    z_in  (complex field amplitude)  --W (complex linear)-->  Wz  --|.|^2 (photodetection)-->  intensity >= 0

Everything here is standard PyTorch so the same modules run on CPU or GPU.

Provides:
  * activation registry: relu, gelu, square (x^2), modsq (|.|^2), abs (|x|),
    scaled_quad (learnable a x^2 + b x + c)
  * ActQuant: straight-through activation quantiser (b-bit)
  * ComplexLinear + ModulusSquared: the coherent-optics linear+detect block
  * ShotNoise: detector shot + thermal noise as an activation-level term, SNR-swept
  * NonNegSplit: differential (two-channel) encoding of signed values as intensities
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scalar activations
# ---------------------------------------------------------------------------
class Square(nn.Module):
    def forward(self, x):
        return x * x


class ModSq(nn.Module):
    """|x|^2 for real input == x^2, but named separately: this is the
    photodetection nonlinearity, the free square-law optics gives you."""
    def forward(self, x):
        return x * x


class AbsAct(nn.Module):
    def forward(self, x):
        return x.abs()


class ScaledQuad(nn.Module):
    """Per-feature learnable quadratic a*x^2 + b*x + c.

    Initialised near-linear (a small, b=1) so it can *learn* how much curvature
    to use -- a fair test of whether a trainable quadratic competes with ReLU."""
    def __init__(self, num_features=None, a0=0.1, b0=1.0):
        super().__init__()
        shape = (num_features,) if num_features else (1,)
        self.a = nn.Parameter(torch.full(shape, float(a0)))
        self.b = nn.Parameter(torch.full(shape, float(b0)))
        self.c = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        # broadcast over the feature dim (last dim for MLP/transformer, channel for CNN)
        if x.dim() == 4 and self.a.numel() > 1:      # NCHW
            a = self.a.view(1, -1, 1, 1); b = self.b.view(1, -1, 1, 1); c = self.c.view(1, -1, 1, 1)
        elif self.a.numel() > 1:
            a, b, c = self.a, self.b, self.c
        else:
            a, b, c = self.a, self.b, self.c
        return a * x * x + b * x + c


def make_activation(name, num_features=None):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("square", "x2"):
        return Square()
    if name in ("modsq", "abs2", "|x|2"):
        return ModSq()
    if name == "abs":
        return AbsAct()
    if name in ("scaled_quad", "squad"):
        return ScaledQuad(num_features)
    raise ValueError(f"unknown activation {name}")


NONNEG_ACTS = {"relu", "square", "x2", "modsq", "abs2", "|x|2", "abs"}


# ---------------------------------------------------------------------------
# Activation quantisation (straight-through)
# ---------------------------------------------------------------------------
class _QuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, levels, lo, hi):
        xc = torch.clamp(x, lo, hi)
        scale = (hi - lo) / (levels - 1)
        q = torch.round((xc - lo) / scale) * scale + lo
        return q

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class ActQuant(nn.Module):
    """b-bit activation quantiser with a running (calibrated) range.

    Uses an EMA of observed [lo, hi] during training; straight-through gradient.
    Set bits=None to disable (identity)."""
    def __init__(self, bits=8, signed=False, momentum=0.99):
        super().__init__()
        self.bits = bits
        self.signed = signed
        self.momentum = momentum
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(1.0))
        self.register_buffer("init", torch.tensor(0.0))

    def forward(self, x):
        if self.bits is None:
            return x
        if self.training:
            with torch.no_grad():
                lo = x.min() if self.signed else torch.zeros((), device=x.device)
                hi = x.max()
                if self.init.item() == 0:
                    self.lo.copy_(lo); self.hi.copy_(hi); self.init.fill_(1.0)
                else:
                    m = self.momentum
                    self.lo.mul_(m).add_((1 - m) * lo)
                    self.hi.mul_(m).add_((1 - m) * hi)
        levels = 2 ** self.bits
        hi = torch.maximum(self.hi, self.lo + 1e-6)
        return _QuantSTE.apply(x, levels, self.lo, hi)


# ---------------------------------------------------------------------------
# Coherent optics: complex linear + modulus-squared detection
# ---------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    """y = W z with complex W and complex z (returned complex).

    Parameter count = 2 * in * out reals (real+imag), so to match a real
    Linear(in,out) at equal parameter count, halve the width elsewhere."""
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        k = 1.0 / math.sqrt(in_f)
        self.wr = nn.Parameter((torch.rand(out_f, in_f) * 2 - 1) * k)
        self.wi = nn.Parameter((torch.rand(out_f, in_f) * 2 - 1) * k)
        if bias:
            self.br = nn.Parameter(torch.zeros(out_f))
            self.bi = nn.Parameter(torch.zeros(out_f))
        else:
            self.br = self.bi = None

    def forward(self, zr, zi):
        # (wr+ i wi)(zr + i zi) = (wr zr - wi zi) + i(wr zi + wi zr)
        yr = F.linear(zr, self.wr) - F.linear(zi, self.wi)
        yi = F.linear(zr, self.wi) + F.linear(zi, self.wr)
        if self.br is not None:
            yr = yr + self.br; yi = yi + self.bi
        return yr, yi


class ModulusSquared(nn.Module):
    """|y|^2 = yr^2 + yi^2  -- the photodetector. Output intensity >= 0."""
    def forward(self, yr, yi):
        return yr * yr + yi * yi


# ---------------------------------------------------------------------------
# Detector noise (shot + thermal) as an activation-level term
# ---------------------------------------------------------------------------
class ShotNoise(nn.Module):
    """Add detector noise to a non-negative intensity signal at a target SNR.

    Photon-count / shot model: for a signal proportional to intensity I with an
    equivalent photon budget n_ph at full scale, shot noise std ~ sqrt(I/full *
    n_ph)/n_ph in normalised units, i.e. relative noise ~ 1/sqrt(signal photons).
    We parameterise directly by `snr_db` (electrical power SNR at full scale) and
    add zero-mean Gaussian noise with std set so full-scale SNR matches; the noise
    scales as sqrt(signal) (shot-like). Set snr_db=None to disable."""
    def __init__(self, snr_db=None):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x):
        if self.snr_db is None or not self.training and self.snr_db is None:
            return x
        if self.snr_db is None:
            return x
        with torch.no_grad():
            fs = x.detach().abs().amax().clamp_min(1e-6)
        snr_lin = 10.0 ** (self.snr_db / 10.0)             # power SNR at full scale
        # shot: variance proportional to signal; std = sqrt(x/fs) * (fs/sqrt(snr))
        sigma = torch.sqrt(torch.clamp(x, min=0.0) / fs) * (fs / math.sqrt(snr_lin))
        return x + torch.randn_like(x) * sigma


class NonNegSplit(nn.Module):
    """Differential encoding of a signed vector as two non-negative channels:
    x -> [relu(x), relu(-x)] concatenated. Doubles the feature width (the cost
    of representing signed values with intensity-only optics)."""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=self.dim)
