"""Track B models — edge inference-only optical architectures vs a ReLU baseline.

Three models, trained through the existing test2/ harness (experiment.train_eval):

  (a) ReLU digital baseline           -> reuse models.MLP (act="relu")
  (b) DiffractiveD2NN                  -> coherent complex propagation through
                                          learnable PHASE mask(s) (a D2NN), depth 1-2,
                                          |.|^2 intensity readout, inference-only.
  (c) SpikingMLP (surrogate-grad LIF)  -> temporally-coded, low T, reports firing
                                          sparsity; matched architecture to the ReLU
                                          baseline (LIF replaces ReLU).

All are ordinary nn.Modules so train_eval() runs them unchanged on CPU or GPU.

Physical honesty notes (flagged in the report):
  * D2NN weights are PHASE-ONLY per plane (a passive phase mask), plus a fixed
    (non-learnable) diffraction operator between planes and a small linear readout
    head on the detected intensity. A phase-only mask is inherently parameter-poor
    vs a dense ReLU layer; we report the true param counts and size the field/planes
    to bring them into the same order, but exact param-matching a phase-only optic to
    a dense MLP is unphysical. Under-parameterising the optic HURTS its accuracy, so
    this is conservative-to-optics on the accuracy axis (and the energy verdict is
    taken at matched accuracy regardless).
  * The D2NN sees a single-wavelength real-amplitude field. For CIFAR that means a
    grayscale 2D field (color is discarded) — a real limitation of a passive
    single-wavelength diffractive optic, and part of why D2NN is expected poor on CIFAR.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# (b) Diffractive optical neural net (D2NN)
# ===========================================================================
def _angular_spectrum_kernel(M, wavelength_px=None, z_px=None):
    """Fixed angular-spectrum free-space propagation transfer function on an MxM
    grid (units of pixels). H(fx,fy) = exp(i 2pi z/lambda sqrt(1 - (lambda fx)^2 -
    (lambda fy)^2)); evanescent components (sqrt arg < 0) are set to zero (they do
    not propagate). This operator is NOT learnable — it is the free-space diffraction
    between mask planes. Returns a complex MxM tensor to multiply in the Fourier plane."""
    if wavelength_px is None:
        wavelength_px = 2.0            # 2 px per wavelength (oversampled, well-sampled)
    if z_px is None:
        z_px = M * 1.0                 # propagate ~one aperture-width downstream
    fx = torch.fft.fftfreq(M).view(1, M)      # cycles / px
    fy = torch.fft.fftfreq(M).view(M, 1)
    arg = 1.0 - (wavelength_px * fx) ** 2 - (wavelength_px * fy) ** 2
    prop = (arg > 0).expand(M, M)
    argc = torch.clamp(arg, min=0.0).expand(M, M)
    phase = (2 * math.pi * z_px / wavelength_px) * torch.sqrt(argc)
    H = torch.exp(1j * phase)
    H = H * prop.to(H.dtype)           # zero the evanescent band
    return H


class DiffractiveLayer(nn.Module):
    """One D2NN plane: free-space diffraction (fixed) then a learnable phase mask
    exp(i*phi) applied element-wise. phi is the only trainable parameter (M*M reals)."""
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.register_buffer("H", _angular_spectrum_kernel(M))
        self.phi = nn.Parameter(torch.zeros(M, M))     # phase mask, init transparent

    def forward(self, field):                          # field: (B, M, M) complex
        F_ = torch.fft.fft2(field)
        field = torch.fft.ifft2(F_ * self.H)           # diffraction
        field = field * torch.exp(1j * self.phi)       # learnable phase modulation
        return field


class DiffractiveD2NN(nn.Module):
    """Coherent diffractive optical net. Input (real, >=0 intensity->amplitude) is
    embedded on an MxM field; light diffracts through `depth` learnable phase masks;
    the detector reads |field|^2; a small linear head maps detected intensity ->
    classes (the standard 'class detector regions + readout' picture).

    in_side: side length of the (square) input image on the grid. For non-square /
    multi-channel inputs the caller passes a pre-flattened vector and we reshape to
    the nearest square, padding with zeros (physically: the image illuminates the
    centre of a larger aperture)."""
    def __init__(self, in_dim, n_classes, M=None, depth=2, head=True, in_side=None):
        super().__init__()
        if in_side is None:
            in_side = int(round(math.sqrt(in_dim)))
        self.in_side = in_side
        self.in_dim = in_dim
        if M is None:
            M = in_side                                # grid = image size (no oversize)
        self.M = M
        self.depth = depth
        self.pad = (M - in_side)
        self.in_norm = nn.BatchNorm1d(in_dim)          # calibrate input field scale
        self.layers = nn.ModuleList([DiffractiveLayer(M) for _ in range(depth)])
        self.head = nn.Linear(M * M, n_classes) if head else None
        self.n_classes = n_classes

    def _embed(self, x):
        B = x.size(0)
        # use exactly in_side*in_side of the (possibly larger) flattened input
        n = self.in_side * self.in_side
        if x.size(1) >= n:
            x = x[:, :n]
        else:
            x = F.pad(x, (0, n - x.size(1)))
        img = x.view(B, self.in_side, self.in_side)
        if self.pad > 0:
            p0 = self.pad // 2; p1 = self.pad - p0
            img = F.pad(img, (p0, p1, p0, p1))
        # amplitude encoding: real, non-negative field amplitude
        amp = img
        return amp.to(torch.complex64)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.in_norm(x)
        field = self._embed(x)
        for lyr in self.layers:
            field = lyr(field)
        inten = (field.real ** 2 + field.imag ** 2)    # |.|^2 photodetection
        inten = inten.flatten(1)
        if self.head is not None:
            return self.head(inten)
        # headless: pool into n_classes equal detector regions (fixed, no params)
        M2 = inten.size(1)
        reg = M2 // self.n_classes
        return inten[:, :reg * self.n_classes].view(inten.size(0), self.n_classes, reg).sum(-1)


# ===========================================================================
# (c) Spiking MLP — surrogate-gradient LIF
# ===========================================================================
class _ATanSurrogate(torch.autograd.Function):
    """Heaviside forward; arctan surrogate gradient (Fang et al., SpikingJelly).
    grad = alpha/2 / (1 + (pi/2 * alpha * u)^2)."""
    alpha = 2.0

    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u >= 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        a = _ATanSurrogate.alpha
        sg = (a / 2.0) / (1.0 + (math.pi / 2.0 * a * u) ** 2)
        return g * sg


def spike_fn(u):
    return _ATanSurrogate.apply(u)


class LIFCell(nn.Module):
    """Leaky integrate-and-fire cell with soft reset and surrogate gradient.
    Tracks a running mean firing rate (spikes/neuron/step) over the last forward,
    for sparsity reporting."""
    def __init__(self, beta=0.9, threshold=1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.last_rate = None

    def forward(self, x_seq):                          # x_seq: (T, B, N) input current
        T = x_seq.size(0)
        u = torch.zeros_like(x_seq[0])
        spikes = []
        for t in range(T):
            u = self.beta * u + x_seq[t]
            s = spike_fn(u - self.threshold)
            u = u - s * self.threshold                 # soft reset
            spikes.append(s)
        out = torch.stack(spikes, 0)                   # (T, B, N)
        self.last_rate = out.detach().mean().item()
        return out


class SpikingMLP(nn.Module):
    """Temporally-coded LIF MLP, matched in architecture (and hence parameter count)
    to the ReLU baseline MLP: Linear -> LIF, `depth` times, then a Linear readout that
    accumulates output membrane current over T (rate readout). Static input is applied
    as a constant current every timestep (direct / constant encoding — the low-latency
    regime, T~4-16). Reports mean firing sparsity across all hidden LIF layers."""
    def __init__(self, in_dim, n_classes, width=256, depth=2, T=8, beta=0.9,
                 threshold=1.0):
        super().__init__()
        self.T = T
        self.fc = nn.ModuleList()
        self.lif = nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            self.fc.append(nn.Linear(d, width))
            self.lif.append(LIFCell(beta, threshold))
            d = width
        self.head = nn.Linear(d, n_classes)
        self.depth = depth

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        B = x.size(0)
        # constant-current (direct) input encoding over T steps
        cur = x
        h = cur.unsqueeze(0).expand(self.T, B, x.size(1))   # (T,B,in)
        for fc, lif in zip(self.fc, self.lif):
            # apply linear per timestep (weight shared across time)
            z = fc(h)                                        # (T,B,width)
            h = lif(z)                                       # (T,B,width) spikes
        out = self.head(h)                                   # (T,B,C) readout current
        return out.mean(0)                                   # rate readout over time

    def firing_rate(self):
        rates = [l.last_rate for l in self.lif if l.last_rate is not None]
        return float(sum(rates) / len(rates)) if rates else float("nan")


def count_params(m):
    return sum(p.numel() for p in m.parameters())
