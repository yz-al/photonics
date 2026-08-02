"""
Mamba long-range head -- a global-context model for the affinity band SOTA fails on.

SOTA's 3D U-Net has a fixed LOCAL receptive field; 79% of its errors are long-range.
A selective state-space model (Mamba) carries a running hidden state along a scan, so
each voxel's prediction integrates evidence from across the whole slice -- global
context at LINEAR cost (vs a Transformer's O(N^2), infeasible on a flattened volume).

This is a minimal, pure-PyTorch Mamba (no CUDA-kernel dependency, so it installs and
runs anywhere): the selective SSM (input-dependent Delta, B, C -- the "S6" mechanism),
a Mamba block (conv + SSM + gate), and a 3D head that scans the y and x axes
bidirectionally (SOTA's long-range errors are dominantly in-plane) on a downsampled
grid for a tractable sequence length, then upsamples to voxel-resolution affinities.
Faithful to the architecture; not the throughput-optimized CUDA kernel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Mamba's selective scan: B, C, and step-size Delta are functions of the input,
    so the model chooses what to write to / read from the running state per position.
    h_t = exp(Delta*A) h_{t-1} + (Delta*B) x_t ; y_t = C h_t + D x_t.  O(L) scan."""
    def __init__(self, d, d_state=8):
        super().__init__()
        self.d, self.n = d, d_state
        # A is a learned, per-channel, NEGATIVE decay (stability): A = -exp(A_log)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()
                                            .repeat(d, 1)))          # (d, n)
        self.x_proj = nn.Linear(d, 2 * d_state)                      # input -> (B, C)
        self.dt_proj = nn.Linear(d, d)                              # input -> Delta (per channel)
        self.D = nn.Parameter(torch.ones(d))                        # skip term

    def forward(self, x):                                            # x: (batch, L, d)
        Bsz, L, d = x.shape
        A = -torch.exp(self.A_log)                                  # (d, n)
        BC = self.x_proj(x)                                         # (batch, L, 2n)
        Bmat, Cmat = BC[..., :self.n], BC[..., self.n:]             # (batch, L, n) each
        delta = F.softplus(self.dt_proj(x))                        # (batch, L, d)
        dA = torch.exp(delta.unsqueeze(-1) * A)                    # (batch, L, d, n)
        dBx = (delta.unsqueeze(-1) * Bmat.unsqueeze(2)) * x.unsqueeze(-1)  # (batch,L,d,n)
        h = torch.zeros(Bsz, d, self.n, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):                                          # linear-time scan
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cmat[:, t].unsqueeze(1)).sum(-1))       # (batch, d)
        return torch.stack(ys, 1) + x * self.D                     # (batch, L, d)


class MambaBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * d)
        self.conv = nn.Conv1d(d, d, 3, padding=1, groups=d)        # local detail before global SSM
        self.ssm = SelectiveSSM(d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x):                                          # (batch, L, d)
        r = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, -1)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)
        x = self.ssm(x)
        x = x * F.silu(gate)                                       # selective gate
        return r + self.out_proj(x)


class MambaLongRange3D(nn.Module):
    """raw subvolume -> affinities, via bidirectional y/x selective scans on a
    downsampled grid (global in-plane context), upsampled to voxel resolution."""
    def __init__(self, n_aff, d=48, down=(2, 8, 8), depth=2):
        super().__init__()
        self.stem = nn.Conv3d(1, d, kernel_size=down, stride=down)  # -> coarse grid (short seq)
        self.blocks = nn.ModuleList([MambaBlock(d) for _ in range(depth)])
        self.head = nn.Conv3d(d, n_aff, 1)
        self.d = d

    def _scan_y(self, g, blk):                                     # g: (B,d,Z,Y,X)
        B, d, Z, Y, X = g.shape
        s = g.permute(0, 2, 4, 3, 1).reshape(B * Z * X, Y, d)     # sequence along Y
        o = blk(s) + blk(s.flip(1)).flip(1)                       # bidirectional
        return o.reshape(B, Z, X, Y, d).permute(0, 4, 1, 3, 2)

    def _scan_x(self, g, blk):
        B, d, Z, Y, X = g.shape
        s = g.permute(0, 2, 3, 4, 1).reshape(B * Z * Y, X, d)     # sequence along X
        o = blk(s) + blk(s.flip(1)).flip(1)
        return o.reshape(B, Z, Y, X, d).permute(0, 4, 1, 2, 3)

    def forward(self, raw):                                        # (B,1,Z,H,W)
        g = self.stem(raw)                                        # (B,d,gz,gy,gx)
        for blk in self.blocks:
            g = g + self._scan_y(g, blk) + self._scan_x(g, blk)   # global in-plane context
        aff = self.head(g)                                        # (B,n_aff,gz,gy,gx)
        aff = F.interpolate(aff, size=raw.shape[2:], mode="trilinear", align_corners=False)
        return aff[0]                                             # (n_aff,Z,H,W)
