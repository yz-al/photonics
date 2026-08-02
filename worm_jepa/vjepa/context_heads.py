"""
Global-context affinity heads for SOTA's long-range failures -- a fair head-to-head.

Three ways to give each voxel context beyond a U-Net's local receptive field, all
sharing the SAME decoder so the comparison is about the CONTEXT MECHANISM, not the
decoder:

  mamba       : selective state-space model, bidirectional z/y/x scans -- linear cost,
                so it can run at higher resolution (the fix for the coarse-output flaw).
  transformer : full self-attention over a downsampled token grid -- most expressive
                per layer, but O(N^2) so it must stay coarse (data-hungry in low-label).
  gnn         : message passing over the affinity graph (short + long edges) -- strong
                RELATIONAL inductive bias matched to segmentation; the cleanest teacher
                to distill a mechanistic rule from (per the symbolic-distillation lit).

Shared decoder (ContextDecoder): upsample the context-feature grid to voxel resolution
and fuse a RAW skip (fine detail the coarse grid lacks) before the affinity head. This
is what `mamba_alone` was missing -- a single trilinear upsample threw away precision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_head import MambaBlock


class ContextDecoder(nn.Module):
    """context-feature grid (d, gz, gy, gx) + raw -> voxel affinities. The raw skip
    restores the resolution the downsampled context grid can't carry."""
    def __init__(self, d, n_aff):
        super().__init__()
        self.raw_stem = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.GELU())
        self.mix = nn.Sequential(nn.Conv3d(d + 16, 32, 3, padding=1), nn.GELU(),
                                 nn.Conv3d(32, 32, 3, padding=1), nn.GELU())
        self.head = nn.Conv3d(32, n_aff, 1)

    def forward(self, g, raw):
        u = F.interpolate(g, size=raw.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([u, self.raw_stem(raw)], 1)
        return self.head(self.mix(x))[0]


class MambaContext3D(nn.Module):
    """Finer grid (down 4x in-plane, not 8x) + bidirectional z/y/x selective scans +
    the shared raw-skip decoder -- the resolution-fixed Mamba head."""
    def __init__(self, n_aff, d=48, down=(2, 4, 4), depth=2):
        super().__init__()
        self.stem = nn.Conv3d(1, d, kernel_size=down, stride=down)
        self.blocks = nn.ModuleList([MambaBlock(d) for _ in range(depth)])
        self.dec = ContextDecoder(d, n_aff)

    @staticmethod
    def _scan(g, blk, axis):                                   # bidirectional along axis in {2,3,4}
        x = g.movedim(1, -1)                                   # (B,Z,Y,X,d): channels last
        x = x.movedim(axis - 1, 3)                             # bring scan axis before channels
        B, a, b, L, d = x.shape
        s = x.reshape(B * a * b, L, d)
        o = blk(s) + blk(s.flip(1)).flip(1)                   # forward + backward
        o = o.reshape(B, a, b, L, d).movedim(3, axis - 1).movedim(-1, 1)
        return o

    def forward(self, raw):
        g = self.stem(raw)
        for blk in self.blocks:
            g = g + self._scan(g, blk, 2) + self._scan(g, blk, 3) + self._scan(g, blk, 4)
        return self.dec(g, raw)


class TransformerContext3D(nn.Module):
    """Full self-attention over a (coarser) token grid -- most expressive context per
    layer, O(N^2) so it stays coarse; shared raw-skip decoder restores resolution."""
    def __init__(self, n_aff, d=48, down=(2, 8, 8), depth=2, nhead=4, max_tokens=4096):
        super().__init__()
        self.stem = nn.Conv3d(1, d, kernel_size=down, stride=down)
        self.pos = nn.Parameter(torch.randn(1, max_tokens, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=2 * d, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.dec = ContextDecoder(d, n_aff)

    def forward(self, raw):
        g = self.stem(raw)                                     # (B,d,gz,gy,gx)
        B, d, Z, Y, X = g.shape
        tok = g.flatten(2).transpose(1, 2)                     # (B, N, d)
        N = tok.shape[1]
        y = self.tf(tok + self.pos[:, :N])                     # global attention
        g2 = y.transpose(1, 2).reshape(B, d, Z, Y, X)
        return self.dec(g2, raw)


class GNNAffinity3D(nn.Module):
    """Message passing over the affinity graph: each grid node aggregates from short-
    AND long-range neighbours (few hops -> long reach), with a learned update. Strong
    relational bias; the best teacher to distill a mechanistic long-range rule from."""
    OFFS = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0),
            (0, 0, 3), (0, 3, 0), (0, 0, -3), (0, -3, 0)]      # short + long edges (grid units)

    def __init__(self, n_aff, d=48, down=(2, 4, 4), rounds=3):
        super().__init__()
        self.stem = nn.Conv3d(1, d, kernel_size=down, stride=down)
        self.msg = nn.ModuleList([nn.Conv3d(d, d, 1) for _ in range(rounds)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Conv3d(2 * d, d, 1), nn.GELU())
                                  for _ in range(rounds)])
        self.rounds = rounds
        self.dec = ContextDecoder(d, n_aff)

    def forward(self, raw):
        g = self.stem(raw)
        for r in range(self.rounds):
            m = self.msg[r](g)
            agg = sum(torch.roll(m, shifts=(-o[0], -o[1], -o[2]), dims=(2, 3, 4))
                      for o in self.OFFS) / len(self.OFFS)     # aggregate neighbour messages
            g = g + self.upd[r](torch.cat([g, agg], 1))        # learned node update
        return self.dec(g, raw)


def make_context_head(kind, n_aff):
    return {"mamba": MambaContext3D, "transformer": TransformerContext3D,
            "gnn": GNNAffinity3D}[kind](n_aff)
