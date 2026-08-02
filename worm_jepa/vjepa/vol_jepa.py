"""
Volumetric hierarchical EM-JEPA -- the hierarchical JEPA made genuinely 3D.

Same recipe as hier_em_jepa (two-scale, VICReg anti-collapse, fine-only auxiliary
loss, coarse-dropout) but the encoder now works on 3D EM SUBVOLUMES:
  - fine 3D patches  (fz x 16 x 16)  on grid (GZf, GF, GF)
  - coarse 3D patches (cz x 32 x 32) on grid (GZc, GC, GC), one coarse token
    aligned to a 2x2x2 block of fine tokens.
Masking picks a 3D coarse block, mapped to its aligned fine block; the predictor
reconstructs the masked fine-token target reps in latent space (EMA target).

This is what "hierarchical JEPA on 3D" means -- perception that sees across z, so
downstream 3D affinities can tie slices into continuous neurites.

Env: WORM_EM_DEVICE, WORM_VOL_CROP, WORM_VOL_ZC, WORM_VOL_DIM, WORM_VOL_DEPTH.
Anti-collapse / fine-pathway knobs are reused from hier_em_jepa (VAR/COV/FINE_AUX/
COARSE_DROP).
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hier_em_jepa as H
from hier_em_jepa import Block, vicreg_terms, VAR_COEF, COV_COEF, FINE_AUX, COARSE_DROP

DEV = H.DEV
CROP = int(os.environ.get("WORM_VOL_CROP", "224"))
ZC = int(os.environ.get("WORM_VOL_ZC", "8"))              # subvolume depth
PF, PC = 16, 32                                            # fine / coarse xy patch
FZ, CZ = 2, 4                                              # fine / coarse z patch (cz=2*fz)
GF, GC = CROP // PF, CROP // PC                            # xy grids
GZF, GZC = ZC // FZ, ZC // CZ                              # z grids
NF, NC = GZF * GF * GF, GZC * GC * GC                      # token counts
FDIM, CDIM = FZ * PF * PF, CZ * PC * PC
DIM = int(os.environ.get("WORM_VOL_DIM", "256"))
DEPTH = int(os.environ.get("WORM_VOL_DEPTH", "6"))
BATCH = int(os.environ.get("WORM_VOL_BATCH", "8"))
# Collapse fix (scale-robust): standardize the prediction TARGETS per dimension
# each step. A collapsed (near-constant) encoder cannot predict unit-variance
# targets, so collapse stops being a winning solution -- this holds even when the
# VICReg weight is too weak to prevent collapse on its own (which is what happened
# at dim-256: std fell to ~0.01). Data2vec/BYOL-style target normalization.
TGT_STD = os.environ.get("WORM_VOL_TGT_STD", "1") == "1"


# ---------------- 3D patchify ----------------
def patch_fine(vol):                                       # vol:(B,ZC,CROP,CROP)->(B,NF,FDIM)
    B = vol.shape[0]
    v = vol.reshape(B, GZF, FZ, GF, PF, GF, PF)
    return v.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, NF, FDIM)


def patch_coarse(vol):
    B = vol.shape[0]
    v = vol.reshape(B, GZC, CZ, GC, PC, GC, PC)
    return v.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, NC, CDIM)


def fine_id(z, y, x):   return (z * GF + y) * GF + x
def coarse_children(zc, yc, xc):
    return [fine_id(2 * zc + a, 2 * yc + b, 2 * xc + c)
            for a in range(2) for b in range(2) for c in range(2)]


def sample_subvols(vol, n, rng):
    Z, Hh, W = vol.shape
    z = rng.integers(0, Z - ZC, n); y = rng.integers(0, Hh - CROP, n); x = rng.integers(0, W - CROP, n)
    return np.stack([vol[z[i]:z[i] + ZC, y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)])


def masks_for(B, rng):
    """One coarse 3D block per sample -> aligned fine block. Global ids: fine
    0..NF-1, coarse NF..NF+NC-1."""
    btz = max(1, GZC // 2); bt = max(1, GC // 3)
    fctx, cctx, ftgt = [], [], []
    for _ in range(B):
        zc0 = rng.integers(0, GZC - btz + 1); r0 = rng.integers(0, GC - bt + 1); c0 = rng.integers(0, GC - bt + 1)
        ct = {((zc0 + dz) * GC + (r0 + dr)) * GC + (c0 + dc)
              for dz in range(btz) for dr in range(bt) for dc in range(bt)}
        ft = set()
        for c in ct:
            zc, rem = divmod(c, GC * GC); yc, xc = divmod(rem, GC)
            ft.update(coarse_children(zc, yc, xc))
        fctx.append([i for i in range(NF) if i not in ft])
        cctx.append([NF + i for i in range(NC) if i not in ct])
        ftgt.append(sorted(ft))
    Kf = min(len(a) for a in fctx); Kc = min(len(a) for a in cctx); T = min(len(a) for a in ftgt)
    to = lambda L, k: torch.tensor([a[:k] for a in L], device=DEV)
    return to(fctx, Kf), to(cctx, Kc), to(ftgt, T)


# ---------------- model ----------------
class VolHierEncoder(nn.Module):
    def __init__(self, d=DIM, depth=DEPTH):
        super().__init__()
        self.pf = nn.Linear(FDIM, d); self.pc = nn.Linear(CDIM, d)
        self.pos_f = nn.Parameter(torch.zeros(1, NF, d)); nn.init.trunc_normal_(self.pos_f, std=0.02)
        self.pos_c = nn.Parameter(torch.zeros(1, NC, d)); nn.init.trunc_normal_(self.pos_c, std=0.02)
        self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def tok_fine(self, pf):    return self.pf(pf) + self.pos_f
    def tok_coarse(self, pc):  return self.pc(pc) + self.pos_c

    def forward(self, fine_tok, coarse_tok):
        x = torch.cat([fine_tok, coarse_tok], 1)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x[:, :fine_tok.shape[1]], x[:, fine_tok.shape[1]:]


class VolPredictor(nn.Module):
    def __init__(self, d=DIM, depth=3):
        super().__init__()
        self.mask = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.mask, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, NF + NC, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, ctx, ctx_ids, tgt_fine_ids):
        B, d = ctx.shape[0], ctx.shape[2]
        pos = self.pos.expand(B, -1, -1)
        ctx = ctx + torch.gather(pos, 1, ctx_ids[:, :, None].expand(-1, -1, d))
        q = self.mask.expand(B, tgt_fine_ids.shape[1], d) + \
            torch.gather(pos, 1, tgt_fine_ids[:, :, None].expand(-1, -1, d))
        x = torch.cat([ctx, q], 1)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)[:, ctx.shape[1]:]


def train_vol_jepa(tr_raw, steps, rng=None):
    rng = rng or np.random.default_rng(0)
    enc = VolHierEncoder().to(DEV)
    tgt = VolHierEncoder().to(DEV); tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = VolPredictor().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3, weight_decay=0.04)
    print(f"[vol] device={DEV} crop={CROP} zc={ZC} fine={GZF}x{GF}x{GF} coarse={GZC}x{GC}x{GC} "
          f"dim={DIM} depth={DEPTH} steps={steps} tokens={NF}+{NC} "
          f"| anti-collapse: VICReg(var={VAR_COEF},cov={COV_COEF}) + EMA/predictor "
          f"+ FINE_AUX={FINE_AUX} + COARSE_DROP={COARSE_DROP}", flush=True)
    std_hist = []
    for step in range(steps):
        vols = torch.tensor(sample_subvols(tr_raw, BATCH, rng), device=DEV)
        pf, pc = patch_fine(vols), patch_coarse(vols)
        fctx, cctx, ftgt = masks_for(BATCH, rng)
        fine_ctx = torch.gather(enc.tok_fine(pf), 1, fctx[:, :, None].expand(-1, -1, DIM))
        coarse_ctx = torch.gather(enc.tok_coarse(pc), 1, (cctx - NF)[:, :, None].expand(-1, -1, DIM))
        fo, co = enc(fine_ctx, coarse_ctx)
        drop = COARSE_DROP > 0 and rng.random() < COARSE_DROP
        if drop:
            ctx, ctx_ids = fo, fctx
        else:
            ctx, ctx_ids = torch.cat([fo, co], 1), torch.cat([fctx, cctx], 1)
        with torch.no_grad():
            tf, _ = tgt(tgt.tok_fine(pf), tgt.tok_coarse(pc))
            target = torch.gather(tf, 1, ftgt[:, :, None].expand(-1, -1, DIM))
            if TGT_STD:                                   # collapse fix: unit-variance targets
                target = (target - target.mean((0, 1), keepdim=True)) / (target.std((0, 1), keepdim=True) + 1e-4)
        var_l, cov_l, on_std = vicreg_terms(fo.reshape(-1, DIM))
        loss = F.smooth_l1_loss(pred(ctx, ctx_ids, ftgt), target) + VAR_COEF * var_l + COV_COEF * cov_l
        if FINE_AUX > 0 and not drop:
            loss = loss + FINE_AUX * F.smooth_l1_loss(pred(fo, fctx, ftgt), target)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pt in zip(enc.parameters(), tgt.parameters()):
                pt.mul_(0.996).add_(pe, alpha=0.004)
        std_hist.append(on_std)
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[vol] step {step+1}/{steps} loss={float(loss.detach()):.4f} std={on_std:.3f}", flush=True)
    enc.eval()
    # collapse monitor: healthy VICReg drives embedding std toward ~1; near-zero == collapse
    enc.std_start = round(float(np.mean(std_hist[:20])), 4)
    enc.std_final = round(float(np.mean(std_hist[-20:])), 4)
    return enc


@torch.no_grad()
def feature_grid(encoder, subvol):
    """subvol:(ZC,CROP,CROP) -> fine features (DIM, GZF, GF, GF)."""
    x = torch.tensor(subvol, device=DEV)[None].float()
    fo, _ = encoder(encoder.tok_fine(patch_fine(x)), encoder.tok_coarse(patch_coarse(x)))
    return fo.reshape(GZF, GF, GF, DIM).permute(3, 0, 1, 2)
