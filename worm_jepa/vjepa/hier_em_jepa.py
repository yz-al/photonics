"""
Hierarchical EM-JEPA on CREMI -- self-supervised perception scored on the tasks
that actually define the connectome: neuron BOUNDARIES (segmentation) and
SYNAPTIC CLEFTS (the connections themselves). Mitochondria are gone; those were a
connectomically-meaningless probe. Boundaries and synapses are context-dependent
(a membrane is defined by what is on either side, a synapse by the apposition of
two processes), so -- unlike mito blobs -- raw pixels and a random encoder should
NOT solve them, giving the representation-quality test real headroom.

Two upgrades over em_jepa.py:
  1. CREMI data (raw EM + neuron_ids -> boundary mask + clefts -> synapse mask).
  2. A HIERARCHICAL (two-scale) encoder: independent patch embeddings at 16px and
     32px on ALIGNED grids (one coarse token == a 2x2 fine block), mixed by joint
     self-attention (cross-scale). Masking is chosen on the coarse grid and mapped
     to the aligned fine block, so both scales are masked consistently with no
     pooling-through-holes leak. Targets are the (EMA) target-encoder's fine-token
     reps at the masked positions; the predictor reconstructs them from the
     multi-scale context + positional mask queries. Latent-space loss (JEPA), not
     pixel reconstruction (MAE) -- EM is noisy, so predicting pixels wastes capacity.

Probes (frozen features, held-out volume): patch-level boundary and synapse
classification, each vs a RANDOM-INIT encoder and RAW PIXELS, swept over label
count (the label-efficiency regime where good representations win).

Env: WORM_EM_DEVICE, WORM_EM_CROP(224), WORM_EM_DIM(256), WORM_EM_DEPTH(6),
WORM_EM_STEPS(4000), WORM_EM_BATCH(64), WORM_CREMI_SAMPLES(A or A,B,C).
"""
import os
import json
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.environ.get("WORM_EM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CROP = int(os.environ.get("WORM_EM_CROP", "224"))
PF, PC = 16, 32                                   # fine / coarse patch sizes
GF, GC = CROP // PF, CROP // PC                    # fine / coarse grid sides
NF, NC = GF * GF, GC * GC                          # token counts
DIM = int(os.environ.get("WORM_EM_DIM", "256"))
DEPTH = int(os.environ.get("WORM_EM_DEPTH", "6"))
STEPS = int(os.environ.get("WORM_EM_STEPS", "4000"))
BATCH = int(os.environ.get("WORM_EM_BATCH", "64"))
SAMPLES = os.environ.get("WORM_CREMI_SAMPLES", "A").split(",")

# Anti-collapse (LeCun's three families for JEPA SSL):
#  1. contrastive        -- explicit negatives (WORM_EM_CONTRAST > 0 turns on an
#                           InfoNCE term over the batch; off by default).
#  2. distillation       -- EMA target + stop-grad + predictor asymmetry (ALWAYS
#                           on; it is the base I-JEPA recipe here).
#  3. information-max     -- VICReg variance + covariance regularisation on the
#                           online embeddings (WORM_EM_VAR / WORM_EM_COV). This is
#                           the family LeCun's team champions and the principled
#                           fix for the near-zero-loss (partial-collapse) regime.
# The modern lineage is VICReg -> SIGReg (LeJEPA) -> VISReg; VICReg is implemented
# here as the well-established, robust default.
# Gentle defaults: a strong variance term (coef 1.0) at full training OVER-
# regularised -- it fixed the collapse metric (std 0.14->0.98) but degraded the
# representation (JEPA fell below the random encoder on the probes). The floor
# should be gentle: prevent collapse without dominating the prediction signal.
VAR_COEF = float(os.environ.get("WORM_EM_VAR", "0.2"))     # variance hinge weight
COV_COEF = float(os.environ.get("WORM_EM_COV", "0.01"))    # covariance decorrelation weight
CONTRAST = float(os.environ.get("WORM_EM_CONTRAST", "0.0"))  # optional InfoNCE weight

CREMI_URL = "https://cremi.org/static/data/sample_{}_20160501.hdf"


def vicreg_terms(z):
    """VICReg variance + covariance on embeddings z:(M,D). Variance term keeps each
    dimension's std >= 1 (prevents dimensional collapse); covariance term
    decorrelates dimensions (prevents informational collapse). Returns
    (var_loss, cov_loss, mean_std) -- mean_std is the collapse monitor."""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std))
    M, D = z.shape
    cov = (z.T @ z) / max(1, M - 1)
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    return var_loss, cov_loss, float(std.mean().detach())


def infonce(pred, target, temp=0.1):
    """Optional contrastive family: InfoNCE pulling each prediction to its own
    target and pushing off the other targets in the batch (L2-normalised)."""
    p = F.normalize(pred.mean(1), dim=-1)                  # (B,D) pooled prediction
    t = F.normalize(target.mean(1), dim=-1)
    logits = p @ t.T / temp
    labels = torch.arange(p.shape[0], device=p.device)
    return F.cross_entropy(logits, labels)


# ---------------- data ----------------
def _boundary(nid):
    """1 where a neuron_id differs from its right/down neighbour (thin membranes)."""
    b = np.zeros(nid.shape, np.uint8)
    b[:, :-1, :] |= (nid[:, :-1, :] != nid[:, 1:, :]).astype(np.uint8)
    b[:, :, :-1] |= (nid[:, :, :-1] != nid[:, :, 1:]).astype(np.uint8)
    return b


def load_cremi(dirpath=None):
    import h5py
    dirpath = dirpath or os.environ.get("WORM_EM_DIR", os.path.join(HERE, "cremi_data"))
    os.makedirs(dirpath, exist_ok=True)
    raws, bnds, syns = [], [], []
    for s in SAMPLES:
        p = os.path.join(dirpath, f"sample_{s}.hdf")
        if not os.path.exists(p):
            print(f"[cremi] downloading sample {s} ...", flush=True)
            urllib.request.urlretrieve(CREMI_URL.format(s), p)
        with h5py.File(p, "r") as f:
            raw = f["volumes/raw"][:].astype(np.float32) / 255.0
            nid = f["volumes/labels/neuron_ids"][:]
            cl = f["volumes/labels/clefts"][:]
        bnd = _boundary(nid)
        syn = (cl != cl.max()).astype(np.uint8)               # non-background cleft voxels
        raws.append(raw); bnds.append(bnd); syns.append(syn)
        del nid, cl
    raw = np.concatenate(raws); bnd = np.concatenate(bnds); syn = np.concatenate(syns)
    return raw, bnd, syn


# ---------------- model ----------------
class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.n1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class HierEncoder(nn.Module):
    """Two-scale encoder. Fine (16px) + coarse (32px) tokens, joint self-attention."""
    def __init__(self, d=DIM, depth=DEPTH):
        super().__init__()
        self.pf = nn.Linear(PF * PF, d); self.pc = nn.Linear(PC * PC, d)
        self.pos_f = nn.Parameter(torch.zeros(1, NF, d)); nn.init.trunc_normal_(self.pos_f, std=0.02)
        self.pos_c = nn.Parameter(torch.zeros(1, NC, d)); nn.init.trunc_normal_(self.pos_c, std=0.02)
        self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def tok_fine(self, pf):     return self.pf(pf) + self.pos_f
    def tok_coarse(self, pc):   return self.pc(pc) + self.pos_c

    def forward(self, fine_tok, coarse_tok):
        x = torch.cat([fine_tok, coarse_tok], 1)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x[:, :fine_tok.shape[1]], x[:, fine_tok.shape[1]:]   # fine, coarse outputs


class Predictor(nn.Module):
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


# ---------------- patchify + masking ----------------
def patch_fine(imgs):      # (B,CROP,CROP)->(B,NF,PF*PF)
    B = imgs.shape[0]
    return imgs.reshape(B, GF, PF, GF, PF).permute(0, 1, 3, 2, 4).reshape(B, NF, PF * PF)


def patch_coarse(imgs):
    B = imgs.shape[0]
    return imgs.reshape(B, GC, PC, GC, PC).permute(0, 1, 3, 2, 4).reshape(B, NC, PC * PC)


def sample_crops(vol, n, rng):
    Z, H, W = vol.shape
    z = rng.integers(0, Z, n); y = rng.integers(0, H - CROP, n); x = rng.integers(0, W - CROP, n)
    return np.stack([vol[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)]), (z, y, x)


def masks_for(B, rng):
    """Pick one coarse block per sample; map to aligned fine block. Returns global
    predictor ids: fine tokens 0..NF-1, coarse tokens NF..NF+NC-1."""
    bt = max(1, GC // 3)                                    # coarse block side
    fctx, cctx, ftgt = [], [], []
    for _ in range(B):
        cr = rng.integers(0, GC - bt + 1); cc = rng.integers(0, GC - bt + 1)
        ct = {(cr + dr) * GC + (cc + dc) for dr in range(bt) for dc in range(bt)}
        ft = set()
        for dr in range(bt):
            for dc in range(bt):
                fr, fc = 2 * (cr + dr), 2 * (cc + dc)
                for a in (0, 1):
                    for b in (0, 1):
                        ft.add((fr + a) * GF + (fc + b))
        fctx.append([i for i in range(NF) if i not in ft])
        cctx.append([NF + i for i in range(NC) if i not in ct])
        ftgt.append(sorted(ft))
    Kf = min(len(x) for x in fctx); Kc = min(len(x) for x in cctx); T = min(len(x) for x in ftgt)
    to = lambda L, k: torch.tensor([x[:k] for x in L], device=DEV)
    return to(fctx, Kf), to(cctx, Kc), to(ftgt, T)


def main():
    rng = np.random.default_rng(0)
    raw, bnd, syn = load_cremi()
    Z = raw.shape[0]; ztr = int(Z * 0.75)
    tr_raw, te_raw = raw[:ztr], raw[ztr:]
    tr_b, te_b = bnd[:ztr], bnd[ztr:]; tr_s, te_s = syn[:ztr], syn[ztr:]
    print(f"[cremi] device={DEV} vols={SAMPLES} Z={Z} crop={CROP} fine={GF}x{GF} "
          f"coarse={GC}x{GC} dim={DIM} depth={DEPTH} steps={STEPS} batch={BATCH} "
          f"bnd_frac={bnd.mean():.3f} syn_frac={syn.mean():.4f}", flush=True)

    enc = HierEncoder().to(DEV)
    tgt = HierEncoder().to(DEV); tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = Predictor().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3, weight_decay=0.04)

    losses = []; std_hist = []
    for step in range(STEPS):
        crops, _ = sample_crops(tr_raw, BATCH, rng)
        crops = torch.tensor(crops, device=DEV)
        pf, pc = patch_fine(crops), patch_coarse(crops)
        fctx_id, cctx_id, ftgt_id = masks_for(BATCH, rng)
        d = pf.shape[-1]
        fine_ctx = torch.gather(enc.tok_fine(pf), 1, fctx_id[:, :, None].expand(-1, -1, DIM))
        # coarse ctx ids are global (offset by NF); local index into coarse grid = id-NF
        cloc = cctx_id - NF
        coarse_ctx = torch.gather(enc.tok_coarse(pc), 1, cloc[:, :, None].expand(-1, -1, DIM))
        fo, co = enc(fine_ctx, coarse_ctx)
        ctx = torch.cat([fo, co], 1)                             # fine + coarse context outputs
        ctx_ids = torch.cat([fctx_id, cctx_id], 1)
        with torch.no_grad():
            tf, _ = tgt(tgt.tok_fine(pf), tgt.tok_coarse(pc))
            target = torch.gather(tf, 1, ftgt_id[:, :, None].expand(-1, -1, DIM))
        p = pred(ctx, ctx_ids, ftgt_id)
        inv = F.smooth_l1_loss(p, target)                        # invariance (prediction)
        var_loss, cov_loss, on_std = vicreg_terms(fo.reshape(-1, DIM))   # info-max reg
        loss = inv + VAR_COEF * var_loss + COV_COEF * cov_loss
        if CONTRAST > 0:
            loss = loss + CONTRAST * infonce(p, target)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            m = 0.996
            for pe, pt in zip(enc.parameters(), tgt.parameters()):
                pt.mul_(m).add_(pe, alpha=1 - m)
            tgt_std = float(torch.sqrt(target.reshape(-1, DIM).var(0) + 1e-4).mean())
        losses.append(float(inv)); std_hist.append((on_std, tgt_std))
        if (step + 1) % max(1, STEPS // 10) == 0:
            print(f"[cremi] step {step+1}/{STEPS} inv={np.mean(losses[-STEPS//10:]):.4f} "
                  f"online_std={on_std:.3f} target_std={tgt_std:.3f} "
                  f"var={float(var_loss):.3f} cov={float(cov_loss):.3f}", flush=True)

    # ---------------- probes: boundary + synapse, patch-level ----------------
    @torch.no_grad()
    def feats(encoder, crops):
        pf, pc = patch_fine(crops), patch_coarse(crops)
        if encoder is None:
            return pf.reshape(pf.shape[0], NF, -1)                  # raw fine-patch pixels
        fo, _ = encoder(encoder.tok_fine(pf), encoder.tok_coarse(pc))
        return fo                                                   # per fine-patch features

    def patch_labels(vol, lab, seed, n, thr):
        r = np.random.default_rng(seed); Z, H, W = vol.shape
        z = r.integers(0, Z, n); y = r.integers(0, H - CROP, n); x = r.integers(0, W - CROP, n)
        imgs = np.stack([vol[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)])
        lm = np.stack([lab[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)]).astype(np.float32)
        lp = lm.reshape(n, GF, PF, GF, PF).mean((2, 4)).reshape(n, NF)
        return imgs, (lp > thr).astype(int)

    def probe_set(vol, lab, n, encoder, seed, thr):
        imgs, y = patch_labels(vol, lab, seed, n, thr)
        f = feats(encoder, torch.tensor(imgs, device=DEV)).cpu().numpy().reshape(n * NF, -1)
        return f, y.reshape(-1)

    rnd = HierEncoder().to(DEV)
    LABELS = [50, 200, 1000, 5000]

    def run_probe(lab_tr, lab_te, thr, name):
        Xtr_s, ytr = probe_set(tr_raw, lab_tr, 200, enc, 31, thr)
        Xte_s, yte = probe_set(te_raw, lab_te, 120, enc, 32, thr)
        Xtr_r, _ = probe_set(tr_raw, lab_tr, 200, rnd, 31, thr)
        Xte_r, _ = probe_set(te_raw, lab_te, 120, rnd, 32, thr)
        Xtr_p, _ = probe_set(tr_raw, lab_tr, 200, None, 31, thr)
        Xte_p, _ = probe_set(te_raw, lab_te, 120, None, 32, thr)

        def fit(X, y, Xt, yt, nl):
            r = np.random.default_rng(0); idx = r.permutation(len(y))[:nl]
            if len(np.unique(y[idx])) < 2:
                return None
            c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[idx], y[idx])
            return round(float(roc_auc_score(yt, c.predict_proba(Xt)[:, 1])), 4)

        out = {"hier_em_jepa": {}, "random_encoder": {}, "raw_pixels": {}, "pos_rate": round(float(yte.mean()), 4)}
        for nl in LABELS:
            if nl > len(ytr):
                continue
            out["hier_em_jepa"][nl] = fit(Xtr_s, ytr, Xte_s, yte, nl)
            out["random_encoder"][nl] = fit(Xtr_r, ytr, Xte_r, yte, nl)
            out["raw_pixels"][nl] = fit(Xtr_p, ytr, Xte_p, yte, nl)
        print(f"[probe:{name}] {json.dumps(out)}", flush=True)
        return out

    on_s = [s[0] for s in std_hist]; tg_s = [s[1] for s in std_hist]
    res = {"device": DEV, "samples": SAMPLES, "crop": CROP, "fine_grid": GF, "coarse_grid": GC,
           "dim": DIM, "depth": DEPTH, "steps": STEPS, "batch": BATCH,
           "anticollapse": {"var_coef": VAR_COEF, "cov_coef": COV_COEF, "contrast_coef": CONTRAST},
           "final_ssl_invariance_loss": round(float(np.mean(losses[-20:])), 4),
           # collapse monitor: healthy embeddings keep std well above 0 (VICReg
           # target ~1). Near-zero std == collapse; the earlier run had no
           # variance term, so its ~0 loss was suspect. These make it verifiable.
           "embedding_std_online_final": round(float(np.mean(on_s[-20:])), 4),
           "embedding_std_target_final": round(float(np.mean(tg_s[-20:])), 4),
           "embedding_std_online_start": round(float(np.mean(on_s[:20])), 4),
           "boundary_probe_AUC_by_labelcount": run_probe(tr_b, te_b, 0.05, "boundary"),
           "synapse_probe_AUC_by_labelcount": run_probe(tr_s, te_s, 0.001, "synapse")}
    print(json.dumps(res, indent=2))
    with open(os.path.join(HERE, "hier_em_jepa.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
