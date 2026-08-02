"""
EM-JEPA: a JEPA (I-JEPA / V-JEPA-style) self-supervised model on real electron
microscopy of neural tissue -- the PERCEPTION-first direction.

Why this exists: every topology-only and activity-only route we tried on the worm
caps out (link-prediction AUC ~0.58, effective-connectivity struct_corr ~0.33),
because the signal that actually distinguishes wiring lives in the IMAGE (the EM
pixels), not in the graph or the activity trace. ConnectomeBench2 makes the same
point: a purpose-trained vision model on EM hits human-level where prompted LLMs
and graph methods do not. So we bring perception in the self-supervised way: a
JEPA that predicts masked-region representations of EM images in latent space, with
NO labels. Then we test whether the learned features carry perceptual structure by
linear-probing a held-out task (mitochondria presence) the model never saw.

Data: EPFL CVLab hippocampus EM (5x5x5um of CA1), 165x768x1024 uint8 volumes with
binary mitochondria masks. Public, direct download (see EM_URLS).

Pipeline:
  1. SSL pretrain: I-JEPA. Context encoder (ViT) sees a subset of patches; target
     encoder (EMA) encodes the full image; a predictor reconstructs target-block
     representations from context + positional mask tokens. Loss = latent MSE.
  2. Probe: freeze the context encoder, extract per-patch features on held-out
     images, logistic-regress patch mito-label. Compare vs raw-pixel features and
     a RANDOM-INIT encoder (the honest "did SSL actually learn perception" test).

Env knobs (small defaults for CPU smoke; scale up on GPU):
  WORM_EM_DIR       cached tif directory (downloaded if missing)
  WORM_EM_DEVICE    cpu / cuda
  WORM_EM_CROP      crop size (default 112)
  WORM_EM_DIM       embed dim (default 128)
  WORM_EM_DEPTH     encoder transformer blocks (default 4)
  WORM_EM_STEPS     SSL training steps (default 300)
  WORM_EM_BATCH     images per step (default 32)
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
CROP = int(os.environ.get("WORM_EM_CROP", "112"))
PATCH = 16
NP1 = CROP // PATCH                       # patches per side
NPAT = NP1 * NP1
DIM = int(os.environ.get("WORM_EM_DIM", "128"))
DEPTH = int(os.environ.get("WORM_EM_DEPTH", "4"))
STEPS = int(os.environ.get("WORM_EM_STEPS", "300"))
BATCH = int(os.environ.get("WORM_EM_BATCH", "32"))

_BASE = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus"
EM_URLS = {f: f"{_BASE}/{f}.tif" for f in
           ["training", "training_groundtruth", "testing", "testing_groundtruth"]}


def load_em(dirpath=None):
    import tifffile
    dirpath = dirpath or os.environ.get("WORM_EM_DIR", os.path.join(HERE, "em_data"))
    os.makedirs(dirpath, exist_ok=True)
    out = {}
    for name, url in EM_URLS.items():
        p = os.path.join(dirpath, name + ".tif")
        if not os.path.exists(p):
            print(f"[em] downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, p)
        out[name] = tifffile.imread(p)
    return (out["training"].astype(np.float32) / 255.0, out["training_groundtruth"] > 127,
            out["testing"].astype(np.float32) / 255.0, out["testing_groundtruth"] > 127)


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


class Encoder(nn.Module):
    """ViT over EM patches. Patch embed + learned pos emb + transformer blocks."""
    def __init__(self, d=DIM, depth=DEPTH):
        super().__init__()
        self.proj = nn.Linear(PATCH * PATCH, d)
        self.pos = nn.Parameter(torch.zeros(1, NPAT, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def embed(self, patches):                       # (B,NPAT,PATCH*PATCH) -> tokens+pos
        return self.proj(patches) + self.pos

    def forward(self, patches, keep=None):
        x = self.embed(patches)
        if keep is not None:                        # keep: (B,K) indices of context patches
            x = torch.gather(x, 1, keep[:, :, None].expand(-1, -1, x.shape[-1]))
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class Predictor(nn.Module):
    """Predict target-block target-encoder reps from context tokens + mask queries."""
    def __init__(self, d=DIM, depth=2):
        super().__init__()
        self.mask = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.mask, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, NPAT, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, ctx, ctx_idx, tgt_idx):
        B, d = ctx.shape[0], ctx.shape[2]
        ctx = ctx + torch.gather(self.pos.expand(B, -1, -1), 1,
                                 ctx_idx[:, :, None].expand(-1, -1, d))
        q = self.mask.expand(B, tgt_idx.shape[1], d) + \
            torch.gather(self.pos.expand(B, -1, -1), 1, tgt_idx[:, :, None].expand(-1, -1, d))
        x = torch.cat([ctx, q], 1)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x[:, ctx.shape[1]:]                   # predictions at target positions


def to_patches(imgs):                                # (B,CROP,CROP) -> (B,NPAT,PATCH*PATCH)
    B = imgs.shape[0]
    x = imgs.reshape(B, NP1, PATCH, NP1, PATCH).permute(0, 1, 3, 2, 4).reshape(B, NPAT, PATCH * PATCH)
    return x


def sample_crops(vol, n, rng):
    Z, H, W = vol.shape
    z = rng.integers(0, Z, n); y = rng.integers(0, H - CROP, n); x = rng.integers(0, W - CROP, n)
    return np.stack([vol[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)])


def mask_indices(B, rng):
    """One contiguous target block (~1/3 of patches); context = the rest."""
    ntg = max(1, NPAT // 3)
    tgt, ctx = [], []
    for _ in range(B):
        bw = max(1, int(round(np.sqrt(ntg))))
        r0 = rng.integers(0, max(1, NP1 - bw + 1)); c0 = rng.integers(0, max(1, NP1 - bw + 1))
        tset = {(r0 + dr) * NP1 + (c0 + dc) for dr in range(bw) for dc in range(bw)}
        tgt.append(sorted(tset)); ctx.append([i for i in range(NPAT) if i not in tset])
    K = min(len(c) for c in ctx); T = min(len(t) for t in tgt)
    ctx = np.array([c[:K] for c in ctx]); tgt = np.array([t[:T] for t in tgt])
    return torch.tensor(ctx, device=DEV), torch.tensor(tgt, device=DEV)


def main():
    rng = np.random.default_rng(0)
    tr_img, tr_lab, te_img, te_lab = load_em()
    print(f"[em] device={DEV} crop={CROP} patches={NPAT} dim={DIM} depth={DEPTH} "
          f"steps={STEPS} batch={BATCH}", flush=True)

    enc = Encoder().to(DEV)
    tgt_enc = Encoder().to(DEV); tgt_enc.load_state_dict(enc.state_dict())
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    pred = Predictor().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3, weight_decay=0.04)

    losses = []
    for step in range(STEPS):
        crops = torch.tensor(sample_crops(tr_img, BATCH, rng), device=DEV)
        patches = to_patches(crops)
        ctx_idx, tgt_idx = mask_indices(BATCH, rng)
        with torch.no_grad():
            full = tgt_enc(patches)                                  # (B,NPAT,d) target reps
            target = torch.gather(full, 1, tgt_idx[:, :, None].expand(-1, -1, full.shape[-1]))
        ctx = enc(patches, keep=ctx_idx)
        p = pred(ctx, ctx_idx, tgt_idx)
        loss = F.smooth_l1_loss(p, target)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                        # EMA target
            m = 0.996
            for pe, pt in zip(enc.parameters(), tgt_enc.parameters()):
                pt.mul_(m).add_(pe, alpha=1 - m)
        losses.append(float(loss))
        if (step + 1) % max(1, STEPS // 10) == 0:
            print(f"[em] step {step+1}/{STEPS} loss={np.mean(losses[-STEPS//10:]):.4f}", flush=True)

    # ---------------- probe: patch-level mitochondria classification ----------------
    def build_probe(vol, lab, n, encoder, seed):
        Z, H, W = vol.shape; r = np.random.default_rng(seed)
        z = r.integers(0, Z, n); y = r.integers(0, H - CROP, n); x = r.integers(0, W - CROP, n)
        imgs = np.stack([vol[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)])
        masks = np.stack([lab[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)]).astype(np.float32)
        mp = masks.reshape(n, NP1, PATCH, NP1, PATCH).mean((2, 4)).reshape(n, NPAT)
        ylab = (mp > 0.15).astype(int).reshape(-1)
        pt = to_patches(torch.tensor(imgs, device=DEV))
        if encoder is None:
            feat = pt.reshape(n, NPAT, -1).cpu().numpy().reshape(n * NPAT, -1)      # raw pixels
        else:
            with torch.no_grad():
                feat = encoder(pt).cpu().numpy().reshape(n * NPAT, -1)
        return feat, ylab

    rnd_enc = Encoder().to(DEV)                                      # random-init control
    # same crops (fixed seeds) across all three feature types -> fair comparison
    Xtr_s, ytr = build_probe(tr_img, tr_lab, 200, enc, 7)
    Xte_s, yte = build_probe(te_img, te_lab, 120, enc, 11)
    Xtr_r, _ = build_probe(tr_img, tr_lab, 200, rnd_enc, 7)
    Xte_r, _ = build_probe(te_img, te_lab, 120, rnd_enc, 11)
    Xtr_p, _ = build_probe(tr_img, tr_lab, 200, None, 7)
    Xte_p, _ = build_probe(te_img, te_lab, 120, None, 11)

    def probe(Xtr, ytr, Xte, yte, nlab, seed):
        # label-efficiency probe: fit on only nlab labeled patches (the regime where
        # good self-supervised representations beat raw pixels, which overfit).
        r = np.random.default_rng(seed); idx = r.permutation(len(ytr))[:nlab]
        c = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr[idx], ytr[idx])
        return round(float(roc_auc_score(yte, c.predict_proba(Xte)[:, 1])), 4)

    LABELS = [20, 50, 100, 300, 1000]
    curve = {"em_jepa_features": {}, "random_encoder": {}, "raw_pixels": {}}
    for nl in LABELS:
        if nl > len(ytr):
            continue
        curve["em_jepa_features"][nl] = probe(Xtr_s, ytr, Xte_s, yte, nl, 3)
        curve["random_encoder"][nl] = probe(Xtr_r, ytr, Xte_r, yte, nl, 3)
        curve["raw_pixels"][nl] = probe(Xtr_p, ytr, Xte_p, yte, nl, 3)

    # ---------------- context-prediction probe (the JEPA-appropriate test) ----------
    # Hide a central patch block; predict ITS mito label from the encoder's
    # representation of the CONTEXT patches only. Raw context pixels cannot copy the
    # hidden region, so this rewards a model that learned to encode context
    # predictive of masked content -- exactly the JEPA pretraining objective.
    cb = max(1, NP1 // 4)
    r0 = (NP1 - cb) // 2
    center = torch.tensor([(r0 + dr) * NP1 + (r0 + dc) for dr in range(cb) for dc in range(cb)],
                          device=DEV)
    ctx_ids = torch.tensor([i for i in range(NPAT) if i not in set(center.tolist())], device=DEV)

    def ctx_probe_data(vol, lab, n, encoder, seed):
        Z, H, W = vol.shape; r = np.random.default_rng(seed)
        z = r.integers(0, Z, n); y = r.integers(0, H - CROP, n); x = r.integers(0, W - CROP, n)
        imgs = np.stack([vol[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)])
        masks = np.stack([lab[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(n)]).astype(np.float32)
        mp = masks.reshape(n, NP1, PATCH, NP1, PATCH).mean((2, 4)).reshape(n, NPAT)
        ylab = (mp[:, center.cpu().numpy()].mean(1) > 0.15).astype(int)          # hidden-block label
        pt = to_patches(torch.tensor(imgs, device=DEV))
        ci = ctx_ids[None].expand(n, -1)
        if encoder is None:                                                       # raw context pixels
            ctxp = torch.gather(pt, 1, ci[:, :, None].expand(-1, -1, pt.shape[-1]))
            feat = ctxp.mean(1).cpu().numpy()
        else:
            with torch.no_grad():
                feat = encoder(pt, keep=ci).mean(1).cpu().numpy()                 # mean-pooled ctx tokens
        return feat, ylab

    Cs_tr, cy_tr = ctx_probe_data(tr_img, tr_lab, 400, enc, 21)
    Cs_te, cy_te = ctx_probe_data(te_img, te_lab, 200, enc, 22)
    Cr_tr, _ = ctx_probe_data(tr_img, tr_lab, 400, rnd_enc, 21)
    Cr_te, _ = ctx_probe_data(te_img, te_lab, 200, rnd_enc, 22)
    Cp_tr, _ = ctx_probe_data(tr_img, tr_lab, 400, None, 21)
    Cp_te, _ = ctx_probe_data(te_img, te_lab, 200, None, 22)

    def cprobe(Xtr, y, Xte, yt):
        c = LogisticRegression(max_iter=1000).fit(Xtr, y)
        return round(float(roc_auc_score(yt, c.predict_proba(Xte)[:, 1])), 4)

    res = {"device": DEV, "crop": CROP, "patches": NPAT, "dim": DIM, "depth": DEPTH,
           "steps": STEPS, "batch": BATCH, "final_ssl_loss": round(float(np.mean(losses[-20:])), 4),
           "mito_probe_AUC_by_labelcount": curve,
           "context_prediction_probe_AUC": {
               "em_jepa_features": cprobe(Cs_tr, cy_tr, Cs_te, cy_te),
               "random_encoder": cprobe(Cr_tr, cy_tr, Cr_te, cy_te),
               "raw_context_pixels": cprobe(Cp_tr, cy_tr, Cp_te, cy_te)},
           "pos_rate": round(float(yte.mean()), 4)}
    print(json.dumps(res, indent=2))
    with open(os.path.join(HERE, "em_jepa.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
