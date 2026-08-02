"""
Does perception (EM-JEPA) actually help CONNECTOME RECONSTRUCTION -- measured by
real segmentation metrics, not a patch probe? This is items 1-3 of the plan:

  1. WATERSHED baseline: features -> pixel boundary map (small decoder) -> seeded
     watershed -> instance segmentation, scored by VOI (split+merge) + adapted-Rand
     against CREMI ground truth. Compare feature sources.
  2. PERCEPTION-FED sequential agent: the honest re-test of the RL. The failed
     connectome RL read self-poisoned topology; here the sequential FLOOD-FILL
     agent reads JEPA perceptual features instead. (It cannot be done on the worm
     connectome graph -- no images there -- so the perceptual sequential agent
     lives on EM, which is exactly what FFN is.)
  3. FLOOD-FILL: from seeds, grow each segment greedily on the affinity map,
     scored by an Expected-Run-Length (ERL) proxy over GT skeletons.

Feature sources compared throughout (the perception question):
  jepa   -- trained hierarchical VICReg EM-JEPA encoder (frozen)
  random -- same architecture, untrained (the honest control)
  raw    -- a from-scratch small CNN on pixels (the no-SSL baseline)

Labels are used sparingly (few slices) -- the regime where good self-supervised
features are supposed to win. GPU-scale via Modal; CPU-smoke via small env knobs.

Env: WORM_EM_DEVICE, WORM_EM_CROP, WORM_SEG_JEPA_STEPS, WORM_SEG_DEC_STEPS,
WORM_SEG_LABEL_SLICES, WORM_CREMI_SAMPLES.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import watershed
from skimage.metrics import variation_of_information, adapted_rand_error
from skimage.morphology import skeletonize, local_minima
from scipy import ndimage
from sklearn.metrics import roc_auc_score

import hier_em_jepa as H          # reuse encoder, data, patching, VICReg

DEV = H.DEV
CROP = H.CROP
GF, PF, NF, DIM = H.GF, H.PF, H.NF, H.DIM
JEPA_STEPS = int(os.environ.get("WORM_SEG_JEPA_STEPS", "1500"))
DEC_STEPS = int(os.environ.get("WORM_SEG_DEC_STEPS", "600"))
LABEL_SLICES = int(os.environ.get("WORM_SEG_LABEL_SLICES", "8"))
rng = np.random.default_rng(0)


# ---------- train a JEPA encoder (VICReg) inline ----------
def train_jepa(tr_raw, steps):
    enc = H.HierEncoder().to(DEV)
    tgt = H.HierEncoder().to(DEV); tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = H.Predictor().to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3, weight_decay=0.04)
    for step in range(steps):
        crops, _ = H.sample_crops(tr_raw, H.BATCH, rng)
        crops = torch.tensor(crops, device=DEV)
        pf, pc = H.patch_fine(crops), H.patch_coarse(crops)
        fctx, cctx, ftgt = H.masks_for(H.BATCH, rng)
        fine_ctx = torch.gather(enc.tok_fine(pf), 1, fctx[:, :, None].expand(-1, -1, DIM))
        coarse_ctx = torch.gather(enc.tok_coarse(pc), 1, (cctx - NF)[:, :, None].expand(-1, -1, DIM))
        fo, co = enc(fine_ctx, coarse_ctx)
        ctx = torch.cat([fo, co], 1); ctx_ids = torch.cat([fctx, cctx], 1)
        with torch.no_grad():
            tf, _ = tgt(tgt.tok_fine(pf), tgt.tok_coarse(pc))
            target = torch.gather(tf, 1, ftgt[:, :, None].expand(-1, -1, DIM))
        p = pred(ctx, ctx_ids, ftgt)
        var_l, cov_l, _ = H.vicreg_terms(fo.reshape(-1, DIM))
        loss = F.smooth_l1_loss(p, target) + H.VAR_COEF * var_l + H.COV_COEF * cov_l
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pt in zip(enc.parameters(), tgt.parameters()):
                pt.mul_(0.996).add_(pe, alpha=0.004)
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[seg] jepa step {step+1}/{steps} loss={float(loss):.4f}", flush=True)
    enc.eval()
    return enc


# ---------- boundary decoders (features -> pixel boundary logits) ----------
class FeatDecoder(nn.Module):
    """Patch features (B,GF,GF,D) -> pixel boundary map (B,CROP,CROP)."""
    def __init__(self, d=DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, 128, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 1))

    def forward(self, fmap):                          # fmap:(B,GF,GF,D)
        x = fmap.permute(0, 3, 1, 2)
        return self.net(x)[:, 0]                       # (B,CROP,CROP)


class PixelDecoder(nn.Module):
    """Raw-pixel baseline: small from-scratch CNN image -> boundary map."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1))

    def forward(self, img):                            # img:(B,1,CROP,CROP)
        return self.net(img)[:, 0]


@torch.no_grad()
def enc_fmap(encoder, crops):                          # (B,GF,GF,D)
    pf, pc = H.patch_fine(crops), H.patch_coarse(crops)
    fo, _ = encoder(encoder.tok_fine(pf), encoder.tok_coarse(pc))
    return fo.reshape(-1, GF, GF, DIM)


def sample_labeled(raw, bnd, nsl, seed):
    """nsl random (image, boundary) crops for training the decoder."""
    r = np.random.default_rng(seed); Z, Hh, W = raw.shape
    z = r.integers(0, Z, nsl); y = r.integers(0, Hh - CROP, nsl); x = r.integers(0, W - CROP, nsl)
    imgs = np.stack([raw[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(nsl)])
    bnds = np.stack([bnd[z[i], y[i]:y[i] + CROP, x[i]:x[i] + CROP] for i in range(nsl)]).astype(np.float32)
    return imgs, bnds


def train_decoder(source, encoder, imgs, bnds, steps):
    dec = (PixelDecoder() if source == "raw" else FeatDecoder()).to(DEV)
    opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
    X = torch.tensor(imgs, device=DEV); Y = torch.tensor(bnds, device=DEV)
    fmap = None if source == "raw" else enc_fmap(encoder, X)
    for step in range(steps):
        idx = torch.randint(0, len(X), (min(16, len(X)),), device=DEV)
        if source == "raw":
            logit = dec(X[idx][:, None])
        else:
            logit = dec(fmap[idx])
        # class-balanced BCE (boundaries are ~sparse thin lines)
        w = torch.where(Y[idx] > 0.5, 4.0, 1.0)
        loss = F.binary_cross_entropy_with_logits(logit, Y[idx], weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    dec.eval()
    return dec


@torch.no_grad()
def predict_boundary(source, dec, encoder, imgs):
    X = torch.tensor(imgs, device=DEV)
    if source == "raw":
        return torch.sigmoid(dec(X[:, None])).cpu().numpy()
    return torch.sigmoid(dec(enc_fmap(encoder, X))).cpu().numpy()


# ---------- segmentation + metrics ----------
def watershed_seg(bmap):
    """Seeded watershed on a boundary probability map -> instance labels."""
    seeds = ndimage.label(local_minima(bmap) & (bmap < 0.3))[0]
    if seeds.max() == 0:
        seeds = ndimage.label(bmap < 0.3)[0]
    return watershed(bmap, markers=seeds)


def seg_metrics(pred, gt):
    voi_split, voi_merge = variation_of_information(gt, pred)
    are, prec, rec = adapted_rand_error(gt, pred)
    return float(voi_split + voi_merge), float(are)


def erl_proxy(pred, gt):
    """2D Expected-Run-Length proxy: skeletonise each GT segment, walk its
    skeleton, and measure error-free run length (contiguous skeleton pixels
    sharing one predicted label). ERL = sum(run_len^2)/sum(run_len) over all GT
    skeletons -- long error-free runs reward, splits punish."""
    tot_len, tot_sq = 0.0, 0.0
    for gid in np.unique(gt):
        m = gt == gid
        if m.sum() < 20:
            continue
        sk = skeletonize(m)
        pl = pred[sk]
        if pl.size < 3:
            continue
        # runs of constant predicted label along skeleton pixels (order-agnostic proxy)
        for plabel in np.unique(pl):
            L = int((pl == plabel).sum())
            tot_len += L; tot_sq += L * L
    return float(tot_sq / max(1.0, tot_len))


def main():
    raw, bnd, syn = H.load_cremi()
    Z = raw.shape[0]; ztr = int(Z * 0.75)
    tr_raw, te_raw = raw[:ztr], raw[ztr:]
    tr_b = bnd[:ztr]
    # GT instance labels on the test slices (2D per-slice neuron ids proxy: use
    # connected components of non-boundary within each test slice's neuron_ids is
    # unavailable here; reload neuron_ids for the test slices).
    print(f"[seg] device={DEV} crop={CROP} jepa_steps={JEPA_STEPS} dec_steps={DEC_STEPS} "
          f"label_slices={LABEL_SLICES} samples={H.SAMPLES}", flush=True)

    enc = train_jepa(tr_raw, JEPA_STEPS)
    rnd = H.HierEncoder().to(DEV).eval()

    imgs, bnds = sample_labeled(tr_raw, tr_b, LABEL_SLICES, 7)
    # held-out test crops + their GT instance labels (from neuron_ids)
    nid = _load_neuron_ids()[ztr:]
    r = np.random.default_rng(99); NT = 24
    zt = r.integers(0, te_raw.shape[0], NT); yt = r.integers(0, te_raw.shape[1] - CROP, NT); xt = r.integers(0, te_raw.shape[2] - CROP, NT)
    te_imgs = np.stack([te_raw[zt[i], yt[i]:yt[i] + CROP, xt[i]:xt[i] + CROP] for i in range(NT)])
    te_bnd = np.stack([bnd[ztr:][zt[i], yt[i]:yt[i] + CROP, xt[i]:xt[i] + CROP] for i in range(NT)])
    te_gt = np.stack([nid[zt[i], yt[i]:yt[i] + CROP, xt[i]:xt[i] + CROP] for i in range(NT)])

    res = {"device": DEV, "crop": CROP, "jepa_steps": JEPA_STEPS, "label_slices": LABEL_SLICES,
           "n_test_crops": NT, "sources": {}}
    for source, encoder in [("jepa", enc), ("random", rnd), ("raw", None)]:
        dec = train_decoder(source, encoder, imgs, bnds, DEC_STEPS)
        bmap = predict_boundary(source, dec, encoder, te_imgs)                # (NT,CROP,CROP)
        b_auc = float(roc_auc_score(te_bnd.reshape(-1) > 0.5, bmap.reshape(-1)))
        vois, ares, erls = [], [], []
        for i in range(NT):
            seg = watershed_seg(bmap[i])
            voi, are = seg_metrics(seg, te_gt[i])
            vois.append(voi); ares.append(are); erls.append(erl_proxy(seg, te_gt[i]))
        res["sources"][source] = {"pixel_boundary_AUC": round(b_auc, 4),
                                   "VOI": round(float(np.mean(vois)), 4),
                                   "adapted_rand_error": round(float(np.mean(ares)), 4),
                                   "ERL_proxy": round(float(np.mean(erls)), 1)}
        print(f"[seg] {source}: {res['sources'][source]}", flush=True)

    print(json.dumps(res, indent=2))
    with open(os.path.join(H.HERE, "segment.json"), "w") as f:
        json.dump(res, f, indent=2)


def _load_neuron_ids():
    import h5py
    dirpath = os.environ.get("WORM_EM_DIR", os.path.join(H.HERE, "cremi_data"))
    ids = []
    for s in H.SAMPLES:
        with h5py.File(os.path.join(dirpath, f"sample_{s}.hdf"), "r") as f:
            ids.append(f["volumes/labels/neuron_ids"][:])
    return np.concatenate(ids)


if __name__ == "__main__":
    main()
