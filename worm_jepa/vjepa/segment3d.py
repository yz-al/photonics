"""
3D affinities + mutex-watershed on the volumetric hierarchical EM-JEPA -- the two
highest-leverage moves toward CREMI SOTA, on a genuinely 3D perception backbone.

Backbone: vol_jepa (hierarchical JEPA on 3D subvolumes) -> per-fine-token 3D
feature grid (DIM, GZF, GF, GF).
Head: a 3D decoder upsamples that grid to voxel-resolution affinities -- SHORT-range
attractive (z, y, x) + LONG-range repulsive -- then MUTEX WATERSHED (parameter-free
learned agglomeration) turns the signed affinities into a 3D instance segmentation.
Scored by 3D VOI + adapted-Rand + ERL proxy, comparing the volumetric JEPA encoder
vs a random-init one vs a raw-pixel 3D CNN.

Env: WORM_EM_DEVICE, WORM_VOL_* (backbone), WORM_SEG_JEPA_STEPS, WORM_SEG_DEC_STEPS,
WORM_S3_EVAL_CROP, WORM_S3_NEVAL, WORM_CREMI_SAMPLES.
"""
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import variation_of_information, adapted_rand_error
from skimage.morphology import skeletonize

import hier_em_jepa as H
import vol_jepa as V

DEV = V.DEV
DIM, GZF, GF, ZC, CROP, FZ = V.DIM, V.GZF, V.GF, V.ZC, V.CROP, V.FZ
JEPA_STEPS = int(os.environ.get("WORM_SEG_JEPA_STEPS", "3000"))
DEC_STEPS = int(os.environ.get("WORM_SEG_DEC_STEPS", "600"))
EVAL_CROP = int(os.environ.get("WORM_S3_EVAL_CROP", "160"))
NEVAL = int(os.environ.get("WORM_S3_NEVAL", "4"))
rng = np.random.default_rng(0)

SHORT = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]                 # attractive (z,y,x)
LONG = [(0, 9, 0), (0, 0, 9), (2, 0, 0)]                  # repulsive (long-range)
OFFS = SHORT + LONG
NAFF = len(OFFS)


class FeatAff3D(nn.Module):
    """U-Net-style: coarse JEPA feature grid (context) upsampled to voxel res, then
    concatenated with the raw subvolume (fine detail) before the affinity head. This
    is how a pretrained backbone is actually used for dense prediction -- features
    supply learned context, the raw skip supplies resolution the 16px patches lack.
    The comparison then asks: does JEPA context add value ON TOP of raw pixels?"""
    def __init__(self, d=DIM):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv3d(d, 128, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=(FZ, 4, 4), mode="trilinear", align_corners=False),
            nn.Conv3d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=(1, 4, 4), mode="trilinear", align_corners=False),
            nn.Conv3d(64, 32, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(
            nn.Conv3d(32 + 8, 32, 3, padding=1), nn.GELU(), nn.Conv3d(32, NAFF, 1))
        self.raw_stem = nn.Sequential(nn.Conv3d(1, 8, 3, padding=1), nn.GELU())  # fine-detail skip

    def forward(self, feat, raw):
        u = self.up(feat)
        x = torch.cat([u, self.raw_stem(raw)], 1)
        return self.head(x)[0]


class RawAff3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv3d(32, NAFF, 1))

    def forward(self, img):
        return self.net(img)[0]


def gt_affinity(seg, offs):
    Z, Hh, W = seg.shape
    aff = np.zeros((len(offs), Z, Hh, W), np.float32); val = np.zeros_like(aff)
    for k, (dz, dy, dx) in enumerate(offs):
        a = seg[:Z - dz, :Hh - dy, :W - dx]; b = seg[dz:, dy:, dx:]
        aff[k, :Z - dz, :Hh - dy, :W - dx] = (a == b).astype(np.float32)
        val[k, :Z - dz, :Hh - dy, :W - dx] = 1.0
    return aff, val


def sample_sub(raw, seg, cxy, seed):
    r = np.random.default_rng(seed); Z, Hh, W = raw.shape
    z = r.integers(0, Z - ZC); y = r.integers(0, Hh - cxy); x = r.integers(0, W - cxy)
    return raw[z:z + ZC, y:y + cxy, x:x + cxy], seg[z:z + ZC, y:y + cxy, x:x + cxy]


def mutex_watershed(aff, offs, n_short):
    NA, Z, Hh, W = aff.shape; N = Z * Hh * W
    def vid(z, y, x): return (z * Hh + y) * W + x
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Hh), np.arange(W), indexing="ij")
    parts = []
    for k, (dz, dy, dx) in enumerate(offs):
        z2, y2, x2 = zz + dz, yy + dy, xx + dx
        m = (z2 < Z) & (y2 < Hh) & (x2 < W)
        u = vid(zz[m], yy[m], xx[m]).astype(np.int64); v = vid(z2[m], y2[m], x2[m]).astype(np.int64)
        a = aff[k][m]; w = a if k < n_short else (1.0 - a)
        att = np.full(w.shape, k < n_short)
        parts.append((w.astype(np.float64), u, v, att))
    w = np.concatenate([p[0] for p in parts]); u = np.concatenate([p[1] for p in parts])
    v = np.concatenate([p[2] for p in parts]); att = np.concatenate([p[3] for p in parts])
    order = np.argsort(-w)
    parent = np.arange(N); rank = np.zeros(N, np.int8); mutex = defaultdict(set)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for i in order:
        ru, rv = find(int(u[i])), find(int(v[i]))
        if ru == rv:
            continue
        if att[i]:
            if rv in mutex[ru]:
                continue
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
            for mm in mutex[rv]:
                mutex[mm].discard(rv); mutex[mm].add(ru); mutex[ru].add(mm)
            mutex.pop(rv, None)
        else:
            mutex[ru].add(rv); mutex[rv].add(ru)
    return np.array([find(i) for i in range(N)]).reshape(Z, Hh, W)


def seg_metrics(pred, gt):
    vs, vm = variation_of_information(gt, pred)
    are, _, _ = adapted_rand_error(gt, pred)
    return float(vs + vm), float(are)


def erl_proxy(pred, gt, max_seg=50):
    tot, sq = 0.0, 0.0
    ids = [i for i in np.unique(gt) if (gt == i).sum() >= 60][:max_seg]
    for gid in ids:
        sk = skeletonize(gt == gid); pl = pred[sk]
        if pl.size < 3:
            continue
        for pv in np.unique(pl):
            L = int((pl == pv).sum()); tot += L; sq += L * L
    return float(sq / max(1.0, tot))


def main():
    raw, bnd, syn = H.load_cremi()
    nid = _load_nid()
    Z = raw.shape[0]; ztr = int(Z * 0.75)
    tr_raw, tr_seg = raw[:ztr], nid[:ztr]; te_raw, te_seg = raw[ztr:], nid[ztr:]
    ec = CROP        # JEPA operates at fixed CROP; eval subvolumes must match it
    # (MWS cost is controlled via CROP / ZC / NEVAL, not a separate eval crop)
    print(f"[s3d] device={DEV} crop={CROP} zc={ZC} eval_crop={ec} n_eval={NEVAL} "
          f"jepa_steps={JEPA_STEPS} dec_steps={DEC_STEPS} affs={NAFF}", flush=True)

    enc = V.train_vol_jepa(tr_raw, JEPA_STEPS, rng)
    rnd = V.VolHierEncoder().to(DEV).eval()

    def run_dec(source, dec, encoder, sub):
        raw_t = torch.tensor(sub, device=DEV)[None, None].float()
        if source == "raw":
            return dec(raw_t)
        return dec(V.feature_grid(encoder, sub.astype(np.float32))[None], raw_t)

    def train_dec(source, encoder):
        dec = (RawAff3D() if source == "raw" else FeatAff3D()).to(DEV)
        opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
        for step in range(DEC_STEPS):
            sub, seg = sample_sub(tr_raw, tr_seg, CROP, 1000 + step)
            aff, val = gt_affinity(seg, OFFS)
            tgt = torch.tensor(aff, device=DEV); vmask = torch.tensor(val, device=DEV)
            logit = run_dec(source, dec, encoder, sub)
            # per-offset class balancing so the decoder can't collapse to the
            # majority class (short-range affinities are mostly "same neuron").
            pos = (tgt * vmask).sum((1, 2, 3)); neg = ((1 - tgt) * vmask).sum((1, 2, 3))
            pw = (neg / (pos + 1)).clamp(0.1, 10)[:, None, None, None]
            bce = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pw, reduction="none")
            loss = (bce * vmask).sum() / vmask.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        dec.eval(); return dec

    @torch.no_grad()
    def predict(source, dec, encoder, sub):
        return torch.sigmoid(run_dec(source, dec, encoder, sub)).cpu().numpy()

    res = {"device": DEV, "crop": CROP, "zc": ZC, "eval_crop": ec, "n_eval": NEVAL,
           "offsets": {"short": SHORT, "long": LONG}, "backbone": "vol_jepa (3D hierarchical)",
           "anticollapse": {"families": ["distillation(EMA+predictor)", "info-max(VICReg var+cov)",
                                         "fine-pathway(FINE_AUX+coarse_drop)"],
                            "var": H.VAR_COEF, "cov": H.COV_COEF, "fine_aux": H.FINE_AUX,
                            "coarse_drop": H.COARSE_DROP,
                            "embedding_std_start": getattr(enc, "std_start", None),
                            "embedding_std_final": getattr(enc, "std_final", None)},
           "sources": {}}
    for source, encoder in [("jepa", enc), ("random", rnd), ("raw", None)]:
        dec = train_dec(source, encoder)
        vois, ares, erls, accs, nseg = [], [], [], [], []
        for j in range(NEVAL):
            sub, seg = sample_sub(te_raw, te_seg, ec, 5000 + j)
            aff = predict(source, dec, encoder, sub)
            gt_a, val = gt_affinity(seg, OFFS)
            accs.append(float((((aff > 0.5).astype(np.float32) == gt_a)[val > 0]).mean()))
            lab = mutex_watershed(aff, OFFS, len(SHORT))
            voi, are = seg_metrics(lab, seg); vois.append(voi); ares.append(are)
            erls.append(erl_proxy(lab, seg)); nseg.append(len(np.unique(lab)))
        res["sources"][source] = {"affinity_acc": round(float(np.mean(accs)), 4),
                                   "VOI": round(float(np.mean(vois)), 4),
                                   "adapted_rand_error": round(float(np.mean(ares)), 4),
                                   "ERL_proxy": round(float(np.mean(erls)), 1),
                                   "n_segments": round(float(np.mean(nseg)), 1)}
        print(f"[s3d] {source}: {res['sources'][source]}", flush=True)

    print(json.dumps(res, indent=2))
    with open(os.path.join(H.HERE, "segment3d.json"), "w") as f:
        json.dump(res, f, indent=2)


def _load_nid():
    import h5py
    dirpath = os.environ.get("WORM_EM_DIR", os.path.join(H.HERE, "cremi_data"))
    ids = []
    for s in H.SAMPLES:
        with h5py.File(os.path.join(dirpath, f"sample_{s}.hdf"), "r") as f:
            ids.append(f["volumes/labels/neuron_ids"][:])
    return np.concatenate(ids)


if __name__ == "__main__":
    main()
