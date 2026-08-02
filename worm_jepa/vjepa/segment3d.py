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

SHORT = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]                 # attractive (z,y,x nearest)
# Repulsive long-range offsets. Mutex watershed merges along ANY attractive edge
# (in weight order) unless a repulsive edge has forced a mutex between the clusters
# first -- so too FEW repulsive offsets under-constrains separation and the result
# under-segments (merges). Use a denser multi-scale set (in-plane 4 & 9, diagonals,
# small z) like standard MWS, not just 3. (Non-negative only: gt_affinity slices
# assume positive offsets.)
LONG = [(0, 9, 0), (0, 0, 9), (0, 9, 9), (0, 4, 0), (0, 0, 4), (0, 4, 4),
        (2, 0, 0), (3, 0, 0)]                             # 8 repulsive offsets
OFFS = SHORT + LONG
NAFF = len(OFFS)


class FeatAff3D(nn.Module):
    """U-Net-style: coarse JEPA feature grid (context) upsampled to voxel res, then
    concatenated with the raw subvolume (fine detail) before the affinity head. This
    is how a pretrained backbone is actually used for dense prediction -- features
    supply learned context, the raw skip supplies resolution the 16px patches lack.
    The comparison then asks: does JEPA context add value ON TOP of raw pixels?"""
    def __init__(self, d=V.FEAT_DIM):                    # dense features -> wider input
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


class _Res3d(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1 = nn.Conv3d(ci, co, 3, padding=1); self.c2 = nn.Conv3d(co, co, 3, padding=1)
        self.n1 = nn.GroupNorm(8, co); self.n2 = nn.GroupNorm(8, co)
        self.sk = nn.Conv3d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x))); h = self.n2(self.c2(h))
        return F.gelu(h + self.sk(x))


class SotaUNet3D(nn.Module):
    """The SOTA-family model: a residual 3D U-Net (PyTorch-Connectomics style)
    predicting affinities from the raw subvolume. Downsamples xy by 4, z by 2.
    Used as the 'sota' base -- pretrained on dense labels, then adapted per budget,
    to test whether supervised-SOTA transfer beats SSL-JEPA / from-scratch."""
    def __init__(self):
        super().__init__()
        self.e1 = _Res3d(1, 32); self.d1 = nn.Conv3d(32, 32, 3, stride=(2, 2, 2), padding=1)
        self.e2 = _Res3d(32, 64); self.d2 = nn.Conv3d(64, 64, 3, stride=(1, 2, 2), padding=1)
        self.bott = _Res3d(64, 128)
        self.u2 = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2)); self.dec2 = _Res3d(128, 64)
        self.u1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2)); self.dec1 = _Res3d(64, 32)
        self.head = nn.Conv3d(32, NAFF, 1)

    def forward(self, img):
        s1 = self.e1(img); x = self.d1(s1)
        s2 = self.e2(x); x = self.d2(s2)
        x = self.bott(x)
        x = self.dec2(torch.cat([self.u2(x), s2], 1))
        x = self.dec1(torch.cat([self.u1(x), s1], 1))
        return self.head(x)[0]


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

    def run_dec(source, dec, encoder, sub):
        raw_t = torch.tensor(sub, device=DEV)[None, None].float()
        if source in ("raw", "sota"):                     # both are image -> affinity nets
            return dec(raw_t)
        return dec(V.feature_grid(encoder, sub.astype(np.float32))[None], raw_t)

    def make_dec(source):
        return {"raw": RawAff3D, "sota": SotaUNet3D}.get(source, FeatAff3D)().to(DEV)

    def _rand_sparse_mask(pool, sparse_k, seed):
        """A FIXED random set of sparse_k annotated voxel-edges on pool[0]."""
        _, seg0 = pool[0]; _, val0 = gt_affinity(seg0, OFFS)
        vi = np.argwhere(val0 > 0)
        r = np.random.default_rng(1234 + seed)
        pick = vi[r.choice(len(vi), min(sparse_k, len(vi)), replace=False)]
        smask = np.zeros_like(val0); smask[tuple(pick.T)] = 1.0
        return smask, val0

    def train_dec(source, encoder, pool, sparse_k=None, smask=None, seed=0,
                  init_state=None, steps=None):
        # pool = fixed labeled (subvol, seg) list. DENSE budget = len(pool) fully
        # labeled subvolumes. SPARSE budget (sparse_k) = ONE subvolume but the loss
        # is restricted to a FIXED set of sparse_k annotated voxel-edges -- the
        # genuinely label-scarce regime where SSL is supposed to win most.
        # smask: a precomputed voxel-edge annotation mask (overrides sparse_k) -- used
        # by active learning to hand in an uncertainty-selected label set.
        # init_state: warm-start from a pretrained model (the SOTA-base transfer test).
        dec = make_dec(source)
        if init_state is not None:
            dec.load_state_dict(init_state)
        opt = torch.optim.Adam(dec.parameters(), lr=2e-3)
        if smask is None and sparse_k is not None:        # fixed random sparse annotation
            smask, _ = _rand_sparse_mask(pool, sparse_k, seed)
        for step in range(steps or DEC_STEPS):
            sub, seg = pool[step % len(pool)]
            aff, val = gt_affinity(seg, OFFS)
            if smask is not None:
                val = val * smask                          # only the annotated labels count
            tgt = torch.tensor(aff, device=DEV); vmask = torch.tensor(val, device=DEV)
            logit = run_dec(source, dec, encoder, sub)
            pos = (tgt * vmask).sum((1, 2, 3)); neg = ((1 - tgt) * vmask).sum((1, 2, 3))
            pw = (neg / (pos + 1)).clamp(0.1, 10)[:, None, None, None]
            bce = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pw, reduction="none")
            loss = (bce * vmask).sum() / vmask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        dec.eval(); return dec

    def train_dec_active(source, encoder, pool, sparse_k, seed=0, init_state=None):
        # Active learning under the SAME K-label budget: spend half the budget on a
        # random seed set, train a warm decoder, then spend the other half on the
        # MOST-UNCERTAIN candidate edges (|p-0.5| smallest) and retrain. Tests whether
        # smart label *selection* beats random selection -- the practical sparse win.
        seed_k = max(1, sparse_k // 2)
        smask_seed, val0 = _rand_sparse_mask(pool, seed_k, seed)
        dec0 = train_dec(source, encoder, pool, smask=smask_seed, seed=seed,
                         init_state=init_state, steps=max(1, DEC_STEPS // 2))
        sub0, _ = pool[0]
        aff0 = predict(source, dec0, encoder, sub0)        # (NAFF,Z,H,W) probabilities
        unc = -np.abs(aff0 - 0.5)                          # higher = more uncertain
        unc[val0 == 0] = -np.inf                           # only annotatable edges
        unc[smask_seed > 0] = -np.inf                      # don't re-query the seed set
        need = min(sparse_k - int(smask_seed.sum()), int(np.isfinite(unc).sum()))
        smask_full = smask_seed.copy()
        if need > 0:
            idx = np.argpartition(unc.ravel(), -need)[-need:]
            smask_full.ravel()[idx] = 1.0
        return train_dec(source, encoder, pool, smask=smask_full, seed=seed,
                         init_state=init_state)

    @torch.no_grad()
    def predict(source, dec, encoder, sub):
        return torch.sigmoid(run_dec(source, dec, encoder, sub)).cpu().numpy()

    POOL = [int(x) for x in os.environ.get("WORM_S3_LABEL_POOL", "1,16").split(",")]
    SPARSE = [int(x) for x in os.environ.get("WORM_S3_SPARSE", "").split(",") if x.strip()]
    ACTIVE = [int(x) for x in os.environ.get("WORM_S3_ACTIVE", "").split(",") if x.strip()]
    SEEDS = [int(x) for x in os.environ.get("WORM_S3_SEEDS", "0,1,2").split(",")]
    # budgets: dense (N fully-labeled subvols) + sparse (K random annotated voxel-edges
    # on 1 subvol) + active (same K, but uncertainty-selected -- the sparse-improvement test)
    BUDGETS = ([(str(P), "dense", P) for P in POOL]
               + [(f"sparse{K}", "sparse", K) for K in SPARSE]
               + [(f"active{K}", "active", K) for K in ACTIVE])
    DENSE_MAX_KEY = str(max(POOL))
    ns = len(SHORT)
    # FIXED data across seeds -> isolates model-init/training variance (the thing we
    # need error bars on). label budget P uses the first P of the fixed pool.
    full_pool = [sample_sub(tr_raw, tr_seg, CROP, 1000 + i) for i in range(max(POOL))]
    te_subs = [sample_sub(te_raw, te_seg, ec, 5000 + j) for j in range(NEVAL)]
    te_gt = [gt_affinity(seg, OFFS) for _, seg in te_subs]
    print(f"[s3d] seeds={SEEDS} label_pool={POOL} jepa_steps={JEPA_STEPS} "
          f"dec_steps={DEC_STEPS} n_eval={NEVAL}", flush=True)

    def evaluate(source, dec, encoder):
        vois, ares, erls, accs, nseg, preds = [], [], [], [], [], []
        for (sub, seg), (gt_a, val) in zip(te_subs, te_gt):
            aff = predict(source, dec, encoder, sub); preds.append(aff)
            accs.append(float((((aff > 0.5).astype(np.float32) == gt_a)[val > 0]).mean()))
            lab = mutex_watershed(aff, OFFS, len(SHORT))
            voi, are = seg_metrics(lab, seg); vois.append(voi); ares.append(are)
            erls.append(erl_proxy(lab, seg)); nseg.append(len(np.unique(lab)))
        return ({"affinity_acc": float(np.mean(accs)), "VOI": float(np.mean(vois)),
                 "adapted_rand_error": float(np.mean(ares)),
                 "ERL_proxy": float(np.mean(erls)), "n_segments": float(np.mean(nseg))}, preds)

    def recovery(full_preds, mask_fn):
        raw_errs, jepa_ok, rand_ok, sota_ok, tot = 0, 0, 0, 0, 0
        for j, (_, (gt_a, val)) in enumerate(zip(te_subs, te_gt)):
            m = (val > 0) & mask_fn(j)
            e = m & ((full_preds["raw"][j] > 0.5) != (gt_a > 0.5))          # raw wrong
            raw_errs += int(e.sum()); tot += int(m.sum())
            jepa_ok += int((e & ((full_preds["jepa"][j] > 0.5) == (gt_a > 0.5))).sum())
            rand_ok += int((e & ((full_preds["random"][j] > 0.5) == (gt_a > 0.5))).sum())
            sota_ok += int((e & ((full_preds["sota"][j] > 0.5) == (gt_a > 0.5))).sum())
        return {"raw_error_rate": raw_errs / max(1, tot),
                "jepa_recovers": jepa_ok / max(1, raw_errs),
                "random_recovers": rand_ok / max(1, raw_errs),
                "sota_recovers": sota_ok / max(1, raw_errs)}

    # ---- multi-seed loop (mean +- std) ----
    SOURCES = ["jepa", "random", "raw", "sota"]
    le_acc = {s: {key: [] for key, _, _ in BUDGETS} for s in SOURCES}
    es_acc = {"all_offsets": [], "long_range_merge_edges": []}
    std_finals = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        enc = V.train_vol_jepa(tr_raw, JEPA_STEPS, np.random.default_rng(seed))
        rnd = V.VolHierEncoder().to(DEV).eval()
        std_finals.append(getattr(enc, "std_final", None))
        encmap = {"jepa": enc, "random": rnd, "raw": None, "sota": None}
        # SOTA base: pretrain the U-Net on the FULL dense training labels, once per
        # seed. Each budget then WARM-STARTS from it (supervised transfer) -- vs raw
        # (from scratch) and jepa (SSL). Answers "should we use SOTA on base?".
        sota_base = train_dec("sota", None, full_pool)
        sota_init = sota_base.state_dict()
        full_preds = {}
        for source in SOURCES:
            for key, kind, B in BUDGETS:
                init = sota_init if source == "sota" else None
                if kind == "dense":
                    dec = train_dec(source, encmap[source], full_pool[:B], init_state=init)
                elif kind == "active":
                    dec = train_dec_active(source, encmap[source], full_pool[:1], sparse_k=B, seed=seed, init_state=init)
                else:
                    dec = train_dec(source, encmap[source], full_pool[:1], sparse_k=B, seed=seed, init_state=init)
                m, preds = evaluate(source, dec, encmap[source])
                le_acc[source][key].append(m)
                if key == DENSE_MAX_KEY:
                    full_preds[source] = preds
        es_acc["all_offsets"].append(recovery(full_preds, lambda j: np.ones_like(te_gt[j][1], bool)))
        es_acc["long_range_merge_edges"].append(recovery(
            full_preds, lambda j: np.concatenate([np.zeros((ns,) + te_gt[j][1].shape[1:], bool),
                                                  np.ones((len(OFFS) - ns,) + te_gt[j][1].shape[1:], bool)])))
        print(f"[s3d] seed {seed} std={std_finals[-1]} "
              f"error_set(all)={es_acc['all_offsets'][-1]}", flush=True)

    def agg(dicts):
        return {k: {"mean": round(float(np.mean([d[k] for d in dicts])), 4),
                    "std": round(float(np.std([d[k] for d in dicts])), 4)} for k in dicts[0]}

    res = {"device": DEV, "crop": CROP, "zc": ZC, "n_eval": NEVAL, "seeds": SEEDS,
           "jepa_steps": JEPA_STEPS, "label_pool": POOL,
           "offsets": {"short": SHORT, "long": LONG}, "backbone": "vol_jepa (3D hierarchical)",
           "anticollapse": {"var": H.VAR_COEF, "cov": H.COV_COEF, "fine_aux": H.FINE_AUX,
                            "coarse_drop": H.COARSE_DROP, "target_std": V.TGT_STD,
                            "embedding_std_final_per_seed": std_finals},
           "label_efficiency": {s: {key: agg(le_acc[s][key]) for key, _, _ in BUDGETS}
                                for s in SOURCES},
           "error_set_analysis": {k: agg(es_acc[k]) for k in es_acc}}
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
