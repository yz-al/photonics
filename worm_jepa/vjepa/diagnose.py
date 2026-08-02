"""
Why does our trained EM-JEPA underperform a RANDOM-INIT encoder on segmentation?
Three cheap CPU diagnostics (no Modal):

  1. Collapse / effective rank -- do trained features use FEWER dimensions than
     random? (participation ratio of the feature covariance + mean per-dim std).
  2. Frozen LINEAR probe -- boundary AUC from a linear head on frozen features
     (no decoder capacity to mask the difference). trained-jepa vs random.
  3. Predictor reliance -- after training, zero the FINE context tokens and see if
     the predictor's error barely moves. If coarse context alone nearly solves the
     task, the fine encoder gets little gradient and never learns useful features
     (a known JEPA shortcut), which would explain why training makes it worse.
"""
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import hier_em_jepa as H

DEV = H.DEV
GF, NF, NC, DIM = H.GF, H.NF, H.NC, H.DIM
rng = np.random.default_rng(0)


def eff_rank(feat):                       # feat: (M,D)
    x = feat - feat.mean(0)
    cov = (x.T @ x) / max(1, len(x) - 1)
    ev = np.linalg.eigvalsh(cov).clip(min=0)
    return float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)), float(np.sqrt(np.diag(cov)).mean())


@torch.no_grad()
def fine_feats(encoder, crops):
    pf, pc = H.patch_fine(crops), H.patch_coarse(crops)
    fo, _ = encoder(encoder.tok_fine(pf), encoder.tok_coarse(pc))
    return fo.reshape(-1, DIM).cpu().numpy()


def patch_boundary(raw, bnd, n, seed):
    r = np.random.default_rng(seed); Z, Hh, W = raw.shape; C = H.CROP
    z = r.integers(0, Z, n); y = r.integers(0, Hh - C, n); x = r.integers(0, W - C, n)
    imgs = np.stack([raw[z[i], y[i]:y[i] + C, x[i]:x[i] + C] for i in range(n)])
    lm = np.stack([bnd[z[i], y[i]:y[i] + C, x[i]:x[i] + C] for i in range(n)]).astype(np.float32)
    lp = lm.reshape(n, GF, H.PF, GF, H.PF).mean((2, 4)).reshape(n, NF)
    return torch.tensor(imgs, device=DEV), (lp > 0.05).astype(int).reshape(-1)


def main():
    from segment import train_jepa
    raw, bnd, syn = H.load_cremi()
    ztr = int(raw.shape[0] * 0.75); tr_raw = raw[:ztr]; te_raw = raw[ztr:]; te_b = bnd[ztr:]
    steps = int(os.environ.get("WORM_DIAG_STEPS", "150"))
    print(f"[diag] device={DEV} crop={H.CROP} dim={DIM} jepa_steps={steps} "
          f"VICReg var={H.VAR_COEF} cov={H.COV_COEF}", flush=True)

    enc = train_jepa(tr_raw, steps)
    rnd = H.HierEncoder().to(DEV).eval()

    # feature sets on held-out crops
    tr_img, tr_y = patch_boundary(tr_raw, bnd[:ztr], 120, 1)
    te_img, te_y = patch_boundary(te_raw, te_b, 80, 2)

    out = {"jepa_steps": steps, "vicreg": {"var": H.VAR_COEF, "cov": H.COV_COEF}, "encoders": {}}
    for name, e in [("jepa", enc), ("random", rnd)]:
        Ftr = fine_feats(e, tr_img); Fte = fine_feats(e, te_img)
        er, mstd = eff_rank(Ftr)
        lin = {}
        for nl in [50, 200, 1000]:
            idx = rng.permutation(len(tr_y))[:nl]
            c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Ftr[idx], tr_y[idx])
            lin[nl] = round(float(roc_auc_score(te_y, c.predict_proba(Fte)[:, 1])), 4)
        out["encoders"][name] = {"effective_rank": round(er, 2), "of_dim": DIM,
                                 "mean_dim_std": round(mstd, 4),
                                 "linear_probe_boundary_AUC": lin}
        print(f"[diag] {name}: eff_rank={er:.1f}/{DIM} std={mstd:.3f} linAUC={lin}", flush=True)

    # 3. predictor reliance: fine-context-zeroed vs full (uses the TRAINED enc+pred)
    out["predictor_reliance"] = predictor_reliance(tr_raw, enc)
    print("[diag] predictor_reliance:", out["predictor_reliance"], flush=True)

    print(json.dumps(out, indent=2))
    with open(os.path.join(H.HERE, "diagnose.json"), "w") as f:
        json.dump(out, f, indent=2)


@torch.no_grad()
def predictor_reliance(tr_raw, enc):
    """Retrain-free proxy: build context, compare predictor error with full context
    vs with fine tokens zeroed (coarse-only). Small gap => fine features barely used."""
    tgt = H.HierEncoder().to(DEV); tgt.load_state_dict(enc.state_dict()); tgt.eval()
    pred = H.Predictor().to(DEV)          # untrained predictor -> compares INPUT informativeness
    crops, _ = H.sample_crops(tr_raw, 32, rng)
    crops = torch.tensor(crops, device=DEV)
    pf, pc = H.patch_fine(crops), H.patch_coarse(crops)
    fctx, cctx, ftgt = H.masks_for(32, rng)
    fine_ctx = torch.gather(enc.tok_fine(pf), 1, fctx[:, :, None].expand(-1, -1, DIM))
    coarse_ctx = torch.gather(enc.tok_coarse(pc), 1, (cctx - NF)[:, :, None].expand(-1, -1, DIM))
    fo, co = enc(fine_ctx, coarse_ctx)
    tf, _ = tgt(tgt.tok_fine(pf), tgt.tok_coarse(pc))
    target = torch.gather(tf, 1, ftgt[:, :, None].expand(-1, -1, DIM))
    full = pred(torch.cat([fo, co], 1), torch.cat([fctx, cctx], 1), ftgt)
    coarse_only = pred(torch.cat([torch.zeros_like(fo), co], 1), torch.cat([fctx, cctx], 1), ftgt)
    e_full = float(F.smooth_l1_loss(full, target)); e_coarse = float(F.smooth_l1_loss(coarse_only, target))
    return {"err_full_context": round(e_full, 4), "err_coarse_only": round(e_coarse, 4),
            "fine_context_value": round(e_coarse - e_full, 4),
            "note": "small fine_context_value => coarse tokens carry most of the signal, "
                    "fine encoder under-trained"}


if __name__ == "__main__":
    main()
