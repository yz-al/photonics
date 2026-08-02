"""
Double black box on the trained EM-JEPA affinity model.

The project methodology (CLAUDE.md): chain two opaque systems -- open the trained
model with mechinterp (first black box), then feed what you pull out into an
automated model-discovery step (second black box) to synthesize a MECHANISTIC model
of the extracted behaviour. Here the behaviour is "predict a boundary affinity from
a pretrained JEPA context feature".

First black box -- mechinterp on the JEPA-feature affinity decoder:
  Per-channel ablation of the FEAT_DIM feature grid. Zero channel c before the
  decoder, measure the drop in held-out affinity accuracy -> an importance ranking
  over the learned feature channels. This is the "circuit" we pull out: the handful
  of JEPA channels the boundary decision actually rides on.

Second black box -- discovery of a compact mechanistic model:
  Fit a sparse (L1) logistic model from ONLY the top-k important channels (+ a local
  raw-intensity-gradient term, the classic boundary cue) to the short-range boundary
  affinity. That sparse rule is an interpretable, cheap program that reconstructs the
  learned mechanism from the extracted signal -- the second opaque system (the L1
  solver) building a mechanistic account of the first.

Payoff ("does pulling the terms through the black box help us?"):
  reproduction -- how much of the full decoder's affinity accuracy the k-channel
  mechanistic model recovers, and the minimal k to reach 90% of it. A small k means
  the boundary mechanism is low-dimensional and distillable -- directly useful for
  scaling the stack (a cheap mechanistic head instead of the full decoder).

CPU-cheap: operates on the already-trained encoder/decoder, no extra training run.
"""
import numpy as np
import torch


def _voxelize(feat_grid, zc, hh, ww):
    """Upsample a (C, gz, gf, gf) feature grid to voxel resolution (C, zc, hh, ww)
    by nearest-neighbour, so features align with voxel-level affinity targets."""
    f = feat_grid[None]                                   # (1,C,gz,gf,gf)
    up = torch.nn.functional.interpolate(f, size=(zc, hh, ww), mode="nearest")
    return up[0]                                           # (C, zc, hh, ww)


def run(enc, dec_jepa, te_subs, te_gt, feature_grid, dev, n_short,
        topks=(4, 8, 16), max_vox=40000):
    """Returns a JSON-able dict of the double-black-box analysis.

    enc          trained JEPA encoder
    dec_jepa     its trained (dense-budget) affinity decoder
    te_subs      list of (raw_subvol, seg) held-out eval subvolumes
    te_gt        list of (gt_affinity[NAFF,Z,H,W], val_mask) aligned with te_subs
    feature_grid V.feature_grid(encoder, subvol_float) -> (FEAT_DIM, gz, gf, gf) torch
    n_short      number of short-range (attractive/boundary) offsets
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    except Exception as e:                                # discovery step needs sklearn
        return {"error": f"sklearn unavailable: {e}"}

    feats, raws, gts = [], [], []
    for (sub, _seg), (gt_a, _val) in zip(te_subs, te_gt):
        fg = feature_grid(enc, sub.astype(np.float32))    # (C, gz, gf, gf) torch on dev
        feats.append(fg)
        raws.append(torch.tensor(sub, device=dev).float())
        gts.append(gt_a)
    C = feats[0].shape[0]

    def bal_acc(pred_bin, gt_bin, valid):
        # BALANCED accuracy of the short-range boundary decision -- boundaries are a
        # rare class (~2% of edges), so plain accuracy is swamped by "same-object";
        # balanced accuracy actually measures boundary detection.
        p = pred_bin[valid]; g = gt_bin[valid]
        if len(np.unique(g)) < 2:
            return 0.5
        return float(balanced_accuracy_score(g, p))

    # short-range boundary GT (1 = boundary / different object) + validity, per sub
    bt, bvalid = [], []
    for gt_a, (_g, val) in zip(gts, te_gt):
        bt.append((gt_a[:n_short].mean(0) <= 0.5).astype(np.int64))   # 1 = boundary
        bvalid.append((val[:n_short] > 0).all(0))

    # decoder's short-range boundary decision given a (possibly ablated) feature grid
    @torch.no_grad()
    def dec_boundary(ablate=None):
        outs = []
        for fg, raw in zip(feats, raws):
            f = fg if ablate is None else fg.clone()
            if ablate is not None:
                f[ablate] = 0.0
            logit = dec_jepa(f[None], raw[None, None])    # (NAFF,Z,H,W)
            p = torch.sigmoid(logit).cpu().numpy()
            outs.append((p[:n_short].mean(0) <= 0.5).astype(np.int64))  # 1 = boundary
        return outs

    base_pred = dec_boundary(None)
    base_bacc = float(np.mean([bal_acc(base_pred[i], bt[i], bvalid[i]) for i in range(len(bt))]))

    # ---- first black box: per-channel ablation importance (on boundary detection) ----
    def imp_of(c):
        pc = dec_boundary(ablate=c)
        bc = float(np.mean([bal_acc(pc[i], bt[i], bvalid[i]) for i in range(len(bt))]))
        return base_bacc - bc
    imp = np.array([imp_of(c) for c in range(C)])
    rank = list(np.argsort(-imp))                         # most-important first

    # ---- second black box: sparse mechanistic model from top-k channels ----
    # voxel table: [top-k channel acts, local raw gradient] -> the DECODER's boundary
    # decision (distillation fidelity: does a compact rule reproduce the black box?),
    # plus grounding vs GT. Balanced acc + AUC dodge the class-imbalance trap.
    Zc, Hh, Ww = gts[0].shape[1:]
    vf_l, grad_l, ymodel_l, ygt_l = [], [], [], []
    for fg, raw, dpred, gtb, vmask in zip(feats, raws, base_pred, bt, bvalid):
        vf = _voxelize(fg, Zc, Hh, Ww).cpu().numpy()      # (C, Z, H, W)
        rr = raw.cpu().numpy()
        gz, gy, gx = np.gradient(rr)
        grad = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)       # local intensity gradient (boundary cue)
        idx = np.argwhere(vmask)
        if len(idx):
            vf_l.append(vf[:, idx[:, 0], idx[:, 1], idx[:, 2]].T)          # (n, C)
            grad_l.append(grad[idx[:, 0], idx[:, 1], idx[:, 2]][:, None])
            ymodel_l.append(dpred[idx[:, 0], idx[:, 1], idx[:, 2]])        # decoder's boundary call
            ygt_l.append(gtb[idx[:, 0], idx[:, 1], idx[:, 2]])            # ground-truth boundary
    Xc = np.concatenate(vf_l); Xg = np.concatenate(grad_l)
    Ym = np.concatenate(ymodel_l); Yg = np.concatenate(ygt_l)
    r = np.random.default_rng(0)
    if len(Ym) > max_vox:
        sel = r.choice(len(Ym), max_vox, replace=False)
        Xc, Xg, Ym, Yg = Xc[sel], Xg[sel], Ym[sel], Yg[sel]
    ntr = int(len(Ym) * 0.7)
    disc = {}
    if len(np.unique(Ym[:ntr])) < 2:
        disc = {"note": "decoder predicts one class on train split; discovery skipped"}
    else:
        for k in topks:
            cols = rank[:k]
            Xk = np.concatenate([Xc[:, cols], Xg], 1)     # top-k channels + raw gradient
            mu, sd = Xk[:ntr].mean(0), Xk[:ntr].std(0) + 1e-6
            Xk = (Xk - mu) / sd
            clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                     class_weight="balanced", max_iter=300)
            clf.fit(Xk[:ntr], Ym[:ntr])
            pm = clf.predict(Xk[ntr:])
            try:
                auc = round(float(roc_auc_score(Ym[ntr:], clf.decision_function(Xk[ntr:]))), 4)
            except Exception:
                auc = None
            disc[f"k{k}"] = {
                "fidelity_bal_acc": round(float(balanced_accuracy_score(Ym[ntr:], pm)), 4),
                "fidelity_auc": auc,
                "vs_gt_bal_acc": round(float(balanced_accuracy_score(Yg[ntr:], pm)), 4),
                "nonzero_terms": int((clf.coef_ != 0).sum()),
            }
        disc["decoder_boundary_bal_acc"] = round(base_bacc, 4)
        disc["best_fidelity_bal_acc"] = round(
            float(max(disc[f"k{k}"]["fidelity_bal_acc"] for k in topks)), 4)

    return {
        "method": "double black box (boundary-ablation -> sparse mechanistic distillation)",
        "feat_dim": int(C),
        "decoder_boundary_bal_acc": round(base_bacc, 4),
        "top_channels": [int(c) for c in rank[:10]],
        "top_channel_importance": [round(float(imp[c]), 4) for c in rank[:10]],
        "importance_mass_top8_frac": round(float(imp[rank[:8]].clip(0).sum() /
                                                (imp.clip(0).sum() + 1e-9)), 4),
        "mechanistic_model": disc,
    }
