"""
Edge-case (failure) analysis of the SOTA base -- the step before scaling to mouse.

SOTA (the dense-pretrained residual 3D U-Net) is our strongest base, but "0.202 is
pretty good" means the remaining value is in its EDGE CASES. This module does the
diagnostic that decides whether a failure-focused corrector is worth building:

  1. Error composition -- of the affinity edges SOTA gets wrong, how many are MERGE
     errors (predicts same-object across a true boundary -- the costly kind for
     connectomes) vs SPLIT errors (predicts a boundary inside one object), split by
     short-range (adjacent) vs long-range (across-distance) offsets, and how many sit
     on a true membrane vs in the interior. Tells us WHAT SOTA fails on.

  2. Error detector -- can we PREDICT, per edge, where SOTA will be wrong? We fit a
     balanced logistic error model on nested feature sets:
        conf         : SOTA's own confidence |p-0.5| (+ offset type)
        conf+raw     : + local raw intensity gradient (the classic boundary cue)
        conf+raw+jepa: + top-variance JEPA feature channels
     If JEPA features raise detector AUC OVER sota-confidence+raw, then the
     hierarchical-JEPA representation SEES blind spots SOTA's own logits miss -- the
     mechanistic-reason signal that justifies a JEPA-driven corrector / active
     re-labelling on exactly SOTA's error set. If they don't, we've saved a GPU run.

CPU-cheap: post-hoc on the already-trained SOTA decoder + JEPA encoder, no training.
"""
import numpy as np
import torch


def _voxelize(feat_grid, zc, hh, ww):
    up = torch.nn.functional.interpolate(feat_grid[None], size=(zc, hh, ww), mode="nearest")
    return up[0]


def run(sota_dec, jepa_enc, te_subs, te_gt, feature_grid, offs, n_short, dev,
        n_jepa_ch=16, max_rows=80000):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    except Exception as e:
        return {"error": f"sklearn unavailable: {e}"}

    NAFF = len(offs)
    Zc, Hh, Ww = te_gt[0][0].shape[1:]

    # ---- gather per-sub tensors: sota prob, gt, validity, jepa features, raw grad ----
    sota_p, gt_b, valm, jfeat, rawg = [], [], [], [], []
    with torch.no_grad():
        for (sub, _seg), (gt_a, val) in zip(te_subs, te_gt):
            raw_t = torch.tensor(sub, device=dev)[None, None].float()
            p = torch.sigmoid(sota_dec(raw_t)).cpu().numpy()          # (NAFF,Z,H,W)
            sota_p.append(p); gt_b.append(gt_a); valm.append(val)
            fg = feature_grid(jepa_enc, sub.astype(np.float32))       # (C,gz,gf,gf)
            jfeat.append(_voxelize(fg, Zc, Hh, Ww).cpu().numpy())     # (C,Z,H,W)
            gz, gy, gx = np.gradient(sub.astype(np.float32))
            rawg.append(np.sqrt(gz ** 2 + gy ** 2 + gx ** 2))

    # ---- 1. error composition ----
    # affinity convention: 1 = same-object, 0 = boundary.
    merge_err = split_err = 0            # merge = pred same where gt boundary; split = opposite
    short_err = long_err = 0
    on_membrane_err = interior_err = 0
    total_err = total_valid = 0
    for p, gt_a, val in zip(sota_p, gt_b, valm):
        pred = (p > 0.5); g = (gt_a > 0.5); v = (val > 0)
        err = v & (pred != g)
        total_err += int(err.sum()); total_valid += int(v.sum())
        merge_err += int((err & (pred & ~g)).sum())      # said same, truly boundary
        split_err += int((err & (~pred & g)).sum())      # said boundary, truly same
        short_err += int(err[:n_short].sum()); long_err += int(err[n_short:].sum())
        # membrane band: any short-range boundary at this voxel (collapsed over offsets)
        memb = (gt_a[:n_short] <= 0.5).any(0) & (val[:n_short] > 0).any(0)
        memb_full = np.broadcast_to(memb, err.shape)
        on_membrane_err += int((err & memb_full).sum())
        interior_err += int((err & ~memb_full).sum())
    composition = {
        "sota_error_rate": round(total_err / max(1, total_valid), 4),
        "merge_frac": round(merge_err / max(1, total_err), 4),      # costliest for connectomes
        "split_frac": round(split_err / max(1, total_err), 4),
        "short_frac": round(short_err / max(1, total_err), 4),
        "long_frac": round(long_err / max(1, total_err), 4),
        "on_membrane_frac": round(on_membrane_err / max(1, total_err), 4),
        "interior_frac": round(interior_err / max(1, total_err), 4),
    }

    # ---- 2. error detector: build per-edge rows ----
    # pick the top-variance JEPA channels once (deterministic, cheap dim reduction)
    allj = np.concatenate([jf.reshape(jf.shape[0], -1) for jf in jfeat], 1)  # (C, Nvox)
    ch = list(np.argsort(-allj.var(1))[:n_jepa_ch])
    rows_conf, rows_raw, rows_jepa, y = [], [], [], []
    is_long = np.array([0] * n_short + [1] * (NAFF - n_short), np.float32)
    for p, gt_a, val, jf, rg in zip(sota_p, gt_b, valm, jfeat, rawg):
        pred = (p > 0.5); g = (gt_a > 0.5); v = (val > 0)
        err = (pred != g).astype(np.int64)
        conf = np.abs(p - 0.5)                                       # (NAFF,Z,H,W)
        idx = np.argwhere(v)                                        # (n,4): k,z,y,x
        if not len(idx):
            continue
        k, z, yy, xx = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        rows_conf.append(np.stack([conf[k, z, yy, xx], is_long[k]], 1))     # (n,2)
        rows_raw.append(rg[z, yy, xx][:, None])                            # (n,1)
        rows_jepa.append(jf[:, z, yy, xx][ch].T)                           # (n,n_jepa_ch)
        y.append(err[k, z, yy, xx])
    Xc = np.concatenate(rows_conf); Xr = np.concatenate(rows_raw)
    Xj = np.concatenate(rows_jepa); Y = np.concatenate(y)
    r = np.random.default_rng(0)
    if len(Y) > max_rows:
        sel = r.choice(len(Y), max_rows, replace=False)
        Xc, Xr, Xj, Y = Xc[sel], Xr[sel], Xj[sel], Y[sel]
    ntr = int(len(Y) * 0.7)

    def fit(X, name):
        if len(np.unique(Y[:ntr])) < 2 or len(np.unique(Y[ntr:])) < 2:
            return {"note": "degenerate error labels"}
        mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
        Xs = (X - mu) / sd
        clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=300)
        clf.fit(Xs[:ntr], Y[:ntr])
        pr = clf.decision_function(Xs[ntr:]); pb = clf.predict(Xs[ntr:])
        return {"auc": round(float(roc_auc_score(Y[ntr:], pr)), 4),
                "bal_acc": round(float(balanced_accuracy_score(Y[ntr:], pb)), 4)}

    det = {
        "conf": fit(Xc, "conf"),
        "conf_raw": fit(np.concatenate([Xc, Xr], 1), "conf_raw"),
        "conf_raw_jepa": fit(np.concatenate([Xc, Xr, Xj], 1), "conf_raw_jepa"),
    }
    jepa_uplift = None
    if "auc" in det["conf_raw"] and "auc" in det["conf_raw_jepa"]:
        jepa_uplift = round(det["conf_raw_jepa"]["auc"] - det["conf_raw"]["auc"], 4)

    # ---- 3. mechanistic correction of the SELF-DETECTED errors ----
    # The hypothesis: SOTA's errors are self-detectable (low |p-0.5|), so let the
    # MECHANISTIC boundary model (sparse JEPA channels + raw gradient -> true boundary)
    # RE-DECIDE exactly on the edges SOTA is unsure about. We fit the mechanistic model
    # on confident/train edges, then on held-out SHORT-range edges compare boundary
    # balanced-accuracy of SOTA-alone vs mechanistic-alone vs a confidence-GATED blend
    # (SOTA where confident, mechanistic where not). The crux number is boundary
    # accuracy ON THE LOW-CONFIDENCE SUBSET: if the mechanistic model beats SOTA there,
    # the mechanistic process fixes the self-detected errors.
    correction = {"note": "no short-range edges"}
    br_f, br_y, br_sotap = [], [], []
    for p, gt_a, val, jf, rg in zip(sota_p, gt_b, valm, jfeat, rawg):
        for k in range(n_short):
            ii = np.argwhere(val[k] > 0)
            if not len(ii):
                continue
            z, yy, xx = ii[:, 0], ii[:, 1], ii[:, 2]
            feat = np.concatenate([jf[:, z, yy, xx][ch].T,               # top JEPA channels
                                   rg[z, yy, xx][:, None]], 1)           # + raw gradient
            br_f.append(feat)
            br_y.append((gt_a[k, z, yy, xx] <= 0.5).astype(np.int64))    # 1 = boundary
            br_sotap.append(p[k, z, yy, xx])                             # SOTA same-object prob
    if br_y:
        Bj = np.concatenate(br_f); By = np.concatenate(br_y); Bsp = np.concatenate(br_sotap)
        rr = np.random.default_rng(1)
        if len(By) > max_rows:
            s = rr.choice(len(By), max_rows, replace=False); Bj, By, Bsp = Bj[s], By[s], Bsp[s]
        m = int(len(By) * 0.7)
        if len(np.unique(By[:m])) > 1 and len(np.unique(By[m:])) > 1:
            mu, sd = Bj[:m].mean(0), Bj[:m].std(0) + 1e-6
            Bs = (Bj - mu) / sd
            mech = LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                      class_weight="balanced", max_iter=300)
            mech.fit(Bs[:m], By[:m])
            p_mech = mech.predict_proba(Bs[m:])[:, 1]         # boundary prob
            yte = By[m:]; sp = Bsp[m:]
            sota_bnd = (sp < 0.5).astype(int)                 # SOTA boundary call
            mech_bnd = (p_mech > 0.5).astype(int)
            conf = np.abs(sp - 0.5)                            # SOTA self-confidence
            thr = np.quantile(conf, 0.30)                     # bottom 30% = self-detected likely errors
            low = conf <= thr
            gated = np.where(low, mech_bnd, sota_bnd)          # correct only where SOTA is unsure
            correction = {
                "sota_boundary_bal_acc": round(float(balanced_accuracy_score(yte, sota_bnd)), 4),
                "mechanistic_boundary_bal_acc": round(float(balanced_accuracy_score(yte, mech_bnd)), 4),
                "gated_correction_bal_acc": round(float(balanced_accuracy_score(yte, gated)), 4),
                "low_conf_frac": round(float(low.mean()), 4),
                "on_low_conf_sota_bal_acc": round(float(balanced_accuracy_score(yte[low], sota_bnd[low])), 4)
                if low.sum() and len(np.unique(yte[low])) > 1 else None,
                "on_low_conf_mechanistic_bal_acc": round(float(balanced_accuracy_score(yte[low], mech_bnd[low])), 4)
                if low.sum() and len(np.unique(yte[low])) > 1 else None,
            }
            lc_s = correction["on_low_conf_sota_bal_acc"]; lc_m = correction["on_low_conf_mechanistic_bal_acc"]
            correction["correction_helps"] = bool(
                correction["gated_correction_bal_acc"] > correction["sota_boundary_bal_acc"]
                and (lc_m or 0) > (lc_s or 1))

    return {
        "method": "sota edge-case analysis (composition + detector + mechanistic correction)",
        "n_edges_scored": int(len(Y)),
        "error_composition": composition,
        "error_detector_auc": det,
        "jepa_auc_uplift_over_conf_raw": jepa_uplift,
        "detector_verdict": ("JEPA features predict SOTA errors beyond confidence+raw -> corrector justified"
                             if (jepa_uplift or 0) > 0.02 else
                             "JEPA adds little to error prediction -> corrector not yet justified"),
        "mechanistic_correction": correction,
    }
