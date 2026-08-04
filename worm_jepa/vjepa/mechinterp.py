"""
FIRST BLACK BOX ONLY -- mechinterp of the trained plain affinity U-Net (our best model).

We open the affinity net and extract WHAT computes the boundary (membrane) decision, and
STOP there. We deliberately DO NOT synthesize a mechanistic model from the extracted
signal (that is the SECOND black box) -- the user has new techniques for that step.

Three complementary probes on the penultimate 32-ch voxel features (the layer feeding the
affinity head), evaluated on held-out subvolumes:

  1. CHANNEL ABLATION importance -- zero each feature channel, measure the drop in
     short-range (membrane) boundary accuracy. Ranks which channels carry the decision and
     how CONCENTRATED it is (few channels = a compact circuit).
  2. SPARSE AUTOENCODER on the activations -- learn an overcomplete, L1-sparse dictionary
     over the 32-ch features. Report sparsity + each dictionary feature's BOUNDARY
     SELECTIVITY (does it fire on membranes vs interior?) -> monosemantic membrane features.
  3. ACTIVATION CLAMPING -- clamp the top channel high/low and measure the causal change in
     boundary decisions (confirms the ablation is causal, not just correlational).

Returns the extracted signal only. No program/rule discovery.
"""
import numpy as np
import torch
import torch.nn.functional as F


def run(dec, test, ctx):
    """dec: trained SotaUNet3D (has .features/.head). test: [(sub, seg)]. ctx supplies
    OFFS, SHORT, gt_affinity, DEV."""
    DEV = ctx["DEV"]; OFFS = ctx["OFFS"]; SHORT = ctx["SHORT"]; ns = len(SHORT)
    gt_affinity = ctx["gt_affinity"]
    dec.eval()
    C = dec.head.in_channels

    @torch.no_grad()
    def feats(sub):
        return dec.features(torch.tensor(sub, device=DEV)[None, None].float())      # (1,C,Z,H,W)

    @torch.no_grad()
    def head_from(feat):
        return torch.sigmoid(dec.head(feat)[0]).cpu().numpy()                       # (NAFF,Z,H,W)

    def boundary_acc(mod=None):
        accs = []
        for sub, seg in test:
            feat = feats(sub)
            if mod is not None:
                feat = mod(feat.clone())
            p = head_from(feat) > 0.5
            ga, val = gt_affinity(seg, OFFS); m = val[:ns] > 0
            accs.append(float((p[:ns][m] == (ga[:ns][m] > 0.5)).mean()))
        return float(np.mean(accs))

    base = boundary_acc()

    # ---- 1. channel ablation importance -----------------------------------------------
    imp = []
    for c in range(C):
        def ab(f, c=c):
            f[:, c] = 0.0; return f
        imp.append({"channel": c, "acc_drop": round(base - boundary_acc(ab), 4)})
    imp.sort(key=lambda d: -d["acc_drop"])
    drops = np.array([d["acc_drop"] for d in imp], float)
    pos = drops[drops > 0].sum() + 1e-9
    cum = np.cumsum([d["acc_drop"] for d in imp if d["acc_drop"] > 0]) / pos
    n_for_80 = int(np.searchsorted(cum, 0.8) + 1) if len(cum) else 0
    concentration = {"n_channels": C, "n_channels_for_80pct_importance": n_for_80,
                     "top5_channels": [d["channel"] for d in imp[:5]],
                     "reading": f"{n_for_80}/{C} channels carry 80% of the boundary decision"}

    # ---- 2. sparse autoencoder over the activations -----------------------------------
    Xs, Bs = [], []
    rng = np.random.default_rng(0)
    for sub, seg in test[:4]:
        f = feats(sub)[0].reshape(C, -1).T.cpu().numpy()                            # (Nvox, C)
        ga, val = gt_affinity(seg, OFFS)
        bnd = (ga[:ns] < 0.5).any(0).reshape(-1)                                    # voxel borders a neighbor
        idx = rng.choice(len(f), min(4000, len(f)), replace=False)
        Xs.append(f[idx]); Bs.append(bnd[idx])
    X = np.concatenate(Xs).astype(np.float32); Bnd = np.concatenate(Bs)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd, device=DEV)
    H = int(ctx.get("sae_hidden", 64))                                              # overcomplete dict
    enc = torch.nn.Linear(C, H).to(DEV); decl = torch.nn.Linear(H, C).to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(decl.parameters()), lr=1e-3)
    l1 = float(ctx.get("sae_l1", 3e-3))
    for step in range(1500):
        code = F.relu(enc(Xn)); recon = decl(code)
        loss = ((recon - Xn) ** 2).mean() + l1 * code.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        code = F.relu(enc(Xn))
        recon = decl(code)
        r2 = float(1 - ((recon - Xn) ** 2).mean() / (Xn.var() + 1e-9))
        active = (code > 1e-3).float()
        sparsity = float(active.mean())                                            # frac of dict active per voxel
        cm = code.cpu().numpy(); b = Bnd.astype(np.float32)
        # per-feature boundary selectivity = correlation of code activation with boundary label
        sel = []
        for h in range(H):
            ch = cm[:, h]
            if ch.std() < 1e-6:
                continue
            corr = float(np.corrcoef(ch, b)[0, 1])
            sel.append({"feature": h, "boundary_corr": round(corr, 3),
                        "fire_rate": round(float((ch > 1e-3).mean()), 3)})
        sel.sort(key=lambda d: -abs(d["boundary_corr"]))
    sae = {"dict_size": H, "recon_r2": round(r2, 3), "mean_active_frac": round(sparsity, 3),
           "n_membrane_selective(|corr|>0.3)": int(sum(abs(s["boundary_corr"]) > 0.3 for s in sel)),
           "top_features": sel[:8],
           "reading": "sparse features with high +|boundary_corr| are membrane detectors; "
                      "clean separation => the net learned near-monosemantic boundary features"}

    # ---- 3. activation clamping (causal check on the top channel) ----------------------
    top = imp[0]["channel"]
    hi = float(np.quantile([feats(s)[0, top].cpu().numpy() for s, _ in test[:1]][0], 0.95))

    def clamp_hi(f):
        f[:, top] = hi; return f

    def clamp_lo(f):
        f[:, top] = 0.0; return f
    clamp = {"top_channel": top, "baseline_acc": round(base, 4),
             "acc_clamp_high": round(boundary_acc(clamp_hi), 4),
             "acc_clamp_zero": round(boundary_acc(clamp_lo), 4),
             "reading": "large acc change under clamping => channel is CAUSAL for the boundary decision"}

    return {"stage": "FIRST BLACK BOX ONLY (mechinterp) -- synthesis intentionally NOT run",
            "model": "plain affinity U-Net (SotaUNet3D)", "penultimate_channels": C,
            "baseline_boundary_acc": round(base, 4),
            "channel_ablation_top": imp[:10], "concentration": concentration,
            "sparse_autoencoder": sae, "activation_clamp": clamp,
            "handoff_note": "extracted signal ready; conversion to a mechanistic model deferred to your new techniques"}
