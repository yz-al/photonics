"""
Error debugging on the SAVED agglomeration checkpoints -- BEFORE spending GPU on a
heavier SOTA run. Answers one question: is the winner's residual error STRUCTURED
(a targetable failure mode with a feature signature) or DIFFUSE (undertraining /
noise -> the lever is more SOTA training, not a cleverer head)?

Runs on the same edge decisions the two-specialist multicut makes. For every LOCAL
fragment-pair edge on the held-out eval subvols we have:
  - the boundary feature vector (mean_short/long aff, contact, sizes, +LSD/JEPA agree),
  - the model's SIGNED confidence edge_w = logit(P_A*(1-P_B)) + merge_bias (>0 => merge),
  - the GROUND TRUTH: do the two fragments actually belong to the same neuron?
An error is a sign disagreement: false_merge (model merges two different neurons) or
false_split (model keeps two same-neuron fragments apart).

Three measurements:
  (1) CONFIDENCE  -- margin |edge_w| for correct vs error edges. Errors near the 0.5
      boundary = the model was UNSURE (calibration / more training helps). Errors far
      from it = CONFIDENTLY WRONG (a different signal is needed, not more of the same).
  (2) STRUCTURE   -- cross-validated AUC of a classifier predicting "is this edge an
      error" from its features. High AUC => errors have a feature signature => a targeted
      fix exists. ~0.5 => not feature-separable => the honest lever is SOTA training.
  (3) CLUSTERS    -- KMeans on the error edges' features (+ silhouette). A tight, large
      cluster with a clear centroid = a concrete failure mode to attack; its merge/split
      composition + dominant features name the modification.

Free: reuses the saved Volume affinities/LSD/JEPA -- no GPU, no retraining.
"""
import numpy as np
import agglomerate as AG


def _clfs(X, ysame, names):
    """Reproduce the two specialists exactly as agglomerate.run trains them, plus the
    standardization, so the confidence we read here matches the production decision."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Ai = [names.index(n) for n in names if n in AG.ATTRACT]
    Bi = [names.index(n) for n in names if n in AG.REPEL]
    clfA = LogisticRegression(class_weight="balanced", max_iter=300).fit((X[:, Ai] - mu[Ai]) / sd[Ai], ysame)
    clfB = LogisticRegression(class_weight="balanced", max_iter=300).fit((X[:, Bi] - mu[Bi]) / sd[Bi], 1 - ysame)

    def conf(feats):                                       # signed merge weight (pre-bias)
        if not len(feats):
            return np.zeros(0)
        pA = clfA.predict_proba((feats[:, Ai] - mu[Ai]) / sd[Ai])[:, 1]
        pB = clfB.predict_proba((feats[:, Bi] - mu[Bi]) / sd[Bi])[:, 1]
        p = np.clip(pA * (1 - pB), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    return conf


def run(train_affs, train_segs, eval_affs, eval_segs, ctx, thr_over=0.9, seed_q=0.6, merge_bias=1.25,
        train_lsds=None, eval_lsds=None, train_jepas=None, eval_jepas=None):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import roc_auc_score, silhouette_score
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return {"error": f"sklearn unavailable: {e}"}
    SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT); LONG = OFFS[ns:]

    def build(affs, lsds, jepas, segs):
        """Per subvol: local edges (pairs, feats) + per-edge GT-same label."""
        rows = []
        for i, aff in enumerate(affs):
            frags = AG._relabel(AG._oversegment(aff, SHORT, thr_over, seed_q))
            lp, lf, _lifp, _liff, names = AG._rag(frags, aff, SHORT, LONG,
                                                  lsds[i] if lsds else None,
                                                  jepas[i] if jepas else None)
            if not lp:
                rows.append((None, names)); continue
            fg = AG._frag_gt(frags, segs[i])
            ys = np.array([1 if fg[a] == fg[b] and fg[a] != 0 else 0 for a, b in lp])
            small = np.array([min(int((frags == a).sum()), int((frags == b).sum())) for a, b in lp])
            rows.append(({"feats": lf, "ysame": ys, "small": small}, names))
        return rows

    tr = build(train_affs, train_lsds, train_jepas, train_segs)
    ev = build(eval_affs, eval_lsds, eval_jepas, eval_segs)
    names = next((n for r, n in tr if r is not None), None) or \
            next((n for r, n in ev if r is not None), ["mean_short", "mean_long", "log_contact", "log_min_size"])

    # train specialists on TRAIN edges (as production does)
    Xtr = np.concatenate([r["feats"] for r, _ in tr if r is not None]) if any(r for r, _ in tr) else None
    ytr = np.concatenate([r["ysame"] for r, _ in tr if r is not None]) if Xtr is not None else None
    if Xtr is None or len(np.unique(ytr)) < 2:
        return {"error": "degenerate/empty train edges"}
    conf = _clfs(Xtr, ytr, names)

    # score the HELD-OUT eval edges
    Xev = np.concatenate([r["feats"] for r, _ in ev if r is not None])
    yev = np.concatenate([r["ysame"] for r, _ in ev if r is not None])
    small = np.concatenate([r["small"] for r, _ in ev if r is not None])
    w = conf(Xev) + merge_bias                             # signed decision weight at the operating point
    pred_same = (w > 0).astype(int)
    margin = np.abs(w)                                     # distance from the merge/split boundary
    err = pred_same != yev
    false_merge = (pred_same == 1) & (yev == 0)            # merged two DIFFERENT neurons
    false_split = (pred_same == 0) & (yev == 1)            # kept two SAME-neuron fragments apart
    n = len(yev)

    def _summ(mask):
        return {"n": int(mask.sum()),
                "mean_margin": round(float(margin[mask].mean()), 4) if mask.any() else None,
                "median_margin": round(float(np.median(margin[mask])), 4) if mask.any() else None}

    # (1) CONFIDENCE: are errors low-margin (unsure) or high-margin (confidently wrong)?
    hi = float(np.median(margin))                          # split errors by their own median margin
    confident_err = err & (margin > hi)
    confidence = {
        "eval_edges": int(n), "error_rate": round(float(err.mean()), 4),
        "correct": _summ(~err), "all_errors": _summ(err),
        "false_merge": _summ(false_merge), "false_split": _summ(false_split),
        "frac_errors_high_confidence": round(float(confident_err.sum() / max(1, err.sum())), 4),
        "interpretation": ("errors sit CLOSER to the decision boundary than correct edges "
                           "-> mostly low-confidence (calibration/more training helps)"
                           if err.any() and margin[err].mean() < margin[~err].mean() else
                           "errors are as/more confident than correct edges "
                           "-> confidently wrong (needs a different signal, not more training)")}

    # (2) STRUCTURE: can we PREDICT which edges are errors from their features? (CV AUC)
    structure = {"note": "AUC of predicting is-error from edge features; >~0.65 => targetable structure"}
    if err.sum() >= 10 and (~err).sum() >= 10:
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=0)
        try:
            auc = cross_val_score(rf, Xev, err.astype(int), cv=5, scoring="roc_auc")
            structure["cv_auc_mean"] = round(float(auc.mean()), 4)
            structure["cv_auc_std"] = round(float(auc.std()), 4)
        except Exception as e:
            structure["cv_auc_error"] = str(e)
        # univariate: which single feature best flags an error? (names the signature)
        uni = {}
        for j, nm in enumerate(names):
            try:
                a = roc_auc_score(err.astype(int), Xev[:, j])
                uni[nm] = round(float(max(a, 1 - a)), 4)    # direction-agnostic separability
            except Exception:
                uni[nm] = None
        structure["per_feature_auc"] = dict(sorted(uni.items(), key=lambda kv: -(kv[1] or 0)))
        a = structure.get("cv_auc_mean", 0.5)
        structure["verdict"] = ("STRUCTURED errors (feature-separable) -> build a targeted specialist / add the "
                                "top feature's signal" if a >= 0.65 else
                                "DIFFUSE errors (not feature-separable) -> the honest lever is heavier SOTA training")
    else:
        structure["verdict"] = "too few errors to test structure"

    # (3) CLUSTERS: group the error edges in feature space; name each failure mode
    clusters = {"note": "KMeans on error-edge features; silhouette>~0.3 => genuine grouping"}
    Xe = Xev[err]
    if len(Xe) >= 12:
        Z = StandardScaler().fit_transform(Xe)
        best = None
        for k in (2, 3, 4):
            if len(Xe) <= k:
                continue
            km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(Z)
            try:
                sil = silhouette_score(Z, km.labels_)
            except Exception:
                sil = -1
            if best is None or sil > best[0]:
                best = (sil, k, km)
        if best:
            sil, k, km = best
            clusters["best_k"] = int(k); clusters["silhouette"] = round(float(sil), 4)
            fm_e = false_merge[err]; fs_e = false_split[err]; sm_e = small[err]
            cl = []
            for c in range(k):
                m = km.labels_ == c
                centroid = {nm: round(float(Xe[m, j].mean()), 3) for j, nm in enumerate(names)}
                cl.append({"cluster": c, "size": int(m.sum()),
                           "false_merge": int(fm_e[m].sum()), "false_split": int(fs_e[m].sum()),
                           "median_small_frag_vox": int(np.median(sm_e[m])),
                           "centroid": centroid})
            clusters["clusters"] = sorted(cl, key=lambda d: -d["size"])
            clusters["verdict"] = ("clustered failure modes present (silhouette high) -> the largest cluster's "
                                   "merge/split mix + centroid names the next fix"
                                   if sil >= 0.3 else
                                   "errors do not cluster tightly -> no single dominant failure mode")
    else:
        clusters["verdict"] = "too few error edges to cluster"

    # dominant residual error type (matches the VOI split/merge story)
    dom = ("split-dominated (over-seg): more false_split -> improve over-seg / attraction"
           if false_split.sum() > false_merge.sum() else
           "merge-dominated (under-seg): more false_merge -> stronger repulsion / specialist B")
    return {"method": "edge-decision error debug (confidence + structure + clustering) on saved checkpoints",
            "features": names, "merge_bias": merge_bias, "thr_over": thr_over, "seed_q": seed_q,
            "dominant_error": dom,
            "confidence": confidence, "structure": structure, "clusters": clusters}
