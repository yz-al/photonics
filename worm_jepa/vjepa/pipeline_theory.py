"""
Non-ODE analytical structure for the segmentation + agglomeration pipeline. Theory about
the quantities we actually measure. Five toolkits, run in order (1-2 load-bearing):

  1. PERCOLATION   -- the agglomeration threshold is a phase transition.
  2. EXTREME VALUE -- ERL is a minimum/worst-case statistic; report the tail, not the mean.
  3. SPECTRAL      -- RAG Laplacian spectral gap = how separable the correct partition is.
  4. SCALE-SPACE   -- which merges are below the imaging resolution limit (unfixable).
  5. DISCRETE      -- per-step error probability vs ERL; independent or CLUSTERED errors.

RULES honored: report distributions not means; sweep widely; call out imaging/signal
limits plainly; never claim an improvement below the section-2 resampling variance.
"""
import numpy as np
from collections import defaultdict


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a


def _components_at(pairs, weights, nf, thr):
    parent = list(range(nf))
    for (a, b), w in zip(pairs, weights):
        if w >= thr:
            ra, rb = _find(parent, a), _find(parent, b)
            if ra != rb:
                parent[rb] = ra
    lab = np.array([_find(parent, i) for i in range(nf)])
    _, inv = np.unique(lab, return_inverse=True)
    return inv


# ---------------------------------------------------------------------------- 1. PERCOLATION
def percolation(subs, ctx, nthr=41):
    AG = ctx["AG"]; SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT); LONG = OFFS[ns:]
    erl = ctx["erl_proxy"]; seg_metrics = ctx["seg_metrics"]
    thrs = np.linspace(0.05, 0.98, nthr)
    lc = np.zeros(nthr); ncomp = np.zeros(nthr); merr = np.zeros(nthr); serr = np.zeros(nthr)
    erls = np.zeros(nthr); allsizes = defaultdict(list)
    for aff, seg in subs:
        frags = AG._relabel(AG._oversegment(aff, SHORT, 0.6))
        lp, lf, _lifp, _liff, _n = AG._rag(frags, aff, SHORT, LONG)
        if not lp:
            continue
        nf = int(frags.max()) + 1
        w = lf[:, 0]                                       # mean short-range affinity per edge
        fsz = np.bincount(frags.ravel(), minlength=nf)
        for ti, thr in enumerate(thrs):
            inv = _components_at(lp, w, nf, thr)
            comp_vox = np.bincount(inv, weights=fsz[:len(inv)] if len(inv) == nf else None)
            comp_vox = np.array([fsz[inv == c].sum() for c in range(inv.max() + 1)])
            lc[ti] += comp_vox.max() / fsz.sum()          # largest component (voxel frac)
            ncomp[ti] += inv.max() + 1
            lab = inv[frags]
            v = seg_metrics(lab, seg)
            # merge/split via contingency
            mm = ss = 0
            for g in np.unique(seg):
                if g == 0:
                    continue
                pv, pc = np.unique(lab[seg == g], return_counts=True)
                if (pc >= 0.1 * pc.sum()).sum() >= 2:
                    ss += 1
            for p in np.unique(lab):
                gl = seg[lab == p]; gv, gc = np.unique(gl[gl > 0], return_counts=True) if (gl > 0).any() else ([], [])
                if len(gc) and (gc >= 0.1 * max(1, gc.sum())).sum() >= 2:
                    mm += 1
            merr[ti] += mm; serr[ti] += ss; erls[ti] += erl(lab, seg)
            if abs(thr - 0.5) < 0.06:
                allsizes["near_mid"].extend(list(comp_vox))
    n = max(1, len(subs))
    lc /= n; ncomp /= n; merr /= n; serr /= n; erls /= n
    dlc = np.gradient(lc, thrs)
    ci = int(np.argmax(np.abs(dlc)))                       # steepest = critical
    thr_c = float(thrs[ci])
    # transition width: threshold range where largest comp goes 0.1 -> 0.9 of its span
    lo, hi = lc.min(), lc.max(); span = hi - lo + 1e-9
    within = thrs[(lc >= lo + 0.1 * span) & (lc <= lo + 0.9 * span)]
    width = float(within.max() - within.min()) if len(within) else float("nan")
    # operating point = threshold that maximises ERL, EXCLUDING the degenerate one-blob
    # regime (largest_comp ~ 1), where ERL is reward-hacked by a single giant merged label.
    nondeg = lc < 0.95
    op = int(np.arange(nthr)[nondeg][np.argmax(erls[nondeg])]) if nondeg.any() else int(np.argmax(erls))
    thr_op = float(thrs[op])
    # power-law exponent of component sizes near criticality
    sizes = np.array([s for s in allsizes["near_mid"] if s > 0])
    expo = None
    if len(sizes) > 30:
        s = np.sort(sizes); xmin = np.percentile(s, 50); s = s[s >= xmin]
        if len(s) > 10:
            expo = round(float(1 + len(s) / np.sum(np.log(s / xmin + 1e-9))), 3)   # MLE power-law alpha
    return {"critical_threshold": round(thr_c, 3), "transition_width": round(width, 3),
            "operating_threshold_maxERL": round(thr_op, 3),
            "distance_op_from_critical": round(abs(thr_op - thr_c), 3),
            "max_dLargestComp_dThr": round(float(np.abs(dlc).max()), 3),
            "size_dist_powerlaw_alpha_near_mid": expo,
            "sweep": [{"thr": round(float(t), 3), "largest_comp_frac": round(float(lc[i]), 3),
                       "merge_err": round(float(merr[i]), 2), "split_err": round(float(serr[i]), 2),
                       "ERL": round(float(erls[i]), 1)} for i, t in enumerate(thrs)][::4],
            "reading": ("operating point within one transition-width of criticality => unstable, "
                        "high-variance merges" if abs(thr_op - thr_c) <= width else
                        "operating point clear of criticality => stable regime")}


# ------------------------------------------------------------------------- 2. EXTREME VALUE
def _run_lengths(lab, seg, merge_cover=0.15):
    """Correct-run lengths along GT skeletons + termination type (merge/split)."""
    from skimage.morphology import skeletonize
    ids = [i for i in np.unique(seg) if i != 0 and (seg == i).sum() >= 60]
    gt_size = {g: int((seg == g).sum()) for g in ids}
    merged = set()
    for pv in np.unique(lab):
        if pv == 0:
            continue
        gv = seg[lab == pv]
        if sum((gv == g).sum() >= merge_cover * gt_size[g] for g in np.unique(gv) if g in gt_size) >= 2:
            merged.add(int(pv))
    runs_m, runs_s = [], []
    for g in ids:
        sk = skeletonize(seg == g); pl = lab[sk]
        if pl.size < 3:
            continue
        pv, pc = np.unique(pl, return_counts=True)
        for val, cnt in zip(pv, pc):
            (runs_m if int(val) in merged else runs_s).append(int(cnt))
    return runs_m, runs_s


def extreme_value(subs, ctx):
    from scipy import stats
    AG = ctx["AG"]; SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT); LONG = OFFS[ns:]
    mws = ctx["mutex_watershed"]; erl = ctx["erl_proxy"]
    per_vol_erl = []; RM, RS = [], []
    for aff, seg in subs:
        lab = mws(np.clip(aff, 0, 1), OFFS, ns)
        per_vol_erl.append(erl(lab, seg))
        rm, rs = _run_lengths(lab, seg); RM += rm; RS += rs

    def fit_tail(x):
        x = np.array([v for v in x if v > 0], float)
        if len(x) < 20:
            return {"n": int(len(x)), "note": "too few"}
        thr = np.percentile(x, 70); ex = x[x >= thr] - thr
        try:
            c, _, sc = stats.genpareto.fit(ex, floc=0)
        except Exception:
            c, sc = float("nan"), float("nan")
        return {"n": int(len(x)), "median": round(float(np.median(x)), 1),
                "p95": round(float(np.percentile(x, 95)), 1), "max": int(x.max()),
                "gpd_shape_xi": round(float(c), 3),
                "tail": ("heavy (xi>0): dominated by rare long runs" if c > 0.1 else
                         "bounded/light (xi<=0)")}
    # ERL resampling variance (bootstrap over subvols)
    rng = np.random.default_rng(0); boot = []
    for _ in range(500):
        idx = rng.integers(0, len(per_vol_erl), len(per_vol_erl))
        boot.append(float(np.mean([per_vol_erl[i] for i in idx])))
    boot = np.array(boot)
    return {"ERL_mean": round(float(np.mean(per_vol_erl)), 2),
            "ERL_resample_CI95": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
            "ERL_resample_std": round(float(boot.std()), 3),
            "split_initiated_runs": fit_tail(RS), "merge_initiated_runs": fit_tail(RM),
            "reading": "compare model-version ERL differences against ERL_resample_std; smaller = noise"}


# ----------------------------------------------------------------------------- 3. SPECTRAL
def spectral(subs, ctx):
    from scipy.linalg import eigh
    AG = ctx["AG"]; SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT); LONG = OFFS[ns:]
    gaps = []; frag_gt_purity = []
    for aff, seg in subs[:4]:
        frags = AG._relabel(AG._oversegment(aff, SHORT, 0.6))
        lp, lf, _a, _b, _n = AG._rag(frags, aff, SHORT, LONG)
        nf = int(frags.max()) + 1
        if not lp or nf < 3:
            continue
        A = np.zeros((nf, nf))
        for (a, b), w in zip(lp, lf[:, 0]):
            A[a, b] = A[b, a] = w
        D = np.diag(A.sum(1)); L = D - A
        deg = np.clip(A.sum(1), 1e-6, None); Dm = np.diag(1 / np.sqrt(deg))
        Ln = Dm @ L @ Dm
        ev = np.sort(np.clip(eigh(Ln, eigvals_only=True), 0, 2))
        if len(ev) > 2:
            gaps.append(float(ev[2] - ev[1]))              # spectral gap (Fiedler region)
    return {"mean_spectral_gap": round(float(np.mean(gaps)), 4) if gaps else None,
            "spectral_gap_p10": round(float(np.percentile(gaps, 10)), 4) if gaps else None,
            "reading": ("small spectral gap => the correct partition is not well separated in the "
                        "affinities; NO thresholding rule finds it reliably (a SIGNAL problem, not an "
                        "agglomeration-parameter problem)")}


# --------------------------------------------------------------------------- 4. SCALE-SPACE
def scale_space(subs, ctx, voxel_xy=1.0, section_z=1.0):
    """Per merge error, local geometry vs resolution. voxel_xy/section_z in voxel units (our
    subvols are already voxel-sampled; ratios are relative to 1 voxel / 1 section)."""
    from scipy.ndimage import distance_transform_edt
    AG = ctx["AG"]; SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT)
    mws = ctx["mutex_watershed"]
    below = 0; total = 0; calibres = []
    for aff, seg in subs:
        lab = mws(np.clip(aff, 0, 1), OFFS, ns)
        for p in np.unique(lab):
            if p == 0:
                continue
            gl = seg[lab == p]; gv, gc = np.unique(gl[gl > 0], return_counts=True) if (gl > 0).any() else (np.array([]), np.array([]))
            spanned = gv[gc >= 0.1 * max(1, gc.sum())]
            if len(spanned) < 2:
                continue
            total += 1
            # calibre of the smaller spanned neuron = its min in-plane thickness (EDT max is radius)
            cals = []
            for g in spanned:
                edt = distance_transform_edt(seg == g)
                cals.append(2 * float(edt.max()))          # diameter in voxels
            cal = min(cals); calibres.append(cal)
            if cal < 2.0:                                  # smaller neurite < ~2 voxels -> imaging-limited
                below += 1
    return {"n_merge_errors": total, "n_below_resolution(<2vox_calibre)": below,
            "frac_merges_below_resolution": round(below / max(1, total), 3),
            "median_smaller_neurite_calibre_vox": round(float(np.median(calibres)), 2) if calibres else None,
            "reading": ("MODEL-limited: ~none of the merges are below the resolution limit "
                        "(the neurons are resolvable) -> training/model CAN fix them"
                        if (below / max(1, total)) < 0.2 else
                        "IMAGING-limited: a large fraction of merges are below the resolution limit -> "
                        "no training fixes them")}


# ----------------------------------------------------------------------------- 5. DISCRETE
def discrete(subs, ctx):
    AG = ctx["AG"]; SHORT = ctx["SHORT"]; OFFS = ctx["OFFS"]; ns = len(SHORT)
    mws = ctx["mutex_watershed"]; erl = ctx["erl_proxy"]; gt_aff = ctx["gt_affinity"]
    perr = []; erls = []; autocorr = []
    for aff, seg in subs:
        ga, val = gt_aff(seg, OFFS)
        wrong = ((aff[:ns] > 0.5) != (ga[:ns] > 0.5)) & (val[:ns] > 0)
        p = float(wrong.sum() / max(1, (val[:ns] > 0).sum())); perr.append(p)
        lab = mws(np.clip(aff, 0, 1), OFFS, ns); erls.append(erl(lab, seg))
        # spatial autocorrelation of the error map (lag-1, in-plane) vs product of means
        e = wrong.any(0).astype(np.float32)
        if e.mean() > 0:
            a = e[:, :-1, :]; b = e[:, 1:, :]
            ac = float(((a * b).mean() - e.mean() ** 2) / (e.var() + 1e-9))
            autocorr.append(ac)
    p = float(np.mean(perr)); measured = float(np.mean(erls)); pred = 1.0 / max(p, 1e-6)
    ratio = measured / pred
    return {"per_step_error_prob_p": round(p, 4), "predicted_ERL_1_over_p": round(pred, 1),
            "measured_ERL": round(measured, 1), "measured_over_predicted": round(ratio, 3),
            "error_spatial_autocorr_lag1": round(float(np.mean(autocorr)), 3) if autocorr else None,
            "reading": ("errors are spatially CLUSTERED (high lag-1 autocorrelation); trace-terminating "
                        "errors are also much rarer than raw affinity errors (ERL >> 1/p) -> a few bad "
                        "regions dominate. TARGET the bad regions, not uniform improvement."
                        if (autocorr and np.mean(autocorr) > 0.2) else
                        "errors roughly independent/spread -> uniform improvement is the route")}


def run(subs, ctx):
    return {"1_percolation": percolation(subs, ctx),
            "2_extreme_value": extreme_value(subs, ctx),
            "3_spectral": spectral(subs, ctx),
            "4_scale_space": scale_space(subs, ctx),
            "5_discrete": discrete(subs, ctx)}
