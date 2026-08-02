"""
Beat SOTA by focusing on where it fails -- a menu of failure-targeted strategies
scored on the REAL segmentation metrics (VOI / adapted-Rand / ERL), not proxies.

What we learned about SOTA's failures (error composition):
  - it OVER-SEGMENTS: ~76% of its wrong edges are SPLITS, only ~24% merges;
  - ~79% of errors are on LONG-RANGE edges;
  - its errors are SELF-DETECTABLE (confidence |p-0.5| predicts them at AUC ~0.85).

So the strategies here attack exactly those weaknesses, and -- crucially -- reuse the
affinity maps segment3d ALREADY predicts for sota / jepa / raw on the eval subvolumes,
so almost no extra training: we just re-combine and re-agglomerate.

  baseline_sota      : SOTA affinities -> MWS (the reference to beat)
  merge_bias         : add a bias to SOTA's SHORT (attractive) affinities before MWS,
                       swept -- directly counters over-segmentation (more merging).
  ens_sota_jepa      : on the edges SOTA is UNSURE about (low |p-0.5|), average in the
                       JEPA-decoder affinities (diversity where SOTA fails).
  ens_sota_raw       : same, blending the from-scratch raw head.
  longrange_from_jepa: keep SOTA's short-range affinities, REPLACE the long-range ones
                       with the JEPA head's -- targets the 79%-of-errors long-range band.

Each returns mean VOI / Rand / ERL over the eval subs, plus the delta vs baseline SOTA.
MWS is the cost driver, so the number of strategies x eval-subs is kept modest.
"""
import numpy as np


def run(full_preds, te_subs, te_gt, ctx, biases=(0.05, 0.1, 0.2), max_subs=2):
    """full_preds[source] = list of predicted affinity arrays (NAFF,Z,H,W) per eval sub.
    ctx carries the segment3d helpers: mutex_watershed, seg_metrics, erl_proxy, OFFS, SHORT."""
    mws = ctx["mutex_watershed"]; seg_metrics = ctx["seg_metrics"]; erl = ctx["erl_proxy"]
    OFFS = ctx["OFFS"]; ns = len(ctx["SHORT"]); NAFF = len(OFFS)
    if "sota" not in full_preds or "jepa" not in full_preds:
        return {"error": "need sota + jepa affinity predictions"}
    nsub = min(max_subs, len(te_subs))
    segs = [te_subs[j][1] for j in range(nsub)]

    def score(aff_list):
        vs, rs, es = [], [], []
        for aff, seg in zip(aff_list, segs):
            lab = mws(np.clip(aff, 0, 1), OFFS, ns)
            v, r = seg_metrics(lab, seg); vs.append(v); rs.append(r); es.append(erl(lab, seg))
        return {"VOI": round(float(np.mean(vs)), 4), "adapted_rand_error": round(float(np.mean(rs)), 4),
                "ERL": round(float(np.mean(es)), 4)}

    sota = [full_preds["sota"][j] for j in range(nsub)]
    jepa = [full_preds["jepa"][j] for j in range(nsub)]
    raw = [full_preds["raw"][j] for j in range(nsub)] if "raw" in full_preds else None

    base = score(sota)

    def merge_bias(b):
        out = []
        for a in sota:
            a2 = a.copy(); a2[:ns] = np.clip(a2[:ns] + b, 0, 1)   # push short (attractive) up -> merge more
            out.append(a2)
        return out

    def ens_lowconf(other):
        out = []
        for a, o in zip(sota, other):
            conf = np.abs(a - 0.5)
            thr = np.quantile(conf, 0.30)                        # bottom 30% conf = self-detected likely errors
            g = (conf <= thr).astype(np.float32)                 # gate: 1 where SOTA unsure
            out.append((1 - g) * a + g * 0.5 * (a + o))          # average in the other source there
        return out

    def longrange_from(other):
        out = []
        for a, o in zip(sota, other):
            a2 = a.copy(); a2[ns:] = o[ns:]                      # replace long-range band with other head
            out.append(a2)
        return out

    results = {"baseline_sota": base}
    # merge-bias sweep (attack over-segmentation)
    mb = {f"merge_bias_{b}": score(merge_bias(b)) for b in biases}
    results.update(mb)
    # confidence-gated ensembles (diversity where SOTA is unsure)
    results["ens_sota_jepa"] = score(ens_lowconf(jepa))
    if raw is not None:
        results["ens_sota_raw"] = score(ens_lowconf(raw))
    # long-range specialist swap (attack the 79%-long-range errors)
    results["longrange_from_jepa"] = score(longrange_from(jepa))

    # deltas vs baseline (VOI/Rand lower=better, ERL higher=better)
    for k, v in results.items():
        v["dVOI_vs_sota"] = round(v["VOI"] - base["VOI"], 4)      # negative = better
        v["dRand_vs_sota"] = round(v["adapted_rand_error"] - base["adapted_rand_error"], 4)
        v["dERL_vs_sota"] = round(v["ERL"] - base["ERL"], 4)      # positive = better
    cand = {k: v for k, v in results.items() if k != "baseline_sota"}
    best_voi = min(cand.items(), key=lambda kv: kv[1]["VOI"])
    best_erl = max(cand.items(), key=lambda kv: kv[1]["ERL"])
    # A REAL win must not buy VOI/Rand by creating compounding merges: require VOI and
    # Rand no worse than SOTA AND ERL within 5% of SOTA's (errors compound -> ERL is the
    # veto). merge-bias that tanks ERL is exactly the trap this guards against.
    erl_floor = base["ERL"] * 0.95
    safe = {k: v for k, v in cand.items()
            if v["VOI"] <= base["VOI"] and v["adapted_rand_error"] <= base["adapted_rand_error"]
            and v["ERL"] >= erl_floor}
    safe_best = min(safe.items(), key=lambda kv: kv[1]["VOI"]) if safe else None
    return {"method": "beat-sota failure-targeted strategies (real VOI/Rand/ERL)",
            "n_subs": nsub, "baseline_sota": base, "strategies": results,
            "best_by_voi": {"name": best_voi[0], "VOI": best_voi[1]["VOI"], "ERL": best_voi[1]["ERL"]},
            "best_by_erl": {"name": best_erl[0], "ERL": best_erl[1]["ERL"], "VOI": best_erl[1]["VOI"]},
            "safe_best": ({"name": safe_best[0], **{m: safe_best[1][m] for m in ("VOI", "adapted_rand_error", "ERL")}}
                          if safe_best else None),
            "verdict": (f"{safe_best[0]} beats SOTA without a merge tradeoff"
                        if safe_best else
                        "no strategy beats SOTA on VOI+Rand without sacrificing ERL (compounding merges)")}
