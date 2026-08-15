"""
remaining_signal_probe.py -- "go find" the ~51% headroom above the 49% perturbation
model. Two physiologically-motivated hypotheses for where it hides, both tested
rigorously, both HONEST NEGATIVES. Recorded so the dead ends are not re-hunted.

Context: perturbation_improved.py reaches 49% of the matched ceiling on held-out
perturbation prediction. The remaining variance is reproducible (it is in the ceiling)
but not captured by the wired connectome. Where is it?

Hypothesis 1 -- WIRELESS (extrasynaptic) signaling.
  The anatomical connectome is synaptic + gap-junction. Add the Bentley 2016
  extrasynaptic connectomes (monoamines + neuropeptides) as extra propagation channels,
  CV-fit their weights on held-out perturbations.
  RESULT: no gain (wired-only 50% -> +wireless 49%). The extrasynaptic map -- itself built
  from receptor co-expression, i.e. anatomical -- does not predict WHERE the functional
  wireless signal goes. NEGATIVE.

Hypothesis 2 -- the headroom is PEPTIDERGIC, isolable via the unc-31 mutant.
  funatlas ships an unc31 matrix: unc-31/CAPS mutants cannot release dense-core vesicles,
  so unc31 is the WIRED-ONLY response. If the wt headroom were peptidergic, the connectome
  should predict unc31 BETTER than wt.
  RESULT (paired, SAME perturbation columns measured in both -- the naive unpaired
  comparison is a confound of different columns/ceilings): the connectome predicts wt
  slightly BETTER than unc31 (0.21 vs 0.16, Wilcoxon n.s.). The headroom does NOT become
  more connectome-predictable when peptidergic release is removed. NEGATIVE.

Conclusion: the ~51% is not recovered by any static connectome we have -- wired,
extrasynaptic, or peptide-release-removed. It is genuinely different signal (per-animal
state / dynamics), consistent with the informational-cap reading. The 49% stands; the
next points are not in this aggregated data.

    python remaining_signal_probe.py     # writes remaining_signal.json
Requires wormneuroatlas, c302, scikit-learn.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_es(N, n2i):
    import wormneuroatlas, pandas as pd
    dd = os.path.join(os.path.dirname(wormneuroatlas.__file__), "data")
    def m2i(name):
        name = str(name).strip()
        name = {"AWCL": "AWCOFF", "AWCR": "AWCON"}.get(name, name)
        return n2i.get(name)
    def build(fn):
        df = pd.read_csv(os.path.join(dd, fn), index_col=0); W = np.zeros((N, N))
        for s, tgt in zip(df.index, df[df.columns[0]]):
            i, j = m2i(s), m2i(tgt)
            if i is not None and j is not None:
                W[i, j] += 1.0
        return W
    return build("esconnectome_monoamines_Bentley_2016.csv"), build("esconnectome_neuropeptides_Bentley_2016.csv")


def run():
    import h5py, wormneuroatlas
    from scipy.stats import pearsonr, wilcoxon
    from sklearn.isotonic import IsotonicRegression
    import perturbation_response as PR
    R, meas, Wc, Wg, ceil = PR.load(); N = PR.N
    p = os.path.join(os.path.dirname(wormneuroatlas.__file__), "data", "funatlas.h5")
    h = h5py.File(p, "r")
    ids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["neuron_ids"][:]]
    n2i = {n: i for i, n in enumerate(ids)}
    Wm, Wp = _load_es(N, n2i)
    nrm = lambda W: W / (W.sum(1, keepdims=True) + 1e-9)

    def prop(g, gap, bm=0.0, bp=0.0):
        A = nrm(Wc) + gap * nrm(Wg) + bm * nrm(Wm) + bp * nrm(Wp)
        A = g * A / (np.abs(np.linalg.eigvals(A)).max() + 1e-12)
        return np.linalg.inv(np.eye(N) - A)

    cols = [j for j in range(N) if (meas[:, j] & (np.arange(N) != j)).sum() >= 10]

    def cs(P, js, iso=None):
        rs = []
        for j in js:
            ii = np.where(meas[:, j])[0]; ii = ii[ii != j]; a = P[ii, j]; b = R[ii, j]
            if iso is not None:
                a = iso.predict(a)
            if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                rs.append(pearsonr(a, b)[0])
        return np.mean(rs) if rs else 0.0

    def fiso(P, js):
        xa, ya = [], []
        for j in js:
            ii = np.where(meas[:, j])[0]; ii = ii[ii != j]; xa += list(P[ii, j]); ya += list(R[ii, j])
        return IsotonicRegression(out_of_bounds="clip").fit(xa, ya)

    # H1: wired vs +wireless, CV
    grid = [(0.9, 2.0, bm, bp) for bm in (0, 1, 2) for bp in (0, 1, 2)]
    def cv(restrict):
        vals = []
        for seed in range(3):
            rng = np.random.RandomState(seed); perm = rng.permutation(cols); hh = len(perm) // 2; t = []
            for tr, te in [(perm[:hh], perm[hh:]), (perm[hh:], perm[:hh])]:
                best = (-9, None, None)
                for pr in grid:
                    if restrict and (pr[2] or pr[3]):
                        continue
                    P = prop(*pr); iso = fiso(P, tr); s = cs(P, tr, iso)
                    if s > best[0]:
                        best = (s, P, iso)
                t.append(cs(best[1], te, best[2]))
            vals.append(np.mean(t))
        return float(np.mean(vals))
    wired = cv(True); full = cv(False)

    # H2: wt vs unc31, PAIRED on same columns
    P = prop(0.9, 2.0)
    Ru = h["unc31/dFF"][:]; occu = h["unc31/occ1"][:]; mu = (occu >= 4) & np.isfinite(Ru)
    def rcol(RR, mm, j):
        ii = np.where(mm[:, j])[0]; ii = ii[ii != j]
        if len(ii) < 10:
            return None
        a = P[ii, j]; b = RR[ii, j]
        return pearsonr(a, b)[0] if (np.std(a) > 1e-9 and np.std(b) > 1e-9) else None
    both = [j for j in range(N) if (meas[:, j] & (np.arange(N) != j)).sum() >= 10
            and (mu[:, j] & (np.arange(N) != j)).sum() >= 10]
    wv, uv = [], []
    for j in both:
        rw, ru = rcol(R, meas, j), rcol(Ru, mu, j)
        if rw is not None and ru is not None:
            wv.append(rw); uv.append(ru)
    wv, uv = np.array(wv), np.array(uv)
    pu = float(wilcoxon(uv, wv, alternative="greater")[1]) if len(wv) > 5 else float("nan")
    h.close()
    return {
        "baseline_49pct_ref": "perturbation_improved.py",
        "H1_wireless": {"wired_only_pct": round(100 * wired / ceil, 1),
                        "plus_wireless_pct": round(100 * full / ceil, 1),
                        "helps": bool(full > wired + 0.01)},
        "H2_unc31_paired": {"n_paired_columns": len(wv),
                            "connectome_vs_wt_r": round(float(wv.mean()), 3),
                            "connectome_vs_unc31_r": round(float(uv.mean()), 3),
                            "unc31_better_p": pu,
                            "peptidergic_explains_headroom": bool(uv.mean() > wv.mean() and pu < 0.05)},
        "both_hypotheses_negative": bool(not (full > wired + 0.01) and not (uv.mean() > wv.mean() and pu < 0.05)),
        "conclusion": "the ~51% headroom is not recovered by any static connectome (wired, extrasynaptic, or "
                      "peptide-release-removed); it needs per-animal state/dynamics this aggregated data lacks.",
    }


if __name__ == "__main__":
    r = run()
    print("=== FIND THE REMAINING SIGNAL above 49% -- two hypotheses, both negative ===\n")
    h1 = r["H1_wireless"]; h2 = r["H2_unc31_paired"]
    print("H1 wireless (extrasynaptic connectome as extra channels, CV):")
    print("   wired-only %.0f%%  ->  +wireless %.0f%%   helps: %s" % (h1["wired_only_pct"], h1["plus_wireless_pct"], h1["helps"]))
    print("\nH2 unc-31 wired-only response (paired, same %d columns):" % h2["n_paired_columns"])
    print("   connectome vs wt r=%.3f   vs unc31 r=%.3f   unc31-better p=%.2f   peptidergic explains headroom: %s"
          % (h2["connectome_vs_wt_r"], h2["connectome_vs_unc31_r"], h2["unc31_better_p"], h2["peptidergic_explains_headroom"]))
    print("\nboth hypotheses negative: %s" % r["both_hypotheses_negative"])
    print(r["conclusion"])
    with open(os.path.join(HERE, "remaining_signal.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("\nwrote remaining_signal.json")
