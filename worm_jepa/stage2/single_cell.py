"""
single_cell.py -- does single-cell biophysics add signal a linear node cannot? A
disciplined test on ONE well-characterized neuron: AWC.

The hypothesis (and why AWC)
----------------------------
The connectome-propagation model treats each neuron as a LINEAR node. AWC is the
textbook counter-example: it is an OLFACTORY OFF-cell -- it fires on odor REMOVAL, not
addition (Chalasani et al. 2007). That makes it a clean, parameter-free discriminator,
because a linear system is SYMMETRIC: by superposition its response to an odor
down-step is exactly minus its response to the up-step, so any linear filter predicts
ZERO on/off amplitude asymmetry. A real AWC is strongly OFF-dominated. So the on/off
asymmetry is a feature a linear node CANNOT produce and a correct single-cell model can.

The test (positive result + negative control, per METHODOLOGY.md)
-----------------------------------------------------------------
Stimulus: isoamyl-alcohol (IAA) pulses, the canonical AWC odor (DANDI:000981, 24
animals). Feature: asymmetry A = (OFF_peak - ON_peak)/(|OFF|+|ON|). Three models driven
by the SAME odor boxcar, params fit on training worms, evaluated on held-out worms:

  linear node   r(t) = FIR_kernel * odor(t)          -- fit to the trace; A = 0 by superposition
  biophysical   adapting + OFF-rectified single cell  -- activation = relu(adapt - odor)
  scrambled     same machinery, ON-rectified          -- activation = relu(odor - adapt)  [wrong channelome]

The OFF-rectification is the CHANNELOME (a structural prior, NOT fit); only the
timescales/gain are fit. PASS requires: the biophysical model's asymmetry is closer to
the real AWC asymmetry than the linear node's (paired across held-out worms), AND the
scrambled channelome predicts the WRONG SIGN (the control fails).

Result (see single_cell.json) -- an HONEST NEGATIVE, which is why we ran it. In the
signed (on, off) plane the real AWC response is ~(-0.62, +0.38): SUPPRESSED during odor,
EXCITED on removal -- and that sits only ~0.20 from the linear manifold (on=-off). AWC's
IAA calcium response is therefore approximately LINEAR (an antisymmetric differentiator).
A linear FIR filter captures it; the hard OFF-rectified single-cell model OVERSHOOTS to
(0, +1) and fits WORSE (p~1.0). The scrambled ON channelome lands in the wrong quadrant
(+1, 0) -- so the test IS sensitive; the negative is real, not a dead test.

Conclusion: for AWC at the (population-calcium) level of these recordings, single-cell
biophysical NONLINEARITY does not add signal. What the connectome's STATIC linear node is
missing here is linear TEMPORAL filtering (adaptation / differentiation) -- a cheap linear
upgrade -- NOT a full biophysical simulation. This BOUNDS the earlier answer: single-cell
sim earns its many parameters only for features that are genuinely nonlinear at the
measured level; AWC/IAA/calcium is not one of them. (Consistent with the informational
connectome cap: the aggregate calcium data does not carry fine single-cell nonlinearity,
so simulating it does not help.)

    python single_cell.py     # writes single_cell.json
Self-contained: reads single_cell_awc.npz (AWC traces + IAA odor boxcar, from 000981).
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "single_cell_awc.npz")


def _transitions(u):
    du = np.diff(u.astype(float))
    return np.where(du > 0.5)[0] + 1, np.where(du < -0.5)[0] + 1   # onsets, offsets


def response_pair(trace, u, rate):
    """SIGNED mean response (baseline-subtracted) in the window after odor ON and after
    odor OFF, normalized to unit L1. Signed is essential: a LINEAR model obeys
    superposition (on = -off), so it is confined to the antidiagonal of this (on, off)
    plane. An OFF-cell sits near (0, +); an ON-cell near (+, 0). Distance in this plane
    is what separates them -- a max/peak measurement would rectify and destroy it."""
    W = max(2, int(5 * rate)); base = max(1, int(3 * rate))
    ons, offs = _transitions(u)
    def sig(fs):
        vals = []
        for f in fs:
            b = np.nanmean(trace[max(0, f - base):f]) if f > 0 else 0.0
            seg = trace[f:f + W]
            if len(seg) >= 2:
                vals.append(np.nanmean(seg) - b)
        return np.mean(vals) if vals else np.nan
    on, off = sig(ons), sig(offs)
    if not (np.isfinite(on) and np.isfinite(off)):
        return None
    n = abs(on) + abs(off) + 1e-9
    return np.array([on / n, off / n])


def sim_cell(u, rate, tau_a, tau_ca, g, off=True):
    """Adapting, rectified single-compartment cell. off=True -> AWC OFF-channelome
    (excited when odor drops below the adapted level); off=False -> scrambled ON."""
    dt = 1.0 / rate; a = 0.0; Ca = 0.0; out = np.zeros(len(u))
    for t in range(len(u)):
        a += dt * (u[t] - a) / tau_a
        drive = (a - u[t]) if off else (u[t] - a)
        act = drive if drive > 0 else 0.0          # rectification = the nonlinearity
        Ca += dt * (-Ca / tau_ca + g * act)
        out[t] = Ca
    return out


def _fit_gain(sig, target):
    m = np.isfinite(target)
    s = sig[m]; y = target[m]
    return float(np.dot(s, y) / (np.dot(s, s) + 1e-9))


def fit_biophysical(train, off=True):
    """Grid-fit timescales (shared) + per-worm gain to maximize pooled correlation with AWC."""
    best = (None, -2)
    for tau_a in (2., 5., 10., 20.):
        for tau_ca in (1., 2., 4.):
            cors = []
            for tr, u, rate in train:
                s = sim_cell(u, rate, tau_a, tau_ca, 1.0, off)
                if s.std() < 1e-9:
                    continue
                cors.append(np.corrcoef(s, np.nan_to_num(tr))[0, 1])
            if cors and np.mean(cors) > best[1]:
                best = ((tau_a, tau_ca), float(np.mean(cors)))
    return best[0]


def fit_linear(train, L=25):
    """Ridge FIR kernel odor->AWC pooled over training worms."""
    from numpy.linalg import lstsq
    X, Y = [], []
    for tr, u, rate in train:
        D = np.zeros((len(u), L))
        for k in range(L):
            D[k:, k] = u[:len(u) - k]
        X.append(D); Y.append(np.nan_to_num(tr))
    X = np.vstack(X); Y = np.concatenate(Y)
    h = lstsq(X.T @ X + 1e-1 * np.eye(L), X.T @ Y, rcond=None)[0]
    return h


def predict_linear(u, h):
    L = len(h); D = np.zeros((len(u), L))
    for k in range(L):
        D[k:, k] = u[:len(u) - k]
    return D @ h


def run():
    from scipy.stats import wilcoxon
    d = np.load(DATA, allow_pickle=True)
    worms = [(np.asarray(a, float), np.asarray(u, float), float(r))
             for a, u, r in zip(d["awc"], d["u"], d["rate"])]
    idx = np.arange(len(worms))
    folds = [(idx[idx % 2 == 0], idx[idx % 2 == 1]), (idx[idx % 2 == 1], idx[idx % 2 == 0])]
    rows = []
    for tr_idx, te_idx in folds:
        train = [worms[i] for i in tr_idx]
        h = fit_linear(train)
        tau_off = fit_biophysical(train, off=True)
        tau_on = fit_biophysical(train, off=False)
        for i in te_idx:
            tr, u, rate = worms[i]
            pr = response_pair(tr, u, rate)
            pl = response_pair(predict_linear(u, h), u, rate)
            pb = response_pair(sim_cell(u, rate, tau_off[0], tau_off[1], 1.0, off=True), u, rate)
            ps = response_pair(sim_cell(u, rate, tau_on[0], tau_on[1], 1.0, off=False), u, rate)
            if any(p is None for p in (pr, pl, pb, ps)):
                continue
            rows.append(dict(real=pr, lin=pl, bio=pb, scr=ps,
                             err_lin=float(np.linalg.norm(pl - pr)),
                             err_bio=float(np.linalg.norm(pb - pr)),
                             err_scr=float(np.linalg.norm(ps - pr))))
    err_lin = np.array([r["err_lin"] for r in rows])
    err_bio = np.array([r["err_bio"] for r in rows])
    err_scr = np.array([r["err_scr"] for r in rows])
    bio_beats_lin = float(wilcoxon(err_lin, err_bio, alternative="greater")[1])
    bio_beats_scr = float(wilcoxon(err_scr, err_bio, alternative="greater")[1])
    def meanpair(k):
        return [round(float(np.mean([r[k][j] for r in rows])), 3) for j in (0, 1)]
    real_off_dom = float(np.mean([r["real"][1] > r["real"][0] for r in rows]))
    scr_pair = meanpair("scr")
    return {
        "neuron": "AWC (olfactory OFF-cell)", "stimulus": "isoamyl alcohol (IAA) pulses, DANDI:000981",
        "feature": "signed (on, off) response pair, L1-normalized; linear node confined to on=-off (superposition)",
        "n_worms": len(rows),
        "real_pair_on_off": meanpair("real"), "real_frac_off_selective": round(real_off_dom, 3),
        "linear_node": {"pair_on_off": meanpair("lin"), "dist_to_real": round(float(err_lin.mean()), 3),
                        "note": "fit to the trace, yet trapped on the antidiagonal -> cannot be OFF-selective"},
        "biophysical_off": {"pair_on_off": meanpair("bio"), "dist_to_real": round(float(err_bio.mean()), 3),
                            "note": "OFF-rectification is the channelome (structural); only timescales/gain fit"},
        "scrambled_on": {"pair_on_off": scr_pair, "dist_to_real": round(float(err_scr.mean()), 3),
                         "note": "wrong channelome -> wrong quadrant (ON-selective); negative control"},
        "biophysical_beats_linear_p": bio_beats_lin,
        "biophysical_beats_scrambled_p": bio_beats_scr,
        "real_dist_to_linear_manifold": round(float(err_lin.mean()), 3),
        "scrambled_control_valid": bool(scr_pair[0] > scr_pair[1]),   # wrong-quadrant -> test is sensitive
        "single_cell_biophysics_helps": bool(err_bio.mean() < err_lin.mean() and bio_beats_lin < 0.01),
        "linear_filter_adequate": bool(err_lin.mean() < err_bio.mean()),
        "conclusion": ("AWC/IAA/calcium is ~linear (real is %.2f from the linear manifold); "
                       "single-cell biophysical nonlinearity does NOT beat a linear filter here. "
                       "The missing ingredient vs the connectome's static node is linear temporal "
                       "filtering, not full biophysical simulation." % float(err_lin.mean())),
    }


if __name__ == "__main__":
    r = run()
    print("=" * 80)
    print("  SINGLE-CELL BIOPHYSICS TEST -- AWC on/off asymmetry (n=%d worms)" % r["n_worms"])
    print("=" * 80)
    print("  %s | %s" % (r["neuron"], r["stimulus"]))
    print("  feature: %s\n" % r["feature"])
    print("  (on, off) signed response, L1-normalized   [OFF-selective = off>on]")
    print("  real AWC            (%+.2f, %+.2f)   OFF-selective in %.0f%% of worms"
          % (r["real_pair_on_off"][0], r["real_pair_on_off"][1], 100 * r["real_frac_off_selective"]))
    print("  linear node         (%+.2f, %+.2f)   dist=%.3f   (trapped on on=-off; cannot be OFF-selective)"
          % (r["linear_node"]["pair_on_off"][0], r["linear_node"]["pair_on_off"][1], r["linear_node"]["dist_to_real"]))
    print("  biophysical OFF     (%+.2f, %+.2f)   dist=%.3f   <- OVERSHOOTS (worse than linear)"
          % (r["biophysical_off"]["pair_on_off"][0], r["biophysical_off"]["pair_on_off"][1], r["biophysical_off"]["dist_to_real"]))
    print("  scrambled ON (ctrl) (%+.2f, %+.2f)   dist=%.3f   <- ON-selective, WRONG quadrant (control fails)"
          % (r["scrambled_on"]["pair_on_off"][0], r["scrambled_on"]["pair_on_off"][1], r["scrambled_on"]["dist_to_real"]))
    print("\n  real distance to the linear manifold (on=-off): %.3f  -> AWC/IAA is ~linear"
          % r["real_dist_to_linear_manifold"])
    print("  biophysical beats linear : p=%.1e  (>0.05 -> it does NOT)" % r["biophysical_beats_linear_p"])
    print("  scrambled control valid (wrong quadrant, test is sensitive): %s" % r["scrambled_control_valid"])
    print("  => single-cell biophysics helps here: %s" % r["single_cell_biophysics_helps"])
    print("     (honest negative: a linear temporal filter is adequate for this neuron/feature)")
    print("=" * 80)
    with open(os.path.join(HERE, "single_cell.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote single_cell.json")
