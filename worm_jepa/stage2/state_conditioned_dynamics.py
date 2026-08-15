"""
state_conditioned_dynamics.py -- does conditioning the perturbation response on the
worm's BRAIN STATE at stimulus onset beat the state-blind connectome? Uses the newly
re-identified single-worm 001075 dynamics (reconnect_1075.py). An HONEST NEGATIVE, saved
because it was a very tempting false positive.

The seductive result -- and why it is a trap
---------------------------------------------
Per stimulation (904 identified trials across 45 worms), predict each downstream neuron's
single-trial response (post - pre activity). Add the pre-stimulus activity of each neuron
("state") as a feature and per-trial prediction leaps from r=0.06 (connectome only) to
r=0.39. That looks like the dynamics recovering the signal the aggregate atlas averages
away.

It isn't. The response is post - pre, and "state" IS pre, so the model exploits
regression-to-the-mean (a high-baseline neuron tends to fall) -- a statistical artifact,
not stimulus-gating. The control that proves it: a PSEUDO-STIMULUS null -- the exact same
pre/post windows at RANDOM non-stimulus timepoints. If state-conditioning were capturing a
real stimulus response, it would predict real trials better than pseudo ones.

Result (see state_conditioned_dynamics.json):
  model                                real r   pseudo r
  connectome only                       0.06      -
  mean-reversion only (state)           0.38      0.41     <- pseudo >= real: PURE artifact
  full (connectome + state + interact)  0.39      0.41     <- still no real-over-pseudo edge
  connectome x state interaction only   0.22      0.20     <- real > pseudo by only ~0.02

The bare-state gain is entirely mean-reversion (pseudo >= real). The ONLY stimulus-specific
component is the connectome x state interaction, and it beats its pseudo-null by ~0.02 --
real but tiny.

What this means (and it is a real scientific conclusion)
--------------------------------------------------------
Single-trial perturbation responses are dominated by stimulus-INDEPENDENT ongoing brain
dynamics. Averaging over trials -- what the funatlas atlas does -- is the EFFICIENT
estimator of the stimulus-evoked response, not a lossy compromise. So the re-identified
single-worm dynamics, while a real new capability, do NOT provide a validated path past
the 49% aggregate cap via state-conditioning: the state-dependent, connectome-predictable
part of the response is ~0.02, not the tens of points the artifact advertised.

This is why "reintegrate the dynamics into every averaged section" is NOT warranted:
bolting mean-reversion into the aggregate benchmarks would add ongoing-activity noise, not
signal. The averaged representation is correct for the stimulus-response question.

    python state_conditioned_dynamics.py     # writes state_conditioned_dynamics.json
Self-contained: reads state_dynamics_trials.npz (distilled from reconnected 001075).
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "state_dynamics_trials.npz")


def _cv(pcol, state, resp, cols, seed=0):
    from scipy.stats import pearsonr
    from numpy.linalg import lstsq
    n = len(resp); idx = np.random.RandomState(seed).permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    def des(i):
        cc = []
        for c in cols:
            if c == "pcol":
                cc.append(pcol[i])
            elif c == "state":
                cc.append(state[i])
            elif c == "inter":
                cc.append(pcol[i] * state[i])
            elif c == "gstate":
                cc.append(np.full(len(pcol[i]), float(np.mean(state[i]))))
        return np.column_stack(cc)
    X = np.vstack([des(i) for i in tr]); y = np.concatenate([resp[i] for i in tr])
    m = np.isfinite(y) & np.all(np.isfinite(X), 1)
    b, *_ = lstsq(np.column_stack([X[m], np.ones(m.sum())]), y[m], rcond=None)
    rs = []
    for i in te:
        Xi = des(i); pred = np.column_stack([Xi, np.ones(len(Xi))]) @ b; obs = resp[i]
        mm = np.isfinite(pred) & np.isfinite(obs)
        if mm.sum() >= 8 and np.std(pred[mm]) > 1e-9 and np.std(obs[mm]) > 1e-9:
            rs.append(float(pearsonr(pred[mm], obs[mm])[0]))
    return float(np.mean(rs)) if rs else 0.0


def run():
    d = np.load(DATA, allow_pickle=True)
    rp, rs, rr = d["real_pcol"], d["real_state"], d["real_resp"]
    pp, ps, pr = d["pseudo_pcol"], d["pseudo_state"], d["pseudo_resp"]
    conn = _cv(rp, rs, rr, ["pcol"])
    mr_real = _cv(rp, rs, rr, ["state", "gstate"]); mr_ps = _cv(pp, ps, pr, ["state", "gstate"])
    full_real = _cv(rp, rs, rr, ["pcol", "state", "inter", "gstate"]); full_ps = _cv(pp, ps, pr, ["pcol", "state", "inter", "gstate"])
    int_real = _cv(rp, rs, rr, ["pcol", "inter"]); int_ps = _cv(pp, ps, pr, ["pcol", "inter"])
    return {
        "n_trials": len(rr), "n_pseudo": len(pr),
        "connectome_only_r": round(conn, 3),
        "mean_reversion": {"real": round(mr_real, 3), "pseudo": round(mr_ps, 3)},
        "full_model": {"real": round(full_real, 3), "pseudo": round(full_ps, 3)},
        "connectome_x_state": {"real": round(int_real, 3), "pseudo": round(int_ps, 3)},
        "bare_state_is_mean_reversion_artifact": bool(mr_ps >= mr_real - 0.02),
        "stimulus_specific_state_gain": round(int_real - int_ps, 3),
        "dynamics_beat_averages": bool((int_real - int_ps) > 0.1),  # would need a real, sizable edge
        "conclusion": "single-trial responses are dominated by stimulus-independent ongoing dynamics; averaging "
                      "(the atlas) is the efficient estimator. State-conditioning's apparent gain is mean-reversion "
                      "(pseudo>=real); the true stimulus-specific state gain is ~0.02. No validated path past 49%.",
    }


if __name__ == "__main__":
    r = run()
    print("=== STATE-CONDITIONED DYNAMICS -- does brain state beat the state-blind connectome? ===\n")
    print("  %d real trials, %d pseudo-stimulus controls\n" % (r["n_trials"], r["n_pseudo"]))
    print("  %-38s %8s %8s" % ("model", "real r", "pseudo r"))
    print("  %-38s %8.3f %8s" % ("connectome only", r["connectome_only_r"], "-"))
    print("  %-38s %8.3f %8.3f  <- pseudo>=real: ARTIFACT" % ("mean-reversion only (state)", r["mean_reversion"]["real"], r["mean_reversion"]["pseudo"]))
    print("  %-38s %8.3f %8.3f" % ("full (conn+state+interaction)", r["full_model"]["real"], r["full_model"]["pseudo"]))
    print("  %-38s %8.3f %8.3f  <- the only stimulus-specific bit" % ("connectome x state interaction", r["connectome_x_state"]["real"], r["connectome_x_state"]["pseudo"]))
    print("\n  bare-state gain is mean-reversion artifact : %s" % r["bare_state_is_mean_reversion_artifact"])
    print("  stimulus-specific state gain (real - pseudo): %.3f" % r["stimulus_specific_state_gain"])
    print("  dynamics beat averages: %s" % r["dynamics_beat_averages"])
    print("  => %s" % r["conclusion"])
    with open(os.path.join(HERE, "state_conditioned_dynamics.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("\nwrote state_conditioned_dynamics.json")
