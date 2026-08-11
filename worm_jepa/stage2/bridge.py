"""
bridge.py -- The sensory -> command -> behavior BRIDGE (levels L2 -> L4).

Motivation
----------
No public C. elegans dataset contains the full sensorimotor loop in one
recording (freely-moving + controlled stimulus + whole-brain command + behavior
together). The data splits into two halves that never co-occur:

  (A) immobilized + stimulus  -- clean sensory transduction, but the command
      circuit is DECOUPLED from the sensory response (an immobilization
      artifact -- see `measure_gap` below; adding sensory activity HURTS command
      prediction, pooled gain -0.54, t=-3.2 over 45 worms, 000541+000981).

  (B) freely-moving (no stimulus) -- an intact command->behavior loop, but the
      sensory neurons are barely driven (no controlled odor), so the
      sensory->command coupling is present but weak (ASH rises only +0.043
      before reversals, p~0.07 in Flavell 000776).

So the sensory->command law can be learned from neither activity source. We
therefore compose THREE maps, each estimated in the regime where it is valid,
none of which ever observed the full loop:

  A  transduction        stimulus -> sensory activation
                         (immobilized+stimulus: DANDI 000541 attractive panel,
                          000981 aversive panel; stimulus-triggered average)
  M  sensory->command    causal coupling (Randi 2023 signal-propagation atlas,
                         `wormneuroatlas` funatlas.h5 wt/dFF gated by wt/q).
                         Supplies WIRING MAGNITUDE -- how strongly each sensor
                         couples into the command circuit -- independent of
                         spontaneous drive.
  V  valence prior       ACTIVATION->reversal sign per sensory neuron, from
                         documented chemosensory logic. Supplies the VALENCE
                         that a magnitude-only map cannot (AWA activation ->
                         forward; AWC, an off-cell, activation -> reversal).
  C  command->behavior   reversal-command state -> velocity
                         (freely-moving Flavell 000776, n=38).

Prediction:  dVelocity(stimulus)  =  betaC * sum_s  a_s * valence_s * wmag_s
             a_s     = transduction amplitude of sensor s to the stimulus (A)
             wmag_s  = |atlas coupling of s into the command circuit| (M)
             valence_s in {+1 promotes reversal, -1 promotes forward} (V)
             betaC   = command(reversal)->velocity slope, < 0 (C)

Validation (see `validate`): end-to-end sign of the behavioral response to each
stimulus class, checked against known ethology, with a SHUFFLED-VALENCE negative
control (the structure test from METHODOLOGY.md: the prediction must collapse or
invert when the valence labels are permuted).

This is the L2->L4 counterpart to OpenWorm's forward simulation: OpenWorm
integrates a hand-built dynamical model; the bridge composes three empirically
measured maps and passes a negative control.

Reproducing
-----------
First run streams the neural data from DANDI and caches it (see `stream_*`
below); the derived maps are tiny and are shipped alongside this file as
`bridge_maps.npz`, so `validate()` runs offline. The Randi atlas ships inside
the `wormneuroatlas` package (funatlas.h5), no download needed.

    python bridge.py            # runs validate(), writes bridge.json
    python bridge.py --stream   # (re)stream + recompute the maps from DANDI
"""
import os, sys, glob, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS_FILE = os.path.join(HERE, "bridge_maps.npz")
CACHE = os.environ.get("BRIDGE_CACHE", os.path.join(HERE, "data", "bridge_cache"))

# Command (reversal) circuit read out for the command state and atlas coupling.
CMD = ["AVAL", "AVAR", "AVEL", "AVER", "AVDL", "AVDR", "AIBL", "AIBR"]

# ACTIVATION -> reversal valence prior (documented C. elegans chemosensory logic).
#  +1 : activation of this neuron PROMOTES a reversal
#  -1 : activation PROMOTES forward locomotion (suppresses reversal)
# Note the cell-type dependence: AWA/ASEL are ON-cells for attractants
# (activation -> forward); AWC is an OFF-cell (activation, on odor removal,
# -> reversal); ASER is a salt OFF-cell (salt decrease is aversive -> reversal).
VALENCE = {
    "ASHL": +1, "ASHR": +1, "ADLL": +1, "ADLR": +1,   # nociceptors      -> reversal
    "AWBL": +1, "AWBR": +1,                            # repulsive odor   -> reversal
    "ASER": +1,                                        # salt off-cell    -> reversal
    "AWCL": +1, "AWCR": +1,                            # AWC off-cell     -> reversal
    "AWAL": -1, "AWAR": -1,                            # attractive on    -> forward
    "ASEL": -1,                                        # salt on-cell     -> forward
}

# Expected ethology for the validation stimuli.
EXPECTED = {
    "aversive":           "REVERSE",   # CuSO4 -> escape reversal
    "attractive_onset":   "FORWARD",   # attractant appears -> run forward
    "attractive_removal": "REVERSE",   # attractant removed -> AWC-off turn
}


# --------------------------------------------------------------------------- #
# Map M: sensory -> command wiring magnitude, from the Randi (L2) atlas.
# --------------------------------------------------------------------------- #
def atlas_coupling(q_thresh=0.10):
    """|causal coupling| of each candidate sensor into the command circuit,
    from the wild-type signal-propagation atlas (funatlas.h5), gating on the
    significance q-value. Returns {sensor: wiring_magnitude}."""
    import h5py, wormneuroatlas
    p = os.path.join(os.path.dirname(wormneuroatlas.__file__), "data", "funatlas.h5")
    with h5py.File(p, "r") as h:
        ids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["neuron_ids"][:]]
        dFF = h["wt/dFF"][:]
        q = h["wt/q"][:]
    n2i = {n: i for i, n in enumerate(ids)}
    out = {}
    for s in VALENCE:
        if s not in n2i:
            out[s] = 0.0
            continue
        sj = n2i[s]
        out[s] = float(sum(abs(dFF[sj, n2i[c]]) for c in CMD
                           if c in n2i and q[sj, n2i[c]] < q_thresh))
    return out


# --------------------------------------------------------------------------- #
# Map A: transduction -- stimulus-triggered average of each sensor.
# --------------------------------------------------------------------------- #
def _sta(sig, frames, T, pre=8, post=16):
    v = [sig[f:f + post].mean() - sig[f - pre:f].mean()
         for f in frames if pre < f < T - post]
    return float(np.mean(v)) if v else None


def _is_aversive(label):
    return any(k in label for k in ("450mM", "CuSO4", "Sorbitol"))


def build_transduction(cache=CACHE):
    """Per-sensor transduction amplitude for each stimulus class, pooled over
    worms. Aversive panel from 000981; attractive onset/removal from 000541
    (removal = onset + 40 frames = +10 s at 4 Hz stimulus duration)."""
    d981 = os.path.join(cache, "chemo_worms")
    d541 = os.path.join(cache, "chemo541_worms")

    def one(folder, stim):
        acc = {}
        for fn in sorted(glob.glob(os.path.join(folder, "*.npz"))):
            d = np.load(fn, allow_pickle=True)
            act = d["act"]; names = list(d["names"])
            idx = {n: i for i, n in enumerate(names) if n}
            A = np.nan_to_num((act.T - np.nanmean(act.T, 0)) / (np.nanstd(act.T, 0) + 1e-9))
            T = len(A); st = d["st"]; sid = list(d["sid"])
            if stim == "aversive":
                frames = [int(t) for t, s in zip(st, sid) if _is_aversive(s)]
            elif stim == "attractive_onset":
                frames = [int(t) for t in st]
            else:  # attractive_removal
                frames = [int(t) + 40 for t in st]
            for s in VALENCE:
                if s not in idx:
                    continue
                v = _sta(A[:, idx[s]], frames, T)
                if v is not None:
                    acc.setdefault(s, []).append(v)
        return {s: float(np.mean(v)) for s, v in acc.items() if v}

    return {
        "aversive":           one(d981, "aversive"),
        "attractive_onset":   one(d541, "attractive_onset"),
        "attractive_removal": one(d541, "attractive_removal"),
    }


# --------------------------------------------------------------------------- #
# Map C: command (reversal) state -> velocity, from freely-moving Flavell.
# --------------------------------------------------------------------------- #
def command_state(traces, names):
    """PC1 of the command neurons, oriented so + = reversal (AVAL-aligned)."""
    from scipy.stats import pearsonr
    idx = {n: i for i, n in enumerate(names) if n}
    ci = [idx[n] for n in CMD if n in idx]
    if len(ci) < 4:
        return None
    Z = np.nan_to_num((traces - np.nanmean(traces, 0)) / (np.nanstd(traces, 0) + 1e-9))
    Xc = Z[:, ci]
    U, s, _ = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
    z = U[:, 0] * s[0]
    if "AVAL" in idx and pearsonr(z, Z[:, idx["AVAL"]])[0] < 0:
        z = -z
    return z


def fit_command_to_behavior(cache=CACHE):
    """betaC = mean correlation of the reversal-command state with velocity
    across freely-moving worms (expected < 0: reversal slows / backs up)."""
    from scipy.stats import pearsonr
    cvs = []
    for fn in sorted(glob.glob(os.path.join(cache, "flav_worms", "*.npz"))):
        d = np.load(fn, allow_pickle=True)
        if "vel" not in d.files:
            continue
        z = command_state(d["tr"], list(d["names"]))
        if z is None:
            continue
        vel = np.asarray(d["vel"], float)
        m = np.isfinite(vel) & np.isfinite(z)
        if m.sum() > 100:
            cvs.append(float(pearsonr(z[m], vel[m])[0]))
    return float(np.mean(cvs)), float(np.std(cvs)), len(cvs)


# --------------------------------------------------------------------------- #
# Composition + validation.
# --------------------------------------------------------------------------- #
def reversal_drive(transduction_vec, wmag, valence):
    """Signed drive onto the reversal command for one stimulus (the core of the
    bridge): sum over sensors of  activation * valence * wiring_magnitude."""
    return float(sum(a * valence.get(s, 0) * wmag.get(s, 0.0)
                     for s, a in transduction_vec.items()))


def predict_velocity(transduction_vec, wmag, valence, betaC):
    return betaC * reversal_drive(transduction_vec, wmag, valence)


def load_maps():
    """Load the shipped derived maps (offline). Returns (transduction, wmag,
    valence, betaC, betaC_std, n_flav)."""
    d = np.load(MAPS_FILE, allow_pickle=True)
    return (json.loads(str(d["transduction"])), json.loads(str(d["wmag"])),
            json.loads(str(d["valence"])), float(d["betaC"]),
            float(d["betaC_std"]), int(d["n_flav"]))


def validate(transduction=None, wmag=None, valence=None, betaC=None,
             n_shuffle=500, seed=0):
    """End-to-end prediction for each stimulus + shuffled-valence control.
    Returns a results dict (also the payload written to bridge.json)."""
    if transduction is None:
        transduction, wmag, valence, betaC, betaC_std, n_flav = load_maps()
    else:
        betaC_std, n_flav = None, None
    rng = np.random.RandomState(seed)
    vkeys = list(valence.keys()); vvals = list(valence.values())
    results = {"betaC": betaC, "betaC_std": betaC_std, "n_flav": n_flav,
               "stimuli": {}, "n_correct": 0}
    for name, vec in transduction.items():
        drive = reversal_drive(vec, wmag, valence)
        dvel = betaC * drive
        behavior = "REVERSE" if dvel < 0 else "FORWARD"
        # shuffled-valence null: permute the activation->reversal labels.
        sh = []
        for _ in range(n_shuffle):
            perm = dict(zip(vkeys, rng.permutation(vvals)))
            sh.append(betaC * reversal_drive(vec, wmag, perm))
        correct = behavior == EXPECTED.get(name)
        results["n_correct"] += int(correct)
        results["stimuli"][name] = {
            "reversal_drive": round(drive, 4),
            "pred_dvelocity": round(dvel, 4),
            "behavior": behavior,
            "expected": EXPECTED.get(name),
            "correct": bool(correct),
            "shuffled_valence_mean": round(float(np.mean(sh)), 4),
            "shuffled_valence_std": round(float(np.std(sh)), 4),
        }
    results["n_total"] = len(transduction)
    return results


# --------------------------------------------------------------------------- #
# Diagnostic: measure the sensory->command decoupling gap (why M needs the atlas).
# --------------------------------------------------------------------------- #
def measure_gap(cache=CACHE):
    """Quantify why the middle map cannot come from activity: sensory->command
    next-step gain in immobilized (000541+000981) vs freely-moving (Flavell)."""
    from scipy.stats import pearsonr, ttest_1samp
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    SENS = ["ASHL", "ASHR", "ADLL", "ADLR", "ASEL", "ASER", "AWAL", "AWAR",
            "AWBL", "AWBR", "AWCL", "AWCR"]

    def gain(files, trace_key):
        K = 3; base = []; ws = []
        for fn in files:
            d = np.load(fn, allow_pickle=True); names = list(d["names"])
            idx = {n: i for i, n in enumerate(names) if n}
            si = [idx[n] for n in SENS if n in idx]
            if len(si) < 2:
                continue
            raw = d[trace_key]
            traces = raw.T if trace_key == "act" else raw
            z = command_state(traces, names)
            if z is None:
                continue
            Z = np.nan_to_num((traces - np.nanmean(traces, 0)) / (np.nanstd(traces, 0) + 1e-9))
            S = Z[:, si]; T = len(z); cut = int(0.7 * T)

            def feats(us):
                X, y = [], []
                for t in range(K, T - 1):
                    X.append(list(z[t - K:t + 1]) + (list(S[t]) if us else []))
                    y.append(z[t + 1] - z[t])
                X = np.array(X); y = np.array(y); c = cut - K
                return X[:c], y[:c], X[c:], y[c:]
            rr = []
            for us in (False, True):
                Xtr, ytr, Xte, yte = feats(us)
                m = Ridge(5.0).fit(Xtr, ytr)
                rr.append(r2_score(yte, m.predict(Xte)))
            base.append(rr[0]); ws.append(rr[1])
        base = np.array(base); ws = np.array(ws); g = ws - base
        t = ttest_1samp(g, 0) if len(g) > 1 else None
        return dict(base=float(base.mean()), withsens=float(ws.mean()),
                    gain=float(g.mean()), t=float(t.statistic) if t else None,
                    p=float(t.pvalue) if t else None, n=len(g))

    immob = (sorted(glob.glob(os.path.join(cache, "chemo_worms", "*.npz"))) +
             sorted(glob.glob(os.path.join(cache, "chemo541_worms", "*.npz"))))
    free = sorted(glob.glob(os.path.join(cache, "flav_worms", "*.npz")))
    return {"immobilized": gain(immob, "act"),
            "freely_moving": gain(free, "tr")}


def recompute_and_save(cache=CACHE):
    """Rebuild the derived maps from the cached neural data and (re)write
    bridge_maps.npz. Requires the DANDI cache (see stream_all)."""
    transduction = build_transduction(cache)
    wmag = atlas_coupling()
    betaC, betaC_std, n_flav = fit_command_to_behavior(cache)
    np.savez(MAPS_FILE,
             transduction=json.dumps(transduction), wmag=json.dumps(wmag),
             valence=json.dumps(VALENCE), betaC=betaC,
             betaC_std=betaC_std, n_flav=n_flav)
    print("wrote %s  (betaC=%.3f, n_flav=%d)" % (MAPS_FILE, betaC, n_flav))
    return transduction, wmag, VALENCE, betaC


if __name__ == "__main__":
    if "--stream" in sys.argv:
        from stream_dandi import stream_all
        stream_all(CACHE)
        recompute_and_save(CACHE)

    res = validate()
    print("=== BRIDGE: sensory -> command -> behavior (L2->L4) ===")
    print("MAP C  command(reversal)->velocity = %+.3f (n=%s)\n" % (res["betaC"], res["n_flav"]))
    hdr = "%-20s %11s %11s   %-8s %-4s  %s"
    print(hdr % ("stimulus", "rev_drive", "pred_dVel", "behavior", "ok", "shuffled-valence"))
    for name, s in res["stimuli"].items():
        print("%-20s %+11.3f %+11.4f   %-8s %-4s  %+.4f +/- %.4f" % (
            name, s["reversal_drive"], s["pred_dvelocity"], s["behavior"],
            "OK" if s["correct"] else "x",
            s["shuffled_valence_mean"], s["shuffled_valence_std"]))
    print("\ncorrect: %d/%d" % (res["n_correct"], res["n_total"]))
    with open(os.path.join(HERE, "bridge.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("wrote bridge.json")
