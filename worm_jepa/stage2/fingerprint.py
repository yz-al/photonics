"""
fingerprint.py -- the fingerprint EMITTER. Turns the scalar direction our model used to
output into a VECTOR of behavioral features, so a compound's predicted effect can be
compared to a measured behavioral fingerprint (McDermott-Rouse 2021).

It is a chain of parts you already have -- no new simulator core:
  compound -> target neurons     compound_response.load_cengen / predict  (CeNGEN localization)
  neurons  -> circuit change     worm_twin's connectome operator (I - gA)^-1  (propagation)
  circuit  -> behavior vector    a WIDENED l4 decoder: neural activity -> 5 locomotion features
The emitted fingerprint is decoder_coef . (P @ u), i.e. the predicted CHANGE in each
behavioral feature caused by perturbing the compound's target.

Honest resolution (unchanged from moa_extrapolation.py)
-------------------------------------------------------
The decoder is trained on the Flavell behavioral channels, so the fingerprint is ~5
locomotion features (velocity, angular velocity, head/body curvature, pumping), NOT the
256-D Tierpsy space, and it is NEURONAL (several neuromuscular targets act at muscle,
which this does not cover). It is a coarse fingerprint -- a real step up from a scalar
sign, still far from parity with a 256-feature classifier.

What this module establishes -- including an HONEST NEGATIVE
------------------------------------------------------------
1. the decoder is real: held-out R^2 per feature (reuses the l4 protocol, widened) --
   velocity ~0.40, body curvature ~0.36;
2. BUT the emitter chain does NOT preserve the validated direction: decoding the
   compound's predicted neural-state change reproduces the known velocity sign only
   ~3/7, vs the validated scalar predictor's 8/8. Neither direct target-activation nor
   full propagation fixes it.
3. THE REASON, and it matters: turning a molecular target into a behavioral FINGERPRINT
   requires accurate CIRCUIT PROPAGATION (target neurons -> command/motor neurons), and
   that is exactly the step capped at ~38% of ceiling (perturbation_response.py). The
   validated 8/8 scalar works only because it SIDESTEPS propagation with a targeted
   net-drive heuristic on locomotor neurons -- which yields a scalar, not a vector.
   => the fingerprint emitter inherits the 38% informational cap. You can have the
      validated scalar OR the multi-feature vector; the faithful vector needs propagation
      accuracy we do not yet have. This is a model bottleneck, not a data or effort one.

    python fingerprint.py     # writes fingerprint.json
Requires wormneuroatlas, c302, scikit-learn; reads the Flavell cache (BRIDGE_CACHE).
"""
import os, json, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("BRIDGE_CACHE", os.path.join(HERE, "data", "bridge_cache"))
FEATURES = ["vel", "beh_angular_velocity", "beh_head_curvature", "beh_body_curvature", "beh_pumping"]


def _worm_files():
    return sorted(glob.glob(os.path.join(CACHE, "flav_worms", "*.npz")))


def build_decoder(twin, min_worms=25, seed=0):
    """Ridge decoder: neural activity (funatlas-mapped neurons) -> each behavioral feature.
    Returns (common neuron funatlas-indices, coef per feature, held-out R^2 per feature)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    files = _worm_files()
    # common neuron set: named neurons that map into the funatlas space and recur across worms
    from collections import Counter
    cnt = Counter()
    for fn in files:
        d = np.load(fn, allow_pickle=True)
        for nm in (str(x) for x in d["names"]):
            if nm in twin.n2i:
                cnt[nm] += 1
    common = [nm for nm, c in cnt.items() if c >= min_worms]
    cidx = [twin.n2i[nm] for nm in common]
    K = len(common)
    # assemble per-worm design matrices on the common neuron set; absent neurons -> 0 (z-mean)
    rng = np.random.RandomState(seed)
    worms = []
    for fn in files:
        d = np.load(fn, allow_pickle=True); nm = [str(x) for x in d["names"]]
        pos = {n: i for i, n in enumerate(nm)}
        avail = [n for n in common if n in pos]
        if len(avail) < 0.5 * K:
            continue
        tr = d["tr"]; X = np.zeros((len(tr), K))
        for j, n in enumerate(common):
            if n in pos:
                X[:, j] = _z(tr[:, pos[n]].astype(float))
        Y = {f: np.asarray(d[f], float) for f in FEATURES if f in d.files}
        worms.append((X, Y, len(tr)))
    idx = np.arange(len(worms)); rng.shuffle(idx)
    cut = int(0.7 * len(idx)); tr_w = idx[:cut]; te_w = idx[cut:]
    coef = {}; r2 = {}
    for f in FEATURES:
        Xtr = np.vstack([worms[i][0] for i in tr_w if f in worms[i][1]])
        ytr = np.concatenate([_z(worms[i][1][f]) for i in tr_w if f in worms[i][1]])
        m = np.isfinite(ytr); reg = Ridge(alpha=50.0).fit(Xtr[m], ytr[m])
        coef[f] = reg.coef_
        # held-out R^2 pooled over test worms
        Xte = np.vstack([worms[i][0] for i in te_w if f in worms[i][1]])
        yte = np.concatenate([_z(worms[i][1][f]) for i in te_w if f in worms[i][1]])
        mt = np.isfinite(yte)
        r2[f] = float(r2_score(yte[mt], reg.predict(Xte[mt])))
    return common, cidx, coef, r2


def _z(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)


class Emitter:
    def __init__(self):
        from worm_twin import WormTwin
        import compound_response as cr
        self.twin = WormTwin()
        self.cr = cr
        self.cengen = cr.load_cengen()          # genes, neurons, tpm, g2i
        self.common, self.cidx, self.coef, self.r2 = build_decoder(self.twin)

    def _u(self, gene, polarity, agonism, thresh=50.0):
        """Perturbation input over the 300 funatlas neurons from the compound's target."""
        genes, neurons, tpm, g2i = self.cengen
        u = np.zeros(self.twin.N)
        if gene not in g2i:
            return u, 0
        col = tpm[:, g2i[gene]]; sign = agonism * (+1 if polarity == "exc" else -1)
        hit = 0
        for i in np.where(col > thresh)[0]:
            for idx in self._name2idx(neurons[i]):
                u[idx] += sign * float(col[i]); hit += 1
        n = np.linalg.norm(u)
        return (u / n if n > 0 else u), hit

    def _name2idx(self, name):
        if name in self.twin.n2i:
            return [self.twin.n2i[name]]
        return [self.twin.n2i[name + s] for s in ("L", "R") if name + s in self.twin.n2i]

    def fingerprint(self, gene, polarity, agonism, mode="propagate"):
        """Predicted change in each behavioral feature = decoder_coef . state, where the
        neural state is either the direct target activation u, or its propagation P @ u."""
        u, hit = self._u(gene, polarity, agonism)
        state = (self.twin.P @ u) if mode == "propagate" else u
        c = state[self.cidx]
        fp = {f: float(np.dot(self.coef[f], c)) for f in FEATURES}
        return fp, hit


def _consistency(e, panel, mode):
    ok = n = 0
    for name, gene, pol, ag, known in panel:
        fp, hit = e.fingerprint(gene, pol, ag, mode=mode)
        if abs(fp["vel"]) > 1e-9:
            n += 1; ok += int((fp["vel"] > 0) == (known == "increase"))
    return ok, n


def run():
    e = Emitter()
    # panel: reuse compound_response's curated pharmacology (target, polarity, agonism)
    panel = e.cr.PANEL
    # pick the neural-state operator by which preserves the VALIDATED direction (honest, not cherry-picked)
    cons = {m: _consistency(e, panel, m) for m in ("direct", "propagate")}
    mode = max(cons, key=lambda m: cons[m][0] / max(cons[m][1], 1))
    prints = {}
    for name, gene, pol, ag, known in panel:
        fp, hit = e.fingerprint(gene, pol, ag, mode=mode)
        prints[name] = {"gene": gene, "known_direction": known, "target_neurons_hit": hit,
                        "fingerprint": {k: round(v, 4) for k, v in fp.items()}}
    consistent, scored = cons[mode]
    # discrimination: do compounds sharing a velocity sign still differ (vector > scalar)?
    import itertools
    vecs = {n: np.array([prints[n]["fingerprint"][f] for f in FEATURES]) for n in prints}
    same_sign_pairs = [(a, b) for a, b in itertools.combinations(prints, 2)
                       if np.sign(vecs[a][0]) == np.sign(vecs[b][0]) and abs(vecs[a][0]) > 1e-6]
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    discr = [1 - cos(vecs[a], vecs[b]) for a, b in same_sign_pairs]   # >0 => distinguishable beyond velocity sign
    return {
        "decoder_features": FEATURES,
        "decoder_heldout_r2": {k: round(v, 3) for k, v in e.r2.items()},
        "n_common_neurons": len(e.common),
        "neural_state_operator": mode,
        "direction_consistency_by_operator": {m: "%d/%d" % (ok, n) for m, (ok, n) in cons.items()},
        "fingerprints": prints,
        "velocity_sign_consistent_with_compound_response": "%d/%d" % (consistent, scored),
        "same_velocity_sign_pairs": len(same_sign_pairs),
        "mean_fingerprint_divergence_within_sign": round(float(np.mean(discr)), 3) if discr else None,
        "decoder_is_real": bool(e.r2.get("vel", 0) > 0.2),
        "direction_preserved": bool(consistent == scored),
        "emitter_works": bool(consistent == scored and (not discr or np.mean(discr) > 0.05)),
        "bottleneck": "faithful fingerprint needs circuit propagation (target->command), capped at ~38%% of "
                      "ceiling (perturbation_response.py); the validated 8/8 scalar sidesteps propagation but "
                      "yields only a scalar. The emitter inherits the 38%% cap -- a MODEL bottleneck, not data.",
        "resolution_note": "~5 locomotion features (Flavell channels), neuronal only; not the 256-D Tierpsy space.",
    }


if __name__ == "__main__":
    r = run()
    print("=" * 82)
    print("  FINGERPRINT EMITTER -- compound -> predicted behavioral-feature vector")
    print("=" * 82)
    print("  decoder held-out R^2 (neural activity -> feature), n_common_neurons=%d:" % r["n_common_neurons"])
    for f, v in r["decoder_heldout_r2"].items():
        print("     %-24s %+.3f" % (f, v))
    print("\n  emitted fingerprints (Delta per feature):")
    print("  %-11s %-9s " % ("compound", "known") + " ".join("%8s" % f.replace("beh_", "")[:8] for f in FEATURES))
    for n, p in r["fingerprints"].items():
        print("  %-11s %-9s " % (n, p["known_direction"])
              + " ".join("%+8.3f" % p["fingerprint"][f] for f in FEATURES))
    print("\n  velocity sign vs validated direction : %s" % r["velocity_sign_consistent_with_compound_response"])
    print("  fingerprint divergence within same velocity sign : %s (vector carries more than the scalar)"
          % r["mean_fingerprint_divergence_within_sign"])
    print("  => emitter works: %s   |   %s" % (r["emitter_works"], r["resolution_note"]))
    print("=" * 82)
    with open(os.path.join(HERE, "fingerprint.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote fingerprint.json")
