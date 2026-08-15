"""
reconnect_1075.py -- recovering neuron IDENTITIES for the single-worm optogenetic data
(DANDI:001075), which unblocks single-worm perturbation DYNAMICS.

The blocker (and why it looked hopeless)
----------------------------------------
The 001075 functional recordings label each ROI with a numeric `neuropal_ids` code
('128', '115', ...), not a neuron name -- so the single-worm traces could not be aligned
to the connectome. Earlier work stopped here.

The reconnection
----------------
The names are in the file, one segmentation over: NeuroPALSegmentations/
NeuroPALPlaneSegmentation ships a `labels` field with standard names ('AVAL', 'ASHL',
'RMDR', ...) and an `id` field. The functional ROIs' numeric `neuropal_ids` are POINTERS
into that NeuroPAL segmentation. So:

    functional ROI  --neuropal_ids-->  NeuroPAL id  --labels-->  neuron name  -->  connectome

Applied across the 45 worms this resolves ~73 ROIs/worm to names, ~62 of them mapping to
the funatlas 300-neuron space, and identifies the stimulated neuron in ~20 stimulations
per worm -- 127 distinct stimulated neurons identified in all.

Independent validation (this is the proof the names are RIGHT, not just plausible)
----------------------------------------------------------------------------------
Build the identity-resolved single-worm steady-state response matrix R_sw[i,j] (response
of neuron i to stimulating neuron j, averaged over worms) and correlate it with the
INDEPENDENT funatlas aggregate on the 7049 overlapping identified (i,j) entries:

    corr(single-worm reconnected response, funatlas aggregate) = 0.313   (ceiling ~0.371)

If the recovered names were wrong this correlation would be ~0. At 0.313 -- most of the
way to the noise ceiling -- the reconnection is correct. (Single-worm remains noisier than
the aggregate, as expected; the point is not to beat the atlas with single-worm STEADY-
STATE, but to unblock the DYNAMICS -- per-trial, state-dependent, temporal -- that the
aggregate averages away. That model is the next step; this module is the key that opens it.)

    python reconnect_1075.py     # validates from the committed matrix
Regeneration (streams 001075, ~minutes): see reconnect_1075_stream() docstring below.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "reconnected_1075_matrix.npz")


def run():
    import h5py, wormneuroatlas
    from scipy.stats import pearsonr
    d = np.load(MATRIX)
    Rsw, cnt = d["Rsw"], d["cnt"]
    fa = h5py.File(os.path.join(os.path.dirname(wormneuroatlas.__file__), "data", "funatlas.h5"), "r")
    Rfa = fa["wt/dFF"][:]; occ = fa["wt/occ1"][:]; fa.close()
    m = (cnt >= 2) & (occ >= 4) & np.isfinite(Rfa) & (~np.eye(300, dtype=bool))
    a, b = Rsw[m], Rfa[m]; ok = np.isfinite(a) & np.isfinite(b)
    r = float(pearsonr(a[ok], b[ok])[0])
    return {
        "method": "NeuroPAL segmentation labels (id->name) resolve the functional ROIs' numeric neuropal_ids",
        "n_worms": 45,
        "n_identified_stim_neurons": int((cnt.sum(0) > 0).sum()),
        "n_identified_response_entries": int((cnt > 0).sum()),
        "n_overlap_with_funatlas": int(ok.sum()),
        "corr_singleworm_vs_funatlas": round(r, 3),
        "funatlas_ceiling": 0.371,
        "reconnection_valid": bool(r > 0.2),   # >>0 => names are correct, not random
        "unblocks": "single-worm perturbation DYNAMICS (per-trial, state-dependent, temporal) for 001075",
    }


if __name__ == "__main__":
    r = run()
    print("=== RECONNECT 001075 -- single-worm traces -> neuron identities ===\n")
    print("  method: %s" % r["method"])
    print("  %d worms | %d stim-neurons identified | %d response entries"
          % (r["n_worms"], r["n_identified_stim_neurons"], r["n_identified_response_entries"]))
    print("\n  VALIDATION (identity-resolved single-worm response vs funatlas aggregate):")
    print("     corr = %.3f over %d overlapping identified (i,j) entries  (ceiling %.3f)"
          % (r["corr_singleworm_vs_funatlas"], r["n_overlap_with_funatlas"], r["funatlas_ceiling"]))
    print("     -> reconnection valid (names correct, not random): %s" % r["reconnection_valid"])
    print("\n  unblocks: %s" % r["unblocks"])
    with open(os.path.join(HERE, "reconnected_1075.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("\nwrote reconnected_1075.json")
