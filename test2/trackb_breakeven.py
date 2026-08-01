"""Track B — the spiking BREAKEVEN SURFACE (the honest deliverable).

Both device inputs in the first pass were the favourable end: E_spike=8 pJ (Xiang gen) and
firing rate f=0.43 (measured). This sweeps BOTH against two matched-accuracy competitors and
reports the win-region boundary — a requirements surface the field does not have, the same
shape as the nonlinearity spec sheet, valid whether or not the single MNIST point survives.

Competitors (both at MATCHED accuracy — the SNN reaches 98.2% on MNIST):
  (A) right-sized digital NPU: the SMALLEST ReLU MLP reaching 98.2% (from trackb_rightsize.json),
      priced at 2-30 TOPS/W INT8. This corrects the earlier unmatched 268.8k-MAC baseline.
  (B) digital neuromorphic (Loihi-class): the SAME spiking net's SOP count at a published
      per-SOP energy, 12-24 pJ/SOP (ODIN 12.7; Loihi 15-23.6, Davies IEEE Micro 2018). This is
      the referee's competitor for an optical SNN, and it is ~1000x cheaper per op than an NPU.

Reports: for each competitor, the (E_spike, f) boundary; where the MEASURED point (8 pJ, 0.43)
sits; the firing rate required to win at measured E_spike; and whether that required f is below
what trained SNNs achieve (~0.3-0.45) — which would close spiking on the sweep, not a benchmark.
Writes data/trackb_breakeven.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "energy_model"))
from model import PARAMS, sample_params, E_PHOTON_1550

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
base = {k: v.nom for k, v in PARAMS.items()}

# ---- network (matches the trained SNN) ----
IN, WIDTH, DEPTH, NC, T = 784, 256, 2, 10, 8
N_NEURONS = DEPTH * WIDTH

def sops(f):
    """synaptic operations: dense input every timestep + spike-driven rest."""
    return T * IN * WIDTH + f * T * ((DEPTH - 1) * WIDTH * WIDTH + WIDTH * NC)

def spikes(f):
    return f * N_NEURONS * T

def optical_snn_j(E_spike, f, p):
    P_bias = p["neuron_bias_mW"] * 1e-3
    bw = p["bandwidth"]
    trans = 10.0 ** (-(2 * p["coupling_db"]) / 10.0)
    E_det1 = p["spike_photons"] * E_PHOTON_1550 / (p["responsivity"] * trans * p["WPE_laser"])
    bias = N_NEURONS * P_bias * T / bw
    gen = spikes(f) * E_spike
    det = sops(f) * E_det1
    comp = N_NEURONS * T * p["comparator_J"]
    io = IN * p["E_DAC_8b"] + NC * T * p["comparator_J"]
    return bias + gen + det + comp + io

# ---- competitors (matched accuracy) ----
def load_matched_macs():
    fp = os.path.join(ROOT, "data", "trackb_rightsize.json")
    if os.path.exists(fp):
        j = json.load(open(fp))
        m = j["matched_targets"].get("acc_0.982") or j["matched_targets"].get("acc_0.9822")
        if m:
            return m["macs"], m["width"]
    return None, None

MATCHED_MACS, MATCHED_W = load_matched_macs()
FULL_MACS = 268800  # the earlier (unmatched) 98.4% baseline, for contrast
EDGE_TOPS_W = (2.0, 5.0, 30.0)          # Orin .. phone NPU, INT8
E_SOP_LOIHI = (12.7e-12, 15e-12, 23.6e-12)  # ODIN / Loihi published pJ/SOP

def npu_j(macs, tops_w):
    return macs * 2.0 / (tops_w * 1e12)

def loihi_j(f, e_sop):
    return sops(f) * e_sop

out = {"network": {"in": IN, "width": WIDTH, "depth": DEPTH, "T": T, "N_neurons": N_NEURONS},
       "matched_macs_98.2": MATCHED_MACS, "matched_width": MATCHED_W, "full_macs_98.4": FULL_MACS,
       "edge_tops_w": EDGE_TOPS_W, "e_sop_loihi_J": E_SOP_LOIHI,
       "sops_at_measured_f0.43": sops(0.43)}

# ---- measured operating point ----
E_MEAS, F_MEAS = 8e-12, 0.43
print(f"SOPs(f=0.43) = {sops(0.43):.3e}   spikes = {spikes(0.43):.0f}")
opt_meas = optical_snn_j(E_MEAS, F_MEAS, base)
npu_full = npu_j(FULL_MACS, EDGE_TOPS_W[1])
npu_matched = npu_j(MATCHED_MACS, EDGE_TOPS_W[1]) if MATCHED_MACS else None
loihi_meas = loihi_j(F_MEAS, E_SOP_LOIHI[1])
print(f"\n=== measured point (E_spike=8pJ, f=0.43), nominal device ===")
print(f"  optical SNN      : {opt_meas*1e9:8.2f} nJ/inf")
print(f"  NPU (unmatched 268.8k MAC, 98.4%): {npu_full*1e9:8.2f} nJ  [the earlier wrong baseline]")
if npu_matched:
    print(f"  NPU (matched {MATCHED_MACS} MAC @98.2%, W={MATCHED_W}): {npu_matched*1e9:8.2f} nJ")
print(f"  Loihi-class SNN  : {loihi_meas*1e9:8.2f} nJ  (same SOPs x 15 pJ/SOP)")
out["measured_point"] = {"E_spike_J": E_MEAS, "f": F_MEAS,
                         "optical_nJ": opt_meas*1e9, "npu_unmatched_nJ": npu_full*1e9,
                         "npu_matched_nJ": (npu_matched*1e9 if npu_matched else None),
                         "loihi_nJ": loihi_meas*1e9,
                         "beats_npu_matched": (bool(opt_meas < npu_matched) if npu_matched else None),
                         "beats_loihi": bool(opt_meas < loihi_meas)}

# ---- breakeven surface over (E_spike, f) ----
Es = np.logspace(-14, -10, 41)      # 10 fJ .. 100 pJ
Fs = np.logspace(-2.3, 0, 41)       # 0.005 .. 1.0
def boundary_vs(competitor_j_at_f):
    """for each f, the max E_spike at which optical still beats the competitor."""
    rows = []
    for f in Fs:
        cj = competitor_j_at_f(f)
        # optical increasing in E_spike; find crossing
        win_E = [E for E in Es if optical_snn_j(E, f, base) < cj]
        rows.append({"f": float(f), "max_E_spike_win_pJ": (max(win_E)*1e12 if win_E else 0.0),
                     "competitor_nJ": cj*1e9})
    return rows

if MATCHED_MACS:
    out["boundary_vs_npu_matched"] = boundary_vs(lambda f: npu_j(MATCHED_MACS, EDGE_TOPS_W[1]))
out["boundary_vs_loihi"] = boundary_vs(lambda f: loihi_j(f, E_SOP_LOIHI[1]))

# ---- required f to win at measured E_spike=8pJ ----
def required_f(competitor_j_at_f):
    for f in Fs:                       # ascending; optical increases with f
        if optical_snn_j(E_MEAS, f, base) < competitor_j_at_f(f):
            continue
        return float(f)                # first f where optical LOSES
    return None
print("\n=== at measured E_spike=8 pJ: firing rate where optical stops winning ===")
if MATCHED_MACS:
    fn = required_f(lambda f: npu_j(MATCHED_MACS, EDGE_TOPS_W[1]))
    print(f"  vs matched NPU : optical wins for f < {fn}  (trained SNNs run f~0.3-0.45)")
    out["max_f_win_vs_npu_at_8pJ"] = fn
fl = required_f(lambda f: loihi_j(f, E_SOP_LOIHI[1]))
print(f"  vs Loihi       : optical wins for f < {fl}  (trained SNNs run f~0.3-0.45)")
out["max_f_win_vs_loihi_at_8pJ"] = fl

# ---- MC at measured point over sourced device ranges ----
rng = np.random.default_rng(31)
wins_npu, wins_loihi = 0, 0
Nmc = 4000
for _ in range(Nmc):
    p = sample_params(rng)
    o = optical_snn_j(E_MEAS, F_MEAS, p)
    tw = np.exp(rng.uniform(np.log(EDGE_TOPS_W[0]), np.log(EDGE_TOPS_W[2])))
    es = np.exp(rng.uniform(np.log(E_SOP_LOIHI[0]), np.log(E_SOP_LOIHI[2])))
    if MATCHED_MACS and o < npu_j(MATCHED_MACS, tw):
        wins_npu += 1
    if o < loihi_j(F_MEAS, es):
        wins_loihi += 1
out["mc_measured"] = {"beats_matched_npu_frac": (wins_npu/Nmc if MATCHED_MACS else None),
                      "beats_loihi_frac": wins_loihi/Nmc}
print(f"\n=== MC at measured point (n={Nmc}) ===")
if MATCHED_MACS:
    print(f"  beats matched-NPU: {wins_npu/Nmc*100:.0f}% of draws")
print(f"  beats Loihi-class: {wins_loihi/Nmc*100:.0f}% of draws")

json.dump(out, open(os.path.join(ROOT, "data", "trackb_breakeven.json"), "w"), indent=2, default=str)
print("\nwrote data/trackb_breakeven.json")
