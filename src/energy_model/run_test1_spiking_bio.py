"""
Spiking bio-sparsity variant — Tasks 2-4 + pre-registered gate.

Gated vs continuous source, biological sparsity sweep, passive PCM weights,
single final readout (loss-wall check).  Same MC protocol / baselines as the
rest of the programme.  Writes data/test1_spiking_bio_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sample_params, crossover_band, spiking_bio_terms,
                   spiking_single_readout_maxlayers, LINK_BUDGET_DB)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
Ns = np.unique(np.round(np.logspace(1.2, 4.5, 60)).astype(int))
N_MC = 3000
out = {"arch": "spiking_bio", "link_budget_db": LINK_BUDGET_DB, "scenarios": {}}
base = {k: v.nom for k, v in PARAMS.items()}


def bio_j(N, p, gated, single_readout=True, n_layers=8):
    # amortisation capped by on-chip laser-bias power budget
    if N * p["neuron_bias_mW"] * 1e-3 > p["laser_power_budget_W"]:
        return np.inf
    return spiking_bio_terms(N, p, gated=gated, single_readout=single_readout,
                             n_layers=n_layers)["j_per_eqmac"]


# ---------------------------------------------------------------------------
# Task 2/3: crossover vs BOTH baselines, continuous vs gated source
# ---------------------------------------------------------------------------
def sweep(gated, dkey, seed=1234):
    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(N_MC):
        p = sample_params(rng)
        dig = p[dkey]; cross = np.inf
        for N in Ns:
            if bio_j(N, p, gated) <= dig:
                cross = N; break
        crossings.append(cross)
    return crossover_band({"crossings": np.array(crossings, dtype=float)})

print("=== spiking bio-variant crossover (MC; passive weights, single readout) ===")
for gated, gtag in ((False, "continuous"), (True, "gated")):
    for dkey, btag in (("J_MAC_digital_4b", "wholechip4b"),
                       ("J_MAC_digital_floor", "FLOOR")):
        cb = sweep(gated, dkey)
        name = f"spk_{gtag}_{btag}"
        out["scenarios"][name] = {"gated": gated, "digital_key": dkey, "crossover": cb}
        msg = "NO crossover (0%)" if cb["exists_frac"] == 0 else \
              f"N50={cb['p50']:.0f} [{cb['p16']:.0f}-{cb['p84']:.0f}] exists {cb['exists_frac']*100:.0f}%"
        print(f"  {name:26s}: {msg}")

# ---------------------------------------------------------------------------
# Task 2: term breakdown continuous vs gated at nominal, N at power-budget cap
# ---------------------------------------------------------------------------
print("\n=== term breakdown (nominal; N=256) vs compute floor 20-40 fJ ===")
brk = {}
for gated, gtag in ((False, "continuous"), (True, "gated")):
    rows = []
    for N in [64, 256, 1024]:
        t = spiking_bio_terms(N, base, gated=gated, single_readout=True, n_layers=8)
        tot = t["j_per_eqmac"]
        rows.append({"N": N, "gated": gated, "source_fJ": t["source"]*1e15,
                     "gen_fJ": t["gen"]*1e15, "det_fJ": t["det"]*1e15, "comp_fJ": t["comp"]*1e15,
                     "total_fJ": tot*1e15, "dominant": t["dominant"], "s": t["s_spikes_per_neuron"]})
        print(f"  {gtag:10s} N={N:5d}: source={t['source']*1e15:8.1f} gen={t['gen']*1e15:6.1f} "
              f"det={t['det']*1e15:6.1f} comp={t['comp']*1e15:6.1f} total={tot*1e15:8.1f}fJ "
              f"dom={t['dominant']}")
    brk[gtag] = rows
out["term_breakdown"] = brk

# ---------------------------------------------------------------------------
# Task 3: sparsity sweep -> J/eqMAC vs activation, both regimes; floor-cross point
# ---------------------------------------------------------------------------
print("\n=== sparsity sweep (N=256, nominal): does any activation cross the floor? ===")
spar = {}
for gated, gtag in ((False, "continuous"), (True, "gated")):
    rows = []
    for act in [0.005, 0.01, 0.05, 0.10, 0.20]:
        p = dict(base); p["snn_activation"] = act
        t = spiking_bio_terms(256, p, gated=gated, single_readout=True, n_layers=8)
        rows.append({"activation": act, "total_fJ": t["j_per_eqmac"]*1e15,
                     "source_fJ": t["source"]*1e15, "dominant": t["dominant"]})
        print(f"  {gtag:10s} act={act*100:4.1f}%: total={t['j_per_eqmac']*1e15:8.1f}fJ "
              f"source={t['source']*1e15:8.1f}fJ dom={t['dominant']} (floor 20-40)")
    spar[gtag] = rows
out["sparsity_sweep"] = spar

# ---------------------------------------------------------------------------
# Task 4: single-readout loss wall — how many layers propagate before < 33 dB?
# ---------------------------------------------------------------------------
print("\n=== Task 4: single-readout loss wall (layers before signal < 33 dB) ===")
lw = {}
for pcm in (0.1, 0.4, 1.0):
    p = dict(base); p["pcm_loss_db"] = pcm
    L = spiking_single_readout_maxlayers(p)
    lw[f"pcm_{pcm}dB"] = {"per_layer_loss_db": pcm + base["coupling_db"], "max_layers": L}
    print(f"  pcm_loss={pcm}dB (+coupling {base['coupling_db']}dB/layer) -> max {L} layers optical")
out["single_readout_maxlayers"] = lw

with open(os.path.join(DATA, "test1_spiking_bio_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/test1_spiking_bio_results.json")
