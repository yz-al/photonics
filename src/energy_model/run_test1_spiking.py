"""
Track B — photonic spiking / time-domain encoding energy model.

Metric CHANGES: joules per synaptic operation (SOP) and per EQUIVALENT MAC (for
the floor comparison), because a spiking receiver is a comparator (fJ), not an
ADC (pJ).  Same MC protocol / baselines as the other architectures.  First
reproduces the published Xiang OEA 2026 anchor (~1 pJ/op) as a model check.
Writes data/test1_spiking_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sample_params, crossover_band, spiking_terms,
                   spiking_j_per_eqmac, reproduce_xiang, LINK_BUDGET_DB)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
Ns = np.unique(np.round(np.logspace(1.2, 4.5, 60)).astype(int))
N_MC = 3000
out = {"arch": "spiking", "scenarios": {}}
base = {k: v.nom for k, v in PARAMS.items()}

# ---------------------------------------------------------------------------
# 0. Model check: reproduce Xiang OEA 2026 (~987 GOPS/W ~ 1 pJ/op) for DFB-SA
# ---------------------------------------------------------------------------
print("=== Xiang OEA 2026 reproduction (DFB-SA spiking array) ===")
xiang = {}
for N in [4, 8, 16, 32]:
    x = reproduce_xiang(base, N=N)
    xiang[f"N{N}"] = {"GOPS_per_W": x["GOPS_per_W"], "pJ_per_op": x["j_per_op"]*1e12}
    print(f"  N={N:3d}: {x['GOPS_per_W']:8.0f} GOPS/W  = {x['j_per_op']*1e12:6.2f} pJ/op  "
          f"(target ~987 GOPS/W ~1 pJ/op)")
out["xiang_reproduction"] = xiang

# ---------------------------------------------------------------------------
# 1. Per-MAC term breakdown (nominal) — which term binds, and vs the floor
# ---------------------------------------------------------------------------
print("\n=== per-equivalent-MAC breakdown (nominal) ===")
brk = {}
for N in [64, 256, 1024, 4096, 16384]:
    t = spiking_terms(N, base)
    brk[str(N)] = {"j_per_sop_fJ": t["j_per_sop"]*1e15, "j_per_eqmac_fJ": t["j_per_eqmac"]*1e15,
                   "bias_fJ": t["bias_sop"]*1e15, "pulse_fJ": t["pulse_sop"]*1e15,
                   "comp_fJ": t["comp_sop"]*1e15, "dominant": t["dominant"]}
    print(f"  N={N:6d}: J/SOP={t['j_per_sop']*1e15:8.1f} J/eqMAC={t['j_per_eqmac']*1e15:8.1f}fJ "
          f"[bias={t['bias_sop']*1e15:.1f} pulse={t['pulse_sop']*1e15:.2f} comp={t['comp_sop']*1e15:.2f}] "
          f"dom={t['dominant']} (floor 20-40)")
out["term_breakdown"] = brk

# ---------------------------------------------------------------------------
# 2. Monte-Carlo crossover of J/eqMAC vs both baselines (loss/accuracy-agnostic)
# ---------------------------------------------------------------------------
def sweep_spk(dkey, seed=1111):
    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(N_MC):
        p = sample_params(rng)
        dig = p[dkey]; cross = np.inf
        for N in Ns:
            if spiking_j_per_eqmac(N, p) <= dig:
                cross = N; break
        crossings.append(cross)
    return crossover_band({"crossings": np.array(crossings, dtype=float)})

print("\n=== spiking J/eqMAC crossover (MC) ===")
for dkey, btag in (("J_MAC_digital_4b", "wholechip4b"), ("J_MAC_digital_8b", "wholechip8b"),
                   ("J_MAC_digital_floor", "FLOOR")):
    cb = sweep_spk(dkey)
    out["scenarios"][f"spk_{btag}"] = {"digital_key": dkey, "crossover": cb}
    msg = "NO crossover (0%)" if cb["exists_frac"] == 0 else \
          f"N50={cb['p50']:.0f} [{cb['p16']:.0f}-{cb['p84']:.0f}] exists {cb['exists_frac']*100:.0f}%"
    print(f"  spk vs {btag:14s}: {msg}")

# ---------------------------------------------------------------------------
# 3. Comparator-vs-ADC: quantify the receiver saving (the track's thesis)
# ---------------------------------------------------------------------------
print("\n=== receiver: comparator vs ADC ===")
recv = {"comparator_J_nom": base["comparator_J"], "E_ADC_4b": base["E_ADC_4b"],
        "E_ADC_8b": base["E_ADC_8b"],
        "comparator_cheaper_than_ADC4b": base["E_ADC_4b"]/base["comparator_J"]}
print(f"  comparator {base['comparator_J']*1e15:.1f} fJ vs ADC-4b {base['E_ADC_4b']*1e12:.2f} pJ "
      f"-> {base['E_ADC_4b']/base['comparator_J']:.0f}x cheaper receiver")
out["receiver"] = recv

with open(os.path.join(DATA, "test1_spiking_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/test1_spiking_results.json")
