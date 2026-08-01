"""
Track B — photonic spiking / time-domain encoding energy model.

Metric CHANGES: joules per EQUIVALENT MAC (comparator receiver, not ADC).
Same MC protocol / baselines as the other architectures, PLUS an on-chip
laser-bias power budget (large-N amortisation is not buildable).  First
reproduces the Xiang OEA 2026 anchor (~1 pJ/op).  Writes
data/test1_spiking_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sample_params, crossover_band, spiking_terms,
                   spiking_j_per_eqmac, reproduce_xiang)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
Ns = np.unique(np.round(np.logspace(1.2, 4.5, 60)).astype(int))
N_MC = 3000
out = {"arch": "spiking", "scenarios": {}}
base = {k: v.nom for k, v in PARAMS.items()}

# 0. Model check: reproduce Xiang OEA 2026 (~987 GOPS/W ~ 1 pJ/op)
print("=== Xiang OEA 2026 reproduction (16-ch DFB-SA, 8 ops/neuron, 5 GHz, 40 mW/neuron) ===")
x = reproduce_xiang(base)
print(f"  total_power={x['total_power_W']:.3f} W  throughput={x['throughput_GOPS']:.0f} GOPS  "
      f"-> {x['GOPS_per_W']:.0f} GOPS/W = {x['j_per_op']*1e12:.2f} pJ/op  (paper: 987.65 GOPS/W, 1.01 pJ/op)")
out["xiang_reproduction"] = {"GOPS_per_W": x["GOPS_per_W"], "pJ_per_op": x["j_per_op"]*1e12,
                             "paper_GOPS_per_W": 987.65, "paper_pJ_per_op": 1.01}

# 1. Per-eqMAC term breakdown (nominal) — which term binds
print("\n=== per-equivalent-MAC breakdown (nominal), and bias-power buildability ===")
brk = {}
for N in [64, 256, 1024, 4096, 16384]:
    t = spiking_terms(N, base)
    brk[str(N)] = {k: (t[k]*1e15 if k in ("j_per_eqmac","bias","gen","det","comp") else t[k])
                   for k in t}
    print(f"  N={N:6d}: J/eqMAC={t['j_per_eqmac']*1e15:8.1f}fJ [bias={t['bias']*1e15:.1f} "
          f"gen={t['gen']*1e15:.2f} det={t['det']*1e15:.3f} comp={t['comp']*1e15:.2f}] "
          f"dom={t['dominant']} bias_power={t['bias_power_W']:.2f}W (floor 20-40)")
out["term_breakdown"] = brk

# 2. Buildability limit: at N capped by laser power budget, does bias/eqMAC reach floor?
print("\n=== bias/eqMAC at the on-chip power-budget limit (N=budget/P_bias) ===")
budget_check = {}
for Pb_mW, label in [(40, "DFB-SA measured"), (10, "VCSEL-class"), (0.5, "nanolaser projection")]:
    for bud in [3.0]:
        Nmax = bud / (Pb_mW*1e-3)
        bias_eqmac = (Pb_mW*1e-3) * base["timesteps_T"] / (Nmax * base["bandwidth"])
        budget_check[f"{label}_{Pb_mW}mW"] = {"N_max": Nmax, "bias_eqmac_fJ": bias_eqmac*1e15,
                                              "reaches_floor": bias_eqmac <= 40e-15}
        print(f"  {label:22s} P_bias={Pb_mW:5.1f}mW: N_max={Nmax:7.0f}  "
              f"bias/eqMAC={bias_eqmac*1e15:8.1f}fJ  reaches_floor={bias_eqmac<=40e-15}")
out["power_budget_check"] = budget_check

# 3. Monte-Carlo crossover vs floor, WITH the laser-power buildability cap
def sweep_spk(dkey, enforce_power, seed=1111):
    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(N_MC):
        p = sample_params(rng)
        dig = p[dkey]; cross = np.inf
        for N in Ns:
            if spiking_j_per_eqmac(N, p, enforce_power=enforce_power) <= dig:
                cross = N; break
        crossings.append(cross)
    return crossover_band({"crossings": np.array(crossings, dtype=float)})

print("\n=== spiking J/eqMAC crossover (MC) — floor, with/without power cap ===")
for dkey, btag in (("J_MAC_digital_4b", "wholechip4b"), ("J_MAC_digital_floor", "FLOOR")):
    for ep in (False, True):
        cb = sweep_spk(dkey, ep)
        name = f"spk_{btag}_{'buildable' if ep else 'unbounded'}"
        out["scenarios"][name] = {"digital_key": dkey, "enforce_power": ep, "crossover": cb}
        msg = "NO crossover (0%)" if cb["exists_frac"] == 0 else \
              f"N50={cb['p50']:.0f} [{cb['p16']:.0f}-{cb['p84']:.0f}] exists {cb['exists_frac']*100:.0f}%"
        print(f"  {name:32s}: {msg}")

# 4. Receiver saving (the real, sourced advantage) — but not where the energy is
out["receiver"] = {"comparator_J": base["comparator_J"],
                   "vs_ADC4b": base["E_ADC_4b"]/base["comparator_J"],
                   "vs_ADC8b": base["E_ADC_8b"]/base["comparator_J"]}
print(f"\n  receiver: comparator {base['comparator_J']*1e15:.0f} fJ = "
      f"{base['E_ADC_4b']/base['comparator_J']:.0f}x cheaper than ADC-4b, "
      f"{base['E_ADC_8b']/base['comparator_J']:.0f}x than ADC-8b (real, but in readout not neuron)")

with open(os.path.join(DATA, "test1_spiking_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/test1_spiking_results.json")
