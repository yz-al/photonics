"""
A1 — rerun the Test-1 energy model with an O(log N) BUTTERFLY/FFT mesh.

Butterfly: log2(N) stages, N/2 MZIs/stage -> optical depth log2(N) (loss stops
being the wall) but only ~N*log2(N) MACs, so conversion amortises over log2(N),
not N.  Question: does an O(log N) mesh beat the COMPUTE FLOOR inside the link
budget, at any N?

Outputs data/test1_butterfly_results.json and prints the question-4 table +
crossover bands (both baselines, both weight models, MC over the same ranges).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sweep, crossover_band, feasibility, optical_terms,
                   amortisation_dim, mesh_depth, LINK_BUDGET_DB)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
Ns = np.unique(np.round(np.logspace(1.2, 4.5, 60)).astype(int))   # ~16 .. ~32000
N_MC = 3000
out = {"arch": "butterfly", "link_budget_db": LINK_BUDGET_DB, "scenarios": {}}

# ---------------------------------------------------------------------------
# Question 4 — buildability of the butterfly mesh (loss is no longer the wall)
# ---------------------------------------------------------------------------
print("=== A1 Q4: butterfly feasibility (depth = log2 N) ===")
q4 = {}
base = {k: v.nom for k, v in PARAMS.items()}
for N in [64, 256, 1024, 4096, 16384]:
    f = feasibility(N, base, arch="butterfly")
    q4[str(N)] = f
    print(f"  N={N:6d}: MZIs={f['n_mzi']:>10,} PS={f['n_phase_shifters']:>10,} depth={f['optical_depth']:>2} "
          f"area={f['area_cm2']:6.2f}cm2 loss[0.1/0.5/2.0]="
          f"{f['loss_db']['best_0p1']:4.1f}/{f['loss_db']['good_0p5']:4.1f}/{f['loss_db']['typ_2p0']:5.1f}dB "
          f"detect@0.1={f['detectable_best']} idle={f['idle_tuning_W']/1e3:.2f}kW")
out["q4_feasibility"] = q4
# max N still inside 33 dB budget at each per-MZI loss (depth = log2 N)
maxN = {}
for mzi in (0.1, 0.5, 2.0):
    Nmax = None
    for N in [2**k for k in range(4, 31)]:
        depth = mesh_depth(N, "butterfly", 0)
        loss = 2*base["coupling_db"] + depth*mzi + depth*base["mzi_cell_len_um"]*1e-4*base["wg_loss_dbcm"]
        if loss <= LINK_BUDGET_DB:
            Nmax = N
    maxN[f"mzi_{mzi}dB"] = Nmax
out["max_buildable_N"] = maxN
print(f"  max buildable N within 33 dB (depth=log2 N): {maxN}")

# ---------------------------------------------------------------------------
# Crossover bands — butterfly, both baselines, both weight models, 4b & 8b
# ---------------------------------------------------------------------------
print("\n=== A1: butterfly crossover (MC, loss enforced) ===")
def run(name, bits, ws, dkey):
    res = sweep(Ns, bits, n_mc=N_MC, weight_stationary=ws, modulator="MRM",
                arch="butterfly", enforce_loss=True, digital_key=dkey, seed=2025)
    cb = crossover_band(res)
    out["scenarios"][name] = {"bits": bits, "weight_stationary": ws, "digital_key": dkey,
                              "crossover": cb}
    if cb["exists_frac"] == 0:
        print(f"  {name:34s}: NO crossover (0% of draws)")
    else:
        print(f"  {name:34s}: N50={cb['p50']:.0f} [16-84%:{cb['p16']:.0f}-{cb['p84']:.0f}] exists {cb['exists_frac']*100:.0f}%")

for bits in (4, 8):
    for ws, wtag in ((True, "pcm"), (False, "thermo")):
        run(f"bfly_{bits}b_wholechip_{wtag}", bits, ws, f"J_MAC_digital_{bits}b")
        run(f"bfly_{bits}b_FLOOR_{wtag}", bits, ws, "J_MAC_digital_floor")

# ---------------------------------------------------------------------------
# Term breakdown at nominal — show conversion/MAC = (E_DAC+E_ADC+E_mod)/log2(N)
# ---------------------------------------------------------------------------
print("\n=== A1: per-MAC term breakdown (nominal, PCM), vs compute floor 40 fJ ===")
brk = {}
for bits in (4, 8):
    rows = []
    for N in [64, 256, 1024, 4096, 16384, 2**20]:
        t = optical_terms(N, bits, base, True, "MRM", "butterfly")
        tot = t["conversion"] + t["modulation"] + t["laser"] + t["thermal"]
        rows.append({"N": N, "amort_log2N": round(amortisation_dim(N, "butterfly", 0), 2),
                     "conversion_fJ": t["conversion"]*1e15, "laser_fJ": t["laser"]*1e15,
                     "thermal_fJ": t["thermal"]*1e15, "total_fJ": tot*1e15,
                     "loss_db": t["loss_db"]})
        print(f"  {bits}b N={N:>8d}: amort(log2N)={amortisation_dim(N,'butterfly',0):5.1f} "
              f"conv={t['conversion']*1e15:9.1f}fJ laser={t['laser']*1e15:7.1f}fJ "
              f"total={tot*1e15:9.1f}fJ  (floor 20-40 fJ, loss={t['loss_db']:.1f}dB)")
    brk[f"{bits}b"] = rows
out["term_breakdown"] = brk

# ---------------------------------------------------------------------------
# The squeeze: dense-equivalent expressivity needs ~log2(N) stacked butterfly
# factors -> depth (log2 N)^2 -> loss grows.  Show it fails the link budget.
# ---------------------------------------------------------------------------
print("\n=== A1: dense-equivalent butterfly stack (G=log2N factors) — loss check ===")
squeeze = {}
for N in [256, 1024, 4096, 16384]:
    g = max(1, int(np.ceil(np.log2(N))))
    depth = g * g                       # G factors x log2(N) depth each
    loss05 = 2*base["coupling_db"] + depth*0.5 + depth*base["mzi_cell_len_um"]*1e-4*base["wg_loss_dbcm"]
    squeeze[str(N)] = {"factors": g, "stacked_depth": depth, "loss_0p5db": loss05,
                       "detectable": loss05 <= LINK_BUDGET_DB}
    print(f"  N={N:6d}: {g} factors -> depth {depth:4d} -> {loss05:6.1f}dB @0.5dB/MZI "
          f"detectable={loss05 <= LINK_BUDGET_DB}")
out["dense_equiv_squeeze"] = squeeze

with open(os.path.join(DATA, "test1_butterfly_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/test1_butterfly_results.json")
