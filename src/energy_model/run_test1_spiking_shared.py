"""
Shared-source spiking variant — one laser feeds many passive spiking elements.
Tests whether dividing the standing bias by the share factor S clears the floor,
or whether the per-element optical drive / bus insertion loss reappears.

Same MC protocol / baselines as the rest of the programme.
Writes data/test1_spiking_shared_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sample_params, crossover_band, spiking_shared_terms,
                   LINK_BUDGET_DB)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
Ns = np.unique(np.round(np.logspace(1.2, 4.5, 60)).astype(int))
N_MC = 3000
out = {"arch": "spiking_shared", "link_budget_db": LINK_BUDGET_DB, "scenarios": {}}
base = {k: v.nom for k, v in PARAMS.items()}


def bio_j_shared(N, p, fully_shared=True):
    S = N if fully_shared else min(p["neurons_per_source"], N)
    if N * p["neuron_bias_mW"] * 1e-3 / max(S, 1) > p["laser_power_budget_W"]:
        # even the shared source(s) must fit the on-chip optical power budget
        pass
    return spiking_shared_terms(N, p, neurons_per_source=S)["j_per_eqmac"]


# ---------------------------------------------------------------------------
# Crossover vs both baselines; fully-shared (S=N) vs partial (S=neurons_per_source)
# ---------------------------------------------------------------------------
def sweep(fully_shared, dkey, seed=4242):
    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(N_MC):
        p = sample_params(rng)
        dig = p[dkey]; cross = np.inf
        for N in Ns:
            if bio_j_shared(N, p, fully_shared) <= dig:
                cross = N; break
        crossings.append(cross)
    return crossover_band({"crossings": np.array(crossings, dtype=float)})

print("=== shared-source spiking crossover (MC) ===")
for fs, ftag in ((True, "fullyshared"), (False, "partial")):
    for dkey, btag in (("J_MAC_digital_4b", "wholechip4b"),
                       ("J_MAC_digital_floor", "FLOOR")):
        cb = sweep(fs, dkey)
        name = f"shared_{ftag}_{btag}"
        out["scenarios"][name] = {"fully_shared": fs, "digital_key": dkey, "crossover": cb}
        msg = "NO crossover (0%)" if cb["exists_frac"] == 0 else \
              f"N50={cb['p50']:.0f} [{cb['p16']:.0f}-{cb['p84']:.0f}] exists {cb['exists_frac']*100:.0f}%"
        print(f"  {name:28s}: {msg}")

# ---------------------------------------------------------------------------
# Term breakdown (nominal, fully shared) — where does the cost relocate?
# ---------------------------------------------------------------------------
print("\n=== term breakdown (nominal, fully-shared S=N) vs floor 20-40 fJ ===")
brk = {}
rows = []
for N in [64, 256, 1024, 4096]:
    t = spiking_shared_terms(N, base, neurons_per_source=N)
    rows.append({"N": N, "S": t["S"], "source_fJ": t["source"]*1e15, "det_fJ": t["det"]*1e15,
                 "gen_fJ": t["gen"]*1e15, "comp_fJ": t["comp"]*1e15, "total_fJ": t["j_per_eqmac"]*1e15,
                 "dominant": t["dominant"], "bus_loss_db": t["bus_loss_db"],
                 "delivery_limited": t["delivery_limited"]})
    print(f"  N={N:5d}: source={t['source']*1e15:8.1f} det={t['det']*1e15:9.1f} gen={t['gen']*1e15:6.1f} "
          f"comp={t['comp']*1e15:5.1f} total={t['j_per_eqmac']*1e15:9.1f}fJ dom={t['dominant']} "
          f"bus={t['bus_loss_db']:.0f}dB deliv_lim={t['delivery_limited']}")
brk["fully_shared"] = rows
out["term_breakdown"] = brk

# ---------------------------------------------------------------------------
# Share-factor sweep at N=1024: how much does sharing buy before the floor reappears?
# ---------------------------------------------------------------------------
print("\n=== share-factor sweep (N=1024): source vs delivery/bus as S grows ===")
sf = []
for S in [1, 8, 64, 256, 1024]:
    t = spiking_shared_terms(1024, base, neurons_per_source=S)
    sf.append({"S": S, "source_fJ": t["source"]*1e15, "det_fJ": t["det"]*1e15,
               "total_fJ": t["j_per_eqmac"]*1e15, "dominant": t["dominant"],
               "bus_loss_db": t["bus_loss_db"], "delivery_limited": t["delivery_limited"]})
    print(f"  S={S:5d}: source={t['source']*1e15:8.1f} det={t['det']*1e15:9.1f} "
          f"total={t['j_per_eqmac']*1e15:9.1f}fJ dom={t['dominant']} bus={t['bus_loss_db']:.0f}dB "
          f"deliv_lim={t['delivery_limited']}")
out["share_sweep_N1024"] = sf

# ---------------------------------------------------------------------------
# Bus-loss wall: S passive elements on the shared bus (1-bit detection tolerant)
# ---------------------------------------------------------------------------
print("\n=== bus-loss wall: S passive elements per shared bus ===")
lw = {}
for elem in (0.05, 0.3, 1.0):
    Smax = int(np.floor((LINK_BUDGET_DB - 2*base["coupling_db"]) / elem))
    lw[f"elem_{elem}dB"] = {"max_S_within_budget": Smax}
    print(f"  elem_loss={elem}dB -> max {Smax} elements per bus within 33 dB "
          f"(1-bit spike detection tolerates more)")
out["bus_loss_wall"] = lw

with open(os.path.join(DATA, "test1_spiking_shared_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/test1_spiking_shared_results.json")
