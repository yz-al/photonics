"""
Crossbar (microring weight-bank) energy model — Task B (Monte Carlo) + Task C.

Same MC protocol and digital baselines as Tests 1 and A1, so all three
architectures are directly comparable.  Writes data/test1_crossbar_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sample_params, crossover_band, crossbar_j_per_mac,
                   crossbar_terms, crossbar_feasibility, crossbar_K_max,
                   crossbar_yield, LINK_BUDGET_DB)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
Ns = np.unique(np.round(np.logspace(1.2, 4.5, 60)).astype(int))
N_MC = 3000
out = {"arch": "crossbar", "link_budget_db": LINK_BUDGET_DB, "scenarios": {}}
base = {k: v.nom for k, v in PARAMS.items()}

# ---------------------------------------------------------------------------
# Feasibility table (K, rings, bus loss, thermal, yield, area)
# ---------------------------------------------------------------------------
print("=== crossbar feasibility (nominal) ===")
print(f"  nominal K_max = {crossbar_K_max(base):.0f}")
q = {}
for N in [64, 256, 1024, 4096, 16384]:
    f = crossbar_feasibility(N, base)
    q[str(N)] = f
    print(f"  N={N:6d}: K={f['K_used']:.0f} rings={f['n_rings']:>12,} "
          f"bus_loss[0.005/0.02/0.05]={f['bus_loss_db']['best_0p005']:.1f}/"
          f"{f['bus_loss_db']['typ_0p02']:.1f}/{f['bus_loss_db']['high_0p05']:.1f}dB "
          f"detect={f['detectable_typ']} yield=10^{f['array_log10_yield']:.0f} "
          f"area={f['area_cm2']:.1f}cm2 idle_stab={f['idle_stab_kW']:.1f}kW")
out["feasibility"] = q

# ---------------------------------------------------------------------------
# Monte-Carlo crossover, both baselines, 4b & 8b, PCM vs thermal
# ---------------------------------------------------------------------------
def cb_sweep(bits, ws, dkey, seed=808):
    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(N_MC):
        p = sample_params(rng)
        dig = p[dkey]
        cross = np.inf
        for N in Ns:
            o = crossbar_j_per_mac(N, bits, p, weight_stationary=ws, enforce_loss=True)
            if o <= dig:
                cross = N; break
        crossings.append(cross)
    return crossover_band({"crossings": np.array(crossings, dtype=float)})

print("\n=== crossbar crossover (MC, loss enforced) ===")
for bits in (4, 8):
    for ws, wtag in ((True, "pcm"), (False, "thermo")):
        for dkey, btag in ((f"J_MAC_digital_{bits}b", "wholechip"),
                           ("J_MAC_digital_floor", "FLOOR")):
            cb = cb_sweep(bits, ws, dkey)
            name = f"xbar_{bits}b_{btag}_{wtag}"
            out["scenarios"][name] = {"bits": bits, "weight_stationary": ws,
                                      "digital_key": dkey, "crossover": cb}
            msg = "NO crossover (0%)" if cb["exists_frac"] == 0 else \
                  f"N50={cb['p50']:.0f} [{cb['p16']:.0f}-{cb['p84']:.0f}] exists {cb['exists_frac']*100:.0f}%"
            print(f"  {name:30s}: {msg}")

# ---------------------------------------------------------------------------
# Term breakdown -> which failure mode binds (Task C q2)
# ---------------------------------------------------------------------------
print("\n=== per-MAC term breakdown (nominal, PCM) — which term dominates ===")
brk = {}
for bits in (4, 8):
    rows = []
    for N in [64, 256, 1024, 4096, 16384]:
        t = crossbar_terms(N, bits, base, True)
        tot = t["conversion"] + t["laser"] + t["thermal_stab"]
        dom = max(("conversion", "laser", "thermal_stab"),
                  key=lambda k: t[k])
        rows.append({"N": N, "K": t["K"], "conversion_fJ": t["conversion"]*1e15,
                     "laser_fJ": t["laser"]*1e15, "thermal_stab_fJ": t["thermal_stab"]*1e15,
                     "total_fJ": tot*1e15, "dominant": dom, "bus_loss_db": t["bus_loss_db"]})
        print(f"  {bits}b N={N:6d}: K={t['K']:.0f} conv={t['conversion']*1e15:7.1f} "
              f"laser={t['laser']*1e15:7.1f} therm_stab={t['thermal_stab']*1e15:10.1f} "
              f"total={tot*1e15:10.1f}fJ dom={dom} bus={t['bus_loss_db']:.1f}dB (floor 20-40)")
    brk[f"{bits}b"] = rows
out["term_breakdown"] = brk

# ---------------------------------------------------------------------------
# Task C q3: does depth~1 survive bus loss? Max N within budget vs through-loss.
# ---------------------------------------------------------------------------
print("\n=== bus-loss check: is K-ring bus traversal within budget? ===")
busck = {}
for thru in (0.005, 0.02, 0.05):
    # bus has K rings; K bounded by K_max(nominal). Worst case K = K_max.
    Kmax = crossbar_K_max(base)
    loss = 2*base["coupling_db"] + Kmax*thru
    busck[f"thru_{thru}"] = {"K_max": Kmax, "bus_loss_db": loss, "ok": loss <= LINK_BUDGET_DB}
    print(f"  through={thru}dB/ring, K={Kmax:.0f} -> bus_loss={loss:.1f}dB ok={loss<=LINK_BUDGET_DB}")
out["bus_loss_check"] = busck

# ---------------------------------------------------------------------------
# Task C q2 detail: thermal-stabilisation-only floor vs compute floor
# ---------------------------------------------------------------------------
print("\n=== thermal-stabilisation-only J/MAC (isolates the binding mode) ===")
tstab = {}
for N in [64, 256, 1024, 4096]:
    row = {}
    for label, pstab in (("athermal_0.5mW", 0.5e-3), ("nominal_4mW", 4e-3), ("active_20mW", 20e-3)):
        pp = dict(base); pp["ring_stab_mW"] = pstab
        t = crossbar_terms(N, 4, pp, True)
        row[label] = t["thermal_stab"]*1e15
    tstab[str(N)] = row
    print(f"  N={N:5d}: thermal_stab/MAC  athermal={row['athermal_0.5mW']:.0f}  "
          f"nominal={row['nominal_4mW']:.0f}  active={row['active_20mW']:.0f} fJ (floor 20-40)")
out["thermal_stab_only"] = tstab

with open(os.path.join(DATA, "test1_crossbar_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/test1_crossbar_results.json")
