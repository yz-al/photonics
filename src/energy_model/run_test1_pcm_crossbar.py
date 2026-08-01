"""
PCM-weight + mode-multiplex crossbar variant — Task B (Monte Carlo) + break-even.

Same MC protocol and digital baselines as Tests 1, A1 and the thermal-ring
crossbar, so all four are directly comparable.  Writes
data/test1_pcm_crossbar_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sample_params, crossover_band, crossbar_pcm_j_per_mac,
                   crossbar_pcm_terms, crossbar_pcm_Keff, crossbar_K_max, LINK_BUDGET_DB)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
Ns = np.unique(np.round(np.logspace(1.2, 4.5, 60)).astype(int))
N_MC = 3000
out = {"arch": "crossbar_pcm_modemux", "link_budget_db": LINK_BUDGET_DB, "scenarios": {}}
base = {k: v.nom for k, v in PARAMS.items()}

print(f"=== nominal K_wave={crossbar_K_max(base):.0f}  M_modes={base['n_modes']:.0f}  "
      f"K_eff(cap N)= up to {crossbar_K_max(base)*base['n_modes']:.0f} ===")

# ---------------------------------------------------------------------------
# Monte-Carlo crossover, both baselines, 4b & 8b, non-resonant vs resonant PCM
# ---------------------------------------------------------------------------
def sweep_pcm(bits, resonant, dkey, seed=909):
    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(N_MC):
        p = sample_params(rng)
        dig = p[dkey]; cross = np.inf
        for N in Ns:
            o = crossbar_pcm_j_per_mac(N, bits, p, resonant=resonant, enforce_loss=True)
            if o <= dig:
                cross = N; break
        crossings.append(cross)
    return crossover_band({"crossings": np.array(crossings, dtype=float)})

print("\n=== PCM+mode-mux crossover (MC, loss enforced) ===")
for bits in (4, 8):
    for resonant, rtag in ((False, "nonres"), (True, "resonant")):
        for dkey, btag in ((f"J_MAC_digital_{bits}b", "wholechip"),
                           ("J_MAC_digital_floor", "FLOOR")):
            cb = sweep_pcm(bits, resonant, dkey)
            name = f"pcm_{bits}b_{btag}_{rtag}"
            out["scenarios"][name] = {"bits": bits, "resonant": resonant,
                                      "digital_key": dkey, "crossover": cb}
            msg = "NO crossover (0%)" if cb["exists_frac"] == 0 else \
                  f"N50={cb['p50']:.0f} [{cb['p16']:.0f}-{cb['p84']:.0f}] exists {cb['exists_frac']*100:.0f}%"
            print(f"  {name:34s}: {msg}")

# ---------------------------------------------------------------------------
# Term breakdown (nominal, non-resonant) — where did the cost relocate?
# ---------------------------------------------------------------------------
print("\n=== per-MAC term breakdown (nominal, non-resonant PCM) ===")
brk = {}
for bits in (4, 8):
    rows = []
    for N in [64, 256, 1024, 4096, 16384]:
        t = crossbar_pcm_terms(N, bits, base, resonant=False)
        tot = t["conversion"] + t["laser"] + t["thermal"] + t["update"]
        dom = max(("conversion", "laser", "thermal", "update"), key=lambda k: t[k])
        rows.append({"N": N, "Keff": t["Keff"], "conversion_fJ": t["conversion"]*1e15,
                     "laser_fJ": t["laser"]*1e15, "update_fJ": t["update"]*1e15,
                     "total_fJ": tot*1e15, "dominant": dom, "bus_loss_db": t["bus_loss_db"]})
        print(f"  {bits}b N={N:6d}: Keff={t['Keff']:5.0f} conv={t['conversion']*1e15:7.1f} "
              f"laser={t['laser']*1e15:8.1f} update={t['update']*1e15:6.2f} "
              f"total={tot*1e15:8.1f}fJ dom={dom} bus={t['bus_loss_db']:.1f}dB (floor 20-40)")
    brk[f"{bits}b"] = rows
out["term_breakdown"] = brk

# ---------------------------------------------------------------------------
# Loss-wall check: bus_loss = 2*coupling + K_eff * pcm_loss.  Max buildable N.
# ---------------------------------------------------------------------------
print("\n=== loss-wall check: does K_eff PCM-element bus stay within 33 dB? ===")
lw = {}
for pcm_loss in (0.05, 0.4, 1.0, 1.5):
    Kw = crossbar_K_max(base); M = base["n_modes"]; Keff = Kw*M
    loss = 2*base["coupling_db"] + Keff*pcm_loss
    lw[f"pcm_{pcm_loss}dB"] = {"K_eff": Keff, "bus_loss_db": loss, "ok": loss <= LINK_BUDGET_DB}
    print(f"  pcm_loss={pcm_loss}dB, K_eff={Keff:.0f} -> bus_loss={loss:.1f}dB ok={loss<=LINK_BUDGET_DB}")
out["loss_wall_check"] = lw

# ---------------------------------------------------------------------------
# Break-even: inferences per weight load before PCM update energy < floor
# ---------------------------------------------------------------------------
print("\n=== weight-update break-even (inferences/load for update < 40 fJ/MAC) ===")
be = {}
for sw in (1e-12, 5e-11, 1e-9):
    n_be = sw / 40e-15   # update/MAC = sw/inferences < 40fJ -> inferences > sw/40fJ
    be[f"switch_{sw:.0e}J"] = n_be
    print(f"  PCM switch={sw:.0e} J -> need > {n_be:.0f} inferences/load (trivially met for fixed models)")
out["update_breakeven"] = be

with open(os.path.join(DATA, "test1_pcm_crossbar_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/test1_pcm_crossbar_results.json")
