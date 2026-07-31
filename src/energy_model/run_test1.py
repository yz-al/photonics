"""
Test 1 driver: run the sweeps, answer the five questions, emit plots + results.

Outputs:
  figures/test1_jmac_vs_N_{4b,8b}.png   -- J/MAC vs N, optical (bands) vs digital
  figures/test1_dominance_{arch}.png     -- which term dominates vs N
  data/test1_results.json                -- machine-readable answers
Run: python src/energy_model/run_test1.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sweep, crossover_band, optical_terms, optical_j_per_mac,
                   digital_j_per_mac, feasibility, sample_params, LINK_BUDGET_DB)

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)

Ns = np.unique(np.round(np.logspace(0.7, 3.9, 60)).astype(int))  # ~5 .. ~8000
N_MC = 3000
results = {"link_budget_db": LINK_BUDGET_DB, "N_grid": Ns.tolist(), "scenarios": {}}


def plot_sweep(res, title, fname, digital_label="digital (whole-chip)"):
    fig, ax = plt.subplots(figsize=(7, 5))
    Ns_ = res["Ns"]
    ax.fill_between(Ns_, res["opt_p05"] * 1e15, res["opt_p95"] * 1e15,
                    alpha=0.15, color="C0", label="optical 5-95%")
    ax.fill_between(Ns_, res["opt_p16"] * 1e15, res["opt_p84"] * 1e15,
                    alpha=0.30, color="C0", label="optical 16-84%")
    ax.plot(Ns_, res["opt_p50"] * 1e15, color="C0", lw=2, label="optical median")
    ax.fill_between(Ns_, res["dig_p16"] * 1e15, res["dig_p84"] * 1e15,
                    alpha=0.20, color="C3")
    ax.plot(Ns_, res["dig_p50"] * 1e15, color="C3", lw=2, ls="--", label=digital_label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("matrix dimension N"); ax.set_ylabel("energy per MAC (fJ)")
    ax.set_title(title); ax.legend(fontsize=8, loc="best"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=130); plt.close(fig)


def run_scenario(name, bits, arch, tile, weight_stationary, modulator,
                 enforce_loss, digital_key, plot=False, title="", ignore_loss=False):
    res = sweep(Ns, bits, n_mc=N_MC, weight_stationary=weight_stationary,
                modulator=modulator, arch=arch, tile=tile,
                enforce_loss=enforce_loss, digital_key=digital_key, seed=1234,
                ignore_loss=ignore_loss)
    cb = crossover_band(res)
    entry = {
        "bits": bits, "arch": arch, "tile": tile, "weight_stationary": weight_stationary,
        "modulator": modulator, "enforce_loss": enforce_loss, "ignore_loss": ignore_loss,
        "digital_key": digital_key, "crossover": cb,
    }
    results["scenarios"][name] = entry
    if plot:
        dl = "digital whole-chip" if digital_key.endswith(("4b", "8b")) else "digital floor (CIM/arith)"
        plot_sweep(res, title, f"test1_{name}.png", digital_label=dl)
    return res, cb


# ---------------------------------------------------------------------------
# 1) Crossover, four scenarios per precision:
#    ideal   = monolithic, loss-free, PCM   -> pure point-(a) amortisation limit
#    mono    = monolithic, loss enforced    -> the loss wall
#    tiled64 = T=64 buildable, loss enforced -> the realistic architecture
#    floor   = tiled64 vs arithmetic/CIM digital floor -> the hard target
# ---------------------------------------------------------------------------
print("=== crossover sweeps ===")
for bits in (4, 8):
    run_scenario(f"ideal_{bits}b", bits, "monolithic", 0, True, "MRM",
                 enforce_loss=False, digital_key=f"J_MAC_digital_{bits}b",
                 ignore_loss=True, plot=True,
                 title=f"Loss-free ideal (point-a limit), {bits}-bit")
    run_scenario(f"mono_{bits}b_loss", bits, "monolithic", 0, True, "MRM",
                 enforce_loss=True, digital_key=f"J_MAC_digital_{bits}b")
    run_scenario(f"tiled64_{bits}b", bits, "tiled", 64, True, "MRM",
                 enforce_loss=True, digital_key=f"J_MAC_digital_{bits}b",
                 plot=True, title=f"Tiled 64x64 (buildable), {bits}-bit, weight-stationary")
    run_scenario(f"tiled64_{bits}b_floor", bits, "tiled", 64, True, "MRM",
                 enforce_loss=True, digital_key="J_MAC_digital_floor")

for name, e in results["scenarios"].items():
    cb = e["crossover"]
    if cb["exists_frac"] == 0:
        print(f"  {name:24s}: NO crossover in {Ns[0]}..{Ns[-1]}")
    else:
        print(f"  {name:24s}: cross N50={cb['p50']:.0f}  [16-84%: {cb['p16']:.0f}-{cb['p84']:.0f}]  "
              f"exists {cb['exists_frac']*100:.0f}%")

# ---------------------------------------------------------------------------
# 2) Dominance analysis: which term dominates, and where it changes hands.
# ---------------------------------------------------------------------------
print("\n=== term dominance (nominal params) ===")
pnom = {k: v.nom for k, v in PARAMS.items()}
dominance = {}
for arch, tile in (("monolithic", 0), ("tiled", 64)):
    for bits in (4, 8):
        rows = []
        for N in [8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
            t = optical_terms(N, bits, pnom, True, "MRM", arch, tile)
            terms = {k: t[k] for k in ("conversion", "modulation", "laser", "thermal")}
            dom = max(terms, key=terms.get)
            rows.append({"N": N, **{k: terms[k] for k in terms}, "dominant": dom,
                         "loss_db": t["loss_db"],
                         "total_fJ": sum(terms.values()) * 1e15})
        dominance[f"{arch}_{bits}b"] = rows
        print(f"  {arch} {bits}b: " +
              " ".join(f"N{r['N']}:{r['dominant']}" for r in rows))
results["dominance"] = dominance

# ---------------------------------------------------------------------------
# 3) Sensitivity: which single parameter most moves the tiled crossover?
#    Hold all at nominal, move one param to its low and high, recompute crossover.
# ---------------------------------------------------------------------------
print("\n=== sensitivity of the loss-free-ideal 8b crossover to each parameter ===")
def crossover_deterministic(pvals, bits, arch, tile, digital_key,
                            enforce_loss=False, ignore_loss=True):
    dig = pvals[digital_key]
    for N in Ns:
        o = optical_j_per_mac(N, bits, pvals, True, "MRM", arch, tile,
                              enforce_loss, ignore_loss)
        if o <= dig:
            return int(N)
    return None

base = {k: v.nom for k, v in PARAMS.items()}
# sensitivity measured on the loss-free ideal (which HAS a crossover at nominal),
# so we can attribute crossover movement to each parameter.
base_cross = crossover_deterministic(base, 8, "monolithic", 0, "J_MAC_digital_8b")
sens = []
for k, prm in PARAMS.items():
    lo = dict(base); lo[k] = prm.low
    hi = dict(base); hi[k] = prm.high
    c_lo = crossover_deterministic(lo, 8, "monolithic", 0, "J_MAC_digital_8b")
    c_hi = crossover_deterministic(hi, 8, "monolithic", 0, "J_MAC_digital_8b")
    def _v(x): return x if x is not None else float("inf")
    span = abs(_v(c_hi) - _v(c_lo))
    sens.append({"param": k, "cross_low": c_lo, "cross_high": c_hi,
                 "base": base_cross, "span": None if not np.isfinite(span) else int(span)})
sens.sort(key=lambda d: (d["span"] is None, -(d["span"] or 0)))
for s in sens[:8]:
    print(f"  {s['param']:20s}: low->{s['cross_low']}  high->{s['cross_high']}  span={s['span']}")
results["sensitivity"] = sens
results["base_crossover_ideal_8b"] = base_cross

# ---------------------------------------------------------------------------
# 4) Feasibility at the monolithic crossover N (component/area/loss).
# ---------------------------------------------------------------------------
print("\n=== feasibility check ===")
feas = {}
mono8 = results["scenarios"]["ideal_8b"]["crossover"]
crossN = int(mono8["p50"]) if np.isfinite(mono8["p50"]) else 1024
for N in sorted(set([crossN, 64, 256, 1024, 4096])):
    f_mono = feasibility(N, base, "monolithic")
    feas[f"mono_N{N}"] = f_mono
    print(f"  monolithic N={N}: MZIs={f_mono['n_mzi']:,} depth={f_mono['optical_depth']} "
          f"area={f_mono['area_cm2']:.1f}cm2 ({f_mono['reticles']:.0f} reticles) "
          f"loss@0.1dB={f_mono['loss_db']['best_0p1']:.0f}dB "
          f"detectable={f_mono['detectable_best']} idleTune={f_mono['idle_tuning_W']/1e3:.1f}kW")
for T in (32, 64, 128, 256):
    f_t = feasibility(4096, base, "tiled", T)
    feas[f"tiled_T{T}_N4096"] = f_t
    print(f"  tiled T={T} (N=4096): tiles={f_t['n_tiles']:,} depth={f_t['optical_depth']} "
          f"loss@0.1dB={f_t['loss_db']['best_0p1']:.0f}dB loss@0.5dB={f_t['loss_db']['good_0p5']:.0f}dB "
          f"detectable={f_t['detectable_best']}")
results["feasibility"] = feas
# max buildable tile size at each per-MZI loss
maxtile = {}
for mzi_loss in (0.1, 0.5, 2.0):
    # 2*coupling + T*mzi_loss + T*cell*wg = LINK_BUDGET
    T = 1
    while True:
        loss = 2*base["coupling_db"] + T*mzi_loss + T*base["mzi_cell_len_um"]*1e-4*base["wg_loss_dbcm"]
        if loss > LINK_BUDGET_DB:
            break
        T += 1
    maxtile[f"mzi_{mzi_loss}dB"] = T - 1
results["max_buildable_tile"] = maxtile
print(f"  max buildable tile depth: {maxtile}")

# ---------------------------------------------------------------------------
# 5) Weight-stationary vs thermo-optic (thermal term on/off).
# ---------------------------------------------------------------------------
print("\n=== weight-stationary (PCM) vs thermo-optic ===")
ws = {}
for bits in (4, 8):
    for stat in (True, False):
        res = sweep(Ns, bits, n_mc=N_MC, weight_stationary=stat, modulator="MRM",
                    arch="tiled", tile=64, enforce_loss=True,
                    digital_key=f"J_MAC_digital_{bits}b", seed=7)
        cb = crossover_band(res)
        key = f"tiled64_{bits}b_{'pcm' if stat else 'thermo'}"
        ws[key] = cb
        tag = "PCM" if stat else "thermo-optic"
        msg = "NO crossover" if cb["exists_frac"] == 0 else \
              f"cross N50={cb['p50']:.0f} [{cb['p16']:.0f}-{cb['p84']:.0f}] exists {cb['exists_frac']*100:.0f}%"
        print(f"  {bits}b {tag:12s}: {msg}")
results["weight_stationary"] = ws

with open(os.path.join(DATA, "test1_results.json"), "w") as fh:
    json.dump(results, fh, indent=2, default=str)
print(f"\nWrote {os.path.join(DATA, 'test1_results.json')}")
print("Wrote figures to", FIG)
