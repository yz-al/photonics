"""
Track B — the decisive encoding test, run at last.

Question: does changing the OUTPUT readout encoding from a b-bit ADC to a comparator
(the one move that directly attacks the data-converter floor) move the crossover against
the digital compute floor, or does the saved converter energy reappear elsewhere?

Controlled A/B: same MVM architecture, same matrix size N, same everything — ONLY the
readout encoding changes. Three encodings at MATCHED effective precision b_eff:
  * adc              : one b_eff-bit ADC per output sample (baseline).
  * comp_bitplane    : recover b_eff bits with b_eff successive comparator decisions
                       (SAR-like), optical energy still precision-bound. OPTIMISTIC for
                       the comparator — it ignores the SAR feedback-DAC energy, so a real
                       b-bit comparator readout costs at least this and at most ~an ADC.
  * comp_stochastic  : a genuine 1-bit comparator (cheap, ~tens of photons), oversampled
                       R = 4^b_eff times to reach b_eff effective bits (0.5·log2 M law).
                       This is the honest cost of a truly 1-bit-cheap readout.

PRE-REGISTERED FALSIFICATION (stated before running):
  The verdict "the converter floor is not fixable by the readout encoding" is FALSIFIED iff
  swapping ADC -> comparator opens floor-crossings that the ADC did not:
     comp_bitplane crosses the 20-40 fJ/MAC floor in > 10% of draws in an architecture where
     adc crosses in ~0%.
  It STANDS iff the crossing fraction does not materially rise — because either (i) the
  binding term after the swap is not the converter (laser-through-loss / thermal), so cutting
  conversion cannot move the floor, or (ii) the 1-bit saving is repaid by R passes of laser
  energy (energy reappears in the source).

Run on the architecture where the converter has the BEST chance of binding (butterfly, where
conversion amortises over only log2 N), plus the crossbar, at 4- and 8-bit. Weight-stationary
PCM (thermal ~0) so the test isolates conversion vs laser, not thermal hold.

Writes data/track_b_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import (PARAMS, sample_params, crossover_band, required_detector_energy,
                   amortisation_dim, mesh_depth, mesh_transmission_db, LINK_BUDGET_DB)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
base = {k: v.nom for k, v in PARAMS.items()}
Ns = np.unique(np.round(np.logspace(1.2, 4.0, 45)).astype(int))
N_MC = 3000
out = {"arch": "track_b_readout_encoding", "link_budget_db": LINK_BUDGET_DB,
       "falsification": "comp_bitplane crosses floor in >10% where adc ~0% -> verdict FALSIFIED"}


def track_b_terms(N, b, p, arch, encoding, modulator="MRM", tile=64, enforce_loss=True):
    """Per-MAC energy + term breakdown for a given readout encoding. Weight-stationary."""
    E_ADC = p[f"E_ADC_{b}b"]; E_DAC = p[f"E_DAC_{b}b"]
    E_mod = {"MZM": p["E_mod_MZM"], "MRM": p["E_mod_MRM"],
             "plasmonic": p["E_mod_plasmonic"]}[modulator]
    E_comp = p["comparator_J"]
    A = amortisation_dim(N, arch, tile)
    depth = mesh_depth(N, arch, tile)
    loss_db = mesh_transmission_db(depth, p["coupling_db"], p["mzi_loss_db"],
                                   p["wg_loss_dbcm"], p["mzi_cell_len_um"])
    if enforce_loss and loss_db > LINK_BUDGET_DB:
        return {"jmac": np.inf, "conv": np.inf, "laser": np.inf, "mod": np.inf,
                "thermal": 0.0, "binding": "loss_wall", "loss_db": loss_db}
    trans = 10.0 ** (-min(loss_db, 300.0) / 10.0)
    WPE = p["WPE_laser"]
    E_det_b = required_detector_energy(b, p["responsivity"], p["tia_noise"], p["bandwidth"])
    E_det_1 = required_detector_energy(1, p["responsivity"], p["tia_noise"], p["bandwidth"])

    E_mod_perMAC = E_mod / A
    thermal = p["P_ps_pcm"] * (1.0 / p["bandwidth"])   # weight-stationary PCM ~0

    if encoding == "adc":
        conv = (E_DAC + E_ADC) / A
        laser = E_det_b / (A * trans * WPE)
    elif encoding == "comp_bitplane":
        conv = (E_DAC + b * E_comp) / A                # b comparator decisions (SAR-like)
        laser = E_det_b / (A * trans * WPE)            # still precision-bound
    elif encoding == "comp_stochastic":
        R = 4.0 ** b                                   # oversample to b effective bits
        conv = (E_DAC + R * E_comp) / A
        laser = R * E_det_1 / (A * trans * WPE)        # R 1-bit-detectable passes
    else:
        raise ValueError(encoding)

    jmac = conv + E_mod_perMAC + laser + thermal
    terms = {"conv": conv, "laser": laser, "mod": E_mod_perMAC, "thermal": thermal}
    binding = max(terms.items(), key=lambda kv: kv[1])[0]
    return {"jmac": jmac, **terms, "binding": binding, "loss_db": loss_db}


# ---------------------------------------------------------------------------
# 1. Term breakdown at nominal — which term binds after the swap?
# ---------------------------------------------------------------------------
print("=== 1. term breakdown (nominal, butterfly, N=1024) — what binds after ADC->comp? ===")
brk = {}
for b in (4, 8):
    for enc in ("adc", "comp_bitplane", "comp_stochastic"):
        t = track_b_terms(1024, b, base, "butterfly", enc)
        brk[f"{enc}_{b}b"] = {k: (t[k] * 1e15 if k in ("jmac", "conv", "laser", "mod", "thermal")
                                  else t[k]) for k in t}
        print(f"  {enc:16s} {b}b: conv={t['conv']*1e15:8.1f} laser={t['laser']*1e15:9.1f} "
              f"mod={t['mod']*1e15:6.1f} total={t['jmac']*1e15:9.1f} fJ  binds={t['binding']} "
              f"(floor 20-40)")
out["term_breakdown_butterfly_N1024"] = brk

# ---------------------------------------------------------------------------
# 2. MC crossover vs the 20-40 fJ compute floor, per encoding and architecture
# ---------------------------------------------------------------------------
print("\n=== 2. MC: does each readout encoding cross the 20-40 fJ floor? ===")
def sweep(arch, b, encoding, seed=909):
    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(N_MC):
        p = sample_params(rng)
        dig = p["J_MAC_digital_floor"]; cross = np.inf
        for N in Ns:
            j = track_b_terms(N, b, p, arch, encoding)["jmac"]
            if j <= dig:
                cross = N; break
        crossings.append(cross)
    return crossover_band({"crossings": np.array(crossings, dtype=float)})

mc = {}
for arch in ("butterfly", "monolithic"):
    for b in (4, 8):
        row = {}
        for enc in ("adc", "comp_bitplane", "comp_stochastic"):
            cb = sweep(arch, b, enc)
            row[enc] = cb["exists_frac"]
            msg = "NO cross (0%)" if cb["exists_frac"] == 0 else \
                  f"N50={cb['p50']:.0f} exists {cb['exists_frac']*100:.0f}%"
            print(f"  {arch:11s} {b}b {enc:16s}: {msg}")
        mc[f"{arch}_{b}b"] = row
        # pre-registered gate check
        opened = row["comp_bitplane"] - row["adc"]
        verdict = "FALSIFIED" if (row["adc"] < 0.02 and row["comp_bitplane"] > 0.10) else "STANDS"
        print(f"    -> gate: adc={row['adc']*100:.0f}% comp_bitplane={row['comp_bitplane']*100:.0f}%"
              f"  verdict {verdict}")
out["mc_crossover"] = mc

# ---------------------------------------------------------------------------
# 3. The isolated question: conversion saving vs where it reappears
# ---------------------------------------------------------------------------
print("\n=== 3. conversion cut vs laser reappearance (butterfly N=1024, nominal) ===")
iso = {}
for b in (4, 8):
    a = track_b_terms(1024, b, base, "butterfly", "adc")
    cbp = track_b_terms(1024, b, base, "butterfly", "comp_bitplane")
    cst = track_b_terms(1024, b, base, "butterfly", "comp_stochastic")
    conv_cut = a["conv"] / cbp["conv"]
    laser_blowup = cst["laser"] / a["laser"]
    iso[f"{b}b"] = {"conv_cut_bitplane_x": conv_cut,
                    "adc_total_fJ": a["jmac"] * 1e15, "bitplane_total_fJ": cbp["jmac"] * 1e15,
                    "stochastic_total_fJ": cst["jmac"] * 1e15,
                    "stochastic_laser_blowup_x": laser_blowup,
                    "adc_binding": a["binding"], "bitplane_binding": cbp["binding"]}
    print(f"  {b}b: comp_bitplane cuts conversion ×{conv_cut:.0f}, but total {a['jmac']*1e15:.0f}"
          f"->{cbp['jmac']*1e15:.0f} fJ (binds on {cbp['binding']}); "
          f"stochastic laser ×{laser_blowup:.0f} -> {cst['jmac']*1e15:.0f} fJ total")
out["isolation"] = iso

with open(os.path.join(DATA, "track_b_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/track_b_results.json")
