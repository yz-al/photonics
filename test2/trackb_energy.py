"""Track B energy accounting — JOULES PER INFERENCE at fixed accuracy, edge-inference.

Metric (stated explicitly): NOT joules-per-MAC. We price joules-per-INFERENCE and
joules-per-synaptic-op for two optical edge architectures against an EDGE digital
competitor (phone-NPU / automotive-SoC), and compare AT THE ACCURACY EACH OPTICAL NET
ACTUALLY REACHES (read from data/trackb_accuracy.json).

Reuses src/energy_model/model.py PARAMS and helpers (no re-invented numbers):
  * required_detector_energy(), reproduce_xiang(), E_PHOTON_1550, PARAMS ranges.

Every energy input is sourced (PARAMS carries the citation). Places where the
accounting is GENEROUS TO OPTICS are flagged in the output under "generous_to_optics".

Writes data/trackb_energy.json.
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "energy_model")))
from model import (PARAMS, sample_params, required_detector_energy, reproduce_xiang,
                   E_PHOTON_1550)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACC_PATH = os.path.join(ROOT, "data", "trackb_accuracy.json")
OUT = os.path.join(ROOT, "data", "trackb_energy.json")
N_MC = 5000

# Edge digital competitor efficiency (INT8), sourced:
#   Jetson Orin-class SoC ~4-5 TOPS/W (AGX Orin 275 TOPS @ ~60 W -> ~4.6; Orin NX
#     100 TOPS @ ~25 W -> ~4). Automotive DRIVE Orin similar.
#   Phone NPU ~10-30 TOPS/W (Apple Neural Engine / Qualcomm Hexagon; ~35 TOPS @ 1-3 W).
# Range 2-30 TOPS/W INT8; sampled log-uniform.
EDGE_TOPS_W = (2.0, 5.0, 30.0)   # (low, nominal=Orin, high=phone-NPU)


def edge_digital_j_per_inf(macs, tops_w):
    """J/inference for an INT8 edge accelerator. 1 MAC = 2 ops."""
    return macs * 2.0 / (tops_w * 1e12)


# ---------------------------------------------------------------------------
# D2NN per-inference energy (passive optical compute ~0; cost is I/O + laser).
# ---------------------------------------------------------------------------
def d2nn_j_per_inf(p, n_in, n_det, planes, bits=8, tops_w_ignored=None):
    """Diffractive optical net, inference-only.
      input encode : n_in  * (E_DAC + E_mod)          -- one DAC+modulator per input pixel
      laser+detect : n_det * E_det_opt / (WPE*trans)  -- deliver detectable optical energy
      output ADC   : n_det * E_ADC                    -- digitise each detector
      compute      : ~0  (passive phase-mask diffraction)
    trans = 10^-((2*coupling + planes*mask_IL)/10). Returns dict of terms (J)."""
    E_DAC = p[f"E_DAC_{bits}b"]; E_ADC = p[f"E_ADC_{bits}b"]; E_mod = p["E_mod_MRM"]
    WPE = p["WPE_laser"]
    mask_il = p["mzi_loss_db"]                       # per phase-plane insertion loss (proxy)
    loss_db = 2 * p["coupling_db"] + planes * mask_il
    trans = 10.0 ** (-loss_db / 10.0)
    E_det_opt = required_detector_energy(bits, p["responsivity"], p["tia_noise"], p["bandwidth"])
    e_in = n_in * (E_DAC + E_mod)
    e_opt = n_det * E_det_opt / (WPE * trans)
    e_out = n_det * E_ADC
    total = e_in + e_opt + e_out
    return {"total": total, "input_dac_mod": e_in, "laser_detect": e_opt, "output_adc": e_out,
            "loss_db": loss_db}


# ---------------------------------------------------------------------------
# Spiking per-inference energy (photonic spiking; bias-dominated).
# ---------------------------------------------------------------------------
def spiking_j_per_inf(p, n_in, width, depth, n_classes, T, firing_rate):
    """Photonic spiking / temporally-coded net, per inference. Uses the SAME primitives
    as model.spiking_terms (bias, gen, det, comp) but at the ACTUAL net structure and the
    MEASURED firing rate. Direct (constant-current) input encoding => the FIRST linear
    layer is dense every timestep (NOT sparse) -- priced honestly.

    N_neurons = depth*width LIF neurons.
      bias : N_neurons * P_bias * T / bandwidth          (standing near-threshold pump)
      SOPs : T*(in_dim*width)  [dense input layer]
             + firing_rate * T * ((depth-1)*width*width + width*n_classes) [spike-driven]
      gen  : total_spikes * E_spike     (total_spikes = firing_rate*N_neurons*T)
      det  : SOPs * E_det_opt_1bit       (1-bit optical spike detection)
      comp : N_neurons * T * comparator_J
      io   : n_in*E_DAC (input) + n_classes*T*comparator_J (rate readout)  -- small
    """
    N_neurons = depth * width
    P_bias = p["neuron_bias_mW"] * 1e-3
    bw = p["bandwidth"]
    trans = 10.0 ** (-(2 * p["coupling_db"]) / 10.0)
    E_det_1bit = p["spike_photons"] * E_PHOTON_1550 / (p["responsivity"] * trans * p["WPE_laser"])
    sops = T * (n_in * width) + firing_rate * T * ((depth - 1) * width * width + width * n_classes)
    total_spikes = firing_rate * N_neurons * T
    bias = N_neurons * P_bias * T / bw
    gen = total_spikes * p["E_spike_J"]
    det = sops * E_det_1bit
    comp = N_neurons * T * p["comparator_J"]
    io = n_in * p[f"E_DAC_8b"] + n_classes * T * p["comparator_J"]
    total = bias + gen + det + comp + io
    return {"total": total, "bias": bias, "gen": gen, "det": det, "comp": comp, "io": io,
            "sops": sops, "total_spikes": total_spikes,
            "dominant": max({"bias": bias, "gen": gen, "det": det, "comp": comp}.items(),
                            key=lambda kv: kv[1])[0]}


def band(a):
    a = np.asarray(a, float)
    return {"p05": float(np.percentile(a, 5)), "p16": float(np.percentile(a, 16)),
            "p50": float(np.percentile(a, 50)), "p84": float(np.percentile(a, 84)),
            "p95": float(np.percentile(a, 95))}


def run():
    with open(ACC_PATH) as fh:
        acc = json.load(fh)
    cfg = acc["config"]; width = cfg["width"]; depth = cfg["depth"]; T = cfg["T"]
    out = {"metric": "joules_per_inference_at_matched_accuracy", "N_MC": N_MC,
           "edge_tops_w_int8": {"low": EDGE_TOPS_W[0], "nominal_orin": EDGE_TOPS_W[1],
                                "high_phone_npu": EDGE_TOPS_W[2],
                                "source": "Jetson/DRIVE Orin ~4-5; phone NPU (ANE/Hexagon) ~10-30 TOPS/W INT8"},
           "generous_to_optics": [], "datasets": {}}

    # model check
    x = reproduce_xiang({k: v.nom for k, v in PARAMS.items()})
    out["xiang_model_check"] = {"GOPS_per_W": x["GOPS_per_W"], "pJ_per_op": x["j_per_op"] * 1e12,
                                "paper_GOPS_per_W": 987.65, "paper_pJ_per_op": 1.01,
                                "reproduced": abs(x["j_per_op"] * 1e12 - 1.01) < 0.1}
    out["generous_to_optics"] = [
        "D2NN readout priced at n_det = n_classes detectors (class-region readout, the "
        "cheapest physical scheme); a full-plane linear-head readout (M^2 detectors) that our "
        "trained accuracy actually uses would add M^2 ADCs -- reported separately as d2nn_fullplane.",
        "D2NN passive optical compute charged at 0 J (no per-op cost) -- upper-bound-favourable.",
        "Digital competitor priced at the ReLU BASELINE's MAC count (~98% MNIST). At MATCHED "
        "accuracy the digital net that only needs the OPTICAL net's (lower) accuracy is smaller "
        "and cheaper, so the true matched-accuracy digital energy is <= what we charge here -- "
        "i.e. every 'optics beats digital' fraction below is an OVER-estimate for optics.",
        "Spiking neuron_bias_mW low end (0.5 mW) is an UNFABRICATED nanolaser projection; the "
        "only measured DFB-SA array is 40 mW/neuron (Xiang 2026).",
    ]

    for dsname, ds in acc["datasets"].items():
        if "relu" not in ds:
            continue
        relu = ds["relu"]; macs = relu["macs_per_inference"]
        n_classes = 10
        n_in = int(round(macs / 1e9)) if False else None
        # input dimension from the ReLU MACs formula: macs = in_dim*width + (depth-1)*w^2 + w*classes
        in_dim = int(round((macs - (depth - 1) * width * width - width * n_classes) / width))
        entry = {"in_dim": in_dim, "relu": {"acc": relu["mean_acc"], "macs": macs}}

        # --- digital competitor J/inference band (over TOPS/W) ---
        rng = np.random.default_rng(0)
        tw = np.exp(rng.uniform(math.log(EDGE_TOPS_W[0]), math.log(EDGE_TOPS_W[2]), N_MC))
        dig = edge_digital_j_per_inf(macs, tw)
        entry["digital_j_per_inf"] = band(dig)
        entry["digital_j_per_inf_nominal_orin"] = edge_digital_j_per_inf(macs, EDGE_TOPS_W[1])

        # --- optical nets: MC over PARAMS ---
        rng = np.random.default_rng(1)
        d2 = ds.get("d2nn", {}); spk = ds.get("spiking", {})
        d2_M = d2.get("grid_M", 56); d2_planes = d2.get("phase_planes", depth)
        fr = spk.get("mean_firing_rate", 0.44)
        d2_in = in_dim if dsname != "cifar10" else d2.get("in_dim_field", 1024)
        # for CIFAR the D2NN input field is grayscale 32x32 = 1024 pixels
        if dsname == "cifar10":
            d2_in = 1024

        d2_class, d2_full = [], []
        spk_e, spk_jsop = [], []
        d2_terms_nom = spk_terms_nom = None
        for i in range(N_MC):
            p = sample_params(rng)
            r_c = d2nn_j_per_inf(p, d2_in, n_classes, d2_planes, bits=8)
            r_f = d2nn_j_per_inf(p, d2_in, d2_M * d2_M, d2_planes, bits=8)
            d2_class.append(r_c["total"]); d2_full.append(r_f["total"])
            s = spiking_j_per_inf(p, in_dim, width, depth, n_classes, T, fr)
            spk_e.append(s["total"]); spk_jsop.append(s["total"] / s["sops"])
            if i == 0:
                d2_terms_nom = r_c; spk_terms_nom = s
        d2_class = np.array(d2_class); d2_full = np.array(d2_full); spk_e = np.array(spk_e)

        # matched-accuracy fractions: draw a digital TOPS/W per optical draw
        tw2 = np.exp(rng.uniform(math.log(EDGE_TOPS_W[0]), math.log(EDGE_TOPS_W[2]), N_MC))
        dig2 = edge_digital_j_per_inf(macs, tw2)

        entry["d2nn"] = {
            "acc": d2.get("mean_acc"), "params": d2.get("params"),
            "gap_vs_relu": ds.get("gap_vs_relu", {}).get("d2nn"),
            "j_per_inf_classreadout": band(d2_class),
            "j_per_inf_fullplane": band(d2_full),
            "terms_nominal_classreadout": {k: d2_terms_nom[k] for k in d2_terms_nom},
            "frac_beats_digital_classreadout": float(np.mean(d2_class < dig2)),
            "frac_beats_digital_fullplane": float(np.mean(d2_full < dig2)),
        }
        entry["spiking"] = {
            "acc": spk.get("mean_acc"), "params": spk.get("params"),
            "firing_rate": fr, "gap_vs_relu": ds.get("gap_vs_relu", {}).get("spiking"),
            "j_per_inf": band(spk_e),
            "j_per_synaptic_op": band(spk_jsop),
            "terms_nominal": {k: (spk_terms_nom[k] if k not in ("dominant",) else spk_terms_nom[k])
                              for k in spk_terms_nom},
            "frac_beats_digital": float(np.mean(spk_e < dig2)),
        }
        out["datasets"][dsname] = entry

        # console
        print(f"\n### {dsname} ###")
        print(f"  ReLU acc={relu['mean_acc']:.4f}  macs={macs}  "
              f"digital J/inf p50={entry['digital_j_per_inf']['p50']*1e9:.2f} nJ "
              f"[{entry['digital_j_per_inf']['p05']*1e9:.2f}-{entry['digital_j_per_inf']['p95']*1e9:.2f}]")
        d = entry["d2nn"]
        print(f"  D2NN acc={d['acc']}  gap={d['gap_vs_relu']}  "
              f"J/inf(class)={d['j_per_inf_classreadout']['p50']*1e9:.2f} nJ "
              f"beats={d['frac_beats_digital_classreadout']*100:.0f}%  |  "
              f"J/inf(fullplane)={d['j_per_inf_fullplane']['p50']*1e9:.2f} nJ "
              f"beats={d['frac_beats_digital_fullplane']*100:.0f}%")
        s = entry["spiking"]
        print(f"  SNN  acc={s['acc']}  gap={s['gap_vs_relu']}  fire={fr:.3f}  "
              f"J/inf={s['j_per_inf']['p50']*1e9:.2f} nJ beats={s['frac_beats_digital']*100:.0f}%  "
              f"dom={spk_terms_nom['dominant']}")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[trackb-energy] wrote {OUT}")
    return out


if __name__ == "__main__":
    run()
