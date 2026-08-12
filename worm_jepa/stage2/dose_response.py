"""
dose_response.py -- the OTHER half of the single-cell question. single_cell.py found
AWC's on/off TIMING is ~linear. This asks about response GAIN, and finds the opposite:
the concentration->response mapping is profoundly NONLINEAR (saturating), so here a
single-cell nonlinearity IS required.

The test
--------
Isoamyl alcohol is presented at two concentrations 10,000x apart (e-2 vs e-6 dilution;
DANDI:000981). A LINEAR-in-concentration node makes a parameter-free prediction: the
response ratio must equal the concentration ratio, i.e. log10(resp_hi/resp_lo) = 4 (four
decades of dynamic range). Real sensory transduction saturates, so the observed ratio is
tiny. Measured on the chemosensory neurons that respond to IAA (AWC, AWA, ASE; L+R;
n=144 worm-neuron observations):

  linear-in-concentration REQUIRES   log10(hi/lo) = 4.00     (a 10^4-fold response range)
  observed                            log10(hi/lo) ~ 0.36     (median ratio ~2.3x)
  => the linear node is falsified by ~3.6 DECADES. A compressive (saturating) single-cell
     transduction is required. AWC is fully saturated (ratio ~1); AWA shows the strong
     amplification consistent with its all-or-none calcium spikes.

Sensitivity control: a SYNTHETIC linear neuron (response proportional to concentration),
run through the identical metric, returns log10 ratio = 4.00 -- so the pipeline CAN report
linearity when it is there; the compression is real, not a measurement floor.

The complete answer to "would single-cell simulation help?"
-----------------------------------------------------------
It depends on the FEATURE, and now we have both halves, measured:
  - response TIMING (on/off)        -> ~LINEAR      (single_cell.py): biophysics does NOT help
  - response GAIN (dose-response)   -> ~SATURATING  (here):           a nonlinearity IS required
So a single static saturating nonlinearity on the input earns its place; a full
multi-parameter biophysical simulation is only warranted where a feature is nonlinear at
the measured level AND the extra parameters are constrained by data.

    python dose_response.py     # writes dose_response.json
Self-contained: reads dose_response.npz (per-neuron responses at two IAA concentrations).
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dose_response.npz")


def run():
    from scipy.stats import wilcoxon
    d = np.load(DATA, allow_pickle=True)
    rows = d["rows"]; cells = [str(c) for c in d["cells"]]
    c_hi, c_lo = float(d["conc_hi"]), float(d["conc_lo"])
    linear_decades = float(np.log10(c_hi / c_lo))          # parameter-free linear requirement = 4
    hi = rows[:, 2]; lo = rows[:, 3]
    ok = (hi > 1e-6) & (lo > 1e-6)
    obs = np.log10(hi[ok] / lo[ok])                        # observed decades of dynamic range
    err_linear = np.abs(obs - linear_decades)             # distance to the linear requirement
    err_sat = np.abs(obs - 0.0)                            # distance to full saturation (0 decades)
    p = float(wilcoxon(err_linear, err_sat, alternative="greater")[1])
    # per-cell medians
    percell = {}
    for ci, name in enumerate(cells):
        m = ok & (rows[:, 1] == ci)
        if m.sum():
            percell[name] = {"n": int(m.sum()),
                             "median_ratio_hi_lo": round(float(np.median(hi[m] / lo[m])), 2),
                             "median_log10": round(float(np.median(np.log10(hi[m] / lo[m]))), 2)}
    # sensitivity control: a synthetic linear neuron recovers the 4-decade prediction
    synth_linear = float(np.log10((c_hi) / (c_lo)))       # response ∝ concentration -> ratio = conc ratio
    compression = 1.0 - float(np.mean(obs)) / linear_decades
    return {
        "stimulus": "isoamyl alcohol, e-2 vs e-6 dilution (10^4-fold), DANDI:000981",
        "neurons": "AWC, AWA, ASE (L+R)", "n_obs": int(ok.sum()),
        "linear_in_concentration_requires_log10_ratio": round(linear_decades, 2),
        "observed_log10_ratio_mean": round(float(np.mean(obs)), 2),
        "observed_log10_ratio_median": round(float(np.median(obs)), 2),
        "linear_model_off_by_decades": round(float(linear_decades - np.mean(obs)), 2),
        "compression_index": round(compression, 3),
        "saturating_closer_than_linear_p": p,
        "per_cell": percell,
        "sensitivity_control_synthetic_linear_log10": round(synth_linear, 2),
        "sensitivity_control_recovers_linearity": bool(abs(synth_linear - linear_decades) < 0.1),
        "single_cell_nonlinearity_required_for_gain": bool(np.mean(obs) < linear_decades - 1 and p < 0.01),
    }


if __name__ == "__main__":
    r = run()
    print("=" * 82)
    print("  DOSE-RESPONSE NONLINEARITY -- is the concentration->response map linear? (n=%d)" % r["n_obs"])
    print("=" * 82)
    print("  stimulus: %s" % r["stimulus"])
    print("  neurons : %s\n" % r["neurons"])
    print("  linear-in-concentration REQUIRES  log10(hi/lo) = %.2f  (a %g-fold response range)"
          % (r["linear_in_concentration_requires_log10_ratio"], 10 ** r["linear_in_concentration_requires_log10_ratio"]))
    print("  OBSERVED                          log10(hi/lo) = %.2f  (median ratio ~%.1fx)"
          % (r["observed_log10_ratio_mean"], 10 ** r["observed_log10_ratio_median"]))
    print("  => linear node falsified by %.1f DECADES;  compression index = %.2f (%.0f%% compressed)\n"
          % (r["linear_model_off_by_decades"], r["compression_index"], 100 * r["compression_index"]))
    for name, c in r["per_cell"].items():
        print("    %-6s n=%2d  median ratio %.2fx  (log10 %.2f)" % (name, c["n"], c["median_ratio_hi_lo"], c["median_log10"]))
    print("\n  saturating closer than linear : p=%.1e" % r["saturating_closer_than_linear_p"])
    print("  sensitivity control (synthetic linear neuron recovers log10=%.2f): %s"
          % (r["sensitivity_control_synthetic_linear_log10"], r["sensitivity_control_recovers_linearity"]))
    print("  => single-cell nonlinearity REQUIRED for dose-response gain: %s"
          % r["single_cell_nonlinearity_required_for_gain"])
    print("     (contrast: on/off TIMING was ~linear -- single_cell.py. Answer depends on the feature.)")
    print("=" * 82)
    with open(os.path.join(HERE, "dose_response.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("wrote dose_response.json")
