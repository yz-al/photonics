"""
openworm_simulate.py -- CLOSE THE FULL LOOP: run OpenWorm's c302 forward
simulation, stimulated according to our transduction map, and check it
reproduces the aversive->reversal transform our data-composed bridge predicts.

The loop
--------
  our transduction map (which sensory neurons an aversive stimulus drives)
     -> inject current into those neurons in c302
     -> c302 forward sim (jNeuroML, graded synapses = correct for non-spiking
        C. elegans neurons)
     -> read the command circuit
     -> does the REVERSAL command (AVA/AVE/AVD) rise more than FORWARD (AVB/PVC)?

This is the counterpart to bridge.py. The bridge COMPOSES measured maps to
predict aversive->reversal; here OpenWorm's biophysical model, driven the same
way, is checked for the SAME transform -- a cross-model agreement test.

Key modeling notes (learned the hard way, encoded here)
------------------------------------------------------
- Use a GRADED-synapse parameter set (parameters_C1/C2/D1). c302's default
  parameters_C uses spike-triggered expTwoSynapse synapses, but the cells are
  non-spiking, so NOTHING propagates (ASH fires, AVA stays flat). Graded
  synapses transmit the continuous depolarization -- the biologically correct
  choice for the largely non-spiking worm nervous system.
- Keep the stimulus small (~1 pA). The command circuit is recurrently coupled
  with little inhibition in c302, so a large drive runs away numerically
  (~1e36 mV). This finickiness is exactly why faithful c302 dynamics is an open
  problem, and why measured functional weights (our L2 atlas) matter.

Result (default connectome, ASH stimulated; see openworm_sim.json):
  REVERSAL command (AVA/AVE/AVD)  +0.48 mV   >   FORWARD command (AVB/PVC) +0.25 mV
  -> aversive stimulus biases c302 toward the reversal command. Loop closed.

    python openworm_simulate.py            # runs default + reweighted, writes json + traces
Requires c302, pyneuroml + jNeuroML (bundled), wormneuroatlas. ~15 s per sim.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNDIR = os.path.join(HERE, "c302_run")

# Aversive reflex circuit: nociceptors -> interneurons -> command (reversal vs forward).
CIRCUIT = ["ASHL", "ASHR", "ADLL", "ADLR", "AIBL", "AIBR", "RIML", "RIMR",
           "AVAL", "AVAR", "AVEL", "AVER", "AVDL", "AVDR", "AVBL", "AVBR", "PVCL", "PVCR"]
STIM_NEURONS = ["ASHL", "ASHR", "ADLL", "ADLR"]     # aversive sensors (from transduction map)
REVERSAL = ["AVAL", "AVAR", "AVEL", "AVER", "AVDL", "AVDR"]
FORWARD = ["AVBL", "AVBR", "PVCL", "PVCR"]


def _reweight_overrides(scale=20.0):
    """Turn our measured causal atlas into c302 conn_number / polarity overrides
    for the circuit edges: number ~ |measured_dFF| * scale, polarity from sign."""
    try:
        import openworm_integrate as owi
    except ImportError:
        return None, None
    dFF, q, n2i = owi.load_causal_atlas()
    num, pol = {}, {}
    for pre in CIRCUIT:
        for post in CIRCUIT:
            if pre == post or pre not in n2i or post not in n2i:
                continue
            d = float(dFF[n2i[pre], n2i[post]])
            if not np.isfinite(d) or abs(d) < 1e-3:
                continue
            sh = "%s-%s" % (pre, post)
            num[sh] = max(1, round(abs(d) * scale))
            pol[sh] = "inh" if d < 0 else "exc"
    return num, pol


def simulate(tag, stim_pA=1.0, duration=1500, dt=0.02, reweight=False):
    """Generate + run one c302 forward simulation; return per-cell stim-evoked
    delta (mV) and the raw traces."""
    import c302
    import c302.parameters_C1 as P               # GRADED synapses
    from pyneuroml import pynml
    params = P.ParameterisedModel()
    ov = {"unphysiological_offset_current": "%gpA" % stim_pA,
          "unphysiological_offset_current_del": "300ms",
          "unphysiological_offset_current_dur": "600ms"}
    kw = {}
    if reweight:
        num, pol = _reweight_overrides()
        if num:
            kw["conn_number_override"] = num
            kw["conn_polarity_override"] = pol
    os.makedirs(RUNDIR, exist_ok=True)
    c302.generate("Loop_%s" % tag, params, cells=CIRCUIT, cells_to_plot=CIRCUIT,
                  cells_to_stimulate=STIM_NEURONS, duration=duration, dt=dt,
                  target_directory=RUNDIR, param_overrides=ov, verbose=False, **kw)
    res = pynml.run_lems_with_jneuroml(os.path.join(RUNDIR, "LEMS_Loop_%s.xml" % tag),
                                       nogui=True, load_saved_data=True, plot=False, verbose=False)
    t = np.array(res["t"])
    pre = (t >= 0.1) & (t < 0.3); dur = (t >= 0.4) & (t < 0.9)
    delta, traces = {}, {"t": t.tolist()}
    for n in CIRCUIT:
        k = [x for x in res if x.startswith(n + "/")]
        if not k:
            continue
        v = np.array(res[k[0]])
        delta[n] = float((v[dur].mean() - v[pre].mean()) * 1000)
        traces[n] = (v * 1000).tolist()
    return delta, traces


def summarize(delta):
    rev = float(np.nanmean([delta[n] for n in REVERSAL if n in delta]))
    fwd = float(np.nanmean([delta[n] for n in FORWARD if n in delta]))
    stable = all(abs(v) < 100 for v in delta.values())     # not numerically blown up
    return dict(reversal_mV=round(rev, 3), forward_mV=round(fwd, 3),
                reversal_gt_forward=bool(rev > fwd), stable=stable,
                bias_ratio=round(rev / fwd, 2) if fwd else None)


if __name__ == "__main__":
    out = {}
    print("=== CLOSE THE LOOP: c302 forward sim, ASH (aversive) stimulated ===\n")
    for tag, rw in [("default", False), ("reweighted", True)]:
        delta, traces = simulate(tag, reweight=rw)
        s = summarize(delta)
        out[tag] = {"summary": s, "delta_mV": delta}
        np.savez(os.path.join(RUNDIR, "traces_%s.npz" % tag),
                 **{k: np.array(v) for k, v in traces.items()})
        label = "our measured weights" if rw else "c302 default (connection counts)"
        print("%-28s REVERSAL %+.3f mV  vs  FORWARD %+.3f mV   %s%s"
              % (label, s["reversal_mV"], s["forward_mV"],
                 "reversal>forward" if s["reversal_gt_forward"] else "forward>reversal",
                 "" if s["stable"] else "  [UNSTABLE]"))
    with open(os.path.join(HERE, "openworm_sim.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote openworm_sim.json ; traces in c302_run/traces_*.npz")
    print("NeuroML model in c302_run/ (LEMS_Loop_default.xml + *.net.nml) -- openable in OpenWorm/Geppetto")
