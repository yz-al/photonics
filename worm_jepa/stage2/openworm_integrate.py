"""
openworm_integrate.py -- feed our measured L2 causal atlas into OpenWorm's c302,
and use the bridge as a validation oracle for c302's forward simulation.

Why this is a real integration, not a demo
------------------------------------------
c302 (OpenWorm's NeuroML nervous-system model) weights each chemical synapse as
`baseline_conductance * (anatomical connection count)` from the connectome, with
a categorical neurotransmitter identity (Glutamate / GABA / ACh / ...). It has no
per-edge *functional* strength and no measured polarity beyond that category.

Our L2 work produced exactly that missing quantity: the Randi 2023 signal-
propagation atlas gives a measured, signed causal coupling for (almost) every
ordered neuron pair. So there are two concrete integration directions:

  (1) DATA -> OPENWORM.  Re-weight c302's synapses with measured causal strengths
      instead of connection counts. `build_reweighted_connectome()` emits a JSON
      edge list (pre, post, c302_count, measured_dFF, q) that a c302 parameter
      override can consume.

  (2) OPENWORM -> US.  c302 is a full biophysical forward simulator; our `bridge`
      (bridge.py) predicts the sign of the stimulus->behavior response from
      measured maps. The bridge is therefore a validation oracle: inject a
      stimulus current into the sensory neurons our transduction map flags, run
      c302, read out AVA/command, and check it reproduces aversive->reversal.
      `c302_validation_targets()` emits those (stimulus -> expected command sign)
      targets.

What the integration reveals (run as a script)
----------------------------------------------
- Coverage: ~3.3k of c302's 2279 chemical + 1084 gap-junction edges get a measured
  causal value from the atlas.
- Anatomical connection count is a POOR proxy for measured functional coupling
  (Spearman r^2 ~ 0.2%). Caveat: the atlas dFF is a propagated network response,
  not a monosynaptic conductance, so some decorrelation is expected -- but the
  gap is large and one-directional, which is the argument for data-driven weights.
- ~28% of the covered edges are net inhibitory; c302 represents inhibition only
  categorically (GABA), so signed functional weights add real information.
- HONEST CAVEAT (cross-check FAILS): c302's GABA edges are NOT more negative than
  its excitatory edges in the atlas (Mann-Whitney p~0.83). The atlas dFF is a
  *propagated network response* to stimulating the presynaptic neuron, not a
  monosynaptic conductance, so it does not track synapse-level NT polarity. Use
  the measured coupling as a NETWORK-level functional weight, not as a claim about
  monosynaptic sign -- that distinction matters for how it feeds c302.

    python openworm_integrate.py     # prints the report, writes reweighted_connectome.json

Requires `c302` (pip install c302) and `wormneuroatlas`. Neither a NeuroML
simulator nor owmeta is needed for the re-weighting; running the c302 forward
model for direction (2) additionally needs pyNeuroML + a simulator.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CHEMICAL_SYNTYPE = "Send"          # c302 chemical synapse (vs "GapJunction")
INHIBITORY_NT = {"GABA"}           # c302 neurotransmitter classes that are inhibitory


def load_c302_connectome(reader="SpreadsheetDataReader"):
    """Return c302's connectome as a list of dicts: pre, post, count, syntype, nt."""
    import c302
    _, conns = c302.get_cell_names_and_connection(reader)
    return [dict(pre=c.pre_cell, post=c.post_cell, count=int(c.number),
                 syntype=c.syntype, nt=c.synclass) for c in conns]


def load_causal_atlas():
    """Return (dFF, q, name->index) from the Randi wild-type signal-propagation atlas."""
    import h5py, wormneuroatlas
    p = os.path.join(os.path.dirname(wormneuroatlas.__file__), "data", "funatlas.h5")
    with h5py.File(p, "r") as h:
        ids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["neuron_ids"][:]]
        dFF = h["wt/dFF"][:]
        q = h["wt/q"][:]
    return dFF, q, {n: i for i, n in enumerate(ids)}


def build_reweighted_connectome(out_path=None, chemical_only=True):
    """Attach the measured causal weight to each c302 synapse. Writes JSON and
    returns (edges, stats)."""
    conns = load_c302_connectome()
    dFF, q, n2i = load_causal_atlas()
    edges, counts, caus = [], [], []
    n_chem = 0
    for c in conns:
        if chemical_only and c["syntype"] != CHEMICAL_SYNTYPE:
            continue
        n_chem += 1
        if c["pre"] not in n2i or c["post"] not in n2i:
            continue
        i, j = n2i[c["pre"]], n2i[c["post"]]
        d, qv = float(dFF[i, j]), float(q[i, j])
        if not np.isfinite(d):          # atlas has no measurement for this pair
            continue
        edges.append(dict(pre=c["pre"], post=c["post"], c302_count=c["count"],
                          nt=c["nt"], measured_dFF=round(d, 4),
                          q=round(qv, 4) if np.isfinite(qv) else None))
        counts.append(c["count"]); caus.append(d)
    counts, caus = np.array(counts), np.array(caus)
    from scipy.stats import spearmanr, mannwhitneyu
    m = np.abs(caus) > 0
    sp = spearmanr(counts[m], np.abs(caus[m]))
    inh = [e["measured_dFF"] for e in edges if e["nt"] in INHIBITORY_NT]
    exc = [e["measured_dFF"] for e in edges if e["nt"] not in INHIBITORY_NT and e["nt"] != "Generic_GJ"]
    nt_test = mannwhitneyu(inh, exc, alternative="less") if inh and exc else None
    stats = dict(
        n_chemical=n_chem,
        n_covered=len(edges),
        n_significant=int(sum(1 for e in edges if e["q"] < 0.05)),
        count_vs_function_spearman=round(float(sp.correlation), 4),
        count_vs_function_r2_pct=round(100 * float(sp.correlation) ** 2, 3),
        pct_net_inhibitory=round(100 * float(np.mean(caus < 0)), 1),
        gaba_more_negative_p=(round(float(nt_test.pvalue), 4) if nt_test else None),
        gaba_median_dFF=(round(float(np.median(inh)), 4) if inh else None),
        exc_median_dFF=(round(float(np.median(exc)), 4) if exc else None),
    )
    out_path = out_path or os.path.join(HERE, "reweighted_connectome.json")
    json.dump({"stats": stats, "edges": edges}, open(out_path, "w"))
    return edges, stats


def c302_validation_targets():
    """Direction (2): stimulus -> expected command-neuron sign, from the bridge.
    Injecting a depolarizing current into `stim_neurons` in c302 and reading out
    AVA should reproduce `expected_command` (UP = reversal command engaged)."""
    import bridge
    tr, wmag, valence, betaC, _, _ = bridge.load_maps()
    targets = {}
    for stim, vec in tr.items():
        # sensors the transduction map flags as strongly driven by this stimulus
        drive = {s: a for s, a in vec.items() if abs(a) > 0.3}
        expected = "UP" if bridge.EXPECTED.get(stim) == "REVERSE" else "DOWN"
        targets[stim] = dict(stim_neurons=sorted(drive, key=lambda s: -abs(drive[s])),
                             expected_command=expected,
                             behavior=bridge.EXPECTED.get(stim))
    return targets


if __name__ == "__main__":
    edges, stats = build_reweighted_connectome()
    print("=== OpenWorm c302  <->  measured L2 causal atlas ===\n")
    print("Direction (1) DATA -> OPENWORM: re-weight c302 synapses with measured coupling")
    print("  chemical synapses in c302        : %d" % stats["n_chemical"])
    print("  with a measured causal value     : %d (%d significant at q<0.05)"
          % (stats["n_covered"], stats["n_significant"]))
    print("  connection count vs |function|   : Spearman r=%.3f  (r^2=%.2f%%)"
          % (stats["count_vs_function_spearman"], stats["count_vs_function_r2_pct"]))
    print("    -> anatomical count barely predicts measured functional strength")
    print("  net-inhibitory measured edges    : %.0f%%  (count-only weights can't represent this)"
          % stats["pct_net_inhibitory"])
    if stats["gaba_more_negative_p"] is not None:
        passed = stats["gaba_more_negative_p"] < 0.05
        print("  cross-check: are c302 GABA edges more negative than excitatory? "
              "median %.3f vs %.3f, p=%.4f -> %s"
              % (stats["gaba_median_dFF"], stats["exc_median_dFF"],
                 stats["gaba_more_negative_p"], "YES" if passed else "NO"))
        print("    -> %s" % ("measured signs track NT polarity" if passed else
              "atlas dFF is a PROPAGATED network response, not monosynaptic polarity;"
              " use as a functional (network) weight, not a synapse-sign claim"))
    print("  wrote reweighted_connectome.json (%d edges)\n" % len(edges))

    print("Direction (2) OPENWORM -> US: bridge as validation oracle for c302 forward sim")
    for stim, t in c302_validation_targets().items():
        print("  %-20s inject-> %-28s expect AVA %-4s (%s)"
              % (stim, ",".join(t["stim_neurons"][:4]), t["expected_command"], t["behavior"]))
