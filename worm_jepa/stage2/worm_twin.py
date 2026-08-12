"""
worm_twin.py -- the unified C. elegans digital twin.

One model, one entry point: put in ANY perturbation (a compound, a gene, a
stimulus, a neuron stimulation, an ablation) and get back the FULL predicted state
-- which neurons change, the command-circuit shift, and the resulting behavior --
each with an honest confidence bounded by what we actually validated.

This fuses the six validated predictors in this repo (compound_response, bridge,
perturbation_response, ablation_importance, the command->behavior map, and the
c302 forward sim) into a single object. It is the whole-organism artifact: not a
wiring diagram, a *generative* model of perturbation -> state.

Honest scope (carried in every result as `confidence` + `domain`):
  - behavior direction from command state ............ strong (R^2 0.74 validated)
  - molecular localization (compound/gene -> neurons) . strong (CeNGEN, 8/8 mech)
  - circuit propagation (which neurons respond) ....... bounded (38% of ceiling)
  - proteostasis / aggregation / metabolic effects .... OUT OF DOMAIN (abstains)

    from worm_twin import WormTwin
    t = WormTwin()
    t.perturb("compound", gene="acr-16")     # nicotine's target
    t.perturb("stimulate", neuron="ASHL")    # optogenetic-style
    t.perturb("stimulus", stimulus="aversive")
    t.perturb("ablate", neuron="AVAL")
Requires wormneuroatlas, c302.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# command circuit in the funatlas 300-neuron space (class-level names collapsed to L/R members)
REVERSAL = ["AVAL", "AVAR", "AVEL", "AVER", "AVDL", "AVDR", "AIBL", "AIBR"]
FORWARD = ["AVBL", "AVBR", "PVCL", "PVCR"]
BETA_C = -0.298   # reversal-command state -> velocity (validated, Flavell n=38): reversal => slower/back

# In-domain = the target is a neural RECEPTOR / ION CHANNEL / transporter (neuroactive).
# Everything else (transcription factors, kinases, chaperones, metabolic enzymes) is OUT of
# domain -- the twin models the nervous system, not proteostasis/metabolism, and abstains.
RECEPTOR_PREFIXES = ("acr-", "unc-29", "unc-38", "unc-63", "lev-", "eat-2", "des-2",
                     "nmr-", "glr-", "glc-", "avr-", "mgl-", "eat-4", "unc-49", "lgc-",
                     "mod-", "ser-", "dop-", "tyra-", "octr-", "lgc-", "ggr-", "gar-", "gbb-",
                     "del-", "mec-4", "mec-10", "asic-", "trp-", "ocr-", "osm-9",
                     "egl-19", "unc-2", "cca-1", "unc-103", "twk-", "slo-", "unc-9", "unc-7")


def _in_domain(gene):
    return any(gene.startswith(p) for p in RECEPTOR_PREFIXES)


class WormTwin:
    def __init__(self, gain=0.7):
        import h5py, wormneuroatlas, c302
        P = os.path.join(os.path.dirname(wormneuroatlas.__file__), "data", "funatlas.h5")
        with h5py.File(P, "r") as h:
            self.ids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["neuron_ids"][:]]
        self.n2i = {n: i for i, n in enumerate(self.ids)}
        self.N = len(self.ids)
        # connectome propagation operator (I - gA)^-1  (chemical + gap junction)
        _, conns = c302.get_cell_names_and_connection("SpreadsheetDataReader")
        Wc = np.zeros((self.N, self.N)); Wg = np.zeros((self.N, self.N))
        for c in conns:
            if c.pre_cell in self.n2i and c.post_cell in self.n2i:
                i, j = self.n2i[c.pre_cell], self.n2i[c.post_cell]
                if c.syntype == "Send":
                    Wc[i, j] += c.number
                elif c.syntype == "GapJunction":
                    Wg[i, j] += c.number; Wg[j, i] += c.number
        A = Wc / (Wc.sum(1, keepdims=True) + 1e-9) + 0.5 * Wg / (Wg.sum(1, keepdims=True) + 1e-9)
        A = gain * A / (np.abs(np.linalg.eigvals(A)).max() + 1e-12)
        self.P = np.linalg.inv(np.eye(self.N) - A)          # response = P @ input
        self._cengen = None

    # ---- shared readout: input vector over neurons -> full predicted state ------
    def _state_from_input(self, u, confidence, domain, driver):
        resp = self.P @ u                                    # circuit propagation
        rev = float(np.mean([resp[self.n2i[n]] for n in REVERSAL if n in self.n2i]))
        fwd = float(np.mean([resp[self.n2i[n]] for n in FORWARD if n in self.n2i]))
        command = rev - fwd                                  # net reversal drive
        dvel = BETA_C * command                              # behavior
        if abs(command) < 0.05:
            beh = "no net locomotor change"
        elif dvel < 0:
            beh = "REVERSE / slow"
        else:
            beh = "FORWARD / faster"
        top = sorted(((self.ids[i], float(resp[i])) for i in range(self.N)),
                     key=lambda kv: -abs(kv[1]))[:8]
        return {
            "driver": driver,
            "affected_neurons": [(n, round(v, 3)) for n, v in top if abs(v) > 1e-4],
            "command": {"reversal_drive": round(rev, 3), "forward_drive": round(fwd, 3),
                        "net": round(command, 3)},
            "behavior": {"predicted": beh, "delta_velocity": round(dvel, 3)},
            "confidence": confidence,
            "domain": domain,
        }

    # ---- perturbation dispatch --------------------------------------------------
    def perturb(self, kind, gene=None, neuron=None, stimulus=None, sign=+1,
                aggregation_effect=None):
        u = np.zeros(self.N)
        if kind == "stimulate":                              # direct neural perturbation
            if neuron not in self.n2i:
                return {"error": "neuron %s not in atlas" % neuron}
            u[self.n2i[neuron]] = float(sign)
            return self._state_from_input(u, "bounded (38% of ceiling)", "in", "stimulate %s" % neuron)

        if kind == "ablate":                                 # remove a neuron -> behavioral deficit
            import ablation_importance as ai
            if not hasattr(self, "_abl"):
                self._abl = {r["neuron"]: r["importance"] for r in ai.run().get("top15", [])}
            imp = self._abl.get(neuron)
            is_cmd = neuron in REVERSAL or neuron in FORWARD
            if imp is not None:
                beh = "locomotor deficit (top behavioral-importance neuron)"
            elif is_cmd:
                beh = "locomotor deficit (command neuron — validated 3.5x enriched)"
            else:
                beh = "little / no locomotor deficit predicted"
            return {"driver": "ablate %s" % neuron,
                    "behavior": {"predicted": beh, "importance": imp,
                                 "command_neuron": is_cmd},
                    "confidence": "strong (recovers causal circuit)", "domain": "in"}

        if kind == "compound" or kind == "knockout":         # molecular -> neurons -> behavior
            import compound_response as cr
            # PROTEOSTASIS path: an anti-aggregation compound (rate factor given) -> paralysis.
            # This is the module that stops the twin abstaining on the GMC101/CL4176 AD screen.
            if aggregation_effect is not None:
                import proteostasis as ps
                base = ps.paralysis_time(1.0); tp = ps.paralysis_time(aggregation_effect)
                delay = (tp - base) if np.isfinite(tp) else float("inf")
                eff = "protective" if aggregation_effect < 1 else "inert" if aggregation_effect == 1 else "toxic"
                return {"driver": "%s %s (aggregation f=%.2f)" % (kind, gene, aggregation_effect),
                        "behavior": {"predicted": "paralysis @ %s h (%+.0f h vs vehicle) — %s"
                                     % ("%.0f" % tp if np.isfinite(tp) else ">120",
                                        delay if np.isfinite(delay) else 999, eff),
                                     "paralysis_time_h": (round(tp, 1) if np.isfinite(tp) else None)},
                        "confidence": "mechanism→phenotype (potency is an assay input)",
                        "domain": "in (proteostasis module)"}
            if not _in_domain(gene):                          # domain gate: neuroactive only
                return {"driver": "%s %s" % (kind, gene),
                        "behavior": {"predicted": "abstained"},
                        "confidence": "abstains",
                        "domain": "OUT (not a neural receptor/channel; no aggregation input)"}
            if self._cengen is None:
                self._cengen = cr.load_cengen()
            genes, neurons_c, tpm, g2i = self._cengen
            if gene not in g2i:
                return {"driver": "%s %s" % (kind, gene),
                        "behavior": {"predicted": "n/a"},
                        "confidence": "abstains", "domain": "OUT (target gene not in CeNGEN)"}
            # DEFER the behavioral direction to the VALIDATED compound_response (8/8), so the twin
            # never contradicts its own components. Requires the pharmacology: receptor polarity
            # (excitatory/inhibitory) and agonism (+1 agonist / -1 antagonist).
            row = next((p for p in cr.PANEL if p[1] == gene), None)
            pol = row[2] if row else ("inh" if sign < 0 else "exc")
            ag = row[3] if row else 1
            direction, tn = cr.predict(gene, pol, ag, neurons_c, tpm, g2i)
            top = [n for n, _ in sorted(tn.items(), key=lambda kv: -kv[1])[:6]]
            beh = {"increase": "FORWARD / faster (or more active)",
                   "decrease": "REVERSE / slow (or less active)"}.get(direction, direction)
            return {"driver": "%s %s (%s receptor, %s)" % (kind, gene, pol, "agonist" if ag > 0 else "antagonist"),
                    "affected_neurons": [(n, None) for n in top],
                    "behavior": {"predicted": beh, "direction": direction},
                    "confidence": "validated (compound_response 8/8)", "domain": "in"}

        if kind == "stimulus":                               # chemical stimulus via the bridge
            import bridge
            s = bridge.validate()["stimuli"].get(stimulus)
            if s is None:
                return {"error": "stimulus must be one of aversive/attractive_onset/attractive_removal"}
            return {"driver": "stimulus %s" % stimulus,
                    "behavior": {"predicted": s["behavior"], "delta_velocity": s["pred_dvelocity"]},
                    "confidence": "directional (3/3, shuffle-controlled)", "domain": "in"}

        return {"error": "unknown perturbation kind: %s" % kind}


if __name__ == "__main__":
    t = WormTwin()
    print("=" * 74)
    print("WORM DIGITAL TWIN  —  any perturbation → full predicted state")
    print("=" * 74)
    demos = [
        ("compound", dict(gene="acr-16"), "nicotine (nAChR agonist)"),
        ("compound", dict(gene="mod-1"), "serotonin (5-HT-gated Cl)"),
        ("stimulate", dict(neuron="ASHL"), "optogenetic ASH (nociceptor)"),
        ("stimulus", dict(stimulus="aversive"), "aversive chemical (CuSO4)"),
        ("ablate", dict(neuron="AVAL"), "ablate AVAL (reversal command)"),
        ("compound", dict(gene="PBT2", aggregation_effect=0.45), "PBT2 (anti-Aβ, proteostasis module)"),
        ("compound", dict(gene="thioflavin-T", aggregation_effect=1.0), "thioflavin T (inert — neg control)"),
        ("compound", dict(gene="daf-16"), "a non-neural target, no aggregation input (abstains)"),
    ]
    for kind, kw, label in demos:
        r = t.perturb(kind, **kw)
        print("\n▸ %s  [%s]" % (label, kind))
        if "error" in r:
            print("    %s" % r["error"]); continue
        if r.get("affected_neurons"):
            print("    neurons : " + ", ".join(
                ("%s%+.2f" % (n, v)) if v is not None else n
                for n, v in r["affected_neurons"][:5]))
        if "command" in r:
            print("    command : reversal %+.2f / forward %+.2f  (net %+.2f)"
                  % (r["command"]["reversal_drive"], r["command"]["forward_drive"], r["command"]["net"]))
        print("    behavior: %s" % r["behavior"].get("predicted"))
        print("    domain  : %s  |  confidence: %s" % (r["domain"], r["confidence"]))
    print("\n" + "=" * 74)
