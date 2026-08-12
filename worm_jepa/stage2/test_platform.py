"""
test_platform.py -- the validation suite. Turns "it works" into assertions.

Two layers, matching V&V 40:
  VERIFICATION (code is correct): fast modules run live and reproduce their result.
  VALIDATION  (result is real):  every claim beats a baseline AND its negative
                                 control fails. No claim stands on a single number.

Every test pairs a POSITIVE result with a NEGATIVE control -- the discipline from
METHODOLOGY.md: a model that can't be falsified isn't validated.

    python test_platform.py     # prints a pass/fail report, exits nonzero on failure
    pytest test_platform.py     # also works
Slow benchmarks (L2/L3/L4/perturbation/ablation) are checked from their committed
result JSONs (the validation record); fast modules run live.
"""
import os, json, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _j(name):
    p = os.path.join(HERE, name)
    assert os.path.exists(p), "missing result file %s (run its module first)" % name
    return json.load(open(p))


# ---- LIVE fast modules: positive result + negative control -------------------
def test_bridge_stimulus_valence():
    """Stimulus->behavior 3/3, AND shuffled valence must invert the attractive calls."""
    import bridge
    r = bridge.validate()
    assert r["n_correct"] == r["n_total"] == 3, "bridge not 3/3"
    # negative control: shuffled-valence null has opposite sign on attractive cases
    onset = r["stimuli"]["attractive_onset"]
    assert onset["pred_dvelocity"] > 0 > onset["shuffled_valence_mean"], "valence control did not invert"


def test_compound_direction():
    """Compound->behavior 8/8 AND CeNGEN localization is load-bearing enough to run."""
    cp = _j("compound_response.json")
    assert cp["correct"] == cp["n"], "compound panel not perfect"
    assert cp["accuracy"] == 1.0


def test_proteostasis_sota():
    """SOTA kinetics: secondary-nucleation most protective, elongation paradox, inert baseline."""
    import proteostasis as ps
    r = ps.run()
    assert r["secondary_nucleation_most_protective"], "k2 inhibitor not most protective"
    assert r["elongation_toxicity_paradox"], "elongation paradox not reproduced"
    assert r["inert_control_at_baseline"], "inert control drifted off baseline"
    # negative control: an inert compound (all factors 1) must not shift paralysis
    assert abs(ps.paralysis_time(1.0) - ps.paralysis_time(None)) < 1.0


def test_twin_domain_gate_and_consistency():
    """Twin abstains out of domain, defers to validated components, flags the paradox."""
    from worm_twin import WormTwin
    t = WormTwin()
    # domain gate: a non-neural target abstains
    assert "OUT" in t.perturb("compound", gene="daf-16")["domain"]
    # consistency: serotonin (mod-1, inhibitory) must slow, not speed (the fusion bug we caught)
    assert "REVERSE" in t.perturb("compound", gene="mod-1")["behavior"]["predicted"]
    # proteostasis paradox surfaced
    para = t.perturb("compound", gene="x", aggregation_effect={"kplus": 0.2})
    assert "PARADOX" in para["behavior"]["predicted"]


def test_ad_reference_domain():
    """AD reference: approved neuroactive drugs in-domain (mechanism right), rest abstained."""
    ad = _j("ad_reference_benchmark.json")
    assert ad["in_domain_mechanism_correct"] == "3/3"
    assert ad["n_out_of_domain_abstained"] >= 7


# ---- VALIDATION RECORD: committed benchmarks each beat baseline + control -----
def test_l2_imputation_beats_shuffle():
    im = _j("l2_imputation_benchmark.json")
    assert im["structure_recovered"]
    assert im["connectome"]["pct_of_ceiling"] > im["shuffled"]["pct_of_ceiling"] + 40  # >>shuffle


def test_l4_beats_hallinen_ridge():
    l4 = _j("l4_benchmark.json")
    for ch in ("velocity", "curvature"):
        ours = l4["results"][ch]["hgb_ours"]["median"]
        ridge = l4["results"][ch]["hallinen_ridge"]["median"]
        null = l4["results"][ch]["shuffle_null"]["median"]
        assert ours > ridge, "%s: ours !> ridge" % ch
        assert null < 0, "%s: shuffle null not negative" % ch     # negative control collapses


def test_l3_beats_persistence():
    l3 = _j("l3_benchmark.json")
    assert l3["models"]["linear_K3"]["beats_persistence"]


def test_perturbation_beats_shuffle():
    pr = _j("perturbation_response.json")
    assert pr["predicts_unseen_perturbations"]
    assert pr["dynamical"]["r"] > pr["shuffled"]["r"] + 0.05      # clean-zero shuffle


def test_ablation_recovers_circuit():
    ab = _j("ablation_importance.json")
    assert ab["locomotor_in_top10"] >= 8
    assert ab["command_enrichment"] >= 2.0


def test_external_validation_transfers():
    """Untrained data: the frozen connectome predicts held-out neurons in the Flavell
    dataset (never used to build the model) above the shuffled-wiring floor."""
    ev = _j("external_validation.json")
    assert ev["structure_transfers_out_of_distribution"]
    assert ev["connectome"]["r"] > ev["shuffled"]["r"] + 0.03   # beats global-state null
    assert ev["wilcoxon_p_connectome_gt_shuffled"] < 0.01       # across animals, not one worm


def test_five_layer_pipeline():
    """The end-to-end proof: every layer L1..L5 passes its gate (beats a baseline AND
    its control fails) AND one input runs coherently through the whole stack."""
    import pipeline
    r = pipeline.run()
    assert r["all_layers_pass"], "a layer failed its gate: %s" % [
        L["level"] for L in r["levels"] if not L["passed"]]
    assert len(r["levels"]) == 5
    assert r["end_to_end"]["coherent"]        # aversive -> reversal, and drivers diverge


TESTS = [test_bridge_stimulus_valence, test_compound_direction, test_proteostasis_sota,
         test_twin_domain_gate_and_consistency, test_ad_reference_domain,
         test_l2_imputation_beats_shuffle, test_l4_beats_hallinen_ridge,
         test_l3_beats_persistence, test_perturbation_beats_shuffle,
         test_ablation_recovers_circuit, test_external_validation_transfers,
         test_five_layer_pipeline]


if __name__ == "__main__":
    print("=" * 70)
    print("PLATFORM VALIDATION SUITE — positive result + negative control each")
    print("=" * 70)
    passed = failed = 0
    for fn in TESTS:
        try:
            fn(); print("  PASS  %s" % fn.__name__); passed += 1
        except AssertionError as e:
            print("  FAIL  %s  -- %s" % (fn.__name__, e)); failed += 1
        except Exception as e:
            print("  ERROR %s  -- %s" % (fn.__name__, str(e)[:70])); failed += 1
    print("=" * 70)
    print("%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
