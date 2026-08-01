"""
Turn a Test-2 results JSON into the markdown tables for reports/photonic_thesis.md.

Usage: python test2/fill_report.py test2/results/test2_results_full.json
Prints the Stage A, Stage B, and headline tables to stdout; also writes them into
reports/photonic_thesis.md by replacing the <!-- TEST2_* --> anchor lines.
"""
import json, os, sys, re

path = sys.argv[1] if len(sys.argv) > 1 else "test2/results/test2_results_full.json"
d = json.load(open(path))
A, B = d.get("stageA", {}), d.get("stageB", {})
ACTS = ["relu", "gelu", "square", "modsq", "abs", "scaled_quad"]


def fmt(e):
    if not e:
        return "—"
    m, s = e.get("mean_best_acc"), e.get("std_best_acc")
    div = e.get("n_diverged", 0)
    tag = f" ⚠{div}div" if div else ""
    return f"{m:.3f}±{s:.3f}{tag}"


def stage_a_table():
    rows = ["| Activation | MLP (Cover-Type) | CNN (CIFAR-10) | Transformer |",
            "|---|---|---|---|"]
    mlp = A.get("mlp_covtype", {}); cnn = A.get("cnn_cifar10", {})
    trk = next((k for k in A if k.startswith("transformer_")), None)
    tr = A.get(trk, {}) if trk else {}
    for a in ACTS:
        rows.append(f"| {a} | {fmt(mlp.get(a))} | {fmt(cnn.get(a))} | {fmt(tr.get(a))} |")
    probe = A.get("depth_stability_square", {})
    if probe:
        rows.append("")
        rows.append("Depth × normalisation stability (x² activation, mean best acc; `div` = diverged seeds):")
        rows.append("")
        depths = sorted({int(k.split("_d")[1]) for k in probe})
        rows.append("| norm | " + " | ".join(f"depth {dp}" for dp in depths) + " |")
        rows.append("|" + "---|" * (len(depths) + 1))
        for nm in ["none", "batch", "layer"]:
            cells = []
            for dp in depths:
                e = probe.get(f"{nm}_d{dp}")
                cells.append(f"{e['mean_best_acc']:.3f}" + (f" ⚠{e['n_diverged']}d" if e.get("n_diverged") else "") if e else "—")
            rows.append(f"| {nm} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def stage_b_table():
    out = []
    ds = B.get("optics_depth_sweep", {})
    if ds:
        out += ["**Optics-native |·|² depth sweep** (complex linear + |·|², full precision):", "",
                "| depth | mean best acc | diverged |", "|---|---|---|"]
        for k in sorted(ds):
            e = ds[k]; out.append(f"| {k.replace('depth','')} | {e['mean_best_acc']:.3f}±{e['std_best_acc']:.3f} | {e['n_diverged']} |")
        out.append("")
    q = B.get("quantization_mlp", {})
    if q:
        out += ["**Quantisation** (real MLP, ReLU vs x², activation bit depth):", "",
                "| activation | full | 8-bit | 4-bit |", "|---|---|---|---|"]
        for a in ["relu", "square"]:
            out.append(f"| {a} | {fmt(q.get(a+'_None'))} | {fmt(q.get(a+'_8b'))} | {fmt(q.get(a+'_4b'))} |")
        out.append("")
    o = B.get("optics_native_mlp", {})
    if o:
        out += ["**Optics-native variants** (complex linear + |·|², matched params):", "",
                "| variant | mean best acc | params |", "|---|---|---|"]
        for k in ["complex_modsq_fp", "complex_modsq_8b", "complex_modsq_4b", "complex_modsq_4b_nonneg"]:
            e = o.get(k)
            if e: out.append(f"| {k} | {fmt(e)} | {e['params']:,} |")
        out.append("")
    n = B.get("noise_sweep_mlp", {})
    if n:
        out += ["**Shot-noise SNR sweep** (optics-native 4-bit):", "",
                "| SNR (dB) | mean best acc |", "|---|---|"]
        for k in ["snr_None", "snr_40", "snr_30", "snr_20", "snr_15", "snr_10"]:
            e = n.get(k)
            if e: out.append(f"| {k.replace('snr_','')} | {fmt(e)} |")
        out.append("")
    nc = B.get("nonneg_cost", {})
    if nc:
        out += [f"**Non-negativity cost**: signed-input {nc['signed_input_params']:,} params → "
                f"differential (two-channel) {nc['nonneg_diff_params']:,} params "
                f"(ReLU ref {nc['relu_ref_params']:,}).", ""]
    return "\n".join(out)


def headline():
    h = B.get("headline_gap_mlp", {})
    if not h:
        return "*(headline gap pending)*"
    r = h.get("relu_8b", {}); o = h.get("optics_native", {})
    gap = h.get("gap_relu8b_minus_optics")
    return (f"**Headline gap (MNIST, matched params):** ReLU 8-bit = {fmt(r)}, "
            f"fully optics-native (complex + |·|² + 4-bit + non-negative + shot noise @20 dB) "
            f"= {fmt(o)} → **accuracy gap = {gap*100:+.1f} points**.")


sa, sb, hd = stage_a_table(), stage_b_table(), headline()
print("=== STAGE A ===\n", sa, "\n\n=== STAGE B ===\n", sb, "\n\n=== HEADLINE ===\n", hd)

rp = os.path.join(os.path.dirname(__file__), "..", "reports", "photonic_thesis.md")
if os.path.exists(rp) and "--write" in sys.argv:
    txt = open(rp).read()
    txt = txt.replace("<!-- TEST2_STAGE_A_TABLE -->\n*(Populated from the 5-seed `local` run; see `test2/results/`.)*",
                      "<!-- TEST2_STAGE_A_TABLE -->\n" + sa)
    txt = txt.replace("<!-- TEST2_STAGE_B_TABLE -->\n*(Populated from the 5-seed `local` run; see `test2/results/`.)*",
                      "<!-- TEST2_STAGE_B_TABLE -->\n" + sb)
    txt = re.sub(r"<!-- TEST2_HEADLINE -->\n\*\(Populated.*?\)\* ", "<!-- TEST2_HEADLINE -->\n" + hd + "\n\n", txt, flags=re.S)
    open(rp, "w").write(txt)
    print("\n[fill_report] wrote tables into", rp)
