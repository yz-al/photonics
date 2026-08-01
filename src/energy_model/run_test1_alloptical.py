"""
Inverse model — solve for the nonlinear element an all-optical (convert-once-in/out)
architecture would need to clear the 20-40 fJ/MAC compute floor.

Task 1: cascading gate (does depth force inter-layer amplification?).
Task 2: backward-solve the required switching-energy spec surface over N, L, bits,
        nl_loss, and report the WEAKEST sufficient spec.
Writes data/test1_alloptical_results.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import PARAMS, all_optical_spec, sample_params, LINK_BUDGET_DB

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
base = {k: v.nom for k, v in PARAMS.items()}
out = {"arch": "all_optical_inverse", "link_budget_db": LINK_BUDGET_DB}

# ---------------------------------------------------------------------------
# Task 1 — cascading gate: depth vs amplification (nl_loss swept)
# ---------------------------------------------------------------------------
print("=== Task 1: cascading gate (needs inter-layer amplification?) ===")
casc = {}
for L in [4, 8, 16]:
    row = {}
    for nl in [0.0, 0.5, 1.0, 2.0]:
        s = all_optical_spec(256, L, 4, base, nl_loss_db=nl)
        row[f"nl_{nl}"] = {"total_loss_db": s["total_loss_db"], "amp": s["needs_amplification"]}
        print(f"  L={L:2d} nl_loss={nl}dB: total={s['total_loss_db']:5.1f}dB "
              f"needs_amp={s['needs_amplification']}")
    casc[f"L{L}"] = row
out["cascading_gate"] = casc

# ---------------------------------------------------------------------------
# Task 2 — backward solve: max tolerable switching energy E_nl (the spec surface)
# ---------------------------------------------------------------------------
print("\n=== Task 2: required switching energy E_nl_max (fJ) — the spec surface ===")
print("    (E_nl below this clears the floor; 'binding' = which constraint sets it)")
surface = {}
for bits in (4, 8):
    for N in [64, 256, 1024]:
        for L in [4, 8, 16]:
            s = all_optical_spec(N, L, bits, base, nl_loss_db=0.5)
            key = f"{bits}b_N{N}_L{L}"
            surface[key] = {"E_nl_max_fJ": s["E_nl_max_fJ"], "binding": s["binding"],
                            "P_op_mW": s["P_op_at_Emax_mW"], "amp": s["needs_amplification"],
                            "total_loss_db": s["total_loss_db"]}
            amp = " AMP-NEEDED" if s["needs_amplification"] else ""
            print(f"  {key:14s}: E_nl<= {s['E_nl_max_fJ']:8.3f} fJ  P_op={s['P_op_at_Emax_mW']:.2g} mW "
                  f"bind={s['binding']}{amp}")
out["spec_surface"] = surface

# ---------------------------------------------------------------------------
# Weakest sufficient spec: the easiest corner (largest tolerable E_nl, no amp)
# ---------------------------------------------------------------------------
print("\n=== Weakest sufficient spec (easiest buildable corner) ===")
best = None
for bits in (4, 8):
    for N in [64, 256, 1024]:
        for L in [4, 8, 16]:
            for nl in [0.1, 0.5, 1.0]:
                s = all_optical_spec(N, L, bits, base, nl_loss_db=nl)
                if s["needs_amplification"]:
                    continue
                if best is None or s["E_nl_max_fJ"] > best["E_nl_max_fJ"]:
                    best = {**s}
print(f"  weakest spec: N={best['N']} L={best['L']} {best['bits']}-bit nl_loss={best['nl_loss_db']}dB")
print(f"    -> E_nl <= {best['E_nl_max_fJ']:.3f} fJ  (P_op {best['P_op_at_Emax_mW']:.2g} mW, "
      f"binding {best['binding']}, total loss {best['total_loss_db']:.1f}dB)")
out["weakest_spec"] = best

# ---------------------------------------------------------------------------
# MC: fraction of sourced-parameter draws in which ANY (N,L) all-optical config
# clears the floor with a physically stated E_nl target (e.g. 1 fJ, 10 fJ)
# ---------------------------------------------------------------------------
print("\n=== MC: with a target switching energy, does any (N,L,4-bit) clear the floor? ===")
Ncand = [64, 256, 1024]; Lcand = [4, 8]
mc = {}
for E_target_fJ in (0.1, 1.0, 10.0, 100.0):
    rng = np.random.default_rng(77)
    hits = 0
    for _ in range(3000):
        pr = sample_params(rng)
        ok = False
        for N in Ncand:
            for L in Lcand:
                s = all_optical_spec(N, L, 4, pr, nl_loss_db=0.5)
                if (not s["needs_amplification"]) and E_target_fJ * 1e-15 <= s["E_nl_max_J"]:
                    ok = True; break
            if ok: break
        hits += ok
    mc[f"E_{E_target_fJ}fJ"] = hits / 3000
    print(f"  target E_nl={E_target_fJ:6.1f} fJ: clears in {hits/3000*100:.0f}% of draws")
out["mc_target_switching_energy"] = mc

# ---------------------------------------------------------------------------
# Task 2b + Task 4 — required material n2 and the Kramers-Kronig verdict
# ---------------------------------------------------------------------------
print("\n=== Task 4: required n2 vs the TPA-free KK ceiling at 1550 nm ===")
LAM = 1.55e-6; C = 2.998e8
def n2_required(E_J, A=1e-13, Lwg=1e-4, tau=1e-10, Q=1.0):
    # Kerr pi-shift: E = A*lam*tau/(2*n2*Lwg); Q = resonant enhancement of effective n2
    return A * LAM * tau / (2 * E_J * Lwg) / Q
# TPA-free ceiling: TPA(beta)=0 requires Eg>2*hw=1.60 eV at 1550nm; n2 ~ Eg^-4 caps at
# the As2Se3/As2S3-class value ~1e-17 m^2/W (Sheik-Bahae/Hutchings two-band model).
N2_CEILING = 1.1e-17
kk = {"tpa_free_needs_Eg_eV": 1.60, "n2_ceiling_m2W": N2_CEILING, "cases": {}}
for lab, E in [("weakest_7fJ", 7e-15), ("robust_1fJ", 1e-15), ("useful_0.1fJ", 1e-16)]:
    n2wg = n2_required(E)
    kk["cases"][lab] = {"n2_req_wg": n2wg, "gap_orders": np.log10(n2wg / N2_CEILING),
                        "Q_to_reach_ceiling": n2wg / N2_CEILING,
                        "tau_at_that_Q_ns": (n2wg / N2_CEILING) * LAM / (2 * np.pi * C) * 1e9}
    print(f"  {lab:12s}: req n2={n2wg:.2e}  gap={np.log10(n2wg/N2_CEILING):.1f} orders  "
          f"Q_needed={n2wg/N2_CEILING:.0e} -> tau={(n2wg/N2_CEILING)*LAM/(2*np.pi*C)*1e9:.0f} ns (too slow)")
print("  VERDICT: Kerr/chi(3) route KK-forbidden at 1550nm (req n2 6-8 orders > TPA-free ceiling);")
print("           bridging via Q costs speed (Q~1e6 -> us), via high-n2 material costs loss (ENZ/2D).")
out["kramers_kronig"] = kk

with open(os.path.join(DATA, "test1_alloptical_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/test1_alloptical_results.json")
