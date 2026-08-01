"""
Can RT single-emitter/polariton blockade actually be reached? — a driven-dissipative
Kerr-mode simulation fed with SOURCED interaction (U) and phonon-broadened linewidth Γ(T).

Physics engine: a single anharmonic (Kerr) cavity mode under weak coherent drive,
    H = Δ a†a + (U/2) a†a†a a + F(a + a†),   collapse c = sqrt(Γ) a,
solved for the Lindblad steady state; blockade quality = g²(0) = <a†a†aa>/<a†a>²
minimised over drive detuning Δ (best case). Blockade threshold: g²(0) < 0.5.

Then:
 1. (U/Γ)_crit for g²(0)=0.5 and 0.1 — the mechanism-independent quantum-optics threshold.
 2. Γ(T) from a sourced 3-term phonon model (residual + acoustic + Fröhlich/LO) anchored to
    measured RT homogeneous linewidths (TMD ~5-15 meV; halide perovskite ~20-70 meV).
 3. Required athermal U at 300 K per material = (U/Γ)_crit · Γ(300 K).
 4. Demonstrated U carried in from the literature (best cold U/Γ≈0.42, Delteil 2019; saturation
    scale 50-300 µeV, arXiv:2501.07899) with dipolar enhancement 10× (bilayer MoS₂,
    Nat. Commun. 2022) and 200× (dipolaritons, PRL 121, 227402 (2018)).
 5. Crossover temperature T* where g²(0)=0.5 for the best demonstrated athermal U.
 6. MC over sourced ranges: fraction of draws reaching RT blockade per mechanism.

Writes data/blockade_sim_results.json.
"""
import json, os
import numpy as np
from numpy import kron

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
KB = 0.0861733  # meV/K

# ---------------------------------------------------------------------------
# Lindblad steady-state g2(0) for a single Kerr mode (all energies in units of Γ)
# ---------------------------------------------------------------------------
def _ops(dim):
    a = np.diag(np.sqrt(np.arange(1, dim)), 1).astype(complex)
    return a, a.conj().T, np.eye(dim, dtype=complex)

def g2_of_UoverGamma(uog, dim=12, F=1e-3, n_det=61):
    """g2(0) minimised over drive detuning, for interaction/linewidth ratio U/Γ (Γ=1)."""
    a, ad, I = _ops(dim)
    ada = ad @ a
    n_op = ada
    kerr = ad @ ad @ a @ a
    best = np.inf
    for delta in np.linspace(-1.5, 2.5, n_det):
        H = delta * n_op + 0.5 * uog * kerr + F * (a + ad)
        # Liouvillian, column-stacking convention: vec(AXB)=(B^T⊗A)vec(X)
        L = -1j * (kron(I, H) - kron(H.T, I))
        L += (kron(a.conj(), a) - 0.5 * kron(I, ada) - 0.5 * kron(ada.T, I))
        # steady state: replace first equation with trace condition Tr(ρ)=1
        M = L.copy()
        vecI = I.flatten(order="F")
        M[0, :] = vecI
        b = np.zeros(dim * dim, dtype=complex); b[0] = 1.0
        rho = np.linalg.solve(M, b).reshape(dim, dim, order="F")
        n = np.real(np.trace(n_op @ rho))
        if n <= 0:
            continue
        g2 = np.real(np.trace(kerr @ rho)) / (n * n)
        best = min(best, g2)
    return best

def uog_for_g2(target, lo=1e-3, hi=50.0):
    """invert: smallest U/Γ giving min-over-detuning g2(0) <= target (bisection)."""
    # g2 decreases with U/Γ; find crossing
    flo, fhi = g2_of_UoverGamma(lo), g2_of_UoverGamma(hi)
    if fhi > target:
        return np.inf
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if g2_of_UoverGamma(mid) > target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)

# ---------------------------------------------------------------------------
# Sourced phonon linewidth Γ(T) = Γ0 + c_ac·T + Γ_LO / (exp(E_LO/kT) - 1)
# nominal / (low, high) anchored to measured RT homogeneous linewidths
# ---------------------------------------------------------------------------
MAT = {
    # material : (Γ0 meV, c_ac meV/K, Γ_LO meV, E_LO meV)  -> Γ(300K) target
    "TMD":        {"G0": (1.0, 1.6, 2.5), "cac": (0.006, 0.010, 0.015),
                   "GLO": (12.0, 17.0, 24.0), "ELO": (28.0, 30.0, 34.0)},   # ~5-15 meV @300K
    "perovskite": {"G0": (2.0, 4.0, 8.0), "cac": (0.010, 0.020, 0.030),
                   "GLO": (30.0, 42.0, 60.0), "ELO": (16.0, 19.0, 22.0)},   # ~20-70 meV @300K
}

def gamma_T(T, p):
    x = p["ELO"] / (KB * T)
    return p["G0"] + p["cac"] * T + p["GLO"] / np.expm1(x)

def _draw(rng, triple):
    lo, nom, hi = triple
    if hi <= lo:
        return float(nom)
    # triangular around nominal within [lo,hi]
    return float(rng.triangular(lo, nom, hi))

def draw_mat(rng, name):
    m = MAT[name]
    return {k: _draw(rng, v) for k, v in m.items()}

def nom_mat(name):
    return {k: v[1] for k, v in MAT[name].items()}

# ---------------------------------------------------------------------------
# Demonstrated athermal interaction U (meV) — sourced, mechanism-dependent
#   saturation baseline (GaAs-class, measured): 0.05-0.30 meV  (arXiv:2501.07899)
#   best confined cold ratio U/Γ≈0.42 (Delteil, Nat. Mater. 18, 219 (2019)) is consistent
#   dipolar enhancement: ×10 (bilayer MoS₂), ×200 (dipolaritons, cryo GaAs) — applied to a
#   TMD-class saturation baseline downscaled by (a_B ratio)² ≈ (1nm/10nm)² = 0.01
# ---------------------------------------------------------------------------
U_SAT_GAAS = (0.05, 0.15, 0.30)          # meV, measured saturation nonlinear scale
A_B_SCALE = (0.006, 0.010, 0.02)         # (a_B_TMD/a_B_GaAs)^2 range, ~0.01
ENH = {"saturation": (1.0, 1.0, 1.0),
       "dipolar_10x": (5.0, 10.0, 20.0),
       "dipolariton_200x": (80.0, 200.0, 300.0)}

def U_demo(rng, mech):
    return _draw(rng, U_SAT_GAAS) * _draw(rng, A_B_SCALE) * _draw(rng, ENH[mech])

out = {"model": "driven-dissipative Kerr mode, Lindblad steady state, g2(0)<0.5 = blockade"}

# ---------------------------------------------------------------------------
# 1. Mechanism-independent quantum-optics threshold
# ---------------------------------------------------------------------------
print("=== 1. blockade threshold in U/Γ units (min over drive detuning) ===")
crit = {}
for tgt in (0.5, 0.1, 0.01):
    r = uog_for_g2(tgt)
    crit[f"g2_{tgt}"] = r
    print(f"  g2(0) <= {tgt:>4}: needs U/Γ >= {r:.3f}")
# a few sample g2 values for the record
curve = {f"{u:g}": g2_of_UoverGamma(u) for u in (0.1, 0.42, 0.7, 1.0, 2.0, 5.0)}
for u, g in curve.items():
    print(f"    U/Γ={u:>4}: g2(0)={g:.3f}")
out["threshold_UoverGamma"] = crit
out["g2_curve"] = curve
UOG_50 = crit["g2_0.5"]

# ---------------------------------------------------------------------------
# 2-3. Γ(300K) and required athermal U per material (nominal)
# ---------------------------------------------------------------------------
print("\n=== 2-3. Γ(300K) and required U for blockade (nominal) ===")
req = {}
for name in MAT:
    p = nom_mat(name)
    g300 = gamma_T(300.0, p); g4 = gamma_T(4.0, p)
    Ureq = UOG_50 * g300
    req[name] = {"gamma_4K_meV": g4, "gamma_300K_meV": g300, "U_req_300K_meV": Ureq}
    print(f"  {name:11s}: Γ(4K)={g4:6.2f} meV  Γ(300K)={g300:6.2f} meV  "
          f"-> U_req(300K) = {Ureq:6.2f} meV  (U/Γ={UOG_50:.2f})")
out["required_U"] = req

# ---------------------------------------------------------------------------
# 4a. Pure thermal penalty (fewest assumptions): take the BEST measured cold ratio
#     U/Γ=0.42 (Delteil) and apply ONLY the measured Γ(300K)/Γ(cold) rise — no
#     material-transfer / a_B extrapolation. Isolates the thermal effect alone.
# ---------------------------------------------------------------------------
print("\n=== 4a. pure thermal penalty on the best measured cold ratio (U/Γ=0.42) ===")
BEST_COLD_RATIO = 0.42  # Delteil, Nat. Mater. 18, 219 (2019)
thermal = {}
for name in MAT:
    p = nom_mat(name)
    g4, g300 = gamma_T(4.0, p), gamma_T(300.0, p)
    uog300 = BEST_COLD_RATIO * g4 / g300           # U athermal, Γ rises
    thermal[name] = {"gamma_ratio_300_over_4": g300 / g4, "U_over_G_300K": uog300,
                     "short_of_threshold_x": UOG_50 / uog300}
    print(f"  {name:11s}: Γ(300K)/Γ(4K)={g300/g4:5.1f}×  -> U/Γ(300K)={uog300:.3f}  "
          f"({UOG_50/uog300:.0f}× short of {UOG_50:.2f} threshold) — thermal penalty alone")
out["thermal_only"] = thermal

# ---------------------------------------------------------------------------
# 4-5. Best demonstrated athermal U, and crossover temperature T* (g2=0.5)
# ---------------------------------------------------------------------------
print("\n=== 4-5. demonstrated U (incl. a_B material-transfer estimate) and T* ===")
tstar = {}
Ts = np.arange(4, 401, 2.0)
for mech in ENH:
    # nominal demonstrated U for this mechanism
    Un = U_SAT_GAAS[1] * A_B_SCALE[1] * ENH[mech][1]
    row = {"U_demo_meV": Un}
    for name in MAT:
        p = nom_mat(name)
        Tc = None
        for T in Ts:
            if (Un / gamma_T(T, p)) < UOG_50:   # dropped below blockade threshold
                Tc = T; break
        # if never below threshold across the whole range, blockade holds throughout
        if Tc is None:
            Tc = ">400"
        # T* is where it CROSSES: find last T with U/Γ>=crit
        Tc2 = 0
        for T in Ts:
            if (Un / gamma_T(T, p)) >= UOG_50:
                Tc2 = T
        row[name] = {"U_over_G_300K": Un / gamma_T(300.0, p), "T_blockade_max_K": Tc2}
    tstar[mech] = row
    print(f"  {mech:16s} U={Un:7.3f} meV: "
          + "  ".join(f"{n}: U/Γ(300K)={row[n]['U_over_G_300K']:.3f}, "
                      f"T*≤{row[n]['T_blockade_max_K']:.0f}K" for n in MAT))
out["crossover_temperature"] = tstar

# ---------------------------------------------------------------------------
# 6. MC: fraction of sourced draws reaching RT blockade (g2(0)<0.5 at 300K)
# ---------------------------------------------------------------------------
print("\n=== 6. MC: fraction of draws reaching RT blockade (U/Γ(300K) >= crit) ===")
mc = {}
N_MC = 4000
for mech in ENH:
    for name in MAT:
        rng = np.random.default_rng(hash((mech, name)) % (2**32))
        hits = 0
        gaps = []
        for _ in range(N_MC):
            U = U_demo(rng, mech)
            g300 = gamma_T(300.0, draw_mat(rng, name))
            uog = U / g300
            gaps.append(UOG_50 / uog)   # factor short (>1 means fails)
            hits += (uog >= UOG_50)
        frac = hits / N_MC
        med_gap = float(np.median(gaps))
        mc[f"{mech}_{name}"] = {"rt_blockade_frac": frac, "median_gap_factor": med_gap}
        print(f"  {mech:16s} + {name:11s}: RT blockade in {frac*100:5.1f}% of draws  "
              f"(median {med_gap:.0f}× short of threshold)")
out["mc_rt_blockade"] = mc

with open(os.path.join(DATA, "blockade_sim_results.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote data/blockade_sim_results.json")
