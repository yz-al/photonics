"""
Phase 3 — the ARRAY-UNIFORMITY bound (the compute-relevant question).

The single-emitter verdict (U/Γ>1 at one deep trap/color center) does NOT give optical
COMPUTE: a neural net needs a SCALABLE, SPECTRALLY-UNIFORM ARRAY of nonlinear elements,
one per activation, all addressable by a shared cavity/drive. This bounds that.

Tolerance: for a shared-cavity blockade array, emitters must sit within ~the linewidth Γ
(a resonance can only blockade what it's resonant with). So the array works only if the
site-to-site inhomogeneous ZPL spread σ_inhom ≲ Γ.

Inputs (measured, sourced):
  Γ (RT homogeneous linewidth, TMD/emitter class): ~10 meV [Selig Nat.Commun.2016; ~14 meV]
  hBN ZPL inhomogeneous spread:
    - typical: 1.5–2.2 eV total range (~700 meV) [Sci.Rep. 2021; strain/Stark/dielectric]
    - BEST demonstrated uniformity: 85% within 580±10 nm -> ±10 nm ≈ ±37 meV
      [structured-defect engineering, ACS Nano 2024, 10.1021/acsnano.4c11413]
Also: the disorder is environmental (strain/dielectric/trapped-charge Stark), NOT the defect
chemistry — so "identical-by-chemistry" defects do NOT remove the dominant term.
"""
import json, os

GAMMA_meV = 10.0                      # RT linewidth tolerance (Selig; ~10-14 meV)
HC_eVnm = 1239.84

def dlambda_to_dE_meV(lam_nm, dlam_nm):
    return HC_eVnm / lam_nm**2 * dlam_nm * 1e3

best_sigma = dlambda_to_dE_meV(580.0, 10.0)      # ±10 nm at 580 nm
typ_sigma  = 700.0                                # ~1.5-2.2 eV total span

out = {
    "gate": "shared-cavity blockade array works iff sigma_inhom <~ Gamma",
    "Gamma_meV": GAMMA_meV,
    "best_demonstrated": {
        "source": "ACS Nano 2024 (structured-defect hBN), 85% within 580+/-10 nm",
        "sigma_inhom_meV": round(best_sigma, 1),
        "sigma_over_gamma": round(best_sigma / GAMMA_meV, 1),
        "verdict": "FAILS" if best_sigma > GAMMA_meV else "ok",
    },
    "typical": {
        "source": "Sci.Rep.2021: hBN ZPL 1.5-2.2 eV (~700 meV), strain/Stark/dielectric",
        "sigma_inhom_meV": typ_sigma,
        "sigma_over_gamma": round(typ_sigma / GAMMA_meV, 0),
        "verdict": "FAILS",
    },
    "disorder_origin": "environmental (strain/dielectric/Stark), NOT defect chemistry -> "
                       "'identical-by-chemistry' color centers do not remove the dominant term",
    "verdict": ("Array uniformity FAILS by ~4x (best demonstrated) to ~70x (typical). "
                "Single-emitter blockade does not scale to a compute array on any current "
                "hBN platform; the escape is unsolved, and the dominant disorder is not the "
                "one the identical-by-chemistry proposal removes."),
}
print(json.dumps(out, indent=2))
os.makedirs("excitonic/data/manifests", exist_ok=True)
json.dump(out, open("excitonic/data/manifests/phase3_array_uniformity.json", "w"), indent=2)

# ---- omega_LO mislabel impact (does the floor move?) ----
import math
def gamma_model(hw_lo, alpha=0.4, C_cal=2.9, G0=2.0, T=300.0):
    n = 1.0/math.expm1(hw_lo/(0.0861733*T))
    return G0 + C_cal*alpha*hw_lo*n
print("\n=== omega_LO mislabel impact on the Gamma floor ===")
for lab, hw in [("DFPT top-mode A2'' (mislabeled) 56.4", 56.4),
                ("Frohlich-active E' LO (correct) 48.0", 48.0)]:
    print(f"  {lab} meV -> Gamma(300K) = {gamma_model(hw):.1f} meV")
print("  Both within measured 10-14 meV RT range -> floor + verdict UNCHANGED.")
