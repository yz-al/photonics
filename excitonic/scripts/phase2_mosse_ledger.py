#!/usr/bin/env python3
"""
Phase 2 — the MoSSe (Janus) blockade ledger: assemble the three legs, don't block on f.

The MoSSe verdict needs three numbers: f (brightness), U (interaction), Γ (linewidth).
Two of them we can pin NOW from data in hand — a forward step while the GW-BSE f cooks:

  Γ_floor(MoSSe): bracketed by the DFPT gate. MoS2 Γ≈10.3, MoSe2 Γ≈10.8 meV (both Mo-TMDs),
                  so the Janus MoSSe (one S, one Se) sits at Γ ≈ 10.5 meV → needs U > 7.5 meV.
  U_dipolar(MoSSe): from the Janus built-in dipole. And here is the catch the solve-for-it
                  optimism glossed: the Janus escapes interlayer DARKNESS by keeping the
                  exciton INTRALAYER — which means its permanent dipole is tiny. The C2DB
                  structural dipole is dipz ≈ 0.04–0.18 e·Å ⇒ charge separation d ≈ 0.004–0.018
                  nm, vs an interlayer exciton's d ≈ 0.6 nm. Since U ∝ d², the Janus dipolar U
                  is ~(0.02–0.3)² ×  smaller ⇒ orders of magnitude below the interlayer's
                  measured 4–20 meV, i.e. WELL below the 7.5 meV threshold.

So the honest forward finding, before f even lands: the Janus does NOT escape the U–f
trade-off — it is the OTHER end of it. Interlayer = dark but big dipole (fails on f);
Janus = bright but tiny dipole (fails on U_dipolar). The BSE f will confirm MoSSe is bright,
but its DIPOLAR-blockade U is structurally too small. Its remaining blockade route is the
SATURATION/moiré channel it shares with every bright TMD (the single-emitter leg) — not a
new door.

    python excitonic/scripts/phase2_mosse_ledger.py
Writes excitonic/data/manifests/phase2_mosse_ledger.json.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests", "phase2_mosse_ledger.json"))

E2_4PIEPS0_eV_nm = 1.43996
THRESHOLD = 0.71


def u_dipolar(d_nm, a_exc_nm=1.0, eps_r=5.0):
    return E2_4PIEPS0_eV_nm * d_nm ** 2 / (eps_r * a_exc_nm ** 3)   # eV


def main() -> int:
    gamma = 10.5                       # meV, bracketed MoS2(10.3)–MoSe2(10.8) DFPT
    need_U = THRESHOLD * gamma         # meV

    # Janus dipole → charge separation (dipz 0.04–0.18 e·Å = 0.004–0.018 nm)
    d_lo, d_hi = 0.004, 0.018          # nm
    U_dip_lo = u_dipolar(d_lo) * 1000  # meV
    U_dip_hi = u_dipolar(d_hi) * 1000  # meV
    # interlayer reference for contrast
    U_interlayer_meas = "4–20 meV (measured, d≈0.6 nm)"

    manifest = {
        "material": "MoSSe (Janus, solve-for-it candidate)",
        "legs": {
            "Gamma_floor_meV": {"value": gamma, "tier": "dfpt (bracketed MoS2/MoSe2)",
                                "need_U_gt_meV": round(need_U, 2), "status": "LOCKED"},
            "f_brightness": {"status": "GW-BSE running (ABINIT); Janus exciton is intralayer "
                             "⇒ expected BRIGHT (unlike the 10–100× dark interlayer exciton)",
                             "tier": "gw_bse (pending)"},
            "U_dipolar_meV": {"range": [round(U_dip_lo, 4), round(U_dip_hi, 3)],
                              "tier": "proxy (Janus dipz → d → U∝d²)",
                              "interlayer_reference": U_interlayer_meas,
                              "status": "≪ threshold: Janus dipole ~15–150× smaller than "
                                        "interlayer ⇒ U_dip ~2–4 orders below the 7.5 meV need"},
            "U_saturation": {"status": "the moiré/single-emitter channel MoSSe shares with any "
                             "bright TMD (E_b·a_B²/area); this is the leg that stays open, not "
                             "the dipole", "tier": "same as TMD saturation leg"},
        },
        "forward_finding": (
            "Before f even lands: the Janus does NOT escape the U–f trade-off, it is the other "
            "END of it. Interlayer = dark + big dipole (fails on f). Janus = bright + tiny "
            f"dipole (U_dip ≈ {U_dip_lo:.3f}–{U_dip_hi:.2f} meV ≪ {need_U:.1f} meV need). The "
            "BSE will confirm MoSSe is bright, but its DIPOLAR blockade U is structurally far "
            "too small. So Janus is not a new dipolar door — it's a bright TMD whose only "
            "blockade route is the same saturation/moiré single-emitter leg as MoS2."),
        "what_the_bse_still_adds": "MoSSe E_b + absolute f (brightness confirmation, and the "
                                   "exciton's own out-of-plane dipole if ABINIT reports it — "
                                   "the real U_dip rather than the dipz proxy).",
        "verdict": "Janus dipolar-blockade route CLOSES on U (tiny dipole). MoSSe remains a "
                   "bright, tightly-bound TMD → its blockade hope is the saturation/moiré "
                   "single-emitter leg, already characterized as open-but-single-emitter. "
                   "The solve-for-it 'escape' was half right (bright ✓) and half wrong (dipole ✗).",
        "tier": "dfpt (Γ) + proxy (U) + gw_bse-pending (f). Honest partial verdict.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[mosse-ledger] Γ_floor ≈ {gamma} meV (DFPT bracket) → need U > {need_U:.1f} meV")
    print(f"[mosse-ledger] U_dipolar ≈ {U_dip_lo:.3f}–{U_dip_hi:.2f} meV  (Janus dipole tiny) "
          f"≪ {need_U:.1f} meV")
    print(f"[mosse-ledger] f: bright expected (intralayer) — BSE pending")
    print(f"[mosse-ledger] VERDICT: Janus dipolar route CLOSES on U; falls back to the "
          f"saturation/moiré single-emitter leg (same as any bright TMD).")
    print(f"[mosse-ledger] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
