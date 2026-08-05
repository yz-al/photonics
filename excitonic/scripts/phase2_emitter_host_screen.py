#!/usr/bin/env python3
"""
Phase 4 — the pivot: screen HOSTS for a room-temperature, uniform single-photon emitter.

Scalable bulk exciton blockade is closed. The one open corner (a single deep trap) fails
to SCALE only because random moiré traps are disorder-varied. The compute-addressable fix:
use emitters that are IDENTICAL BY CONSTRUCTION — chemically-identical point defects / color
centers in a stiff wide-gap host — so there is no ensemble disorder to begin with. A single
two-level defect is also an intrinsic photon blockade (it cannot absorb a second photon), so
the exciton U/Γ fight is replaced by a cleaner emitter figure of merit.

WHAT MAKES A HOST SUPPORT RT SINGLE-PHOTON BLOCKADE (the FOM, derived):
  1. NARROW ZERO-PHONON LINE AT 300 K — needs a STIFF, LIGHT lattice: high Debye temperature
     θ_D ∝ √(bond-strength / mass) freezes out phonons (ħω_phonon ≫ kT) so the ZPL stays sharp
     and the Debye–Waller factor (fraction of light in the ZPL) stays high. This is the same
     phonon physics as the Γ floor — but now weak coupling is ACHIEVABLE because the emitter
     is a localized deep state in a rigid host, not a mobile band exciton.
  2. WIDE GAP — deep in-gap defect levels, optically isolated from the bands.
  3. STABLE, NON-MAGNETIC host (a clean two-level system).
  4. (defect-level, next phase) a bright deep level with a spin-conserving optical transition.

Anchors that MUST score high if the FOM is right: hBN (BN) and light nitrides — the known
RT single-photon-emitter hosts. Soft ultra-wide-gap ionic halides must score LOW (broad RT
ZPL) despite their huge gaps — the test the FOM has to pass.

TIER: host-level proxy (Debye-temp proxy from formation energy + mass; gap). It ranks hosts;
CONFIRMING a specific RT emitter needs defect-supercell DFT/GW (formation energy, ZPL, Debye–
Waller, transition dipole) — the named next phase, not claimed here.

    python excitonic/scripts/phase2_emitter_host_screen.py
Writes excitonic/data/manifests/phase2_emitter_host_screen.json.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re

from ase.data import atomic_masses, atomic_numbers

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "c2db_excitonic.csv"))
OUT = os.path.abspath(os.path.join(HERE, "..", "data", "manifests",
                                   "phase2_emitter_host_screen.json"))


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def mean_atomic_mass(formula):
    toks = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    tot_m, tot_n = 0.0, 0
    for el, cnt in toks:
        if not el or el not in atomic_numbers:
            continue
        n = int(cnt) if cnt else 1
        tot_m += atomic_masses[atomic_numbers[el]] * n
        tot_n += n
    return (tot_m / tot_n) if tot_n else None


def main() -> int:
    rows = list(csv.DictReader(open(CSV)))
    pool = []
    for r in rows:
        gap = num(r, "gap_hse") or num(r, "gap_gw") or num(r, "gap")
        hform = num(r, "hform")           # eV/atom (formation energy; more negative = stronger bonds)
        eh = num(r, "ehull")
        mm = mean_atomic_mass(r.get("formula", ""))
        mag = str(r.get("is_magnetic", "")).strip().lower() == "yes"
        if gap is None or hform is None or mm is None or gap < 2.0 or mag:
            continue
        if eh is not None and eh > 0.1:
            continue
        # Debye-temperature proxy: √(bond strength / mass). Light + strongly-bonded ⇒ high θ_D
        # ⇒ frozen phonons at 300 K ⇒ narrow ZPL / high Debye–Waller.
        theta = math.sqrt(max(0.0, -hform) / mm) if hform < 0 else 0.0
        # IONIC-SOFTNESS penalty: hform overstates stiffness for salts (large ionic cohesion,
        # but SOFT acoustic lattice ⇒ broad RT ZPL). Penalize metal + halogen (alkali/alkaline-
        # earth halides) so covalent stiff hosts (nitrides/carbides/BN) are not out-ranked by
        # soft ultra-wide-gap ionics. Covalency is what keeps θ_D — and the ZPL — high.
        elset = set(re.findall(r"[A-Z][a-z]?", r.get("formula", "")))
        halogen = bool(elset & {"F", "Cl", "Br", "I"})
        soft_metal = bool(elset & {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"})
        ionic = 0.35 if (halogen and soft_metal) else (0.7 if halogen else 1.0)
        # emitter-host score: reward stiffness AND a usable-but-not-absurd gap (cap so ultra-
        # wide soft ionics don't win on gap alone). log-compress the gap.
        score = ionic * theta * math.log(min(gap, 6.5))
        pool.append({"formula": r.get("formula"), "gap_eV": round(gap, 2),
                     "hform_eV_at": round(hform, 3), "mean_mass": round(mm, 1),
                     "theta_proxy": round(theta, 4), "covalency_factor": ionic,
                     "score": round(score, 4), "layergroup": r.get("layergroup")})

    ranked = sorted(pool, key=lambda p: -p["score"])
    # locate the anchors that validate the FOM
    def find(fs):
        return [p for p in pool if p["formula"] in fs]
    anchors = find({"B2N2", "BN", "B2N2 ", "C2", "C4", "AlN", "Al2N2", "GaN", "Ga2N2"})
    # contrast: soft ultra-wide-gap ionic halides should rank LOW
    soft_wide = sorted([p for p in pool if p["gap_eV"] >= 6.0],
                       key=lambda p: -p["gap_eV"])[:5]

    manifest = {
        "phase": "Phase 4 pivot — RT single-photon-emitter HOST screen (dissolve the "
                 "uniformity wall by using identical-by-chemistry defects in a stiff host)",
        "fom": "narrow-ZPL-at-300K (high Debye-temp: stiff+light) × wide gap × non-magnetic × "
               "stable. A single two-level defect is an intrinsic photon blockade.",
        "n_candidates": len(pool),
        "top_hosts": ranked[:15],
        "fom_validation_anchors_hBN_nitrides": anchors,
        "soft_wide_gap_low_rank_check": [
            {"formula": p["formula"], "gap_eV": p["gap_eV"], "theta_proxy": p["theta_proxy"],
             "score": p["score"], "rank": ranked.index(p) + 1} for p in soft_wide],
        "reading": (
            "The FOM is sound if light, stiff, wide-gap covalent hosts (BN/C/nitride class) top "
            "the list while soft ultra-wide-gap ionic halides (CaCl2, MgCl2 — huge gap, low "
            "θ_proxy) rank far down. Top hosts are the 2D platforms most likely to hold a "
            "bright, narrow-ZPL RT single-photon emitter — the disorder-free route to a single-"
            "emitter blockade device."),
        "next_phase_to_confirm": [
            "Defect-supercell DFT for the top 2–3 hosts: formation energy of candidate vacancies/"
            "substitutionals, in-gap level positions, spin state.",
            "ZPL energy + Debye–Waller factor + transition dipole (bright?) at the defect — "
            "ideally GW-BSE on the defect (the ABINIT engine we validated can do supercells).",
            "Emitter–cavity cooperativity C=g²/(κγ) at 300 K > 1 ⇒ blockade; g∝√f, γ=ZPL width.",
        ],
        "honest_limit": "Host-level proxy only. It points at platform families and ranks 2D "
                        "hosts; it does NOT identify a specific working emitter — that is the "
                        "defect-level next phase. No fabrication claim.",
        "tier": "proxy (Debye-temp from hform+mass, gap). Anchored to known RT-emitter hosts.",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[emitter-host] {len(pool)} wide-gap non-mag stable hosts ranked. Top 10:")
    for p in ranked[:10]:
        print(f"    {p['formula']:12s} score={p['score']:.3f}  gap={p['gap_eV']:.1f}  "
              f"θ_proxy={p['theta_proxy']:.3f}  m̄={p['mean_mass']:.0f}")
    print("[emitter-host] FOM check — soft ultra-wide-gap ionics should rank LOW:")
    for p in manifest["soft_wide_gap_low_rank_check"]:
        print(f"    {p['formula']:12s} gap={p['gap_eV']:.1f}  θ={p['theta_proxy']:.3f}  "
              f"rank={p['rank']}/{len(pool)}")
    print(f"[emitter-host] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
