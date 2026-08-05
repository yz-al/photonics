"""
Phase-2 seed-set selection.

Every material in the C2DB BSE set (370) carries an exciton-binding label but
NONE carry Γ(300 K) or U — so they are all legitimate seed candidates for the
GW-BSE + EPW label-generation runs. This module ranks them by how promising they
are for a room-temperature blockade material, so expensive first-principles jobs
go to the highest-value candidates first (and the active-learning loop later
replaces this static score with surrogate uncertainty).

Priority favors, per the project brief:
  - large exciton binding  E_b            (thermal stability at 300 K)
  - low expected Fröhlich coupling        (small lattice/IR response -> narrow Γ)
  - a useful optical gap window           (telecom-adjacent, not too wide)
  - enough oscillator strength            (to strong-couple and read out)
  - thermodynamic + dynamic stability     (ehull small)
  - family diversity across the brief's target families (TMD, halide perovskite,
    chalcogenide, oxide, organic, dipolar/interlayer)

Nothing here computes Γ or U — it only *prioritizes* what to compute.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .features import parse_formula

# Element sets for family tagging.
GROUP6 = {"Cr", "Mo", "W"}
GROUP4_5 = {"Ti", "Zr", "Hf", "V", "Nb", "Ta"}
CHALCOGEN = {"S", "Se", "Te"}
HALOGEN = {"F", "Cl", "Br", "I"}
PEROVSKITE_METALS = {"Pb", "Sn", "Ge", "Bi", "Sb", "Cu", "Ag"}


def tag_family(formula: str) -> str:
    comp = parse_formula(formula)
    els = set(comp)
    metals6 = els & GROUP6
    chal = els & CHALCOGEN
    hal = els & HALOGEN
    # group-6 transition-metal dichalcogenide (the MoS2/WSe2 anchor family)
    if metals6 and chal:
        # check ~1:2 metal:chalcogen
        m = sum(comp[e] for e in metals6)
        x = sum(comp[e] for e in chal)
        if m > 0 and 1.6 <= x / m <= 2.4:
            return "TMD_group6"
    if (els & GROUP4_5) and chal:
        return "TMD_other"
    if hal and (els & PEROVSKITE_METALS):
        return "halide_perovskite_like"
    if hal:
        return "halide"
    if chal:
        return "chalcogenide_2D"
    if "O" in els:
        return "oxide"
    if {"C", "H"} <= els or ("C" in els and len(els) <= 3):
        return "organic_like"
    return "other"


def _minmax(x, lo, hi):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return max(0.0, min(1.0, (x - lo) / (hi - lo))) if hi > lo else 0.0


@dataclass
class Weights:
    binding: float = 0.34       # large E_b
    low_frohlich: float = 0.24  # small lattice/IR response
    gap_window: float = 0.16    # useful optical gap
    oscillator: float = 0.12    # transition strength
    stability: float = 0.14     # small ehull


@dataclass
class SeedScorer:
    weights: Weights = field(default_factory=Weights)
    # gap window (eV): soft box; centered on the visible/near-IR, telecom-adjacent
    gap_lo: float = 0.7
    gap_hi: float = 2.6

    def gap_score(self, gap_gw, gap_hse, gap):
        g = next((v for v in (gap_gw, gap_hse, gap)
                  if v is not None and not (isinstance(v, float) and math.isnan(v))), None)
        if g is None:
            return 0.0, None
        if g <= 0:
            return 0.0, g
        mid = 0.5 * (self.gap_lo + self.gap_hi)
        half = 0.5 * (self.gap_hi - self.gap_lo)
        return max(0.0, 1.0 - abs(g - mid) / (half + 1e-9)), g

    def frohlich_proxy(self, alphax_lat, alphax_el):
        """Lower is better. Ratio of lattice(IR) to electronic polarizability is a
        rough proxy for LO/Fröhlich strength; small -> weak Fröhlich -> narrow Γ.
        Returns (score in [0,1], raw ratio or None)."""
        def ok(v):
            return v is not None and not (isinstance(v, float) and math.isnan(v))
        if not (ok(alphax_lat) and ok(alphax_el)) or alphax_el <= 0:
            return 0.5, None  # neutral when unknown
        ratio = alphax_lat / alphax_el
        # map ratio 0 -> score 1, ratio >=1 -> score 0
        return max(0.0, 1.0 - min(ratio, 1.0)), ratio

    def score_row(self, r: dict) -> dict:
        w = self.weights
        E_b = r.get("E_B")
        s_bind = _minmax(E_b, 0.1, 1.5) or 0.0
        s_fro, fro_ratio = self.frohlich_proxy(r.get("alphax_lat"), r.get("alphax_el"))
        s_gap, gap_used = self.gap_score(r.get("gap_gw"), r.get("gap_hse"), r.get("gap"))
        # oscillator strength: in-plane interband polarizability
        alx = r.get("alphax_el"); aly = r.get("alphay_el")
        f_in = None
        if all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in (alx, aly)):
            f_in = 0.5 * (alx + aly)
        s_osc = _minmax(f_in, 0.0, 8.0) or 0.0
        # stability: small ehull good
        eh = r.get("ehull")
        s_stab = 0.0
        if eh is not None and not (isinstance(eh, float) and math.isnan(eh)):
            s_stab = max(0.0, 1.0 - _minmax(eh, 0.0, 0.3))  # ehull 0 ->1, 0.3+ ->0
        total = (w.binding * s_bind + w.low_frohlich * s_fro + w.gap_window * s_gap
                 + w.oscillator * s_osc + w.stability * s_stab)
        reasons = []
        if E_b is not None and E_b >= 0.3:
            reasons.append(f"large E_b={E_b:.2f} eV")
        if fro_ratio is not None and fro_ratio < 0.3:
            reasons.append(f"low Fröhlich proxy (α_lat/α_el={fro_ratio:.2f})")
        if gap_used is not None and self.gap_lo <= gap_used <= self.gap_hi:
            reasons.append(f"gap {gap_used:.2f} eV in window")
        return {
            "score": round(total, 4),
            "subscores": {"binding": round(s_bind, 3), "low_frohlich": round(s_fro, 3),
                          "gap_window": round(s_gap, 3), "oscillator": round(s_osc, 3),
                          "stability": round(s_stab, 3)},
            "frohlich_proxy_ratio": (None if fro_ratio is None else round(fro_ratio, 3)),
            "gap_used_eV": (None if gap_used is None else round(gap_used, 3)),
            "reasons": reasons,
        }


def select_seed_set(rows: list[dict], n: int = 150,
                    per_family_cap: int | None = None) -> list[dict]:
    """Rank materials and return the top-n with priority metadata.

    rows: list of dicts with numeric C2DB features (already parsed to floats/None).
    per_family_cap: if set, keep at most this many per family (diversity).
    """
    scorer = SeedScorer()
    scored = []
    for r in rows:
        fam = tag_family(str(r.get("formula", "")))
        s = scorer.score_row(r)
        scored.append({"formula": r.get("formula"), "family": fam,
                       "E_b_eV": r.get("E_B"), **s})
    scored.sort(key=lambda d: d["score"], reverse=True)
    if per_family_cap:
        kept, counts = [], {}
        for d in scored:
            c = counts.get(d["family"], 0)
            if c < per_family_cap:
                kept.append(d); counts[d["family"]] = c + 1
        scored = kept
    return scored[:n]
