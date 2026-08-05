"""
The dipolar leg, closed (or not) by an oscillator-strength SUM RULE — no BSE.

This is the analytic route the BSE was standing in for. The dipolar interlayer
exciton faces a Kramers-Kronig / sum-rule trade-off: the SAME interlayer
hybridization that brightens the transition (imaginary part of χ → oscillator
strength f) shrinks the charge-separation dipole (real/static part of χ →
interaction U). You cannot independently maximize both. A discrete two-site
model makes this exact.

TWO-SITE MODEL (leading order, the microscopic content of the KK linkage).
An interlayer exciton has the hole localized in layer B and the electron in a
hybridized state |ψ_e⟩ = √(1-x)|A⟩ + √x|B⟩, where x∈[0,1] is the electron weight
in the hole's layer (set by interlayer tunneling t and band offset Δ: x ≈ (t/Δ)²
for weak coupling). Then:

  * Oscillator strength (bright recombination needs electron amplitude in the
    hole layer B):            f(x) = x · f0        [f0 = intralayer reference]
  * Charge-separation dipole (electron centroid vs hole):
        d_eff(x) = (1-x)·d0   ⇒   U(x) = (1-x)² · U_max
    (U ∝ dipole², U_max = interaction of the fully-separated x→0 exciton).

So brightness (f∝x) is "borrowed" from the intralayer transition — an
oscillator-strength sum-rule budget — at the direct cost of the dipole (U∝(1-x)²).
This is the anti-correlation the whole dipolar leg turns on, now in ONE parameter.

TWO NECESSARY CONDITIONS for a room-temperature dipolar blockade:
  (I)  Strong coupling — you must resolve the polariton to use it:
           Ω(f) = Ω0·√(f/f0) = Ω0·√x  >  Γ(300K)     ⇒   x > (Γ/Ω0)²
       (Ω0 = intralayer RT Rabi splitting, ~30–50 meV, demonstrated.)
  (II) Blockade — the interaction must beat the linewidth (project convention):
           U(x) = (1-x)²·U_max  >  threshold·Γ        ⇒   x < 1 − √(threshold·Γ/U_max)

A usable exciton exists iff the window (x_lo, x_hi) is non-empty:

        (Γ/Ω0)²  <  1 − √(threshold·Γ/U_max)          ... (★)  the master inequality

Corollary (best case x→0): even the fully-dark, maximum-dipole exciton needs
U_max > threshold·Γ; if that fails, the leg is CLOSED at every x with NO strong-
coupling analysis required.

TIERS. The two-site trade-off f∝x, U∝(1-x)² is a model result (leading order,
exact for the discrete model). Ω0, Γ, U_max are measured/literature. So the
verdict is 'model + measured' tier — a genuine semi-analytic bound, not a GW-BSE
number, and it is reported as such.
"""
from __future__ import annotations

import math

BLOCKADE_THRESHOLD = 0.71


def f_of_x(x: float, f0: float = 1.0) -> float:
    """Oscillator strength f(x) = x·f0 (brightness borrowed via layer-B weight x)."""
    return x * f0


def U_of_x(x: float, U_max: float) -> float:
    """Dipolar interaction U(x) = (1-x)²·U_max (dipole ∝ (1-x); U ∝ dipole²)."""
    return (1.0 - x) ** 2 * U_max


def feasible_window(gamma_meV: float, U_max_meV: float, omega0_meV: float,
                    threshold: float = BLOCKADE_THRESHOLD, sc_margin: float = 1.0) -> dict:
    """The (x_lo, x_hi) brightness window where BOTH conditions hold, and the verdict.

    x_lo (strong coupling) = (sc_margin·Γ/Ω0)² ;  x_hi (blockade) = 1 − √(threshold·Γ/U_max).
    Open iff x_lo < x_hi and x_hi > 0.  sc_margin=1 is the bare onset Ω>Γ; sc_margin=2 is
    a resolved doublet Ω>2Γ (device-realistic) — a stricter, honest criterion.
    """
    if min(gamma_meV, U_max_meV, omega0_meV) <= 0:
        return {"verdict": "invalid", "open": False}
    x_lo = (sc_margin * gamma_meV / omega0_meV) ** 2         # strong-coupling floor
    r = threshold * gamma_meV / U_max_meV
    blockade_possible = r <= 1.0                              # U_max > threshold·Γ ?
    x_hi = (1.0 - math.sqrt(r)) if blockade_possible else float("-inf")
    open_ = blockade_possible and (x_lo < x_hi) and (x_hi > 0.0)
    width = max(0.0, x_hi - x_lo) if blockade_possible else 0.0
    if not blockade_possible:
        why = ("CLOSED at every x: even the maximum-dipole (x→0) exciton has "
               f"U_max={U_max_meV:g} < threshold·Γ={threshold*gamma_meV:.1f} meV.")
    elif open_:
        why = (f"OPEN in a window x∈({x_lo:.3f}, {x_hi:.3f}) (width {width:.3f}): a "
               "brightness fraction exists that is both bright enough to strong-couple "
               "and separated enough to blockade.")
    else:
        why = (f"CLOSED: strong-coupling floor x_lo={x_lo:.3f} ≥ blockade ceiling "
               f"x_hi={x_hi:.3f} — brightening past the SC threshold already kills the "
               "dipole below blockade.")
    return {"verdict": "open" if open_ else "closed", "open": open_,
            "x_lo_strong_coupling": round(x_lo, 4),
            "x_hi_blockade": (None if not blockade_possible else round(x_hi, 4)),
            "window_width": round(width, 4),
            "U_at_best_meV": round(U_of_x(max(0.0, x_lo), U_max_meV), 3),
            "f_at_best_over_f0": round(f_of_x(x_lo), 4),
            "reason": why}


def scan(gamma_range, U_max_range, omega0_range, threshold: float = BLOCKADE_THRESHOLD,
         n: int = 21, sc_margin: float = 1.0) -> dict:
    """Grid-scan the literature-plausible parameter box; report the fraction OPEN.

    Returns the open fraction, the verdict at the nominal midpoint, and the
    optimistic/pessimistic corners — so the knife-edge character is explicit.
    """
    def lin(a, b):
        return [a + (b - a) * i / (n - 1) for i in range(n)]

    gs, us, os = lin(*gamma_range), lin(*U_max_range), lin(*omega0_range)
    total = opened = 0
    for g in gs:
        for u in us:
            for o in os:
                total += 1
                if feasible_window(g, u, o, threshold, sc_margin)["open"]:
                    opened += 1
    mid = feasible_window((gamma_range[0] + gamma_range[1]) / 2,
                          (U_max_range[0] + U_max_range[1]) / 2,
                          (omega0_range[0] + omega0_range[1]) / 2, threshold, sc_margin)
    # optimistic = low Γ, high U_max, high Ω0 ; pessimistic = the opposite corner
    opt = feasible_window(gamma_range[0], U_max_range[1], omega0_range[1], threshold, sc_margin)
    pes = feasible_window(gamma_range[1], U_max_range[0], omega0_range[0], threshold, sc_margin)
    return {"sc_margin": sc_margin, "open_fraction": round(opened / total, 3),
            "n_grid": total, "nominal_midpoint": mid,
            "optimistic_corner": opt, "pessimistic_corner": pes}
