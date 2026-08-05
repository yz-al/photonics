"""
The dipolar-branch window: does a usable separation exist?

The interlayer (dipolar) exciton trades one quantity for another as the layer
separation d grows:

    U(d)  RISES   — the interaction comes from a static dipole ∝ d (not from
                    wavefunction overlap), so it grows with separation.
    f(d)  FALLS   — separating electron and hole shrinks the transition-dipole
                    overlap, so the oscillator strength decays, ~exp(-d/λ_f).

f is not optional: it sets the light-matter coupling g ∝ √f, hence whether you
can strong-couple at all (Ω = √N·g > Γ) and whether U survives the Hopfield
weighting (usable FOM ~ |X|²·U_exc/Γ_x). So the branch lives or dies on ONE
comparison — how fast f falls versus how fast U rises. This module fits both
slopes from a few (d, U, f) points and returns the window verdict.

RIGOR. U and f MUST come from GW-BSE at ≥2 separations (the flagship deliverable).
With one point there is no slope and the verdict is `undetermined` — the geometric
d²/ε prior used to RANK candidates assumes this tradeoff is favorable WITHOUT
checking it; this module is what checks it.
"""
from __future__ import annotations

import math


def fit_f_decay(ds, fs):
    """Exponential-overlap fit f(d) = f0·exp(-d/λ_f). Returns (f0, λ_f, per-point).

    λ_f is the oscillator-strength decay length [same unit as d]; small λ_f = f
    collapses fast = the branch-killing regime. Uses a log-linear least squares.
    """
    pts = [(d, f) for d, f in zip(ds, fs) if f is not None and f > 0 and d is not None]
    if len(pts) < 2:
        return None, None, "need ≥2 positive f points for a decay slope"
    xs = [d for d, _ in pts]
    ys = [math.log(f) for _, f in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None, None, "degenerate d spacing"
    slope = sxy / sxx                     # d(ln f)/dd  (negative if f falls)
    ln_f0 = my - slope * mx
    lam = (-1.0 / slope) if slope < 0 else float("inf")  # decay length (inf if f rises)
    return math.exp(ln_f0), lam, "ok"


def fit_U_rise(ds, Us):
    """Power-law fit U(d) = U0·d^p. Returns (U0, exponent p, note).

    p>0 means U rises with separation (the dipolar gain). Log-log least squares.
    """
    pts = [(d, U) for d, U in zip(ds, Us) if U is not None and U > 0 and d and d > 0]
    if len(pts) < 2:
        return None, None, "need ≥2 positive U points for a rise slope"
    xs = [math.log(d) for d, _ in pts]
    ys = [math.log(U) for _, U in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None, None, "degenerate d spacing"
    p = sxy / sxx
    U0 = math.exp(my - p * mx)
    return U0, p, "ok"


def window_verdict(ds, Us, fs, *, alpha=0.5, tier="gw_bse"):
    """Does a usable dipolar window exist across the measured separations?

    The usable per-polariton nonlinearity near the strong-coupling threshold scales
    like U·f^α (g ∝ √f ⇒ α=0.5 is the default coupling weight; α=0 ignores f, α=1
    weights it as the full coupling). We report the NET figure N(d)=U·f^α and its
    trend:

      * N rises with d  → the U gain outruns the f loss → **window OPEN** (larger
        separation is favorable; the dipolar branch is real).
      * N falls with d  → the f loss dominates every U gain → **window CLOSED**
        (the branch shuts for the same structural reason saturation-U did).

    Also returns the raw slopes (U exponent p, f decay length λ_f) so the verdict is
    auditable, and d* = the separation maximizing N over the measured points.
    """
    pts = [(d, U, f) for d, U, f in zip(ds, Us, fs)
           if None not in (d, U, f) and U > 0 and f > 0]
    if len(pts) < 2:
        return {"verdict": "undetermined", "provenance": tier,
                "reason": f"need ≥2 (d,U,f) points; have {len(pts)}. "
                          "One point cannot give a slope — this is exactly the gap "
                          "the geometric prior leaves open.",
                "n_points": len(pts)}
    pts.sort(key=lambda t: t[0])
    U0, p, uN = fit_U_rise([d for d, _, _ in pts], [U for _, U, _ in pts])
    f0, lam, fN = fit_f_decay([d for d, _, _ in pts], [f for _, _, f in pts])
    net = [(d, U * (f ** alpha)) for d, U, f in pts]
    d_star, net_star = max(net, key=lambda t: t[1])
    net_lo, net_hi = net[0][1], net[-1][1]      # smallest-d vs largest-d net
    net_rises = net_hi > net_lo
    # net exponent: d(ln N)/d(ln d) between the extreme points
    d1, N1 = net[0]; d2, N2 = net[-1]
    net_loglog = (math.log(N2 / N1) / math.log(d2 / d1)) if (d2 > 0 and d1 > 0 and N1 > 0) else None
    return {
        "verdict": "window_open" if net_rises else "window_closed",
        "provenance": tier,
        "alpha_coupling_weight": alpha,
        "U_rise_exponent_p": (None if p is None else round(p, 3)),
        "f_decay_length_lambda": (None if lam is None else round(lam, 3)),
        "net_figure_U_times_f_alpha": [[round(d, 3), round(N, 6)] for d, N in net],
        "net_loglog_slope": (None if net_loglog is None else round(net_loglog, 3)),
        "d_star": round(d_star, 3),
        "interpretation": (
            "U gain outruns f loss → dipolar window OPEN (larger d favorable)"
            if net_rises else
            "f loss dominates U gain → dipolar window CLOSED (branch shuts)"),
        "caveat": ("Slopes from GW-BSE at multiple separations."
                   if tier == "gw_bse" else
                   f"Inputs are tier '{tier}', NOT GW-BSE — illustrative, not a verdict."),
        "n_points": len(pts),
    }
