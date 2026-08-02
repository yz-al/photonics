"""
Provenance tiers — the backbone of the project's rigor rule.

Every numerical property this project emits carries a tier saying *how* it was
obtained, so a model estimate is never confused with a GW-BSE or EPW result.
The two decision-critical Phase-2 targets (Γ, U) have NO real labels yet; until
a GW-BSE/EPW job actually runs, their values are either `None` (not run) or an
explicitly `*_model` estimate — never a fabricated first-principles number.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# Ordered worst -> best trust for a *first-principles* label.
TIERS = (
    "not_run",          # no calculation performed; value is None
    "frohlich_model",   # analytic Fröhlich/LO model estimate for Γ (NOT EPW)
    "saturation_model", # analytic phase-space-filling / dipolar estimate for U (NOT BSE)
    "dft_proxy",        # PBE/RPA-level proxy
    "hybrid",           # HSE06
    "gw",               # G0W0 electronic structure
    "gw_bse",           # GW-BSE (exciton binding, biexciton/saturation U)
    "epw",              # DFPT + EPW electron-phonon (Γ(300 K))
    "measured",         # experiment
)

# Which tier *counts* as a real first-principles label for each target.
FIRST_PRINCIPLES = {"gw", "gw_bse", "epw", "measured"}
MODEL_ESTIMATE = {"frohlich_model", "saturation_model", "dft_proxy", "hybrid"}


@dataclass
class Label:
    """A single property value with full provenance.

    value is None iff tier == 'not_run'. `computed` is True only for a real
    first-principles calculation that actually ran to completion; a model
    estimate has computed=False by construction.
    """
    name: str
    value: float | None
    unit: str
    tier: str
    source: str = ""              # code + version, or model reference
    uncertainty: float | None = None
    converged: bool | None = None
    notes: str = ""

    def __post_init__(self):
        if self.tier not in TIERS:
            raise ValueError(f"unknown provenance tier: {self.tier!r}")
        if self.tier == "not_run" and self.value is not None:
            raise ValueError("tier 'not_run' must have value=None (no fabrication)")

    @property
    def computed(self) -> bool:
        """True only for a real first-principles label that ran to convergence."""
        return self.tier in FIRST_PRINCIPLES and self.value is not None

    @property
    def is_model_estimate(self) -> bool:
        return self.tier in MODEL_ESTIMATE and self.value is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["computed"] = self.computed
        d["is_model_estimate"] = self.is_model_estimate
        return d


def not_run(name: str, unit: str, reason: str = "") -> Label:
    """Construct an honest 'no calculation performed' placeholder."""
    return Label(name=name, value=None, unit=unit, tier="not_run", notes=reason)
