"""
Structures for the Phase-2 first-principles runs.

Production runs use the *relaxed* C2DB structures (obtained from the downloadable
C2DB ASE database; the web front-end does not expose a clean per-material
structure endpoint, so that download is a documented manual/CI step). For
scaffolding and for the validation anchors we build 2D prototypes with ASE so
the whole input-generation + Modal pipeline is runnable and testable end to end
without any gated download.

All structures are returned as ASE Atoms with a vacuum gap along z (2D).
"""
from __future__ import annotations

import numpy as np

DEFAULT_VACUUM = 15.0  # Å of vacuum for a 2D slab


def _add_vacuum(atoms, vacuum: float = DEFAULT_VACUUM):
    atoms.center(vacuum=vacuum, axis=2)
    atoms.pbc = (True, True, True)
    return atoms


def tmd_monolayer(formula_metal: str, chalcogen: str, phase: str = "2H",
                  a: float | None = None, thickness: float | None = None,
                  vacuum: float = DEFAULT_VACUUM):
    """Build an MX2 TMD monolayer (e.g. MoS2) via ASE's mx2 builder.

    phase: '2H' (trigonal-prismatic, semiconducting) or '1T' (octahedral).
    a, thickness default to sensible MoS2-like values when not given.
    """
    from ase.build import mx2
    a = a or 3.16
    thickness = thickness or 3.19
    kind = "2H" if phase.upper() in ("2H", "H") else "1T"
    atoms = mx2(formula=f"{formula_metal}{chalcogen}2", kind=kind, a=a,
                thickness=thickness, size=(1, 1, 1), vacuum=vacuum / 2)
    atoms.pbc = (True, True, True)
    return atoms


def prototype_from_formula(formula: str, vacuum: float = DEFAULT_VACUUM):
    """Best-effort 2D prototype for a binary MX2/MX formula, for scaffolding only.

    Recognizes an MX2 dichalcogenide -> TMD 2H prototype. Anything else raises,
    so callers must supply a real (C2DB-relaxed) structure for those — we never
    silently fabricate a geometry and pass it off as the real material.
    """
    from .features import parse_formula
    comp = parse_formula(formula)
    syms = list(comp.keys())
    if len(syms) == 2:
        (a_sym, a_n), (b_sym, b_n) = sorted(comp.items(), key=lambda kv: kv[1])
        if abs(b_n / a_n - 2.0) < 0.05:
            return tmd_monolayer(a_sym, b_sym, phase="2H", vacuum=vacuum)
    raise NotImplementedError(
        f"no built-in 2D prototype for {formula!r}; supply the C2DB-relaxed "
        f"structure for production runs (do not fabricate a geometry)")


def anchor_structures(vacuum: float = DEFAULT_VACUUM) -> dict:
    """The TMD validation anchors as ASE Atoms (2H monolayers)."""
    specs = {
        "MoS2": ("Mo", "S", 3.16, 3.19), "MoSe2": ("Mo", "Se", 3.29, 3.34),
        "WS2": ("W", "S", 3.15, 3.14), "WSe2": ("W", "Se", 3.28, 3.36),
    }
    out = {}
    for name, (m, x, a, th) in specs.items():
        out[name] = tmd_monolayer(m, x, phase="2H", a=a, thickness=th, vacuum=vacuum)
    return out


def structure_summary(atoms) -> dict:
    cell = np.array(atoms.cell)
    return {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "cell_a_A": float(np.linalg.norm(cell[0])),
        "cell_b_A": float(np.linalg.norm(cell[1])),
        "cell_c_A": float(np.linalg.norm(cell[2])),
        "pbc": [bool(x) for x in atoms.pbc],
    }
