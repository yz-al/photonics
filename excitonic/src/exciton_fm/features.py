"""
Composition featurization for the Phase-1 baseline.

Turns a chemical formula into a fixed-length vector of composition-weighted
elemental-property statistics (mean / std / min / max / range), plus a couple of
whole-composition descriptors. Element properties come from `mendeleev`
(a vetted periodic-table source) with an in-process cache; formula parsing is a
small regex so no structure file is needed.

This is the "coGN/ALIGNN-style structure->property" baseline at the composition
level — appropriate for n~370. Structure-graph models (ALIGNN/MACE) are the
Phase-1b upgrade and need atomic structures + a Modal GPU.
"""
from __future__ import annotations

import functools
import re
from typing import Iterable

import numpy as np

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

# Elemental scalar properties used for the composition statistics.
ELEMENT_PROPS = (
    "en_pauling",             # electronegativity
    "atomic_weight",
    "group_id",               # column
    "period",                 # row
    "covalent_radius",        # pm
    "nvalence",               # valence electrons
    "dipole_polarizability",  # atomic polarizability (bohr^3) — optical relevance
    "electron_affinity",
)

_STATS = ("mean", "std", "min", "max", "range")


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a simple formula like 'Tl2Br2' or 'C2F2' -> {symbol: count}."""
    out: dict[str, float] = {}
    for sym, num in _FORMULA_TOKEN.findall(formula.strip()):
        if not sym:
            continue
        n = float(num) if num not in ("", ".") else 1.0
        out[sym] = out.get(sym, 0.0) + n
    return out


@functools.lru_cache(maxsize=256)
def _element_vector(symbol: str) -> tuple[float, ...]:
    """Return the ELEMENT_PROPS vector for one element (cached)."""
    from mendeleev import element  # local import keeps module import cheap
    try:
        e = element(symbol)
    except Exception:
        return tuple(np.nan for _ in ELEMENT_PROPS)
    vals = []
    for p in ELEMENT_PROPS:
        v = getattr(e, p, None)
        try:
            v = float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            v = np.nan
        vals.append(v)
    return tuple(vals)


def feature_names() -> list[str]:
    names = [f"{p}_{s}" for p in ELEMENT_PROPS for s in _STATS]
    names += ["n_elements", "n_atoms_reduced", "mean_stoich_frac_max"]
    return names


def featurize_formula(formula: str) -> np.ndarray:
    """Composition -> fixed-length feature vector (see feature_names())."""
    comp = parse_formula(formula)
    n_names = len(feature_names())
    if not comp:
        return np.full(n_names, np.nan)

    syms = list(comp.keys())
    counts = np.array([comp[s] for s in syms], dtype=float)
    weights = counts / counts.sum()
    mat = np.array([_element_vector(s) for s in syms], dtype=float)  # (n_el, n_prop)

    feats: list[float] = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        w = weights.copy()
        good = ~np.isnan(col)
        if good.any():
            col_g = col[good]
            w_g = w[good]
            w_g = w_g / w_g.sum() if w_g.sum() > 0 else w_g
            mean = float(np.sum(col_g * w_g))
            std = float(np.sqrt(np.sum(w_g * (col_g - mean) ** 2)))
            mn, mx = float(col_g.min()), float(col_g.max())
            feats += [mean, std, mn, mx, mx - mn]
        else:
            feats += [np.nan] * 5

    feats += [
        float(len(syms)),                       # number of distinct elements
        float(counts.sum()),                    # atoms in the (reduced) cell
        float(weights.max()),                   # dominance of the top element
    ]
    return np.array(feats, dtype=float)


def featurize_many(formulas: Iterable[str]) -> np.ndarray:
    return np.vstack([featurize_formula(f) for f in formulas])
