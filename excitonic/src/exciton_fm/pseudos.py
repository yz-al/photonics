"""
Pseudopotential staging.

Fetches PSlibrary (PAW, PBE) UPF files from the official Quantum ESPRESSO
pseudopotential mirror — free, no login — into a local pseudo directory, so a
real pw.x/ph.x run has the potentials it needs. The mirror is reachable from the
Modal container (open egress).

Recommended per-element cutoffs (ecutwfc/ecutrho, Ry) accompany each entry; a
real production run should still verify convergence per material.
"""
from __future__ import annotations

import os
import urllib.request

QE_MIRROR = "https://pseudopotentials.quantum-espresso.org/upf_files"

# Curated PSlibrary v1.0.0 PAW/PBE pseudopotentials (filenames verified on the
# mirror) with conservative cutoffs. Extend as the seed set grows.
PSEUDO_TABLE: dict[str, dict] = {
    "Ga": {"file": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF", "ecutwfc": 70, "ecutrho": 560},
    "As": {"file": "As.pbe-n-kjpaw_psl.1.0.0.UPF",  "ecutwfc": 70, "ecutrho": 560},
    "Al": {"file": "Al.pbe-nl-kjpaw_psl.1.0.0.UPF", "ecutwfc": 60, "ecutrho": 480},
    "N":  {"file": "N.pbe-n-kjpaw_psl.1.0.0.UPF",   "ecutwfc": 80, "ecutrho": 640},
    "Mo": {"file": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF", "ecutwfc": 70, "ecutrho": 560},
    "S":  {"file": "S.pbe-nl-kjpaw_psl.1.0.0.UPF",  "ecutwfc": 60, "ecutrho": 480},
    "Si": {"file": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",  "ecutwfc": 50, "ecutrho": 400},
    "Se": {"file": "Se.pbe-n-kjpaw_psl.1.0.0.UPF",  "ecutwfc": 60, "ecutrho": 480},
    "W":  {"file": "W.pbe-spn-kjpaw_psl.1.0.0.UPF", "ecutwfc": 70, "ecutrho": 560},
    "Te": {"file": "Te.pbe-n-kjpaw_psl.1.0.0.UPF",  "ecutwfc": 60, "ecutrho": 480},
}


def pseudo_filename(symbol: str) -> str:
    if symbol not in PSEUDO_TABLE:
        raise KeyError(f"no curated pseudopotential for {symbol!r}; add it to "
                       f"PSEUDO_TABLE (do not run without a real UPF)")
    return PSEUDO_TABLE[symbol]["file"]


def recommended_cutoffs(symbols) -> tuple[int, int]:
    """Max recommended (ecutwfc, ecutrho) over the given elements [Ry]."""
    ecw = max(PSEUDO_TABLE[s]["ecutwfc"] for s in symbols)
    ecr = max(PSEUDO_TABLE[s]["ecutrho"] for s in symbols)
    return ecw, ecr


def stage_pseudos(symbols, dest: str, mirror: str = QE_MIRROR) -> dict:
    """Download the UPFs for `symbols` into `dest`. Returns {symbol: local_path}.

    Skips a file already present. Raises on a missing curated entry — we never
    run DFT with a fabricated or absent potential.
    """
    os.makedirs(dest, exist_ok=True)
    out = {}
    for s in sorted(set(symbols)):
        fname = pseudo_filename(s)
        path = os.path.join(dest, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            url = f"{mirror}/{fname}"
            urllib.request.urlretrieve(url, path)
        out[s] = path
    return out
