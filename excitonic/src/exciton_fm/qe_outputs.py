"""
Parsers for real Quantum ESPRESSO / DFPT output.

Extracts the genuine first-principles Fröhlich ingredients from a completed
`ph.x` run (with `epsil=.true.`): the electronic dielectric tensor ε∞, the Born
effective charges Z*, and the phonon frequencies (the highest polar mode ≈ ω_LO).
These are tier-`dfpt` (real computed numbers). If the output is missing or the
run did not finish, the parsers return `not_run` — never a guess.
"""
from __future__ import annotations

import re

from .provenance import Label, not_run

CM1_TO_meV = 0.1239842  # 1 cm^-1 in meV


def _finished(text: str) -> bool:
    return "JOB DONE" in text


def parse_epsilon_inf(ph_out: str) -> Label:
    """Electronic (clamped-ion) dielectric constant ε∞ = trace/3 from ph.x."""
    if not _finished(ph_out):
        return not_run("eps_inf", "1", "ph.x run did not finish (no JOB DONE)")
    m = re.search(r"Dielectric constant in cartesian axis\s*\n\s*\n(.*?)\n\s*\n",
                  ph_out, re.S)
    if not m:
        return not_run("eps_inf", "1", "no dielectric tensor block found")
    nums = re.findall(r"[-+]?\d+\.\d+", m.group(1))
    if len(nums) < 9:
        return not_run("eps_inf", "1", "dielectric tensor incomplete")
    diag = [float(nums[0]), float(nums[4]), float(nums[8])]
    eps = sum(diag) / 3.0
    return Label("eps_inf", round(eps, 4), "1", "dfpt",
                 source="QE ph.x (epsil=.true.)",
                 notes=f"trace/3 of ε∞; diagonal={['%.3f'%d for d in diag]}")


_NUM = re.compile(r"[-+]?\d+\.\d+")
_ATOM_LINE = re.compile(r"\batom\s+\d+", re.I)


def parse_born_charges(ph_out: str) -> Label:
    """Born effective charge |Z*_iso| = |trace/3| at the MOST polar site.

    Non-polar (Si) -> ~0; polar (GaAs) -> ~2.2. Robust to QE spacing: locate the
    'Effective charges' section, then for each 'atom N' block read the diagonal of
    the next three numeric rows (Z*_xx, Z*_yy, Z*_zz). Reports the max over atoms so
    a polar bond is not averaged away against its counter-ion.
    """
    if not _finished(ph_out):
        return not_run("Z_born", "e", "ph.x run did not finish")
    idx = ph_out.rfind("Effective charges")
    if idx < 0:
        return not_run("Z_born", "e", "no 'Effective charges' section")
    lines = ph_out[idx:idx + 6000].splitlines()
    zs = []
    i = 0
    while i < len(lines) - 3:
        if _ATOM_LINE.search(lines[i]):
            r1 = _NUM.findall(lines[i + 1])
            r2 = _NUM.findall(lines[i + 2])
            r3 = _NUM.findall(lines[i + 3])
            if len(r1) >= 3 and len(r2) >= 3 and len(r3) >= 3:
                diag = (float(r1[0]) + float(r2[1]) + float(r3[2])) / 3.0
                zs.append(abs(diag))
                i += 4
                continue
        i += 1
    if not zs:
        return not_run("Z_born", "e", "no per-atom Born-charge tensor parsed")
    return Label("Z_born", round(max(zs), 4), "e", "dfpt",
                 source="QE ph.x", notes=f"max |Z*_iso| (diag trace/3) over {len(zs)} atoms")


def parse_phonon_omega_LO(ph_out: str) -> Label:
    """Highest phonon frequency at Γ ≈ ω_LO [meV] (the polar LO mode).

    Note: exact LO–TO splitting needs the non-analytic term (dynmat.x); this
    takes the top ph.x frequency as ω_LO, adequate for the Fröhlich estimate.
    """
    if not _finished(ph_out):
        return not_run("omega_LO", "meV", "ph.x run did not finish")
    freqs = [float(x) for x in re.findall(r"freq\s*\(.*?\)\s*=.*?=\s*([-+]?\d+\.\d+)\s*\[cm-1\]",
                                          ph_out)]
    freqs = [f for f in freqs if f > 1.0]  # drop acoustic ~0 (and imaginary<0)
    if not freqs:
        return not_run("omega_LO", "meV", "no phonon frequencies parsed")
    w_cm = max(freqs)
    w_meV = w_cm * CM1_TO_meV
    return Label("omega_LO", round(w_meV, 3), "meV", "dfpt",
                 source="QE ph.x (top Γ mode)",
                 notes=f"ω_LO≈{w_cm:.1f} cm^-1; LO-TO exactness needs dynmat.x")


def parse_total_energy(scf_out: str) -> Label:
    """Total energy from a pw.x scf run [Ry] — a quick 'did it run' check."""
    if "JOB DONE" not in scf_out:
        return not_run("etot", "Ry", "pw.x scf did not finish")
    m = re.findall(r"!\s+total energy\s*=\s*([-+]?\d+\.\d+)\s*Ry", scf_out)
    if not m:
        return not_run("etot", "Ry", "no total energy line")
    return Label("etot", float(m[-1]), "Ry", "dfpt", source="QE pw.x scf")
