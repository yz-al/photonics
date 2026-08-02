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


# Per-atom Born-charge tensor block:
#   atom    1   Ga
#      Ex  (  2.190   0.000   0.000 )
#      Ey  (  0.000   2.190   0.000 )
#      Ez  (  0.000   0.000   2.190 )
# The isotropic Z* is the trace/3 = (Ex_x + Ey_y + Ez_z)/3 (the DIAGONAL), not
# the mean of all components (which dilutes with the zero off-diagonals and the
# opposite-sign second atom).
_BORN_ATOM = re.compile(
    r"atom\s+\d+\s+\S+\s*\n"
    r"\s*Ex\s*\(\s*([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s*\)\s*\n"
    r"\s*Ey\s*\(\s*[-+]?\d+\.\d+\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+\s*\)\s*\n"
    r"\s*Ez\s*\(\s*[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+([-+]?\d+\.\d+)\s*\)",
    re.I)


def parse_born_charges(ph_out: str) -> Label:
    """Born effective charge |Z*_iso| = |trace/3| at the MOST polar site.

    Non-polar (Si) -> ~0; polar (GaAs) -> ~2.2. Reports the max over atoms so a
    polar bond is not averaged away against its counter-ion.
    """
    if not _finished(ph_out):
        return not_run("Z_born", "e", "ph.x run did not finish")
    zs = []
    for m in _BORN_ATOM.finditer(ph_out):
        a, e, i = float(m.group(1)), float(m.group(2)), float(m.group(3))
        zs.append(abs((a + e + i) / 3.0))
    if not zs:
        return not_run("Z_born", "e", "no per-atom Born-charge tensor parsed")
    z = max(zs)
    return Label("Z_born", round(z, 4), "e", "dfpt",
                 source="QE ph.x", notes=f"max |Z*_iso| over {len(zs)} atoms "
                        f"(diagonal trace/3)")


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
