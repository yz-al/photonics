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


def parse_born_charges(ph_out: str) -> Label:
    """Representative Born effective charge |Z*| (mean over atoms, trace/3)."""
    if not _finished(ph_out):
        return not_run("Z_born", "e", "ph.x run did not finish")
    blocks = re.findall(r"E\*x\s*\(.*?\)|Z\*?\(\s*\d+\)|atom\s+\d+.*?Z\*", ph_out)
    # Parse the standard 'Effective charges (d Force / dE)' section.
    sec = re.search(r"Effective charges .*?axis.*?\n(.*?)(?:\n\s*\n|Electric)",
                    ph_out, re.S)
    if not sec:
        return not_run("Z_born", "e", "no Born-charge block found")
    zdiag = re.findall(r"Ez?x?\s*\(\s*[123]\s*\)\s*=?\s*([-+]?\d+\.\d+)", sec.group(1))
    vals = [abs(float(x)) for x in re.findall(r"[-+]?\d+\.\d+", sec.group(1))]
    if not vals:
        return not_run("Z_born", "e", "Born-charge values not parsed")
    z = sum(vals) / len(vals)
    return Label("Z_born", round(z, 4), "e", "dfpt",
                 source="QE ph.x", notes="mean |Z*| component over the block")


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
