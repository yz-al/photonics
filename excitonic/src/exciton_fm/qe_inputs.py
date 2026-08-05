"""
Quantum ESPRESSO input generation for the electron-phonon (Γ) branch.

Chain:  pw.x (scf) -> ph.x (DFPT phonons + dvscf) -> epw.x (Wannier + e-ph).
These produce the LO-phonon spectrum and the exciton/carrier-phonon coupling
from which the 300 K Fröhlich linewidth Γ is obtained.

The decks are real and runnable in principle; the pseudopotential *library* and
convergence parameters are chosen conservatively and must be validated per
material before a production run. Generating a deck is NOT a calculation — no Γ
is produced here.
"""
from __future__ import annotations

import os

# A default plane-wave cutoff (Ry) and a coarse phonon q-grid for scaffolding.
DEFAULTS = dict(ecutwfc=60.0, ecutrho=480.0, kgrid=(12, 12, 1), qgrid=(6, 6, 1),
                pseudo_dir="./pseudo", pseudo_suffix=".upf")


def _pseudo_map(atoms, suffix: str) -> dict:
    return {s: f"{s}{suffix}" for s in sorted(set(atoms.get_chemical_symbols()))}


def write_scf(atoms, outdir: str, prefix: str = "mat", **kw) -> str:
    """Write a pw.x scf input for a 2D system via ASE. Returns the path."""
    from ase.io.espresso import write_espresso_in
    os.makedirs(outdir, exist_ok=True)
    d = {**DEFAULTS, **kw}
    pseudos = _pseudo_map(atoms, d["pseudo_suffix"])
    input_data = {
        "control": {"calculation": "scf", "prefix": prefix, "verbosity": "high",
                    "pseudo_dir": d["pseudo_dir"], "outdir": "./out", "tprnfor": True,
                    "tstress": True},
        "system": {"ibrav": 0, "ecutwfc": d["ecutwfc"], "ecutrho": d["ecutrho"],
                   "occupations": "smearing", "smearing": "cold", "degauss": 0.01,
                   "assume_isolated": "2D"},  # 2D Coulomb cutoff (Sohier et al.)
        "electrons": {"conv_thr": 1e-10, "mixing_beta": 0.7},
    }
    path = os.path.join(outdir, "scf.in")
    with open(path, "w") as fh:
        write_espresso_in(fh, atoms, input_data=input_data, pseudopotentials=pseudos,
                          kpts=d["kgrid"])
    return path


def write_ph(outdir: str, prefix: str = "mat", qgrid=None, fildvscf="dvscf") -> str:
    """Write a ph.x (DFPT) input on a q-grid (produces dvscf for EPW)."""
    os.makedirs(outdir, exist_ok=True)
    q = qgrid or DEFAULTS["qgrid"]
    txt = f"""Phonons on a {q[0]}x{q[1]}x{q[2]} grid for {prefix}
&inputph
  prefix    = '{prefix}'
  outdir    = './out'
  fildyn    = '{prefix}.dyn'
  fildvscf  = '{fildvscf}'
  ldisp     = .true.
  epsil     = .true.        ! dielectric tensor + Born charges (Fröhlich!)
  trans     = .true.
  nq1 = {q[0]}
  nq2 = {q[1]}
  nq3 = {q[2]}
  tr2_ph    = 1.0d-16
/
"""
    path = os.path.join(outdir, "ph.in")
    with open(path, "w") as fh:
        fh.write(txt)
    return path


def write_epw(outdir: str, atoms, prefix: str = "mat", kgrid=None, qgrid=None,
              temps=(300.0,)) -> str:
    """Write an epw.in for the electron-phonon / Fröhlich linewidth evaluation.

    lpolar=.true. + the 2D Fröhlich treatment (Sohier/Verdi/Giustino) is the key:
    it captures the polar LO coupling that sets Γ(300 K). Fine interpolation grids
    are placeholders and must be converged per material.
    """
    os.makedirs(outdir, exist_ok=True)
    k = kgrid or DEFAULTS["kgrid"]
    q = qgrid or DEFAULTS["qgrid"]
    projections = "\n".join(f"  proj({i+1}) = '{s}'"
                            for i, s in enumerate(sorted(set(atoms.get_chemical_symbols()))))
    temp_list = " ".join(f"{t:.1f}" for t in temps)
    txt = f"""EPW: electron-phonon + Fröhlich linewidth for {prefix}
&inputepw
  prefix      = '{prefix}'
  outdir      = './out'
  elph        = .true.
  epbwrite    = .true.
  epwwrite    = .true.
  lpolar      = .true.          ! polar (Fröhlich) LO coupling — essential
  vme         = 'wannier'
  nbndsub     =  8
{projections}
  wannierize  = .true.
  num_iter    = 500
  dis_win_max = 20.0
  ! coarse grids must match the ph.x q-grid and the nscf k-grid
  nk1 = {k[0]}
  nk2 = {k[1]}
  nk3 = {k[2]}
  nq1 = {q[0]}
  nq2 = {q[1]}
  nq3 = {q[2]}
  ! fine interpolation grids (placeholder; converge before production)
  nkf1 = 48
  nkf2 = 48
  nkf3 = 1
  nqf1 = 48
  nqf2 = 48
  nqf3 = 1
  ! finite-T carrier linewidth from the imaginary e-ph self-energy
  degaussw    = 0.01
  temps       = {temp_list}
  efermi_read = .false.
/
"""
    path = os.path.join(outdir, "epw.in")
    with open(path, "w") as fh:
        fh.write(txt)
    return path


def write_eph_chain(atoms, outdir: str, prefix: str = "mat", **kw) -> dict:
    """Write the full scf -> ph -> epw deck set. Returns the file paths."""
    return {
        "scf": write_scf(atoms, outdir, prefix=prefix, **kw),
        "ph": write_ph(outdir, prefix=prefix, qgrid=kw.get("qgrid")),
        "epw": write_epw(outdir, atoms, prefix=prefix,
                         kgrid=kw.get("kgrid"), qgrid=kw.get("qgrid")),
    }
