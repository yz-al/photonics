"""
BerkeleyGW input generation for the GW-BSE (U / E_b) branch.

Chain:  pw.x (scf -> wfn/wfnq via pw2bgw) -> epsilon -> sigma (G0W0)
        -> kernel -> absorption (BSE).
This yields the quasiparticle gap, the exciton binding E_b (a cross-check on the
C2DB label), the exciton wavefunction/Bohr radius, and — from the BSE two-particle
kernel — the exciton–exciton interaction U (saturation / biexciton channel).

Decks are real but parameter placeholders (bands, cutoffs, k-grids, q0) MUST be
converged per material. Writing a deck is not a calculation; no U or E_b is
produced here.
"""
from __future__ import annotations

import os

DEFAULTS = dict(
    epsilon_cutoff=10.0,     # Ry
    number_bands=200,
    sigma_bands=200,
    nk=(12, 12, 1),
    q0=(0.0, 0.0, 0.001),    # small q0 for the head of epsilon (2D)
    bse_valence=4,
    bse_conduction=4,
)


def write_epsilon(outdir: str, **kw) -> str:
    d = {**DEFAULTS, **kw}
    os.makedirs(outdir, exist_ok=True)
    txt = f"""# epsilon.inp — dielectric matrix (2D truncation)
epsilon_cutoff {d['epsilon_cutoff']}
number_bands {d['number_bands']}
cell_slab_truncation
begin qpoints
  {d['q0'][0]} {d['q0'][1]} {d['q0'][2]} 1.0 1
end
"""
    p = os.path.join(outdir, "epsilon.inp")
    open(p, "w").write(txt)
    return p


def write_sigma(outdir: str, **kw) -> str:
    d = {**DEFAULTS, **kw}
    os.makedirs(outdir, exist_ok=True)
    txt = f"""# sigma.inp — G0W0 self-energy (2D truncation)
number_bands {d['sigma_bands']}
cell_slab_truncation
screening_semiconductor
band_index_min 1
band_index_max {d['sigma_bands']}
"""
    p = os.path.join(outdir, "sigma.inp")
    open(p, "w").write(txt)
    return p


def write_kernel(outdir: str, **kw) -> str:
    d = {**DEFAULTS, **kw}
    os.makedirs(outdir, exist_ok=True)
    txt = f"""# kernel.inp — BSE kernel (electron-hole interaction)
number_val_bands {d['bse_valence']}
number_cond_bands {d['bse_conduction']}
cell_slab_truncation
screening_semiconductor
"""
    p = os.path.join(outdir, "kernel.inp")
    open(p, "w").write(txt)
    return p


def write_absorption(outdir: str, **kw) -> str:
    d = {**DEFAULTS, **kw}
    os.makedirs(outdir, exist_ok=True)
    txt = f"""# absorption.inp — BSE (exciton binding, wavefunction; U from the kernel)
number_val_bands_coarse {d['bse_valence']}
number_cond_bands_coarse {d['bse_conduction']}
number_val_bands_fine {d['bse_valence']}
number_cond_bands_fine {d['bse_conduction']}
cell_slab_truncation
screening_semiconductor
use_velocity
diagonalization
write_eigenvectors 10          # exciton wavefunctions -> Bohr radius, and U kernel
"""
    p = os.path.join(outdir, "absorption.inp")
    open(p, "w").write(txt)
    return p


def write_pw2bgw(outdir: str, prefix: str = "mat", nk=None) -> str:
    """pw2bgw.x input to export QE wavefunctions to BerkeleyGW format."""
    os.makedirs(outdir, exist_ok=True)
    nk = nk or DEFAULTS["nk"]
    txt = f"""&input_pw2bgw
  prefix = '{prefix}'
  outdir = './out'
  real_or_complex = 2
  wfng_flag = .true.
  wfng_file = 'WFN'
  rhog_flag = .true.
  rhog_file = 'RHO'
  wfng_kgrid = .true.
  wfng_nk1 = {nk[0]}
  wfng_nk2 = {nk[1]}
  wfng_nk3 = {nk[2]}
/
"""
    p = os.path.join(outdir, "pw2bgw.in")
    open(p, "w").write(txt)
    return p


def write_bse_chain(outdir: str, prefix: str = "mat", **kw) -> dict:
    """Write the full BerkeleyGW deck set. Returns the file paths."""
    return {
        "pw2bgw": write_pw2bgw(outdir, prefix=prefix, nk=kw.get("nk")),
        "epsilon": write_epsilon(outdir, **kw),
        "sigma": write_sigma(outdir, **kw),
        "kernel": write_kernel(outdir, **kw),
        "absorption": write_absorption(outdir, **kw),
    }
