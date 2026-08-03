"""
Phase 2 label generation on Modal — GW-BSE (BerkeleyGW) + DFPT/EPW (Quantum
ESPRESSO). These are CPU/MPI-bound, so they run as many-core Modal CPU jobs
(NOT GPU), launched from a GitHub Action (`modal run`).

Functions
---------
- smoke_test()      : verify the QE + BerkeleyGW binaries installed and report
                      versions. Proves the container builds. No science claimed.
- label_material()  : orchestrate scf -> ph -> epw (Γ) and scf -> pw2bgw ->
                      epsilon -> sigma -> kernel -> absorption (E_b, U) for one
                      material, then parse. Returns provenance-tagged labels;
                      any step that did not actually complete yields a 'not_run'
                      label — never a fabricated value.

Rigor: this module WILL run first-principles codes when given a structure + a
pseudopotential library + converged parameters. Until a run completes, it emits
no Γ/U number. The analytic model estimates live in the exciton_fm.frohlich /
exciton_u modules and are tiered 'frohlich_model' / 'saturation_model'.

    modal run excitonic/modal_phase2.py::smoke
    modal run excitonic/modal_phase2.py::generate            # (structure-gated)
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

# Many-core CPU image. Quantum ESPRESSO (pw.x, ph.x, epw.x, pw2bgw.x) comes from
# conda-forge and covers the whole electron-phonon / Γ branch plus the QE→BGW
# export. BerkeleyGW is NOT packaged on conda-forge, so the GW-BSE branch needs a
# source build — added as an optional layer (heavy: MPI+ScaLAPACK+FFTW+HDF5).
# GW-BSE/EPW are MPI/CPU-bound, hence a CPU image, not GPU.
# BerkeleyGW source build (GW-BSE branch). Its source is distribution-gated
# (registration on berkeleygw.org; GitLab needs auth), so it cannot be fetched
# unattended. Provide a reachable tarball URL via the BGW_TARBALL_URL env var to
# compile it into the image; otherwise the QE/EPW Γ branch is fully functional
# and the GW-BSE branch is a documented, opt-in add-on.
BGW_TARBALL_URL = os.environ.get("BGW_TARBALL_URL", "").strip()

# conda-forge `qe` is built ONLY against OpenMPI (there is no nompi/mpich variant).
# OpenMPI's launcher wants ssh/rsh even for a single-node run and fails in this
# container. Standard fix: a "fake ssh" shim — mpirun invokes `<agent> <host>
# orted…`; the shim drops the hostname and execs the rest LOCALLY, so orted runs
# on localhost with no real ssh/daemon. Point OMPI/PRTE at it via env in the run.
_FAKE_SSH = (
    "printf '#!/bin/sh\\nshift\\nexec \"$@\"\\n' > /usr/local/bin/fake_ssh "
    "&& chmod +x /usr/local/bin/fake_ssh"
)
# GW-BSE engine: Yambo (conda-forge, free) replaces the distribution-gated
# BerkeleyGW. qe -> p2y -> yambo gives G0W0 + BSE (E_b, oscillator strengths, and
# the exciton CT character the interlayer test needs).
qe_bgw_image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install(
        "qe", "yambo", "openmpi", "fftw", "hdf5", "numpy", "ase",
        channels=["conda-forge"],
    )
    .pip_install("requests==2.33.1")
    .run_commands(_FAKE_SSH)
)
if BGW_TARBALL_URL:  # pragma: no cover - opt-in heavy build, needs a source URL
    qe_bgw_image = qe_bgw_image.run_commands(
        f"curl -L -o /opt/bgw.tar.gz '{BGW_TARBALL_URL}'",
        "mkdir -p /opt/bgw && tar xzf /opt/bgw.tar.gz -C /opt/bgw --strip-components=1",
        # Generic arch.mk against the conda MPI/ScaLAPACK/FFTW/HDF5 stack.
        "cp /root/excitonic/build/berkeleygw_arch.mk /opt/bgw/arch.mk || true",
        "cd /opt/bgw && make -j 8 all || (echo 'BGW build failed; see log' && false)",
        "cp /opt/bgw/bin/*.x /opt/conda/bin/ 2>/dev/null || true",
    )
qe_bgw_image = qe_bgw_image.add_local_dir(HERE, remote_path="/root/excitonic", copy=True)

app = modal.App("exciton-fm-phase2")

N_CORES = 16  # many-core CPU; GW-BSE + EPW are MPI-parallel


@app.function(image=qe_bgw_image, cpu=N_CORES, timeout=1800)
def smoke_test() -> dict:
    """Confirm the first-principles binaries are present and runnable."""
    import shutil
    import subprocess

    def probe(binary: str, version_flag: str = "") -> dict:
        path = shutil.which(binary)
        info = {"binary": binary, "found": bool(path), "path": path or ""}
        if path and version_flag:
            try:
                out = subprocess.run([path, version_flag], capture_output=True,
                                     text=True, timeout=60)
                info["banner"] = (out.stdout or out.stderr).splitlines()[:3]
            except Exception as e:  # pragma: no cover
                info["banner_error"] = str(e)
        return info

    binaries = {
        "pw.x": probe("pw.x"),
        "ph.x": probe("ph.x"),
        "epw.x": probe("epw.x"),
        "pw2bgw.x": probe("pw2bgw.x"),
        "yambo": probe("yambo"),
        "p2y": probe("p2y"),
        "ypp": probe("ypp"),
        "mpirun": probe("mpirun", "--version"),
    }
    qe_ok = all(binaries[k]["found"] for k in ("pw.x", "ph.x"))
    epw_ok = binaries["epw.x"]["found"]
    bgw_ok = all(binaries[k]["found"] for k in ("yambo", "p2y"))  # GW-BSE via Yambo
    print(f"[phase2/smoke] QE (pw/ph) present: {qe_ok}; EPW present: {epw_ok}; "
          f"Yambo GW-BSE present: {bgw_ok}")
    for k, b in binaries.items():
        print(f"[phase2/smoke]   {k:20s} found={b['found']} {b.get('path','')}")
    return {
        "qe_ok": qe_ok, "epw_branch_ok": qe_ok and epw_ok, "gwbse_branch_ok": bgw_ok,
        "core_eph_chain_ok": qe_ok and epw_ok,
        "binaries": binaries,
        "note": ("binary-presence check only; no science computed. GW-BSE branch "
                 "now uses Yambo (conda-forge, free) instead of the gated BerkeleyGW."),
    }


@app.function(image=qe_bgw_image, cpu=N_CORES, timeout=36000)
def label_material(spec: dict) -> dict:
    """Run the GW-BSE + EPW chain for one material and return tiered labels.

    `spec` must include an atomic structure and a pseudopotential source; without
    a real, converged run every label comes back 'not_run'. This function does
    the orchestration and parsing — it does not invent numbers.
    """
    import sys
    sys.path.insert(0, "/root/excitonic/src")
    from exciton_fm.provenance import not_run
    from exciton_fm.frohlich import parse_epw_linewidth
    from exciton_fm.exciton_u import parse_bse_biexciton

    workdir = spec.get("workdir", "/root/run")
    os.makedirs(workdir, exist_ok=True)

    # A production run executes, in `workdir`, with mpirun -np N:
    #   pw.x  < scf.in         ;  ph.x < ph.in   ;  epw.x < epw.in       (-> Γ)
    #   pw.x  < wfn/wfnq       ;  pw2bgw.x       ;  epsilon/sigma/kernel/absorption (-> E_b, U)
    # These require the pseudopotential library and converged grids to be staged
    # into `spec`. That staging is intentionally NOT auto-fabricated here.
    ran = bool(spec.get("staged_and_converged"))
    if not ran:
        gamma = not_run("Gamma_300K", "meV",
                        "structure + pseudopotentials + converged grids not staged; "
                        "no EPW run performed")
        U = not_run("U", "eV",
                    "structure + pseudopotentials + converged grids not staged; "
                    "no BSE run performed")
    else:  # pragma: no cover - exercised only with a fully staged run
        gamma = parse_epw_linewidth(os.path.join(workdir, "epw.out"))
        U = parse_bse_biexciton(os.path.join(workdir, "absorption.out"))

    print(f"[phase2/label] {spec.get('name','?')}: "
          f"Γ tier={gamma.tier} value={gamma.value}; U tier={U.tier} value={U.value}")
    return {"name": spec.get("name"), "Gamma_300K": gamma.to_dict(), "U": U.to_dict()}


# Two zincblende/diamond validation anchors that DIRECTLY test the phonon side:
# GaAs is polar (Born charge Z*~±2.2, LO-TO splitting, Fröhlich>0); Si is NON-polar
# (Z*~0, no LO-TO, Fröhlich must be ZERO). If Si returns a large Z*, the pipeline
# is not computing what we think.
DFT_NP = 8            # MPI ranks
DFT_NPOOL = 8         # k-point pools == ranks -> no G-vector split -> avoids the
                      # cdiaghg cholesky failure on tiny cells.
MATERIALS = {
    "GaAs": {"prefix": "gaas", "celldm": 10.6829, "polar": True,
             "species": [("Ga", 69.723, "Ga"), ("As", 74.9216, "As")],
             "positions": [("Ga", 0.0, 0.0, 0.0), ("As", 0.25, 0.25, 0.25)],
             "ref": "polar: ω_LO≈36 meV (~292 cm⁻¹), ε∞≈10.9, Z*≈±2.2"},
    "Si": {"prefix": "si", "celldm": 10.26, "polar": False,
           "species": [("Si", 28.0855, "Si")],
           "positions": [("Si", 0.0, 0.0, 0.0), ("Si", 0.25, 0.25, 0.25)],
           "ref": "NON-polar: Z*≈0, no LO-TO splitting, Fröhlich=0, ε∞≈11.7"},
}


@app.function(image=qe_bgw_image, cpu=N_CORES, timeout=7200)
def run_dft_dfpt(materials=("GaAs", "Si")) -> dict:
    """Run REAL scf + DFPT (ph.x, epsil+trans) for each material; parse ε∞, Z*, ω.

    Genuine first-principles numbers (tier 'dfpt') or 'not_run' on failure — never
    fabricated. GaAs (polar) + Si (non-polar) together test that the pipeline gets
    Born charges right: Z*(Si)≈0, Z*(GaAs)≈2.2.
    """
    import subprocess
    import sys
    import time
    sys.path.insert(0, "/root/excitonic/src")
    from exciton_fm.pseudos import stage_pseudos, pseudo_filename, recommended_cutoffs
    from exciton_fm.qe_outputs import (parse_epsilon_inf, parse_born_charges,
                                       parse_phonon_omega_LO, parse_total_energy)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    env["OMPI_MCA_plm_rsh_agent"] = "/usr/local/bin/fake_ssh"   # OpenMPI 4
    env["PRTE_MCA_plm_ssh_agent"] = "/usr/local/bin/fake_ssh"   # OpenMPI 5 (PRRTE)
    env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    MPI = ["mpirun", "--allow-run-as-root", "-np", str(DFT_NP)]
    POOL = ["-nk", str(DFT_NPOOL)]

    def one(name: str) -> dict:
        spec = MATERIALS[name]
        wd = f"/root/run_{name}"
        os.makedirs(os.path.join(wd, "pseudo"), exist_ok=True)
        os.makedirs(os.path.join(wd, "out"), exist_ok=True)
        syms = [s[0] for s in spec["species"]]
        stage_pseudos(syms, os.path.join(wd, "pseudo"))
        ecutwfc, ecutrho = recommended_cutoffs(syms)
        pfx = spec["prefix"]
        splines = "\n".join(f" {s} {m} {pseudo_filename(el)}"
                            for s, m, el in spec["species"])
        plines = "\n".join(f" {s} {x} {y} {z}" for s, x, y, z in spec["positions"])
        scf = f"""&control
  calculation='scf'
  prefix='{pfx}'
  outdir='./out'
  pseudo_dir='./pseudo'
/
&system
  ibrav=2
  celldm(1)={spec['celldm']}
  nat={len(spec['positions'])}
  ntyp={len(spec['species'])}
  ecutwfc={ecutwfc}
  ecutrho={ecutrho}
/
&electrons
  conv_thr=1.0d-10
  mixing_beta=0.7
  diagonalization='cg'
/
ATOMIC_SPECIES
{splines}
ATOMIC_POSITIONS crystal
{plines}
K_POINTS automatic
 8 8 8 0 0 0
"""
        ph = f"""{name}: dielectric + Born charges + phonons at Gamma
&inputph
  prefix='{pfx}'
  outdir='./out'
  fildyn='{pfx}.dyn'
  epsil=.true.
  trans=.true.
  asr=.true.
  tr2_ph=1.0d-15
/
0.0 0.0 0.0
"""
        open(os.path.join(wd, "scf.in"), "w").write(scf)
        open(os.path.join(wd, "ph.in"), "w").write(ph)

        def run(cmd, infile, outfile):
            with open(os.path.join(wd, outfile), "w") as fo:
                p = subprocess.run(cmd, stdin=open(os.path.join(wd, infile)),
                                   stdout=fo, stderr=subprocess.STDOUT, cwd=wd, env=env)
            return p.returncode

        t0 = time.time()
        rc_scf = run(MPI + ["pw.x"] + POOL, "scf.in", "scf.out")
        t1 = time.time()
        rc_ph = run(MPI + ["ph.x"] + POOL, "ph.in", "ph.out")
        t2 = time.time()
        wall = {"scf_s": round(t1 - t0, 1), "ph_s": round(t2 - t1, 1),
                "total_s": round(t2 - t0, 1), "ranks": DFT_NP}
        scf_out = open(os.path.join(wd, "scf.out")).read()
        ph_out = open(os.path.join(wd, "ph.out")).read()
        err_tail = ""
        if rc_scf != 0 or rc_ph != 0:
            err_tail = ("SCF tail:\n" + "\n".join(scf_out.splitlines()[-12:])
                        + "\nPH tail:\n" + "\n".join(ph_out.splitlines()[-12:]))
            print(f"[phase2/dfpt] {name} QE FAILED:\n" + err_tail)
        etot = parse_total_energy(scf_out)
        eps = parse_epsilon_inf(ph_out)
        zb = parse_born_charges(ph_out)
        wlo = parse_phonon_omega_LO(ph_out)
        print(f"[phase2/dfpt] {name}: rc_scf={rc_scf} rc_ph={rc_ph} "
              f"eps_inf={eps.value} Z*={zb.value} omega={wlo.value} meV "
              f"wall={wall['total_s']}s")
        return {"material": name, "polar_reference": spec["polar"],
                "rc_scf": rc_scf, "rc_ph": rc_ph, "wall_clock": wall,
                "etot_Ry": etot.to_dict(), "eps_inf": eps.to_dict(),
                "Z_born": zb.to_dict(), "omega_max_meV": wlo.to_dict(),
                "error_tail": err_tail, "reference": spec["ref"]}

    results = {name: one(name) for name in materials}
    # Polarity test verdict: Z*(non-polar) should be ~0, Z*(polar) clearly nonzero.
    zsi = results.get("Si", {}).get("Z_born", {}).get("value")
    zga = results.get("GaAs", {}).get("Z_born", {}).get("value")
    verdict = None
    if zsi is not None and zga is not None:
        verdict = {"Z_Si": zsi, "Z_GaAs": zga,
                   "pass": bool(zsi < 0.3 and zga > 1.0),
                   "criterion": "Z*(Si)<0.3 (non-polar) and Z*(GaAs)>1.0 (polar)"}
    return {"results": results, "polarity_test": verdict}


@app.local_entrypoint()
def dfpt():
    """Run the GaAs+Si DFPT validation on Modal; write results (tier dfpt)."""
    res = run_dft_dfpt.remote(("GaAs", "Si"))
    out = os.path.join(HERE, "data", "manifests", "phase2_dfpt_validation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[phase2/dfpt] wrote {out}")
    for name, r in res["results"].items():
        print(f"[phase2/dfpt]   {name}: eps_inf={r['eps_inf']['value']} "
              f"Z*={r['Z_born']['value']} omega={r['omega_max_meV']['value']} meV")
    print(f"[phase2/dfpt] polarity test: {res.get('polarity_test')}")


# --- 2D TMD DFPT-Γ sweep (Suggestion 2): real ω_LO/Z*/ε∞ -> DFPT-grounded Γ ---
# The question: do the RT linewidths of the TMD family cluster near ~10 meV? If not,
# the FOM is out of reach before U is ever computed. Cheap (DFPT, no EPW/BSE).
TMDS = {
    # formula: (metal, chalcogen, a[Å], layer-thickness[Å])
    "MoS2":  ("Mo", "S", 3.16, 3.17),
    "MoSe2": ("Mo", "Se", 3.29, 3.34),
    "WS2":   ("W", "S", 3.15, 3.14),
    "WSe2":  ("W", "Se", 3.28, 3.36),
}


@app.function(image=qe_bgw_image, cpu=N_CORES, timeout=14400)
def dft_2d_one(name: str) -> dict:
    """One TMD monolayer: scf + ph (epsil+trans) -> ω_LO, Z*, ε∞ + DFPT-grounded Γ.
    Runs in its own container so the sweep parallelizes across materials (Modal
    .map): wall-clock = slowest single material, not the sum."""
    import subprocess
    import sys
    import time
    import numpy as np
    sys.path.insert(0, "/root/excitonic/src")
    from ase.build import mx2
    from exciton_fm.pseudos import stage_pseudos, pseudo_filename, recommended_cutoffs
    from exciton_fm.qe_outputs import (parse_epsilon_inf, parse_born_charges,
                                       parse_phonon_omega_LO)
    from exciton_fm.frohlich import estimate_gamma_300K

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    env["OMPI_MCA_plm_rsh_agent"] = "/usr/local/bin/fake_ssh"
    env["PRTE_MCA_plm_ssh_agent"] = "/usr/local/bin/fake_ssh"
    MPI = ["mpirun", "--allow-run-as-root", "-np", str(N_CORES)]
    POOL = ["-nk", str(N_CORES)]

    def one(name: str) -> dict:
        m, x, a, th = TMDS[name]
        wd = f"/root/tmd_{name}"
        os.makedirs(os.path.join(wd, "pseudo"), exist_ok=True)
        os.makedirs(os.path.join(wd, "out"), exist_ok=True)
        syms = [m, x]
        stage_pseudos(syms, os.path.join(wd, "pseudo"))
        ecutwfc, ecutrho = recommended_cutoffs(syms)
        atoms = mx2(formula=name, kind="2H", a=a, thickness=th, vacuum=9.0)
        atoms.pbc = (True, True, True)
        cell = np.array(atoms.cell)
        spos = atoms.get_scaled_positions()
        chem = atoms.get_chemical_symbols()
        pfx = name.lower()
        cellblk = "\n".join(" %.10f %.10f %.10f" % tuple(cell[i]) for i in range(3))
        posblk = "\n".join(" %s %.10f %.10f %.10f" % (chem[i], *spos[i])
                           for i in range(len(chem)))
        spblk = "\n".join(" %s %.4f %s" % (s, 1.0, pseudo_filename(s)) for s in syms)
        scf = f"""&control
  calculation='scf'
  prefix='{pfx}'
  outdir='./out'
  pseudo_dir='./pseudo'
/
&system
  ibrav=0
  nat={len(chem)}
  ntyp={len(syms)}
  ecutwfc={ecutwfc}
  ecutrho={ecutrho}
  assume_isolated='2D'
  occupations='fixed'
/
&electrons
  conv_thr=1.0d-9
  mixing_beta=0.7
  diagonalization='cg'
/
CELL_PARAMETERS angstrom
{cellblk}
ATOMIC_SPECIES
{spblk}
ATOMIC_POSITIONS crystal
{posblk}
K_POINTS automatic
 12 12 1 0 0 0
"""
        ph = f"""{name} 2D: dielectric + Born charges + phonons at Gamma
&inputph
  prefix='{pfx}'
  outdir='./out'
  fildyn='{pfx}.dyn'
  epsil=.true.
  trans=.true.
  asr=.true.
  tr2_ph=1.0d-14
/
0.0 0.0 0.0
"""
        open(os.path.join(wd, "scf.in"), "w").write(scf)
        open(os.path.join(wd, "ph.in"), "w").write(ph)

        def run(cmd, i, o):
            with open(os.path.join(wd, o), "w") as fo:
                return subprocess.run(cmd, stdin=open(os.path.join(wd, i)),
                                      stdout=fo, stderr=subprocess.STDOUT,
                                      cwd=wd, env=env).returncode

        t0 = time.time()
        rc_scf = run(MPI + ["pw.x"] + POOL, "scf.in", "scf.out")
        rc_ph = run(MPI + ["ph.x"] + POOL, "ph.in", "ph.out")
        wall = round(time.time() - t0, 1)
        ph_out = open(os.path.join(wd, "ph.out")).read()
        scf_out = open(os.path.join(wd, "scf.out")).read()
        eps = parse_epsilon_inf(ph_out)
        zb = parse_born_charges(ph_out)
        wlo = parse_phonon_omega_LO(ph_out)
        # DFPT-grounded Γ: use the real ω_LO; α held at the anchor value (the real
        # per-material 2D α / linewidth needs the 2D Fröhlich + EPW). Flagged.
        gamma = (estimate_gamma_300K(wlo.value, alpha=0.4).to_dict()
                 if wlo.value else None)
        err = ""
        if rc_scf != 0 or rc_ph != 0:
            err = ("SCF:\n" + "\n".join(scf_out.splitlines()[-10:]) + "\nPH:\n"
                   + "\n".join(ph_out.splitlines()[-10:]))
        print(f"[phase2/2d] {name}: omega_LO={wlo.value} Z*={zb.value} "
              f"eps_inf={eps.value} Gamma~{(gamma or {}).get('value')} meV wall={wall}s")
        return {"formula": name, "rc": [rc_scf, rc_ph], "wall_s": wall,
                "omega_LO_meV": wlo.to_dict(), "Z_born": zb.to_dict(),
                "eps_inf": eps.to_dict(), "gamma_300K_model": gamma,
                "error_tail": err[:400]}

    return one(name)


def _cluster_summary(results: dict) -> dict:
    good = [r["gamma_300K_model"]["value"] for r in results.values()
            if r.get("gamma_300K_model")]
    if not good:
        return None
    return {"gamma_values_meV": good, "mean": round(sum(good) / len(good), 2),
            "range": [round(min(good), 2), round(max(good), 2)],
            "near_10meV": bool(all(3 <= g <= 25 for g in good))}


@app.local_entrypoint()
def gamma2d():
    """Run the TMD DFPT-Γ sweep — all monolayers CONCURRENTLY (Modal .map), so
    wall-clock ≈ one material's time instead of the sum."""
    formulas = ["MoS2", "MoSe2", "WS2", "WSe2"]
    results = {}
    for name, r in zip(formulas, dft_2d_one.map(formulas)):
        results[name] = r
        print(f"[phase2/2d] {name}: omega_LO={r['omega_LO_meV']['value']} "
              f"Z*={r['Z_born']['value']} Gamma~{(r.get('gamma_300K_model') or {}).get('value')} "
              f"meV wall={r['wall_s']}s")
    manifest = {"results": results, "gamma_cluster": _cluster_summary(results),
                "note": "ω_LO/Z*/ε∞ are real DFPT (tier dfpt); Γ is a DFPT-grounded "
                        "model estimate (α at anchor; real linewidth needs 2D-Fröhlich EPW)."}
    out = os.path.join(HERE, "data", "manifests", "phase2_tmd_gamma.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[phase2/2d] wrote {out}; cluster={manifest['gamma_cluster']}")


# --- Capped GW-BSE cost measurement (Yambo). LAUNCH ONLY ON APPROVAL. ---
# APPROVED budget ~$300 with leeway -> $400 hard ceiling. 48 cores (sweet spot for
# a 3-atom cell; more cores lose efficiency), wall-clock cap 20 h. Expected ACTUAL
# for a converged monolayer-MoS2 G0W0+BSE: ~500-1500 core-hours (~$50-150). The
# generous cap lets it CONVERGE instead of reject-on-cost. See
# reports/phase2_gwbse_cost_harness.md for the pre-declared convergence criteria.
GWBSE_CORES = 48
# Pre-committed spend CHECKPOINT = the wall cap: 36 h × 48 cores = 1728 core-hours
# ≈ $173 (at $0.10) to $259 (at $0.15). If the run hits this without a convergence
# signal it is KILLED and reviewed, not ridden to the ceiling — the fix for the
# sunk-cost trap. Expected ACTUAL is ~$50-150 (500-1500 core-hours), well inside it.
GWBSE_WALL_CAP_S = 129600                # 36 h hard wall-clock cap (~$250 checkpoint)
GWBSE_SPEND_CEILING_USD = 400            # backstop $ ceiling (rarely binds; wall cap first)
GWBSE_RATE_USD_PER_CORE_HR = 0.15        # conservative Modal CPU rate for the ceiling


GWBSE_PARAMS = {
    # mode -> (nbnd for nscf, bands for screening, screening cutoff [Ry],
    #          bands for GW self-energy, k-grid, BSE bands v/c)
    "debug":      dict(nbnd=60,  bnd_x=60,  ng_x=4,  bnd_gw=60,  kgrid=(6, 6, 1),  bse_v=2, bse_c=2),
    "production": dict(nbnd=300, bnd_x=300, ng_x=10, bnd_gw=300, kgrid=(18, 18, 1), bse_v=6, bse_c=6),
}


@app.function(image=qe_bgw_image, cpu=GWBSE_CORES, timeout=GWBSE_WALL_CAP_S)
def gwbse_cost(material: str = "MoS2", mode: str = "debug", vacuum: float = 10.0) -> dict:
    """Real G0W0+BSE (Yambo) on monolayer MoS2 with 2D Coulomb truncation, under a
    hard wall-clock cap. Chain: pw.x scf -> pw.x nscf(+bands) -> p2y -> yambo setup
    -> yambo GW -> yambo BSE -> parse E_b. Records per-stage wall-clock + cost.
    Deliverable = cost + convergence status; E_b is banked ONLY if the pre-declared
    gates pass, else 'not_run' (never fabricated). `mode`: 'debug' (cheap) or
    'production' (converged, the ~$300 run)."""
    import subprocess
    import sys
    import time
    import re as _re
    import numpy as np
    sys.path.insert(0, "/root/excitonic/src")
    from ase.build import mx2
    from exciton_fm.provenance import not_run, Label
    from exciton_fm.pseudos import stage_pseudos, pseudo_filename, recommended_cutoffs

    p = GWBSE_PARAMS[mode]
    wd = f"/root/gwbse_{material}_{mode}_v{int(vacuum)}"
    os.makedirs(os.path.join(wd, "pseudo"), exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    env["OMPI_MCA_plm_rsh_agent"] = "/usr/local/bin/fake_ssh"
    env["PRTE_MCA_plm_ssh_agent"] = "/usr/local/bin/fake_ssh"
    MPI = ["mpirun", "--allow-run-as-root", "-np", str(GWBSE_CORES)]
    stages, t_start = {}, time.time()

    def sh(name, cmd, cwd=wd, infile=None, outfile=None):
        t0 = time.time()
        stdin = open(os.path.join(cwd, infile)) if infile else None
        fo = open(os.path.join(cwd, outfile), "w") if outfile else subprocess.DEVNULL
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdin=stdin,
                            stdout=fo, stderr=subprocess.STDOUT).returncode
        stages[name] = round(time.time() - t0, 1)
        return rc

    # --- structure + QE scf/nscf (MoS2 monolayer, 2D) ---
    m, x, a, th = TMDS["MoS2"]
    atoms = mx2(formula="MoS2", kind="2H", a=a, thickness=th, vacuum=vacuum)
    atoms.pbc = (True, True, True)
    stage_pseudos([m, x], os.path.join(wd, "pseudo"))
    ecw, ecr = recommended_cutoffs([m, x])
    cell = np.array(atoms.cell); spos = atoms.get_scaled_positions(); chem = atoms.get_chemical_symbols()
    cellblk = "\n".join(" %.10f %.10f %.10f" % tuple(cell[i]) for i in range(3))
    posblk = "\n".join(" %s %.10f %.10f %.10f" % (chem[i], *spos[i]) for i in range(len(chem)))
    spblk = "\n".join(" %s 1.0 %s" % (s, pseudo_filename(s)) for s in [m, x])
    k = p["kgrid"]
    common = f"""  ibrav=0
  nat={len(chem)}
  ntyp=2
  ecutwfc={ecw}
  ecutrho={ecr}
  assume_isolated='2D'
"""
    for calc, nb, fn in (("scf", None, "scf.in"), ("nscf", p["nbnd"], "nscf.in")):
        nbnd = f"  nbnd={nb}\n" if nb else ""
        occ = "  occupations='fixed'\n" if calc == "scf" else "  occupations='fixed'\n  nosym=.true.\n"
        body = f"""&control
  calculation='{calc}'
  prefix='mos2'
  outdir='./out'
  pseudo_dir='./pseudo'
/
&system
{common}{nbnd}{occ}/
&electrons
  conv_thr=1.0d-9
  diagonalization='cg'
/
CELL_PARAMETERS angstrom
{cellblk}
ATOMIC_SPECIES
{spblk}
ATOMIC_POSITIONS crystal
{posblk}
K_POINTS automatic
 {k[0]} {k[1]} {k[2]} 0 0 0
"""
        open(os.path.join(wd, fn), "w").write(body)
    POOL = ["-nk", str(min(GWBSE_CORES, k[0] * k[1]))]
    rc = sh("scf", MPI + ["pw.x"] + POOL, infile="scf.in", outfile="scf.out")
    rc |= sh("nscf", MPI + ["pw.x"] + POOL, infile="nscf.in", outfile="nscf.out")

    # --- p2y (QE -> Yambo SAVE) + yambo setup ---
    save_dir = os.path.join(wd, "out", "mos2.save")
    rc |= sh("p2y", ["p2y"], cwd=save_dir)
    # move SAVE up to the yambo working dir
    ydir = os.path.join(wd, "yambo")
    os.makedirs(ydir, exist_ok=True)
    subprocess.run(["bash", "-c", f"cp -r {save_dir}/SAVE {ydir}/ 2>/dev/null || true"])
    rc |= sh("y_setup", ["yambo"], cwd=ydir)

    # --- GW input (2D truncation 'slab z') + run ---
    gw_in = f"""gw
rim_cut
gw0
ppa
HF_and_locXC
em1d
CUTGeo= "slab z"
EXXRLvcs= 8000            RL
% BndsRnXp
  1 | {p['bnd_x']} |
%
NGsBlkXp= {p['ng_x']}     Ry
% GbndRnge
  1 | {p['bnd_gw']} |
%
% QPkrange
  1 | {k[0] * k[1]} | 1 | {p['bnd_gw']} |
%
"""
    open(os.path.join(ydir, "gw.in"), "w").write(gw_in)
    rc |= sh("gw", MPI + ["yambo", "-F", "gw.in", "-J", "GW"], cwd=ydir)

    # --- BSE input (use GW db) + run ---
    bse_in = f"""optics
bss
bse
bsk
CUTGeo= "slab z"
BSEmod= "resonant"
BSKmod= "SEX"
BSSmod= "d"
KfnQP_E= "GW"
% BSEBands
  {max(1, p['nbnd'] // 2 - p['bse_v'] + 1)} | {p['nbnd'] // 2 + p['bse_c']} |
%
BSENGBlk= {p['ng_x']}    Ry
% BEnRange
  0.0 | 5.0 |  eV
%
"""
    open(os.path.join(ydir, "bse.in"), "w").write(bse_in)
    rc |= sh("bse", MPI + ["yambo", "-F", "bse.in", "-J", "BSE"], cwd=ydir)

    # --- parse: GW direct gap + lowest exciton -> E_b ---
    def _read(path):
        try:
            return open(path).read()
        except OSError:
            return ""
    reports = "\n".join(_read(os.path.join(ydir, f)) for f in os.listdir(ydir)
                        if f.startswith("r-")) if os.path.isdir(ydir) else ""
    # exciton energies: yambo writes o-BSE.exc_qpt1_E_sorted (col 1 = energy eV)
    exc = ""
    for f in (os.listdir(ydir) if os.path.isdir(ydir) else []):
        if "exc" in f.lower() and ("sorted" in f.lower() or f.startswith("o-")):
            exc = _read(os.path.join(ydir, f))
    exc_e = None
    for line in exc.splitlines():
        nums = _re.findall(r"[-+]?\d+\.\d+", line)
        if nums and not line.strip().startswith("#"):
            exc_e = float(nums[0]); break
    gw_gap = None
    mgw = _re.search(r"GW.*?gap.*?([-+]?\d+\.\d+)\s*eV", reports, _re.I | _re.S)
    if mgw:
        gw_gap = float(mgw.group(1))

    total_s = round(time.time() - t_start, 1)
    core_hours = round(total_s * GWBSE_CORES / 3600, 3)
    cost = {"mode": mode, "total_wall_s": total_s, "cores": GWBSE_CORES,
            "core_hours": core_hours,
            "usd_estimate": [round(core_hours * 0.05, 2), round(core_hours * GWBSE_RATE_USD_PER_CORE_HR, 2)],
            "per_stage_s": stages, "wall_cap_s": GWBSE_WALL_CAP_S,
            "spend_ceiling_usd": GWBSE_SPEND_CEILING_USD,
            "hit_wall_cap": total_s >= GWBSE_WALL_CAP_S * 0.98,
            "over_ceiling": core_hours * GWBSE_RATE_USD_PER_CORE_HR > GWBSE_SPEND_CEILING_USD}

    # Bank E_b only if we have both numbers AND (production) the anchor gate passes.
    if exc_e is not None and gw_gap is not None:
        E_b = gw_gap - exc_e
        anchor_ok = 0.3 <= E_b <= 0.8
        banked = (mode == "production" and anchor_ok)
        lab = (Label("E_b", round(E_b, 4), "eV", "gw_bse",
                     source=f"Yambo G0W0+BSE ({mode}), GW_gap={gw_gap:.3f}, exc={exc_e:.3f}",
                     notes=("banked" if banked else "NOT banked: "
                            + ("debug params" if mode != "production" else "anchor gate 0.3-0.8 eV failed")))
               if banked or mode == "debug"
               else not_run("E_b", "eV", "anchor gate failed; rejected"))
    else:
        lab = not_run("E_b", "eV",
                      f"Yambo chain incomplete (exc_e={exc_e}, gw_gap={gw_gap}); "
                      f"stages={stages}; needs input tuning")
    print(f"[phase2/gwbse] mode={mode} E_b={lab.value} gw_gap={gw_gap} exc={exc_e} "
          f"cost={cost['usd_estimate']} stages={stages}")
    return {"material": material, "mode": mode, "cost": cost,
            "gw_gap_eV": gw_gap, "exciton_eV": exc_e, "E_b": lab.to_dict(),
            "report_tail": reports[-1500:] if reports else ""}


@app.local_entrypoint()
def gwbse(mode: str = "debug"):
    """Capped GW-BSE on Modal.

    debug: run MoS2 at TWO vacuum spacings (10 & 16 Å) with the same (cheap) params
      and VERIFY 2D Coulomb truncation is actually engaged — E_b must PLATEAU
      (agree within ~5%), not climb. Also check magnitude (~hundreds of meV) and
      E_b>0 (lowest exciton below the GW gap). Reading the CUTGeo flag is not
      verification; this is. Cheap (~$2-4).
    production: the ~$300 converged run (single converged vacuum), launched ONLY
      after (a) the TMD sweep gate passes and (b) debug verified truncation.
    """
    if mode == "debug":
        r10, r16 = gwbse_cost.remote("MoS2", "debug", 10.0), None
        r16 = gwbse_cost.remote("MoS2", "debug", 16.0)
        eb10 = r10["E_b"]["value"]; eb16 = r16["E_b"]["value"]
        verdict = {"E_b_vac10_eV": eb10, "E_b_vac16_eV": eb16}
        if eb10 and eb16:
            drift = abs(eb16 - eb10) / abs(eb10)
            verdict.update({
                "vacuum_drift_frac": round(drift, 4),
                "truncation_engaged (plateau <5%)": bool(drift < 0.05),
                "magnitude_sane (0.3-0.8 eV)": bool(all(0.3 <= e <= 0.8 for e in (eb10, eb16))),
                "E_b_positive (exciton below GW gap)": bool(eb10 > 0 and eb16 > 0),
            })
            verdict["debug_pass"] = bool(verdict.get("truncation_engaged (plateau <5%)")
                                         and verdict["magnitude_sane (0.3-0.8 eV)"]
                                         and verdict["E_b_positive (exciton below GW gap)"])
        else:
            verdict["debug_pass"] = False
            verdict["reason"] = "Yambo chain did not return E_b at both vacua (needs input tuning)"
        res = {"mode": "debug", "vac10": r10, "vac16": r16, "verdict": verdict}
        print(f"[phase2/gwbse] DEBUG verdict: {verdict}")
    else:
        res = gwbse_cost.remote("MoS2", "production", 16.0)
        print(f"[phase2/gwbse] PRODUCTION E_b={res['E_b']['value']} cost={res['cost']['usd_estimate']}")
    out = os.path.join(HERE, "data", "manifests", f"phase2_gwbse_{mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[phase2/gwbse] wrote {out}")


@app.local_entrypoint()
def smoke():
    """Verify the Modal QE+BerkeleyGW container (run from CI)."""
    res = smoke_test.remote()
    out = os.path.join(HERE, "data", "manifests", "phase2_smoke.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[phase2/smoke] core_eph_chain_ok={res['core_eph_chain_ok']}; wrote {out}")


@app.local_entrypoint()
def generate():
    """Run the label chain over the seed set (structure-gated; honest by default)."""
    seed_path = os.path.join(HERE, "data", "manifests", "phase2_seed_set.json")
    if not os.path.exists(seed_path):
        print("[phase2/generate] no seed set found; run phase2_seed_select.py first.")
        return
    seed = json.load(open(seed_path))["seed_set"]
    # Without staged structures/pseudos, this reports 'not_run' for every material
    # rather than fabricating labels. Wire structure + pseudo staging here to run.
    results = [label_material.remote({"name": s["formula"]}) for s in seed[:5]]
    out = os.path.join(HERE, "data", "manifests", "phase2_labels.json")
    with open(out, "w") as fh:
        json.dump({"results": results,
                   "note": "not_run until structures + pseudopotentials staged"}, fh, indent=2)
    print(f"[phase2/generate] wrote {out} ({len(results)} materials, all not_run "
          f"until staging is wired)")
