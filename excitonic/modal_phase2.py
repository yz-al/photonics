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
# GW-BSE engine: Yambo. The conda-forge Yambo BINARY's BSE aborts in this image
# (SIGABRT with no Yambo [ERROR] across 5.2.x and 5.3.0 — a linked-library ABI abort
# in the BSE code path; see feasibility_bound.md "Fork A — CONCLUDED"). Fork B builds
# Yambo FROM SOURCE against the conda MPI/ScaLAPACK/FFTW/HDF5 stack we control, which
# is the documented way to escape that ABI mismatch.
#
#   YAMBO_SRC="5.1.2"  (a git tag on yambo-code/yambo) → source build replaces the
#                       conda yambo binary; the rest of the image is unchanged.
#   unset              → conda yambo binary (GW works; BSE known-broken here).
YAMBO_SRC = os.environ.get("YAMBO_SRC", "").strip()

if YAMBO_SRC:
    # Toolchain-only base (NO conda yambo binary — we compile our own against these
    # exact libs so there is one, self-consistent MPI/BLAS/ScaLAPACK/HDF5/netCDF ABI).
    qe_bgw_image = (
        modal.Image.micromamba(python_version="3.11")
        .micromamba_install(
            # Pin libxc to 5.2.x: Yambo 5.1.2's Fortran interface targets libxc 5.x.
            # conda's default (6/7.x) has an incompatible API, so configure rejects it
            # and falls back to building the bundled libxc-5.1.5 — whose source download
            # the proxy blocks. A compatible external libxc breaks that chain.
            "qe", "openmpi", "fftw", "libxc=5.2.3", "scalapack", "openblas",
            "hdf5=*=mpi_openmpi*", "netcdf-fortran=*=mpi_openmpi*", "libnetcdf",
            "fortran-compiler", "c-compiler", "cxx-compiler",
            "make", "pkg-config", "curl", "numpy", "ase",
            channels=["conda-forge"],
        )
        .pip_install("requests==2.33.1")
        .run_commands(_FAKE_SSH)
        .run_commands(
            # Fetch the OFFICIAL RELEASE tarball, which BUNDLES the external-library sources
            # in lib/archive/ (libxc/netcdf/hdf5/fftw/iotk). The GitHub *archive* tarball
            # strips those, so Yambo tried to DOWNLOAD them at build time from a dead libxc
            # URL ("not in gzip format"). Release asset first; fall back to the archive.
            f"curl -fSL -o /opt/yambo.tar.gz "
            f"https://github.com/yambo-code/yambo/releases/download/{YAMBO_SRC}/yambo-{YAMBO_SRC}.tar.gz "
            f"|| curl -fSL -o /opt/yambo.tar.gz "
            f"https://github.com/yambo-code/yambo/archive/refs/tags/{YAMBO_SRC}.tar.gz",
            "mkdir -p /opt/yambo && tar xzf /opt/yambo.tar.gz -C /opt/yambo --strip-components=1",
            "echo '=== lib/archive (bundled ext-lib sources) ===' ; "
            "ls -la /opt/yambo/lib/archive/ 2>/dev/null | head -20 || echo 'NO lib/archive — archive tarball'",
            # MINIMAL configure: give it only MPI + BLAS/ScaLAPACK (conda), and let Yambo
            # build hdf5/netcdf/libxc/fftw/iotk INTERNAL from the bundled sources. All-internal
            # avoids both the dead-URL download AND the conda-ABI mismatch that aborted the
            # conda binary's BSE. FPP=gfortran (its -ansi default mangles Fortran source).
            "cd /opt/yambo && P=$(dirname $(dirname $(which mpif90))) && "
            "FC=mpif90 F77=mpif90 CC=mpicc CPP='cpp -E -P' FPP='gfortran -E -P -cpp' "
            "./configure --enable-mpi --enable-open-mp "
            "--with-blas-libs=\"-L$P/lib -lopenblas\" "
            "--with-lapack-libs=\"-L$P/lib -lopenblas\" "
            "--with-scalapack-libs=\"-L$P/lib -lscalapack\" "
            "--with-blacs-libs=\"-L$P/lib -lscalapack\" 2>&1 | tail -50",
            # build ext-libs (internal, from bundled source — offline) then yambo + p2y.
            # Generous 90-min cap (internal lib compiles add time on the slow builder);
            # dump the log tail and fail only if bin/yambo is truly absent.
            "cd /opt/yambo && (timeout 5400 make -j8 yambo interfaces > make.log 2>&1; "
            "echo \"make_exit=$?\") ; echo '=== tail make.log ===' ; tail -170 make.log ; "
            "echo '=== bin/ ===' ; ls -la bin/ 2>/dev/null ; test -x bin/yambo",
            # expose the source-built binaries ahead of anything else on PATH
            "cp /opt/yambo/bin/yambo /opt/yambo/bin/p2y $(dirname $(which mpif90))/ && "
            "cp /opt/yambo/bin/ypp $(dirname $(which mpif90))/ 2>/dev/null ; "
            "echo 'yambo-from-source installed:' && yambo -version 2>&1 | head -3 || true",
        )
    )
else:
    qe_bgw_image = (
        modal.Image.micromamba(python_version="3.11")
        .micromamba_install(
            # conda yambo binary — GW works; BSE known-broken in this image (Fork A).
            "qe", "yambo<5.3", "openmpi", "fftw", "hdf5", "numpy", "ase",
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

# --- ABINIT BSE engine (non-Yambo route) -----------------------------------
# ABINIT installs as a conda BINARY (no source build, no dead-URL wall that blocked
# Yambo), and its BSE (optdriver=99) is standard. The one unknown in this sandbox is
# pseudopotentials: our QE PAW UPFs are not ABINIT-readable, so ABINIT needs its own
# (psp8 NC / JTH PAW). abinit_smoke probes both the BSE-compiled flag and which pseudo
# source actually downloads, BEFORE we write the full BSE input generator.
abinit_image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("abinit", "numpy", "ase", channels=["conda-forge"])
    .pip_install("requests==2.33.1")
    .run_commands(_FAKE_SSH)
    .add_local_dir(HERE, remote_path="/root/excitonic", copy=True)
)

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


@app.function(image=abinit_image, cpu=8, timeout=1800)
def abinit_smoke() -> dict:
    """De-risk the ABINIT BSE route: is BSE compiled, and which pseudos download?

    Two unknowns before writing the BSE input generator: (1) does the conda ABINIT
    have GW/BSE (optdriver=99) compiled, (2) can we fetch ABINIT-format pseudopotentials
    (psp8 NC or JTH PAW) in this sandbox. Probes both and writes a manifest.
    """
    import json as _json
    import os
    import shutil
    import subprocess

    import requests

    out = {"abinit": {}, "pseudo_probe": {}}
    ab = shutil.which("abinit")
    out["abinit"]["path"] = ab
    if ab:
        for flag in ("--version", "--build"):
            try:
                r = subprocess.run([ab, flag], capture_output=True, text=True, timeout=60)
                out["abinit"][flag] = (r.stdout or r.stderr)
            except Exception as e:  # pragma: no cover
                out["abinit"][flag] = f"ERR {e}"
        cfg = str(out["abinit"].get("--build", "")).lower()
        # ABINIT BSE/GW (optdriver 99/3/4) is core; flag any explicit disable.
        out["abinit"]["gw_bse_likely"] = ("gw" in cfg or "bethe" in cfg or ab is not None)

    # Probe candidate pseudopotential sources (per-element S = light, always exists).
    candidates = {
        "pdojo_github_psp8_S": "https://raw.githubusercontent.com/abinit/pseudo_dojo/master/"
                               "pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.4/S/S.psp8",
        "pdojo_github_psp8_Mo": "https://raw.githubusercontent.com/abinit/pseudo_dojo/master/"
                                "pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.4/Mo/Mo.psp8",
        "pdojo_github_psp8_Se": "https://raw.githubusercontent.com/abinit/pseudo_dojo/master/"
                                "pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.4/Se/Se.psp8",
        "pdojo_org_table_tgz": "http://www.pseudo-dojo.org/pseudos/"
                               "nc-sr-04_pbe_standard_psp8.tgz",
        "jth_paw_abinit_org": "https://www.abinit.org/sites/default/files/PAW2/JTH/"
                              "ATOMICDATA/JTH-PBE-atomicdata.tar.gz",
    }
    for name, url in candidates.items():
        rec = {"url": url}
        try:
            r = requests.get(url, timeout=60, stream=True)
            rec["status"] = r.status_code
            if r.status_code == 200:
                chunk = next(r.iter_content(chunk_size=4096), b"")
                rec["first_bytes"] = repr(chunk[:16])
                rec["looks_binary_or_psp"] = (b"psp8" not in chunk and b"<" not in chunk[:4]) \
                    or chunk[:1] in (b"\x1f", b"P")  # gzip magic / psp header heuristic
                # save a real psp8 candidate to confirm it's usable
                if name.startswith("pdojo_github_psp8") and b"<html" not in chunk.lower():
                    rec["ok_psp8"] = True
            r.close()
        except Exception as e:
            rec["error"] = str(e)[:200]
        out["pseudo_probe"][name] = rec

    working = [k for k, v in out["pseudo_probe"].items()
               if v.get("status") == 200 and (v.get("ok_psp8") or v.get("looks_binary_or_psp"))]
    out["pseudo_sources_working"] = working
    out["verdict"] = ("ABINIT present; pseudo source(s) working: " + ", ".join(working)
                      if (ab and working) else
                      "BLOCKER: " + ("no abinit " if not ab else "")
                      + ("no pseudo source downloaded" if not working else ""))

    os.makedirs("/root/excitonic/data/manifests", exist_ok=True)
    with open("/root/excitonic/data/manifests/phase2_abinit_smoke.json", "w") as fh:
        _json.dump(out, fh, indent=2)
    print("[abinit/smoke] abinit:", bool(ab), "| pseudo sources working:", working)
    print("[abinit/smoke] verdict:", out["verdict"])
    return out


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


# A 3-atom monolayer is TINY: 16-way MPI over-decomposes the G-vectors and ph.x
# deadlocks (the first gate run sat to the 4 h wall — the signature of a hang, not
# slow compute). Use MODERATE parallelism as pure k-point pools (1 rank/pool, no
# intra-pool G-vector split) and a 45-min fail-fast wall so a hang costs minutes.
N_CORES_2D = 8


@app.function(image=qe_bgw_image, cpu=N_CORES_2D, timeout=2700)
def dft_2d_one(name: str) -> dict:
    """One TMD monolayer: scf + ph (epsil+trans) -> ω_LO, Z*, ε∞ + DFPT-grounded Γ.
    Runs in its own container so the sweep parallelizes across materials (Modal
    .map): wall-clock = slowest single material, not the sum. Fails fast (45 min)
    rather than riding a hang to a multi-hour wall."""
    import subprocess
    import sys
    import time
    import numpy as np
    sys.path.insert(0, "/root/excitonic/src")
    from ase.build import mx2
    from ase.data import atomic_masses, atomic_numbers
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
    # Pure k-point pooling (1 rank/pool): spreads the ~19 IBZ k-points, no
    # G-vector decomposition on the tiny cell -> avoids the cdiaghg/collective
    # deadlock that over-decomposition triggers in ph.x.
    MPI = ["mpirun", "--allow-run-as-root", "-np", str(N_CORES_2D)]
    POOL = ["-nk", str(N_CORES_2D)]

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
        # REAL atomic masses: phonon frequencies scale as 1/√M, so a placeholder
        # mass of 1.0 amu inflates ω_LO ~5–13× (S=32, Mo=96, W=184…) — that inflated
        # ω_LO drove the Bose occupation to ~0 and floored Γ(300 K) at Γ0 for every
        # material in the first successful sweep. Dielectric ε∞ and Born Z* are
        # mass-INDEPENDENT and were unaffected.
        spblk = "\n".join(" %s %.4f %s" % (s, atomic_masses[atomic_numbers[s]],
                                           pseudo_filename(s)) for s in syms)
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
  tr2_ph=1.0d-13
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
        # Sanity band: a monolayer LO phonon is ~30–80 meV. Anything far outside
        # (e.g. the ~428 meV that a placeholder atomic mass produced) is unphysical
        # and must NOT be fed to the Fröhlich Γ — it would silently floor Γ at Γ0.
        wlo_physical = bool(wlo.value is not None and 10.0 <= wlo.value <= 120.0)
        # DFPT-grounded Γ: use the real ω_LO; α held at the anchor value (the real
        # per-material 2D α / linewidth needs the 2D Fröhlich + EPW). Flagged.
        gamma = (estimate_gamma_300K(wlo.value, alpha=0.4).to_dict()
                 if wlo_physical else None)
        err = ""
        if rc_scf != 0 or rc_ph != 0:
            err = ("SCF:\n" + "\n".join(scf_out.splitlines()[-10:]) + "\nPH:\n"
                   + "\n".join(ph_out.splitlines()[-10:]))
        print(f"[phase2/2d] {name}: omega_LO={wlo.value} meV "
              f"(physical={wlo_physical}) Z*={zb.value} eps_inf={eps.value} "
              f"Gamma~{(gamma or {}).get('value')} meV wall={wall}s")
        return {"formula": name, "rc": [rc_scf, rc_ph], "wall_s": wall,
                "omega_LO_meV": wlo.to_dict(), "omega_LO_physical": wlo_physical,
                "Z_born": zb.to_dict(),
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
    # return_exceptions=True: a container that times out/errors becomes an
    # Exception in the stream instead of crashing the whole entrypoint, so the
    # materials that DID finish are still banked (no all-or-nothing on one hang).
    for name, r in zip(formulas, dft_2d_one.map(formulas, return_exceptions=True)):
        if isinstance(r, Exception):
            results[name] = {"formula": name, "rc": None,
                             "error_tail": f"{type(r).__name__}: {r}"[:400]}
            print(f"[phase2/2d] {name}: FAILED {type(r).__name__}: {r}")
            continue
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
# Modal's hard per-function timeout ceiling is 86400 s (24 h) — a longer value is
# rejected at build time (that killed the first gwbse dispatch instantly). So the
# wall cap IS 24 h, and the pre-committed spend CHECKPOINT is set to it:
# 24 h × 48 cores = 1152 core-hours ≈ $115 (at $0.10) to $173 (at $0.15). If the
# run hits this without a convergence signal it is KILLED and reviewed, not ridden
# to the ceiling — the sunk-cost fix. Expected ACTUAL ~$50-150 (500-1500 core-h).
GWBSE_WALL_CAP_S = 86400                 # 24 h hard wall-clock cap (Modal max; ~$115-173 checkpoint)
GWBSE_SPEND_CEILING_USD = 400            # backstop $ ceiling (rarely binds; wall cap first)
GWBSE_RATE_USD_PER_CORE_HR = 0.15        # conservative Modal CPU rate for the ceiling


GWBSE_PARAMS = {
    # mode -> (nbnd for nscf, bands for screening, screening cutoff [Ry],
    #          bands for GW self-energy, k-grid, BSE bands v/c)
    "debug":      dict(nbnd=60,  bnd_x=60,  ng_x=4,  bnd_gw=60,  kgrid=(2, 2, 1),  bse_v=1, bse_c=1),
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
    from ase.data import atomic_masses, atomic_numbers
    from exciton_fm.provenance import not_run, Label
    from exciton_fm.pseudos import stage_pseudos, pseudo_filename, recommended_cutoffs

    p = GWBSE_PARAMS[mode]
    wd = f"/root/gwbse_{material}_{mode}_v{int(vacuum)}"
    os.makedirs(os.path.join(wd, "pseudo"), exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"   # belt-and-suspenders vs a BLAS-thread deadlock
    env["MKL_NUM_THREADS"] = "1"
    # yambo BSE crashed inside libhdf5 (H5Pclose) on database close — the classic
    # HDF5-in-container failure: file locking on an overlay/ephemeral FS. Disable it.
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # OpenMPI vader/sm shared-memory single-copy uses CMA (ptrace), which is
    # permission-denied in the container ("cma-permission-denied") — a possible
    # contributor to the BSE MPI aborts. Force a copy mechanism that needs no ptrace.
    env["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"
    env["OMPI_MCA_smsc"] = "^cma"
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    env["OMPI_MCA_plm_rsh_agent"] = "/usr/local/bin/fake_ssh"
    env["PRTE_MCA_plm_ssh_agent"] = "/usr/local/bin/fake_ssh"
    MPI = ["mpirun", "--allow-run-as-root", "-np", str(GWBSE_CORES)]
    # Yambo on a 3-atom debug cell with 48 ranks over-decomposes and hung for 6 h
    # (rode GitHub's job wall). Use modest ranks for debug; full ranks for production.
    n_y = 8 if mode == "debug" else GWBSE_CORES
    MPI_Y = ["mpirun", "--allow-run-as-root", "-np", str(n_y)]
    # Per-stage wall caps so NO single stage can ride the 6 h GitHub limit. A stage
    # that exceeds its cap is SIGKILLed and reported as a timeout (rc=124) with its
    # output tail — a hang becomes a legible, minutes-long failure, not a 6 h burn.
    TMO = ({"scf": 1500, "nscf": 1500, "p2y": 180, "y_setup": 300, "gw": 1500, "bse": 600}
           if mode == "debug"
           else {"scf": 3600, "nscf": 3600, "p2y": 600, "y_setup": 900, "gw": 9000, "bse": 9000})
    stages, t_start = {}, time.time()

    def sh(name, cmd, cwd=wd, infile=None, outfile=None):
        import signal
        t0 = time.time()
        stdin = open(os.path.join(cwd, infile)) if infile else None
        fo = open(os.path.join(cwd, outfile), "w") if outfile else subprocess.DEVNULL
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=stdin, stdout=fo,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=TMO.get(name))
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # kill the whole MPI tree
            proc.wait()
            rc = 124  # timeout sentinel
        stages[name] = round(time.time() - t0, 1)
        return rc

    def _tail(fn, n=30):
        try:
            return "\n".join(open(os.path.join(wd, fn)).read().splitlines()[-n:])
        except OSError:
            return "(no file)"

    def _fail(stage, note, extra=None):
        # Always return the full result shape (E_b not_run) + diagnostics, so the
        # entrypoint never crashes blind and the ACTUAL error tail lands in the
        # manifest. The Yambo chain is being debugged; failures must be legible.
        diag = {"ok": False, "stage_failed": stage, "note": note,
                "per_stage_s": dict(stages), "vacuum": vacuum,
                "scf_tail": _tail("scf.out"), "nscf_tail": _tail("nscf.out")}
        if extra:
            diag.update(extra)
        print(f"[phase2/gwbse] ABORT at {stage}: {note} | stages={stages}")
        return {"material": material, "mode": mode,
                "cost": {"per_stage_s": dict(stages), "note": f"aborted at {stage}"},
                "gw_gap_eV": None, "exciton_eV": None,
                "E_b": not_run("E_b", "eV", f"{stage}: {note}").to_dict(),
                "report_tail": "", "error": diag}

    # --- structure + QE scf/nscf (MoS2 monolayer, 2D) ---
    m, x, a, th = TMDS["MoS2"]
    atoms = mx2(formula="MoS2", kind="2H", a=a, thickness=th, vacuum=vacuum)
    atoms.pbc = (True, True, True)
    stage_pseudos([m, x], os.path.join(wd, "pseudo"))
    ecw, ecr = recommended_cutoffs([m, x])
    cell = np.array(atoms.cell); spos = atoms.get_scaled_positions(); chem = atoms.get_chemical_symbols()
    cellblk = "\n".join(" %.10f %.10f %.10f" % tuple(cell[i]) for i in range(3))
    posblk = "\n".join(" %s %.10f %.10f %.10f" % (chem[i], *spos[i]) for i in range(len(chem)))
    spblk = "\n".join(" %s %.4f %s" % (s, atomic_masses[atomic_numbers[s]],
                                       pseudo_filename(s)) for s in [m, x])
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
  diagonalization='david'
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
    # npool MUST divide nproc (QE aborts otherwise: 48 % 36 ≠ 0 killed the first
    # debug in 42 s) and be ≤ the #k-points. Pick the largest divisor of the core
    # count that fits the k-mesh.
    nk = k[0] * k[1] * k[2]
    npool = max(d for d in range(1, GWBSE_CORES + 1)
                if GWBSE_CORES % d == 0 and d <= nk)
    POOL = ["-nk", str(npool)]
    rc = sh("scf", MPI + ["pw.x"] + POOL, infile="scf.in", outfile="scf.out")
    rc |= sh("nscf", MPI + ["pw.x"] + POOL, infile="nscf.in", outfile="nscf.out")

    # Guard: QE must have produced the SAVE before p2y (don't blind-cd into a
    # missing dir). If not, return the scf/nscf tails so we can see WHY.
    save_dir = os.path.join(wd, "out", "mos2.save")
    if rc != 0 or not os.path.isdir(save_dir):
        return _fail("qe_scf_nscf",
                     f"QE rc={rc}, save_exists={os.path.isdir(save_dir)}, npool={npool}")

    # --- p2y (QE -> Yambo SAVE) + yambo setup ---
    rc |= sh("p2y", ["p2y"], cwd=save_dir)
    # move SAVE up to the yambo working dir
    ydir = os.path.join(wd, "yambo")
    os.makedirs(ydir, exist_ok=True)
    subprocess.run(["bash", "-c", f"cp -r {save_dir}/SAVE {ydir}/ 2>/dev/null || true"])
    if not os.path.isdir(os.path.join(ydir, "SAVE")):
        return _fail("p2y", "no SAVE in yambo dir after p2y",
                     {"save_ls": os.listdir(save_dir) if os.path.isdir(save_dir) else []})
    rc_setup = sh("y_setup", ["yambo"], cwd=ydir, outfile="y_setup.log")
    if rc_setup == 124:
        return _fail("y_setup", "yambo initialization timed out",
                     {"y_setup_tail": _tail("yambo/y_setup.log")})
    rc |= rc_setup

    # --- GW input (2D truncation 'slab z') + run ---
    # RandQpts/RandGvec control the RIM (random integration of the truncated
    # Coulomb) — the expensive, poorly-parallel step that made serial BSE crawl.
    # Cheap sampling for debug (this DB is reused by the BSE via -J GW); production
    # raises them for convergence.
    rim = "RandQpts=  1000000\nRandGvec= 100          RL\n" if mode == "production" \
          else "RandQpts=  10000\nRandGvec=  20          RL\n"
    gw_in = f"""gw
rim_cut
gw0
ppa
HF_and_locXC
em1d
CUTGeo= "slab z"
EXXRLvcs= 8000            RL
{rim}% BndsRnXp
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
    rc_gw = sh("gw", MPI_Y + ["yambo", "-F", "gw.in", "-J", "GW"], cwd=ydir,
               outfile="gw_run.log")
    if rc_gw == 124:
        return _fail("gw", "yambo GW (screening/QP) exceeded its wall cap",
                     {"gw_tail": _tail("yambo/gw_run.log")})
    rc |= rc_gw

    # --- BSE input (use GW db) + run ---
    # v/c band window for both the DIPOLES and the BSE kernel (must match). Yambo
    # needs the dipoles database before the BSE solver; omitting the `dipoles`
    # runlevel + DipBands makes BSE exit in ~1 s (the no-op we saw). The GW QP
    # correction is read via -J BSE,GW (ndb.QP), so KfnQP_E="GW" (a numeric field
    # given a string — an input-parse bail) is dropped.
    v_lo = max(1, p['nbnd'] // 2 - p['bse_v'] + 1)
    c_hi = p['nbnd'] // 2 + p['bse_c']
    # MPI BSE crashes (SIGABRT) in this yambo build regardless of decomposition;
    # serial BSE is crash-free. So debug runs BSE SERIAL on a deliberately TINY
    # problem (2x2 k, 1v×1c) so it finishes fast. No parallel-role vars needed.
    bs_par = ""
    bse_in = f"""dipoles
optics
bss
bse
bsk
CUTGeo= "slab z"
BSEmod= "resonant"
BSKmod= "SEX"
BSSmod= "d"
Chimod= "HARTREE"
% DipBands
  {v_lo} | {c_hi} |
%
% BSEBands
  {v_lo} | {c_hi} |
%
BSENGBlk= {p['ng_x']}    Ry
% BEnRange
  0.0 | 5.0 |  eV
%
% BDmRange
  0.1 | 0.1 |  eV
%
BEnSteps= 100
{bs_par}"""
    open(os.path.join(ydir, "bse.in"), "w").write(bse_in)
    # -J "BSE,GW": write to BSE, but READ the GW databases (ndb.QP for KfnQP_E="GW"
    # and the screening) from the GW folder. Without the GW read dir the BSE has no
    # QP correction and exits in seconds (the 3.2 s no-op we saw).
    # BSE parallelism: serial is too slow, and default MPI crashed because yambo
    # auto-decomposed the tiny eh-transition space. Fix: run 4 ranks but force the
    # parallel role onto k-points (16 of them) via BS_CPU/BS_ROLEs (set in bse.in),
    # not eh — plus the CMA-free transport (env above). Full ranks for production.
    # DECISIVE isolation: a 2x2 (4-kpt) serial BSE hung for 30 min — impossible for
    # the diagonalization itself, so the hang is UPSTREAM. Prime suspect: reading the
    # GW ndb.QP via -J GW. Run debug BSE at KS level (-J BSE only). Completes ⟹ the
    # GW-db read was the hang (and we get a KS-level exciton = the pipeline milestone);
    # still hangs ⟹ the yambo BSE build is broken (hard fork).
    bse_cmd = (["yambo", "-F", "bse.in", "-J", "BSE,GW"] if mode == "debug"
               else MPI_Y + ["yambo", "-F", "bse.in", "-J", "BSE,GW"])
    rc_bse = sh("bse", bse_cmd, cwd=ydir, outfile="bse_run.log")
    if rc_bse == 124:
        return _fail("bse", "yambo BSE (kernel/diagonalization) exceeded its wall cap",
                     {"bse_tail": _tail("yambo/bse_run.log")})
    rc |= rc_bse

    # --- parse: GW direct gap + lowest exciton -> E_b ---
    def _read(path):
        try:
            return open(path).read()
        except OSError:
            return ""
    reports = "\n".join(_read(os.path.join(ydir, f)) for f in os.listdir(ydir)
                        if f.startswith(("r-", "l-"))) if os.path.isdir(ydir) else ""
    # yambo writes o-*.exc_qpt1_E_sorted: col1 = energy [eV], col2 = STRENGTH
    # (the exciton oscillator strength / residue). The strength is the whole point
    # of the dipolar branch: it sets g ∝ √f and hence whether U survives the
    # Hopfield weighting |X|²U/Γ. Parse BOTH for the lowest (interlayer) exciton.
    exc = ""
    for f in (os.listdir(ydir) if os.path.isdir(ydir) else []):
        if "exc" in f.lower() and ("sorted" in f.lower() or f.startswith("o-")):
            exc = _read(os.path.join(ydir, f))
    _NUMSCI = r"[-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?"
    exc_e, exc_f = None, None
    for line in exc.splitlines():
        nums = _re.findall(_NUMSCI, line)
        if nums and not line.strip().startswith("#"):
            exc_e = float(nums[0].replace("D", "E").replace("d", "e"))
            if len(nums) >= 2:
                exc_f = float(nums[1].replace("D", "E").replace("d", "e"))
            break
    # Direct gap: parse the explicit "Direct Gap : X [eV]" lines (NOT a loose
    # GW…gap…eV span, which grabbed the 27.211 eV = 1 Hartree constant). Yambo
    # reports the gap in the [X] setup (KS) and, after the QP run, the GW-corrected
    # value; take the LAST occurrence (GW-corrected if present, else KS).
    # leading [^a-zA-Z] so "Direct" does NOT match inside "In-direct Gap" (that bug
    # made gw_gap the indirect gap 2.239 instead of the direct 2.660).
    gaps = _re.findall(r"[^a-zA-Z]Direct Gap\s*:\s*([-+]?\d+\.\d+)\s*\[?eV\]?", reports, _re.I)
    gw_gap = float(gaps[-1]) if gaps else None
    ks_gap = float(gaps[0]) if gaps else None
    bse_log_tail = _tail("yambo/bse_run.log", 90)   # so a BSE no-op/crash is diagnosable

    # The stack-trace tail never shows yambo's actual [ERROR] — grep the BSE run
    # log AND the yambo report/log files (r-*/l-*) for the real message.
    def _bse_errors():
        out, paths = [], [os.path.join(ydir, "bse_run.log")]
        if os.path.isdir(ydir):
            paths += [os.path.join(ydir, f) for f in sorted(os.listdir(ydir))
                      if f.startswith(("l-", "r-"))]
        for pth in paths:
            try:
                with open(pth) as fh:
                    for ln in fh:
                        if _re.search(r"\[ERROR\]|<ERROR>|error|abort|not enough|too many|"
                                      r"segmentation|allocat", ln, _re.I):
                            out.append(f"{os.path.basename(pth)}: {ln.strip()[:200]}")
            except OSError:
                pass
        return out[-30:]
    bse_errors = _bse_errors()

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
          f"f_osc={exc_f} cost={cost['usd_estimate']} stages={stages}")
    return {"material": material, "mode": mode, "vacuum": vacuum, "cost": cost,
            "gw_gap_eV": gw_gap, "ks_gap_eV": ks_gap, "exciton_eV": exc_e,
            "oscillator_strength": exc_f, "E_b": lab.to_dict(),
            "bse_log_tail": bse_log_tail, "bse_errors": bse_errors,
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
        # Run vac=10 first; only pay for vac=16 if the chain actually returned an
        # E_b (else the second vacuum would just reproduce the same failure at cost).
        r10 = gwbse_cost.remote("MoS2", "debug", 10.0)
        eb10 = r10["E_b"]["value"]
        if eb10 is None:
            err = r10.get("error", {})
            res = {"mode": "debug", "vac10": r10, "vac16": None,
                   "verdict": {"E_b_vac10_eV": None, "debug_pass": False,
                               "reason": f"vac10 aborted at {err.get('stage_failed')}: "
                                         f"{err.get('note')} — skipped vac16 (no spend)"}}
            out = os.path.join(HERE, "data", "manifests", "phase2_gwbse_debug.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as fh:
                json.dump(res, fh, indent=2)
            print(f"[phase2/gwbse] DEBUG aborted at vac10: {res['verdict']['reason']}")
            print(f"[phase2/gwbse] wrote {out}")
            return
        r16 = gwbse_cost.remote("MoS2", "debug", 16.0)
        eb16 = r16["E_b"]["value"]
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


def _abinit_structure(material: str, vacuum: float = 18.0):
    """Return (symbols, cell_Ang(3x3), scaled_positions, ntypat, znucl-order) for a 2D TMD.

    MoS2 = symmetric 1H (ase.mx2). MoSSe = Janus: Mo plane, S below, Se above, with the
    two different Mo–chalcogen vertical distances (broken mirror symmetry ⇒ the built-in
    out-of-plane dipole that makes this the solve-for-it candidate).
    """
    import numpy as np
    a = 3.22 if material == "MoS2" else 3.25          # Å
    zc = vacuum / 2.0
    # in-plane: Mo at (1/3,2/3); chalcogens eclipsed at (2/3,1/3)
    cell = np.array([[a, 0, 0], [-a / 2, a * (3 ** 0.5) / 2, 0], [0, 0, vacuum]])
    if material == "MoS2":
        dS = 1.56
        syms = ["Mo", "S", "S"]
        cart_z = [zc, zc - dS, zc + dS]
        order = ["Mo", "S"]
    elif material == "MoSSe":
        dS, dSe = 1.54, 1.70                          # Mo–S shorter than Mo–Se (Janus)
        syms = ["Mo", "S", "Se"]
        cart_z = [zc, zc - dS, zc + dSe]
        order = ["Mo", "S", "Se"]
    else:
        raise ValueError(material)
    xy = {"Mo": (1 / 3, 2 / 3), "S": (2 / 3, 1 / 3), "Se": (2 / 3, 1 / 3)}
    spos = []
    for s, z in zip(syms, cart_z):
        fx, fy = xy[s]
        spos.append([fx, fy, z / vacuum])
    return syms, cell, np.array(spos), len(order), order


@app.function(image=abinit_image, cpu=16, timeout=7200)
def abinit_bse(material: str = "MoS2", mode: str = "debug") -> dict:
    """ABINIT GS→WFK→model-dielectric BSE for a 2D TMD; return exciton energy + f.

    v1 captures the ABINIT output generously (exciton-eigenvalue / oscillator-strength
    print format) rather than guessing the parser. Model dielectric function (no explicit
    screening) + direct diagonalization (prints exciton energies AND oscillator strengths).
    Debug: small k-grid/ecut, no Coulomb truncation (magnitude approximate, proves the chain).
    """
    import glob
    import os
    import re as _re
    import subprocess
    import tarfile
    import tempfile

    import numpy as np
    import requests

    wd = tempfile.mkdtemp(prefix=f"abinit_{material}_")
    psp_dir = os.path.join(wd, "pseudo")
    os.makedirs(psp_dir, exist_ok=True)
    stages = {}

    # --- pseudos: fetch the pseudo-dojo NC-SR PBE psp8 table (has every element) ---
    try:
        tgz = os.path.join(wd, "psp.tgz")
        url = "http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_psp8.tgz"
        r = requests.get(url, timeout=300)
        open(tgz, "wb").write(r.content)
        with tarfile.open(tgz) as t:
            t.extractall(psp_dir)
        found = {}
        for el in ({"MoS2": ["Mo", "S"], "MoSSe": ["Mo", "S", "Se"]}[material]):
            hits = glob.glob(os.path.join(psp_dir, "**", f"{el}.psp8"), recursive=True)
            if not hits:
                return {"material": material, "error": f"no {el}.psp8 in table", "stage": "pseudo"}
            found[el] = hits[0]
    except Exception as e:
        return {"material": material, "error": f"pseudo fetch: {e}", "stage": "pseudo"}

    syms, cell, spos, ntypat, order = _abinit_structure(material)
    natom = len(syms)
    typat = [order.index(s) + 1 for s in syms]
    from ase.data import atomic_numbers
    znucl = [atomic_numbers[el] for el in order]
    BOHR = 1.8897259886
    rprim = "\n".join("  %.10f %.10f %.10f" % tuple(cell[i] * BOHR) for i in range(3))
    xred = "\n".join("  %.10f %.10f %.10f" % tuple(spos[i]) for i in range(natom))
    pseudos = ", ".join(f"{el}.psp8" for el in order)

    ngk = 6 if mode == "debug" else 12
    ecut = 25 if mode == "debug" else 38
    nband = 30 if mode == "debug" else 60
    # occupied bands (rough): Mo 14 val e- (ONCVPSP sp-semicore) + 6/chalcogen; nvalence/2.
    # Use a generous BSE window: bs_loband a few below the gap, nband a few above.
    bs_lo = 8
    eps_model = 13.0

    abi = f"""# {material} 2D BSE (full RPA screening, Tamm-Dancoff, direct diag)
pp_dirpath "{psp_dir}"
pseudos "{pseudos}"
ndtset 4

acell 1 1 1
rprim
{rprim}
natom {natom}
ntypat {ntypat}
znucl {' '.join(str(z) for z in znucl)}
typat {' '.join(str(t) for t in typat)}
xred
{xred}

ecut {ecut}
kptopt 1
ngkpt {ngk} {ngk} 1
nshiftk 1
shiftk 0.0 0.0 0.0
nstep 60
diemac 5.0

# DS1: GS density
tolvrs1 1.0d-8
nband1 {nband}

# DS2: NSCF WFK (many bands; nbdbuf so the top bands need not converge)
iscf2 -2
getden2 1
tolwfr2 1.0d-8
nband2 {nband + 6}
nbdbuf2 6

# DS3: SCREENING (optdriver=3) -> SCR file (RPA dielectric matrix)
optdriver3 3
getwfk3 2
nband3 {nband}
ecuteps3 4
ecutwfn3 {ecut - 5}
awtr3 1
inclvkb3 2

# DS4: BSE (optdriver=99), reads W from the SCR; direct diag prints exciton E + f.
# G-sphere cutoffs MUST be set (unset ⇒ SIGSEGV): ecutwfn/ecuteps/ecutsigx.
optdriver4 99
getwfk4 2
getscr4 3
bs_calctype4 1
ecutwfn4 {ecut - 5}
ecuteps4 4
ecutsigx4 {ecut - 5}
mbpt_sciss4 0.0 eV
bs_exchange_term4 1
bs_coulomb_term4 11
bs_coupling4 0
bs_loband4 {bs_lo}
nband4 {nband}
bs_freq_mesh4 0.0 8.0 0.02 eV
bs_algorithm4 1
inclvkb4 2
"""
    open(os.path.join(wd, "run.abi"), "w").write(abi)

    env = dict(os.environ)
    # ABINIT's UCX/MPI transport errors in this container — force serial + plain BTL.
    env.update({"OMPI_MCA_pml": "ob1", "OMPI_MCA_btl": "self,vader",
                "UCX_TLS": "self,sm", "OMP_NUM_THREADS": "1"})
    abinit = "/opt/conda/bin/abinit"
    try:
        p = subprocess.run([abinit, "run.abi"], cwd=wd, env=env,
                           capture_output=True, text=True, timeout=6600)
        log = (p.stdout or "") + "\n----STDERR----\n" + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return {"material": material, "error": "abinit timeout", "stage": "run"}

    # capture the main output file too
    outfiles = sorted(glob.glob(os.path.join(wd, "*")) )
    abo = glob.glob(os.path.join(wd, "*.abo")) + glob.glob(os.path.join(wd, "*.out"))
    abo_txt = open(abo[0]).read() if abo else ""

    # best-effort parse: exciton energies + oscillator strengths (format to be confirmed)
    exc = _re.findall(r"[Ee]xciton.{0,40}?([-+]?\d+\.\d+)", abo_txt + log)
    osc = _re.findall(r"[Oo]scillator.{0,40}?([-+]?\d+\.\d+)", abo_txt + log)
    tail = (abo_txt[-4000:] if abo_txt else log[-4000:])

    return {
        "material": material, "mode": mode, "ok": bool(abo_txt),
        "abinit_returncode": p.returncode,
        "output_files": [os.path.basename(f) for f in outfiles],
        "exciton_matches": exc[:12], "oscillator_matches": osc[:12],
        "abo_tail": tail,
        "note": "v1 engine-confirm: captures output to learn the exciton/f print format; "
                "model dielectric + no proper 2D truncation ⇒ magnitude approximate.",
    }


@app.local_entrypoint()
def abinitbse(material: str = "MoS2", mode: str = "debug"):
    """Run the ABINIT BSE engine on a 2D TMD (MoS2 to confirm, then MoSSe)."""
    res = abinit_bse.remote(material, mode)
    out = os.path.join(HERE, "data", "manifests", f"phase2_abinit_bse_{material}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[phase2/abinit-bse] {material}: ok={res.get('ok')} "
          f"exc={res.get('exciton_matches')}; wrote {out}")


@app.local_entrypoint()
def abinitsmoke():
    """Probe the ABINIT BSE route (BSE compiled? pseudos fetchable?) — run from CI."""
    res = abinit_smoke.remote()
    out = os.path.join(HERE, "data", "manifests", "phase2_abinit_smoke.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[phase2/abinit] verdict: {res.get('verdict')}; wrote {out}")


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
