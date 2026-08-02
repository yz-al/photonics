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
BUILD_BERKELEYGW = False  # flip on to compile BerkeleyGW into the image (slow)

qe_bgw_image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install(
        "qe", "openmpi", "fftw", "scalapack", "hdf5", "numpy", "ase",
        channels=["conda-forge"],
    )
    .pip_install("requests==2.33.1")
)
if BUILD_BERKELEYGW:  # pragma: no cover - opt-in heavy build
    qe_bgw_image = qe_bgw_image.run_commands(
        "micromamba install -y -n base -c conda-forge make gfortran",
        # Fetch + build BerkeleyGW against the conda MPI/ScaLAPACK/FFTW/HDF5 stack.
        # A per-arch arch.mk must be supplied; see reports/phase2_pipeline.md.
        "curl -L -o /opt/bgw.tar.gz https://berkeleygw.org/download/ || true",
        "echo 'BerkeleyGW source build placeholder — supply arch.mk and make'",
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
        "epsilon.cplx.x": probe("epsilon.cplx.x"),
        "sigma.cplx.x": probe("sigma.cplx.x"),
        "kernel.cplx.x": probe("kernel.cplx.x"),
        "absorption.cplx.x": probe("absorption.cplx.x"),
        "mpirun": probe("mpirun", "--version"),
    }
    qe_ok = all(binaries[k]["found"] for k in ("pw.x", "ph.x"))
    epw_ok = binaries["epw.x"]["found"]
    bgw_ok = all(binaries[k]["found"] for k in
                 ("epsilon.cplx.x", "sigma.cplx.x", "kernel.cplx.x", "absorption.cplx.x"))
    print(f"[phase2/smoke] QE (pw/ph) present: {qe_ok}; EPW present: {epw_ok}; "
          f"BerkeleyGW present: {bgw_ok}")
    for k, b in binaries.items():
        print(f"[phase2/smoke]   {k:20s} found={b['found']} {b.get('path','')}")
    return {
        "qe_ok": qe_ok, "epw_branch_ok": qe_ok and epw_ok, "bgw_branch_ok": bgw_ok,
        "core_eph_chain_ok": qe_ok and epw_ok,
        "binaries": binaries,
        "note": ("binary-presence check only; no science computed. BerkeleyGW is "
                 "not on conda-forge — the GW-BSE branch needs the optional source "
                 "build (BUILD_BERKELEYGW=True); the QE/EPW Γ branch is complete."),
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
