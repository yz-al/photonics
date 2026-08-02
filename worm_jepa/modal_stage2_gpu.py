"""
Modal GPU launcher for the stage-2 PERCEPTION + self-play runs.

Separate from modal_app.py (which runs the JEPA train/extract pipeline) so these
can be fired on their own sentinel without re-running the whole pipeline. Runs, on
one A10G:
  1. vjepa/em_jepa.py       -- EM-JEPA self-supervised on real EPFL EM, at real
                               scale, then the frozen-feature mito + context probes.
  2. stage2/stage2_alphazero.py -- the self-play connectome builder at high episode
                               count on GPU: empirical confirmation that its test
                               AUC plateaus below the static baseline (the
                               inverted-feature fixed point diagnosed on CPU).
Collects the two result JSONs into artifacts/ and returns them to the runner.

Launched from CI: modal run worm_jepa/modal_stage2_gpu.py
(gRPC -> must run from GitHub Actions, not the sandboxed dev container.)
"""
import os
import base64
import modal

HERE = os.path.dirname(os.path.abspath(__file__))

# Real-scale defaults for the GPU; overridable from the workflow env.
_PASS = {
    "WORM_EM_CROP": os.environ.get("WORM_EM_CROP", "224"),
    "WORM_EM_DIM": os.environ.get("WORM_EM_DIM", "256"),
    "WORM_EM_DEPTH": os.environ.get("WORM_EM_DEPTH", "6"),
    "WORM_EM_BATCH": os.environ.get("WORM_EM_BATCH", "64"),
    "WORM_CREMI_SAMPLES": os.environ.get("WORM_CREMI_SAMPLES", "A,B,C"),
    # segmentation experiment (gentle-VICReg encoder -> watershed/flood-fill -> VOI/Rand/ERL)
    "WORM_SEG_JEPA_STEPS": os.environ.get("WORM_SEG_JEPA_STEPS", "3000"),
    "WORM_SEG_DEC_STEPS": os.environ.get("WORM_SEG_DEC_STEPS", "800"),
    "WORM_SEG_LABEL_SLICES": os.environ.get("WORM_SEG_LABEL_SLICES", "12"),
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy<2.3", "scikit-learn", "scikit-image",
                 "scipy", "tifffile", "h5py")
    .env(_PASS)
    .add_local_dir(HERE, remote_path="/root/worm_jepa", copy=True)
)

app = modal.App("worm-stage2-gpu")


@app.function(gpu="A10G", image=image, timeout=10800)
def run() -> str:
    import io
    import sys
    import shutil
    import tarfile
    import runpy
    import traceback

    os.chdir("/root/worm_jepa")
    if "/root/worm_jepa" not in sys.path:
        sys.path.insert(0, "/root/worm_jepa")
    os.environ["WORM_EM_DEVICE"] = "cuda"
    os.environ["WORM_AZ_DEVICE"] = "cuda"

    import torch
    print("[modal] torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)

    art = "/root/worm_jepa/artifacts"
    os.makedirs(art, exist_ok=True)

    stages = [
        # Segmentation experiment: gentle-VICReg EM-JEPA encoder -> pixel boundary
        # decoder -> watershed/flood-fill -> VOI / adapted-Rand / ERL, comparing
        # jepa vs random-encoder vs raw-pixel features. The connectome-reconstruction
        # metric that actually matters, and the honest perception-vs-topology test.
        ("/root/worm_jepa/vjepa/segment.py",
         "/root/worm_jepa/vjepa/segment.json", "segment.json"),
    ]
    if "/root/worm_jepa/vjepa" not in sys.path:      # segment.py imports hier_em_jepa
        sys.path.insert(0, "/root/worm_jepa/vjepa")
    failures = []
    for script, out_json, art_name in stages:
        try:
            print(f"[modal] === running {os.path.basename(script)} ===", flush=True)
            runpy.run_path(script, run_name="__main__")
            if os.path.exists(out_json):
                shutil.copy(out_json, os.path.join(art, art_name))
        except Exception:
            print(f"[modal] STAGE FAILED: {script}", flush=True)
            traceback.print_exc()
            failures.append(script)
    if failures:
        print(f"[modal] completed with stage failures: {failures}", flush=True)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(art, arcname="artifacts")
    return base64.b64encode(buf.getvalue()).decode()


@app.local_entrypoint()
def main():
    import io
    import tarfile

    payload = run.remote()
    raw = base64.b64decode(payload)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(path=HERE)
    print(f"[modal] unpacked artifacts under {os.path.join(HERE, 'artifacts')}")
