"""
Modal launcher for the C. elegans JEPA on a GPU.

Mirrors test2/modal_app.py. Runs worm_jepa/train.py then worm_jepa/extract.py
on a Modal GPU against the *real* HuggingFace dataset, tars the artifacts dir,
and returns it (base64) to the local entrypoint, which unpacks it under
worm_jepa/artifacts/ so CI can commit it back to the branch.

Launched from CI:  modal run worm_jepa/modal_app.py
(Requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment. This process
uses gRPC, so it must run from GitHub Actions -- the sandboxed dev container
blocks gRPC through its proxy.)

Env knobs (captured on the runner, baked into the image):
  WORM_JEPA_EPOCHS, WORM_JEPA_MAX_WORMS, WORM_JEPA_DMODEL, WORM_JEPA_DEPTH,
  WORM_JEPA_SYNTHETIC (set to 1 to run the offline toy data on GPU too).
"""
import os
import base64
import modal

HERE = os.path.dirname(os.path.abspath(__file__))

_PASSTHROUGH = {
    k: os.environ.get(k, "")
    for k in (
        "WORM_JEPA_EPOCHS", "WORM_JEPA_BENCH_EPOCHS", "WORM_JEPA_MAX_WORMS",
        "WORM_JEPA_DMODEL", "WORM_JEPA_DEPTH", "WORM_JEPA_WINDOW", "WORM_JEPA_PATCH",
        "WORM_JEPA_BATCH", "WORM_JEPA_SYNTHETIC", "WORM_JEPA_REAL",
    )
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "numpy<2.3", "pandas", "pyarrow",
        "huggingface_hub", "scikit-learn",
    )
    .env(_PASSTHROUGH)
    .add_local_dir(HERE, remote_path="/root/worm_jepa", copy=True)
)

app = modal.App("worm-jepa")


@app.function(gpu="A10G", image=image, timeout=7200)
def run() -> str:
    import io
    import tarfile
    import runpy

    os.chdir("/root/worm_jepa")
    os.environ["WORM_JEPA_DEVICE"] = "cuda"
    os.environ["WORM_JEPA_OUT"] = "/root/worm_jepa/artifacts"

    import torch
    print("[modal] torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")

    # Head-to-head: flat JEPA vs hierarchical JEPA vs forecasting NN
    # (synthetic ground-truth metrics + real-dataset foundation-model metrics).
    runpy.run_path("/root/worm_jepa/benchmark.py", run_name="__main__")
    # Also train + open up the flat JEPA on the real data (checkpoint + SAE).
    runpy.run_path("/root/worm_jepa/train.py", run_name="__main__")
    runpy.run_path("/root/worm_jepa/extract.py", run_name="__main__")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/root/worm_jepa/artifacts", arcname="artifacts")
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
