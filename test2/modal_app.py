"""
Modal launcher for Test 2 on a GPU.

Runs test2/run_all.py on a Modal GPU and returns the results JSON to the caller,
which writes it under test2/results/.  TEST2_ONLY (from the runner env, baked into
the container image) selects a targeted block, e.g. "cnn" for the CIFAR-10 CNN
only (fits a short GPU window); empty runs the full sweep.

Launched from CI:  modal run test2/modal_app.py
(Requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment.)
"""
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ONLY = os.environ.get("TEST2_ONLY", "").strip()   # captured on the runner
MODE = os.environ.get("TEST2_MODE", "full").strip()
RESULT_FILE = "test2_results_cnn.json" if ONLY == "cnn" else f"test2_results_{MODE}.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy<2.3",
        "scikit-learn", "datasets", "huggingface_hub",
    )
    # bake the block selector + mode into the container, then ship the source
    .env({"TEST2_ONLY": ONLY, "TEST2_MODE": MODE})
    .add_local_dir(HERE, remote_path="/root/app", copy=True)
)

app = modal.App("photonic-test2")


@app.function(gpu="A10G", image=image, timeout=7200)
def run():
    import runpy
    os.chdir("/root/app")
    os.environ["TEST2_DEVICE"] = "cuda"
    os.environ["TEST2_DATA"] = "/root/data"
    os.makedirs("/root/data", exist_ok=True)
    import torch
    print("[modal] torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
          "| TEST2_ONLY=", os.environ.get("TEST2_ONLY", ""))
    runpy.run_path("/root/app/run_all.py", run_name="__main__")
    with open(f"/root/app/results/{RESULT_FILE}") as f:
        return f.read()


@app.local_entrypoint()
def main():
    js = run.remote()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, RESULT_FILE)
    with open(path, "w") as f:
        f.write(js)
    print(f"[modal] wrote {path} ({len(js)} bytes)")
