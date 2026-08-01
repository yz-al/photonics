"""
Modal launcher for Test 2 (full run on a GPU).

Runs test2/run_all.py with TEST2_MODE=full on a Modal GPU, then returns the
results JSON to the caller, which writes it to test2/results/test2_results_full.json.

Launched from CI:  modal run test2/modal_app.py
(Requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment.)
"""
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy<2.3",
        "scikit-learn", "datasets", "huggingface_hub",
    )
    # ship the test2 source into the container
    .add_local_dir(HERE, remote_path="/root/app", copy=True)
)

app = modal.App("photonic-test2")


@app.function(gpu="A10G", image=image, timeout=7200)
def run_full():
    import runpy
    os.chdir("/root/app")
    os.environ["TEST2_MODE"] = "full"
    os.environ["TEST2_DEVICE"] = "cuda"
    os.environ["TEST2_DATA"] = "/root/data"
    os.makedirs("/root/data", exist_ok=True)
    import torch
    print("[modal] torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    runpy.run_path("/root/app/run_all.py", run_name="__main__")
    with open("/root/app/results/test2_results_full.json") as f:
        return f.read()


@app.local_entrypoint()
def main():
    js = run_full.remote()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "test2_results_full.json")
    with open(path, "w") as f:
        f.write(js)
    print(f"[modal] wrote {path} ({len(js)} bytes)")
