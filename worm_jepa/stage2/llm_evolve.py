"""
Automated Claude-as-operator evolutionary search (AlphaEvolve-style recipe,
reimplemented from scratch -- AlphaEvolve itself is DeepMind-internal).

This is the same loop the agent runs by hand in discover.py's candidates/, but
driven programmatically: Claude is the mutation operator. Each generation we show
the model the best programs so far + their scores + failure feedback, ask for an
improved candidate, evaluate it on the held-out problem, and keep an archive.

Requires ANTHROPIC_API_KEY (+ `pip install anthropic`). With neither, it prints
how to run the manual loop instead. The scoring/evaluation is identical to the
hand-driven loop, so results are directly comparable.

    python llm_evolve.py --generations 12
"""
from __future__ import annotations

import os
import argparse

import discover

SYSTEM = """You are the mutation operator in an evolutionary program search that
discovers a mechanistic model of C. elegans neural dynamics.

Write a Python function:

    def build(X_train, D_train):
        # X_train, D_train: numpy arrays (M, N). D = x[t+1]-x[t] (the delta).
        # fit anything on the training pairs, then return:
        def predict(X):   # (B, N) -> (B, N) predicted delta
            ...
        return predict

Only `np` and `softplus` are in scope (no imports, no other builtins beyond
range/len/float/int/abs/min/max/sum/enumerate/zip). Return ONLY the code.

Scored on a held-out split by: pred_r2 (R^2 predicting the delta) and struct_corr
(corr of the model's effective linear operator with the true neuron coupling).
Reason about the mechanism; beat the incumbents shown."""


def _prompt(archive):
    lines = ["Incumbents (higher pred_r2 is better):"]
    for name, code, m in archive[:6]:
        lines.append(f"\n## {name}: pred_r2={m.get('pred_r2'):.4f} "
                     f"struct_corr={m.get('struct_corr', 0):.4f}\n{code}")
    lines.append("\nPropose ONE improved `build` program. Code only.")
    return "\n".join(lines)


def evolve(generations=12, model="claude-opus-4-8"):
    try:
        import anthropic
    except ImportError:
        print("anthropic SDK not installed. Run the manual loop: write candidates "
              "to stage2/candidates/*.py and score with discover.eval_program().")
        return
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("No ANTHROPIC_API_KEY. Run the manual loop instead (see module docstring).")
        return

    client = anthropic.Anthropic()
    prob = discover.get_problem()

    # seed archive from the committed hand-driven candidates
    archive = []
    cdir = os.path.join(os.path.dirname(__file__), "candidates")
    for fn in sorted(os.listdir(cdir)):
        if fn.endswith(".py"):
            code = open(os.path.join(cdir, fn)).read()
            archive.append((fn[:-3], code, discover.eval_program(code, prob)))
    archive.sort(key=lambda a: a[2].get("pred_r2", -9), reverse=True)

    for g in range(generations):
        msg = client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM,
            messages=[{"role": "user", "content": _prompt(archive)}])
        code = msg.content[0].text
        if "```" in code:
            code = code.split("```")[1].lstrip("python").strip()
        m = discover.eval_program(code, prob)
        archive.append((f"evo_gen{g}", code, m))
        archive.sort(key=lambda a: a[2].get("pred_r2", -9), reverse=True)
        print(f"[evolve] gen {g}: pred_r2={m.get('pred_r2'):.4f} "
              f"struct={m.get('struct_corr', 0):.4f} | best={archive[0][0]} "
              f"{archive[0][2].get('pred_r2'):.4f}")
    best = archive[0]
    print(f"[evolve] winner: {best[0]}  {best[2]}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--model", default="claude-opus-4-8")
    evolve(**vars(ap.parse_args()))
