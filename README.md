# Photonic-compute thesis: honest energy accounting + optics-native architecture

Research track testing whether optical neural-network compute survives (1) an
honest, fully-sourced energy accounting versus digital, and (2) an architecture
designed around the free square-law (|E|^2) nonlinearity rather than ported from GPUs.

No fab, no synthesis, no proprietary data. Two falsifiable computational tests.

- `src/energy_model/` — Test 1: parameterized J/MAC model, optical vs digital, vs matrix size N.
- `test2/` — Test 2: activation-function comparison (ReLU-family vs quadratic / |.|^2), with optics constraints.
- `reports/photonic_thesis.md` — the deliverable: energy model + crossover band, feasibility check, architecture comparison, verdict.

See `reports/photonic_thesis.md` for the full writeup and sources.
