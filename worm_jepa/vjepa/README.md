# EM-JEPA: self-supervised perception on real connectomics EM

A JEPA (I-JEPA / V-JEPA-style) model trained self-supervised on real electron
microscopy of neural tissue -- the **perception-first** direction. Every
topology-only and activity-only route on the worm caps out (link-prediction AUC
~0.58, effective-connectivity struct_corr ~0.33) because the discriminative signal
lives in the EM **image**, not the graph. ConnectomeBench2 makes the same point: a
purpose-trained vision model on EM reaches human level where prompted LLMs and graph
methods do not.

`em_jepa.py` pretrains a context encoder to predict masked-region representations of
EM patches in latent space (EMA target encoder, predictor, no labels), then probes
the frozen features on a held-out mitochondria task -- with a **random-init encoder**
and **raw pixels** as honest controls.

- Data: EPFL CVLab hippocampus EM (public direct download; auto-cached).
- Run (CPU smoke): `WORM_EM_CROP=64 WORM_EM_STEPS=100 python em_jepa.py`
- Run (GPU, real scale): fired on Modal via the `worm_jepa/RUN_GPU2` sentinel.

Finding so far (small-scale): EPFL-mito is pixel-trivial and spatially
autocorrelated (random encoder ~ SSL ~ raw pixels ~ 0.87), so it lacks the
representation headroom to *differentiate* SSL quality; the GPU run tests whether
real capacity + a context-prediction probe separates them. A cleanly differentiating
result needs a context-dependent task (neuron boundaries / synaptic clefts, e.g.
CREMI) rather than large blob detection.
