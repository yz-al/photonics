# Project memory

## Working conventions
- Whenever a section of work finishes, provide a copyable write-up (fenced code
  block) summarizing everything the user should know happened: what changed, why,
  key results, current state, and open items.

## Concepts / methodology

### Double black box
A two-stage strategy for turning a trained model into a mechanistic model:

1. **First black box — open it with mechinterp.** Take a model and apply
   interpretability techniques (sparse autoencoders / SAEs, ablations, activation
   clamping, etc.) to extract a specific behavior, circuit, feature, or mechanism.
2. **Second black box — synthesize a mechanistic model.** Feed what you pulled out
   into an automated program/model discovery system (e.g. AlphaEvolve or a similar
   implementation) to create a new mechanistic model of that behavior, or to update
   an existing mechanistic model.

The name "double black box" refers to chaining two opaque systems: the model under
study (opened via mechinterp) and the automated discovery engine that builds the
mechanistic account from the extracted signal.
