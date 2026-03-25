# GPU Rollout Plan

This folder currently contains small `wgpu` bring-up experiments:

- `wgpu_helloworld.rs` proves out device creation, buffer setup, dispatch, and readback.
- `introduction.wgsl` is the matching starter shader.

The goal is to evolve that into a real GPU execution path for the predictive coding model while preserving the existing CPU implementation as a first-class option.

This document describes the recommended rollout for this repository.

## Goals

- Keep the CPU path as a maintained, tested backend.
- Add a GPU backend using `wgpu` and hand-written WGSL compute shaders.
- Keep model files and configuration backend-neutral.
- Reuse the existing training and inference orchestration instead of forking the whole application flow.
- Make the GPU path incremental so each stage can be validated against the CPU implementation.

## Non-goals

- Do not introduce a tensor library.
- Do not build a generic tensor framework just to support two backends.
- Do not rewrite training, inference, evaluation, and serialization all at once.
- Do not optimize for peak GPU throughput before correctness and parity are established.

## Current State

At the moment, the CPU implementation is centered around `PredictiveCodingModel` in `src/model/model_structure.rs`.

That type currently owns three different responsibilities:

1. Portable model state that should be serializable.
2. Runtime state used during convergence and training.
3. CPU-specific math implemented with `ndarray`.

This is manageable for a CPU-only implementation, but it becomes awkward once a GPU backend is added because the orchestration code in `src/training` and `src/inference` depends directly on that concrete CPU representation.

The playground code in this folder is a good starting point, but it is intentionally not yet a reusable backend. The next step is not to expand the playground in place until it becomes production code by accident. The next step is to separate the backend boundary cleanly and then move GPU work behind that boundary.

## Recommended Target Architecture

The maintainable shape for this project is:

- one shared set of configs and portable model snapshots
- one shared training and inference flow
- two execution backends: CPU and GPU

### 1. Portable model state stays backend-neutral

The serialized representation should not contain GPU resources, staging buffers, bind groups, or adapter-specific details.

Keep these concepts portable:

- layer sizes
- activation function selection
- learning rates and convergence settings
- pinned layer flags
- values, predictions, errors, and weights when writing a snapshot

GPU-specific runtime resources should be rebuilt from the portable snapshot when a GPU backend is instantiated.

In practice, this usually means splitting the current model into two concepts:

- a portable snapshot or state struct used for config loading and serialization
- a runtime implementation that knows how to execute that state on CPU or GPU

### 2. Backends own execution, not orchestration

Training loops, evaluation flow, data loading, reporting, and artifact writing should remain shared.

What should differ by backend is the execution of model operations such as:

- setting input and output values
- reinitialising latent values
- computing predictions and errors
- advancing one convergence timestep
- computing and applying weight updates
- reading output activations
- exporting a snapshot

That keeps the application logic stable while allowing CPU and GPU implementations to diverge internally where they need to.

### 3. The abstraction boundary should be narrow

The right boundary is a model runtime interface, not a general-purpose array API.

Avoid trying to invent a custom `Array` type that mirrors `ndarray` on CPU and GPU. That usually creates a third abstraction that is harder to maintain than either backend.

Instead, define a small runtime contract around the operations the predictive coding algorithm actually needs.

For example:

```rust
pub enum ExecutionBackend {
    Cpu,
    Gpu,
    Auto,
}

pub trait ModelRuntime {
    fn backend(&self) -> ExecutionBackend;
    fn config(&self) -> PredictiveCodingModelConfig;

    fn set_input(&mut self, input: &[f32]) -> Result<()>;
    fn set_output(&mut self, output: &[f32]) -> Result<()>;
    fn pin_input(&mut self) -> Result<()>;
    fn pin_output(&mut self) -> Result<()>;
    fn unpin_output(&mut self) -> Result<()>;
    fn reinitialise_latents(&mut self) -> Result<()>;

    fn compute_predictions_and_errors(&mut self) -> Result<()>;
    fn timestep(&mut self) -> Result<f32>;
    fn converge_values(&mut self) -> Result<u32>;
    fn update_weights(&mut self) -> Result<()>;

    fn output_activations(&mut self) -> Result<Vec<f32>>;
    fn total_energy(&mut self) -> Result<f32>;
    fn snapshot(&mut self) -> Result<ModelSnapshot>;
}
```

The exact signatures can change, but the principle should hold: expose operations in terms of the model algorithm, not raw tensor primitives.

### 4. CPU stays the reference backend

The current CPU path should become the reference implementation of the runtime trait.

That gives you three important properties:

- a correctness oracle for GPU parity tests
- a fallback path on machines with no suitable GPU support
- a simpler debugging environment when GPU behavior is unclear

The CPU backend should remain easy to run in CI, on laptops, and during debugging.

### 5. GPU backend should own its own layout and buffers

The GPU implementation does not need to mirror `ndarray` internally.

A better approach is to flatten model state into contiguous buffers and maintain explicit metadata describing layer boundaries and weight regions.

Typical GPU-owned buffers will include:

- a values buffer
- a predictions buffer
- an errors buffer
- a weights buffer
- a metadata buffer containing per-layer offsets and sizes
- optional staging buffers for upload and readback

This makes shader logic explicit and avoids carrying CPU-centric data structures into the GPU path.

## Suggested Module Layout

One reasonable destination shape for this repository is:

```text
src/
  runtime/
    mod.rs
    backend.rs
    cpu.rs
    gpu/
      mod.rs
      context.rs
      buffers.rs
      pipelines.rs
      layout.rs
      shaders/
        predict.wgsl
        errors.wgsl
        timestep.wgsl
        weights.wgsl
  model/
    mod.rs
    snapshot.rs
    config.rs
    maths.rs
```

This is only a shape suggestion, not a requirement. The important point is that the GPU implementation should become a real module with a clean ownership boundary, not remain a loose playground forever.

## Rollout Stages

The safest way to land this is in stages where each stage keeps the repo working.

### Stage 0: Record a baseline

Before refactoring, lock down the current CPU behavior.

Work items:

- Keep the current unit and integration tests green.
- Add a small set of deterministic parity fixtures if they do not already exist.
- Record CPU timing baselines for:
  - single inference pass
  - one training step
  - mini-batch training step
  - model sizes small enough to run frequently
- Log adapter and backend details in the GPU playground so hardware behavior is explicit.

Exit criteria:

- You can compare later GPU results against a known-correct CPU baseline.
- You know whether the GPU path is actually faster for the sizes you care about.

### Stage 1: Split portable model state from CPU execution

This is the most important refactor.

Refactor the current `PredictiveCodingModel` so the serializable state is separate from the CPU execution engine.

Work items:

- Introduce a portable `ModelSnapshot` or similar struct.
- Move serialization and deserialization to that portable type.
- Keep the CPU implementation functionally identical.
- Update loading and saving helpers to round-trip through the portable type.

Why this stage matters:

- The GPU backend should not need to serialize `wgpu` resources.
- Training and inference code should stop assuming the runtime object is also the storage format.

Exit criteria:

- Existing snapshot files still load.
- Saving and loading remain backend-neutral.
- CPU behavior is unchanged.

### Stage 2: Introduce the runtime interface with CPU only

Do not write GPU execution code yet. First make the CPU path use the new abstraction.

Work items:

- Add the runtime trait and backend factory.
- Implement the trait for the CPU backend.
- Update training handlers and inference handlers to depend on the runtime trait instead of the concrete CPU model.
- Keep all runtime selection pointed at CPU at first.

This stage should be a pure architectural refactor with no intended behavior change.

Exit criteria:

- `train`, `eval`, and tests still work using the CPU backend only.
- The orchestration layers no longer depend directly on CPU-specific internals.

### Stage 3: Add backend selection and feature gating

Once the abstraction is real, add the configuration surface for backend choice.

Recommended behavior:

- `cpu` means always use the CPU backend.
- `gpu` means fail clearly if no suitable GPU backend is available.
- `auto` means prefer GPU when available and fall back to CPU otherwise.

Recommended configuration points:

- a config enum for execution backend
- a CLI flag for `train` and `eval`
- a default of `cpu` or `auto`, depending on how stable the GPU path becomes

Also recommended:

- make the GPU dependency feature opt-in rather than enabled by default while the backend is still maturing

Exit criteria:

- Backend choice is explicit and testable.
- CPU-only builds remain straightforward.

### Stage 4: Build the GPU host-side scaffold

Now turn the playground into reusable infrastructure.

Work items:

- create a reusable `GpuContext` for adapter, device, and queue setup
- create reusable helpers for storage buffers, staging buffers, and readback
- load WGSL shaders from the backend module rather than a one-off example
- define the flattened layer and weight layout used by the GPU backend
- add clear error messages for missing adapter support or unsupported limits

This stage should still be small in scope. The goal is to make GPU resource management boring and reusable.

Exit criteria:

- You can construct a GPU runtime instance from a portable snapshot.
- You can upload model state and read it back reliably.
- The host-side GPU code is no longer tied to a hello-world example.

### Stage 5: Implement inference-first GPU execution

Start with inference and convergence before training.

That means implementing GPU support for:

- prediction computation
- error computation
- value timestep updates
- convergence loops
- output readback

This is the right first target because it exercises the core model dynamics while keeping the scope smaller than full training.

Recommended kernel split for the first pass:

- one kernel for predictions
- one kernel for errors
- one kernel for value updates

Do not over-fuse kernels immediately. Small, explicit kernels are easier to debug and compare against the CPU reference.

Testing strategy for this stage:

- compare predictions layer-by-layer against CPU on tiny deterministic models
- compare one timestep of value updates against CPU within a numeric tolerance
- compare full convergence outputs on tiny fixtures

Exit criteria:

- GPU inference produces numerically close results to CPU.
- The outer inference flow can switch backends without changing application code.

### Stage 6: Implement training kernels

Once inference is correct, add weight update support.

Work items:

- implement the weight update kernel path
- validate weight updates against CPU for deterministic fixtures
- support single-sample training first
- only then address mini-batch GPU execution

Important note for this repository:

The current mini-batch flow clones the model per sample and averages updates afterwards. That is fine for CPU parallelism, but it should not dictate the long-term GPU design. The GPU backend may eventually want a different batching strategy that keeps data resident and computes batch statistics without full model cloning.

Exit criteria:

- GPU single-sample training matches CPU within tolerance.
- The weight update path is correct before mini-batch optimization begins.

### Stage 7: Revisit mini-batch strategy for GPU

Only after correctness is established should batching be redesigned for GPU efficiency.

Possible directions:

- keep batch elements in a packed batch dimension on the GPU
- accumulate weight deltas on device
- read back only reduced results when necessary
- maintain a separate optimized GPU batch path while preserving the shared outer API

At this stage, profile before changing architecture. The right batching approach depends on model size, dataset size, and the ratio of compute to transfer costs.

Exit criteria:

- The GPU backend is faster than CPU on the workloads it is meant to accelerate.
- Host-device copies are not dominating the steady-state training loop.

### Stage 8: Hardening, docs, and operational polish

Once the backend works, finish the operational details.

Work items:

- add backend-specific smoke tests where hardware is available
- document failure modes and fallback behavior
- benchmark upload time, dispatch time, readback time, and total wall time separately
- keep a small CPU/GPU parity suite for regression detection
- decide whether `auto` should become the user-facing default

Exit criteria:

- The backend is usable by someone other than the author.
- Performance claims are backed by measurements.

## GPU Data Layout Recommendations

Do not start by trying to store a vector of `Layer` structs directly on the GPU.

A more practical layout is:

- one flat `values` buffer for all layers
- one flat `predictions` buffer for all layers
- one flat `errors` buffer for all layers
- one flat `weights` buffer for all inter-layer connections
- one metadata buffer that describes:
  - layer size
  - value offset
  - error offset
  - prediction offset
  - weight offset
  - lower and upper layer relationships

That gives the shader enough information to operate over each layer while keeping the host-to-device data format simple.

For early versions, favor clarity over clever packing. Once the path is correct, you can optimize buffer packing, alignment, and dispatch granularity with profiling data.

## Shader Design Recommendations

Keep the first WGSL kernels explicit and algorithm-specific.

Recommended first kernels:

- `predict.wgsl`
  - compute lower-layer predictions from upper-layer values and weights
- `errors.wgsl`
  - compute `values - predictions`
- `timestep.wgsl`
  - compute per-node value updates using the existing predictive coding update rule
- `weights.wgsl`
  - compute per-weight update contributions

Avoid these mistakes in the first iteration:

- reading model state back to CPU after every convergence step
- over-generalizing kernels into a homemade tensor DSL
- hiding buffer layout details so deeply that debugging becomes difficult
- fusing many algorithm stages into one shader before the math has been validated

## Testing Strategy

The CPU backend should be treated as the correctness oracle.

Recommended tests:

- serialization tests that prove snapshots stay backend-neutral
- layout tests that verify flattening and unflattening preserve the model state
- parity tests for predictions, errors, one timestep, and weight updates
- inference tests comparing converged outputs between CPU and GPU within tolerance
- benchmark-style tests or scripts that separate upload, dispatch, and readback costs

For parity tests, use tiny deterministic fixtures first. A 1-1-1 or 4-3-2 model with fixed weights is more useful for debugging than a large random network.

## Design Rules To Keep The Codebase Maintainable

- Keep serialization types free of backend-specific resources.
- Keep the runtime trait narrow and algorithm-oriented.
- Keep CPU as a fully supported backend, not a compatibility afterthought.
- Prefer feature-gated modules over scattering `cfg` attributes through business logic.
- Avoid inventing a fake tensor library unless repeated evidence shows it is necessary.
- Keep GPU buffers resident across repeated dispatches whenever possible.
- Treat batching and performance tuning as later stages, not day-one architecture requirements.

## First Concrete Tasks For This Repository

If this plan is followed literally, the next tasks should be:

1. Extract a portable model snapshot from the current `PredictiveCodingModel` implementation.
2. Introduce a CPU runtime behind a trait without changing behavior.
3. Update training and inference handlers to use the runtime trait.
4. Move the code in `wgpu_helloworld.rs` into reusable GPU infrastructure rather than expanding it in place.
5. Implement inference-only GPU parity before touching GPU training.

## Summary

The core idea is simple:

- keep one model format
- keep one training and inference flow
- keep CPU as the reference backend
- add GPU as a backend implementation, not as a second copy of the whole application

That gives this repository a path to GPU acceleration without turning it into a split codebase.
