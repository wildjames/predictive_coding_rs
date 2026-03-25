# Intel GPU Programming in Rust: Reading and Exercise Ladder

This is a curated ladder for learning GPU compute in Rust on Intel GPUs. It is intentionally resource-first: the goal is to point you at strong material and give you exercises that force you to build the model yourself.

## Recommended First Path

For a Rust learner targeting Intel GPUs today, start with `wgpu` and WGSL, running on Intel's Vulkan stack.

- `wgpu` has the best Rust-first ergonomics and documentation.
- It teaches the GPU concepts that actually matter: adapter selection, device and queue creation, buffers, bind groups, compute pipelines, workgroups, dispatch, synchronization, and readback.
- On Linux and Windows, this maps well to Intel GPUs via Vulkan.
- It is a better first implementation path than Intel Level Zero, whose Rust ecosystem is much thinner and much lower-level.

Read Intel-native APIs later, not first.

- OpenCL is a useful concept bridge because its host-device-kernel-queue model is explicit and well documented.
- Level Zero is Intel's low-level execution model. It is worth reading once you already understand buffers, queues, kernels, and synchronization, but it is a poor starting point.

Treat `rust-gpu` as optional and late-stage.

- It is interesting if you specifically want shader code in Rust.
- It is still explicitly early-stage and is not the best first route for practical compute on Intel hardware today.

## Stage 0: Platform Sanity Check

Estimated time: 1 session.

### Goal

Confirm that your machine can see the Intel GPU and that a Rust program can target it.

### Read

- `wgpu` crate docs: <https://docs.rs/wgpu/latest/wgpu/>
- `wgpu` wiki home: <https://github.com/gfx-rs/wgpu/wiki>
- Intel driver overview for GPGPU on Linux: <https://dgpu-docs.intel.com/driver/overview.html>

### Exercises

1. Make a tiny Rust binary that requests an adapter with `wgpu` and prints the adapter name, backend, limits, and features.
2. If you are on Linux, run `vulkaninfo` and compare the device it sees with the device your `wgpu` program reports.
3. Identify which backend you are actually using on your machine. Do not assume it is Vulkan until you verify it.

### Exit Criteria

- You can explain the difference between an adapter, a device, and a queue.
- You have verified that your Intel GPU is visible to a Rust program.
- You know which backend your system is using.

## Stage 1: Learn the Host-Side GPU Model with `wgpu`

Estimated time: 2-3 sessions.

### Goal

Learn how a Rust program sets up compute work on the GPU.

### Read

- `Learn Wgpu` introduction: <https://sotrh.github.io/learn-wgpu/>
- Official `wgpu` examples index: <https://github.com/gfx-rs/wgpu/tree/v28/examples>
- `wgpu` docs for the core types:
  - `Instance`
  - `Adapter`
  - `Device`
  - `Queue`
  - `Buffer`
  - `ComputePipeline`

Focus on the official compute examples first:

- `hello_compute`
- `repeated_compute`
- `hello_workgroups`
- `hello_synchronization`

### Exercises

1. Run the official `hello_compute` example and trace the lifetime of the data from CPU allocation to GPU buffer to CPU readback.
2. Draw the object graph for a minimal compute program:
   `Instance -> Adapter -> Device + Queue -> Buffer/BindGroup/Pipeline -> CommandEncoder -> Submission`.
3. Change the input size, workgroup size, or number of dispatches in an official example and predict what will happen before you run it.
4. Replace the example's computation with a very simple operation of your own choice, such as vector add, scalar multiply, or clamp.

### Exit Criteria

- You can explain what gets created once and what gets created per dispatch.
- You can explain why GPU work is submitted through command buffers instead of called like an ordinary Rust function.
- You understand how results get back to the CPU.

## Stage 2: Learn WGSL Well Enough to Stop Guessing

Estimated time: 2 sessions.

### Goal

Build a working mental model of the shader language used by `wgpu`.

### Read

- Tour of WGSL: <https://google.github.io/tour-of-wgsl/>
- WebGPU Fundamentals:
  - WGSL: <https://webgpufundamentals.org/webgpu/lessons/webgpu-wgsl.html>
  - Storage buffers: <https://webgpufundamentals.org/webgpu/lessons/webgpu-storage-buffers.html>
  - Data memory layout: <https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html>
  - Copying data: <https://webgpufundamentals.org/webgpu/lessons/webgpu-copying-data.html>
  - Compute shader basics: <https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html>

### Exercises

1. Work through the Tour of WGSL sections on functions, types, variables, control flow, and binding points.
2. Create a host-side Rust struct and a matching WGSL buffer struct, then verify that the fields line up correctly in memory.
3. Write a tiny compute shader that reads one storage buffer, writes another, and uses `global_invocation_id` correctly.
4. Deliberately break alignment or binding declarations and read the error messages until you can explain them.

### Exit Criteria

- You can read a WGSL compute shader without treating it as opaque syntax.
- You understand the difference between workgroup size and dispatch size.
- You understand why buffer layout and alignment bugs are common.

## Stage 3: Build Real Compute Intuition on Intel Hardware

Estimated time: 3-4 sessions.

### Goal

Learn the performance shape of simple GPU compute jobs on your Intel GPU.

### Read

- WebGPU Fundamentals:
  - How it works: <https://webgpufundamentals.org/webgpu/lessons/webgpu-how-it-works.html>
  - Timing performance: <https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html>
  - Speed and optimization: <https://webgpufundamentals.org/webgpu/lessons/webgpu-optimization.html>
- `wgpu` wiki:
  - Debugging wgpu Applications: <https://github.com/gfx-rs/wgpu/wiki/Debugging-wgpu-Applications>
  - Debugging performance issues: <https://github.com/gfx-rs/wgpu/wiki/Debugging-performance-issues>
  - Do's and Don'ts: <https://github.com/gfx-rs/wgpu/wiki/Do%27s-and-Dont%27s>

### Exercises

1. Benchmark a CPU vector operation against a GPU version on small, medium, and large inputs.
2. Measure separately: upload time, dispatch time, readback time, and total wall time.
3. Run the same kernel many times with data kept resident on the GPU, then compare that against re-uploading every time.
4. Experiment with different workgroup sizes and note when performance changes meaningfully.

### Exit Criteria

- You can explain when the GPU loses to the CPU.
- You can explain why host-device traffic can dominate runtime.
- You have at least one benchmark where keeping data resident clearly helps.

## Stage 4: Learn Intel-Native Execution Models Without Implementing Them First

Estimated time: 2-3 sessions.

### Goal

Understand how Intel's lower-level runtimes think about devices, memory, queues, synchronization, and kernels.

### 4A: OpenCL as the Concept Bridge

#### Read

- Khronos OpenCL overview: <https://www.khronos.org/opencl/>
- OpenCL Guide: <https://github.com/KhronosGroup/OpenCL-Guide>
- Rust `ocl` crate docs: <https://docs.rs/ocl/latest/ocl/>

#### Exercises

1. Build a translation table from `wgpu` concepts to OpenCL concepts.
2. Identify which OpenCL concepts feel simpler than `wgpu`, and which feel more exposed.
3. Sketch how your stage-3 vector program would look in OpenCL terms: platform, device, context, queue, program, kernel, buffers, and readback.

#### Exit Criteria

- You can explain the host-device-kernel model in OpenCL terms.
- You can map OpenCL queues and buffers back to the ideas you learned through `wgpu`.

### 4B: Level Zero as the Intel-Native Low-Level Model

#### Read

- Level Zero Core Programming Guide: <https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html>
- oneAPI toolkit overview: <https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html>

Read these Level Zero sections in order:

1. Drivers and Devices
2. Contexts
3. Memory and Images
4. Command Queues and Command Lists
5. Synchronization Primitives
6. Modules and Kernels

Pay extra attention to:

- host, device, and shared allocations
- command queue groups
- command lists vs queues
- fences vs events
- sub-devices
- memory residency

#### Exercises

1. Write a concept map from `wgpu` to Level Zero terms.
2. Explain to yourself why Level Zero separates command lists from command queues.
3. Explain the difference between host, device, and shared allocations in Level Zero, and what kinds of mistakes each makes easier.
4. Read the sections on sub-devices and ask whether they might matter for future batching or model sharding on Intel hardware.

#### Exit Criteria

- You can explain what Level Zero is trying to expose that `wgpu` intentionally hides.
- You understand why Level Zero is better read after a Rust-first compute path, not before it.

## Stage 5: Learn How Intel Wants You to Profile Compute

Estimated time: 1-2 sessions, then ongoing.

### Goal

Learn to analyze offload quality, kernel time, and memory traffic on Intel GPUs.

### Read

- Intel VTune GPU compute profiling article:
  <https://www.intel.com/content/www/us/en/developer/articles/technical/optimize-applications-for-intel-gpus-with-intel-vtune-profiler.html>
- GPU Offload Analysis:
  <https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-offload-analysis.html>
- GPU Compute and Media Hotspots:
  <https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-compute-media-hotspots-analysis.html>
- oneAPI GPU Optimization Guide:
  <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/overview.html>

### Notes

- Intel explicitly positions VTune as the current tool for GPU compute analysis.
- Intel GPA is being phased out and is not the right default tool for compute learning.

### Exercises

1. Profile one of your stage-3 compute programs and look separately at CPU time, GPU kernel time, transfer time, and synchronization time.
2. Identify whether the program is CPU-bound, GPU-bound, or transfer-bound.
3. Find one kernel where the GPU is busy but the utilization is still poor, and write down your current hypothesis for why.
4. Repeat the profile after changing input size or batching strategy.

### Exit Criteria

- You can talk about offload quality instead of just runtime.
- You can distinguish a slow kernel from a bad offload pattern.

## Stage 6: Map the Ladder Back onto This Repository

Estimated time: 2 sessions.

### Goal

Use the GPU concepts you learned to reason about this codebase before touching implementation.

### Read and Inspect

- [Predictive coding model core](../src/model_structure/model.rs)
- [Math helpers](../src/model_structure/maths.rs)
- [Single-threaded training path](../src/training/cpu/singlethreaded.rs)
- [Mini-batch training path](../src/training/cpu/minibatch.rs)
- [Training loop orchestration](../src/training/train_handler.rs)

### Exercises

1. Classify every math-heavy operation in the model core as one of: elementwise, reduction, matrix-vector multiply, transpose-matrix-vector multiply, outer product, buffer copy, or synchronization point.
2. Mark which arrays must live for a whole convergence loop and which only need to exist for one kernel or one layer update.
3. Identify every place where a naive GPU port would accidentally copy data back to the CPU too often.
4. Decide which parts of the code should stay CPU-side in the first port: data loading, run orchestration, logging, serialization, and evaluation control flow.
5. Write a short design note answering this question: if you were forced to port only one operation first, which would it be, and why?

### Exit Criteria

- You have a candidate backend boundary for this project.
- You can explain why the convergence loop makes data residency important.
- You can name the first one or two operations you would port for learning value.

## Optional Stage 7: Explore Rust-Authored Shaders Later

### Goal

Explore writing shader-side code in Rust after you already understand the execution model.

### Read

- Rust GPU Dev Guide: <https://rust-gpu.github.io/rust-gpu/book/>
- Rust GPU repository: <https://github.com/Rust-GPU/rust-gpu>

### Why This Is Optional

- `rust-gpu` is explicitly still early-stage.
- It is great if your question becomes "can I author shader code in Rust?"
- It is not the best first route for "how do GPUs, buffers, dispatch, and synchronization work on Intel hardware?"

### Exercises

1. Only after stage 3 or later, decide whether shader authorship in Rust solves a real problem for you.
2. Compare the cost of learning WGSL now versus the cost of depending on a more experimental shader toolchain.

## Resource Shortlist by Role

### Primary Rust-First Path

- `wgpu` docs: <https://docs.rs/wgpu/latest/wgpu/>
- `Learn Wgpu`: <https://sotrh.github.io/learn-wgpu/>
- Official `wgpu` examples: <https://github.com/gfx-rs/wgpu/tree/v28/examples>
- WebGPU Fundamentals: <https://webgpufundamentals.org/>
- Tour of WGSL: <https://google.github.io/tour-of-wgsl/>

### Intel-Native Concepts

- OpenCL overview: <https://www.khronos.org/opencl/>
- OpenCL Guide: <https://github.com/KhronosGroup/OpenCL-Guide>
- `ocl` docs: <https://docs.rs/ocl/latest/ocl/>
- Level Zero guide: <https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html>
- oneAPI toolkits: <https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html>

### Profiling and Optimization

- VTune Intel GPU article: <https://www.intel.com/content/www/us/en/developer/articles/technical/optimize-applications-for-intel-gpus-with-intel-vtune-profiler.html>
- VTune GPU Offload Analysis: <https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-offload-analysis.html>
- VTune GPU Compute and Media Hotspots: <https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/gpu-compute-media-hotspots-analysis.html>
- oneAPI GPU Optimization Guide: <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/overview.html>

## What Not to Do First

- Do not start by trying to port the whole predictive coding model.
- Do not start with Level Zero unless your real goal is runtime internals.
- Do not assume GPU means faster until you measure transfers, dispatch overhead, and kernel occupancy.
- Do not treat WGSL layout rules as optional details.
- Do not mix "learn the API" and "optimize the whole application" in the same first week.

## First Checkpoint Question

After stages 0 through 3, you should be able to answer this without looking anything up:

"If I keep my tensors on the GPU across several steps, which costs disappear, which costs remain, and which bugs become more likely?"

If you cannot answer that yet, stay on the `wgpu` and WGSL path a bit longer before reading more Level Zero.