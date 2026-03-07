# Architecture Refactoring TODO

## Goals

- Add enough test coverage that refactors can be done safely
- Reduce duplication
- Make failures explicit instead of panic-driven
- the `utils` modules are a mess. Sort them out into appropriate files
- Make sure that the interfaces are flexible enough for more datasets, training strategies, and evaluation modes

## Job order

1. Build a safety net.
2. Centralize run setup and validation.
3. Replace panic-heavy boundaries with structured errors.
4. Split oversized utility modules.
5. Generalize data loading and evaluation boundaries.
6. Unify the training and benchmarking control flow.
7. Clean up abstractions that remain unused or duplicated.

## Phase 1: Build a Safety Net

### Outcome

Refactors can proceed without guessing whether behavior changed.

### Tasks

- [x] Add unit tests for training-config parsing, model-config parsing, and invalid-shape rejection.
- [x] Add snapshot round-trip tests for model serialization and deserialization.
- [x] Add tests for training-loop behavior: report intervals, snapshot intervals, and hook ordering.
- [x] Add tests for mini-batch aggregation behavior on a tiny deterministic fixture.
- [x] Add a smoke test path for the `train`, `bench`, and `eval` binaries using very small test data or fixtures.
  - [x] Test data should go in a `test_data` directory, not the main `data` one.

### Notes

- Keep fixtures small enough to run in normal `cargo test` workflows.
- Enforce deterministic seeds or deterministic fixtures.

## Phase 2: Centralize Run Setup

### Outcome

There is one place that prepares a training or benchmark run: config loading, model loading, dataset loading, compatibility validation, output path creation, and handler construction.

### Tasks

- [x] Move shared startup logic out of `src/train.rs` and `src/benchmark.rs`.
- [x] Replace duplicated dataset/model compatibility checks with one validation function.
- [x] Use a single handler factory in both training and benchmarking code paths.

## Phase 3: Replace Panic-Driven Boundaries

### Outcome

Loaders, validators, and binaries return explicit errors with context instead of aborting through `unwrap`, `expect`, or `panic!`.

### Tasks

- [x] Convert loader functions to return `Result` all the way to the binary entry points.
- [x] Replace startup panics for model/data mismatches with validation errors.
- [x] Add context to file IO and serialization failures so the failing path is visible.
- [x] Make `main` functions return `Result<(), _>` where appropriate.

## Phase 4: Split Oversized Utility Modules

### Outcome

Modules are organized by responsibility instead of accumulating unrelated helper functions.

### Tasks

- [ ] Split `src/training/utils.rs` into narrower modules such as config, loading, and validation.
- [ ] Split `src/model_structure/model_utils.rs` into narrower modules such as persistence, math, activation, and sampling.
- [ ] Keep type definitions close to the code that owns them.
- [ ] Rename modules away from generic `utils` where the responsibility is now clear.

## Phase 5: Generalize the Data Boundary

### Outcome

Training and evaluation do not depend on a single hardcoded in-memory MNIST path.

### Tasks

- [ ] Define a dataset or sample-provider abstraction that exposes shape metadata and sample access.
- [ ] Keep the current MNIST loader as one implementation of that abstraction.
- [ ] Decide whether batching should operate on sampled records, iterators, or owned matrices.
- [ ] Add support for distinct training and evaluation dataset configuration instead of hardcoded evaluation paths.
- [ ] Ensure evaluation can reuse config-driven dataset loading rather than loading MNIST directly.

### Notes

- The current `TrainingDataset` struct is acceptable for now, but it should stop being the only shape the rest of the system understands.

## Phase 6: Make Inference and Evaluation First-Class

### Outcome

Inference and evaluation live behind shared library code instead of ad hoc logic in the binary.

### Tasks

- [ ] Move shared evaluation logic into a real inference module.
- [ ] Define a reusable API for: load model, prepare inputs, converge, read outputs, compute metrics.
- [ ] Keep the binary thin: argument parsing, setup, and output formatting only.

### Notes

- Empty modules should either become real extension points or be deleted to avoid false signals about architecture maturity.

## Phase 7: Unify Training and Benchmarking Control Flow

### Outcome

There is one canonical training loop, with benchmarking layered on via callbacks, event sinks, or instrumentation.

### Tasks

- [ ] Refactor benchmarking so it observes the shared training loop instead of reimplementing it.
- [ ] Define a reporting interface for per-step timing, snapshots, and summary metrics.
- [ ] Ensure benchmarking can opt into timing without owning a forked loop.
- [ ] Preserve current CSV and JSON benchmark artifacts through the new reporting path.

### Notes

- The training loop should remain the source of truth for hook ordering and step behavior.

## Phase 8: Clean Up Remaining Abstractions

### Outcome

The public surface matches actual usage and is easier to extend deliberately.

### Tasks

- [ ] Remove or justify trait methods that are implemented but not used.
- [ ] Extract shared handler setup code used by both single-threaded and mini-batch handlers.
- [ ] Review whether dynamic dispatch through `Box<dyn TrainingHandler>` is still the right choice after setup is centralized.
- [ ] Revisit naming so modules describe domain concepts rather than implementation leftovers.

## Definition of Done

- [ ] `train`, `bench`, and `eval` binaries are thin composition layers.
- [ ] Shared setup, validation, and artifact-path logic live in one place.
- [ ] Benchmarking reuses the canonical training loop.
- [ ] Dataset loading and evaluation are not hardcoded to MNIST-only paths in binaries.
- [ ] Error handling is `Result`-based at boundaries.
- [ ] Test coverage exists for the architectural seams introduced by the refactor.
