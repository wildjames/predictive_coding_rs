# Training Configuration

This document describes the JSON consumed by the `train` binary and parsed in `src/training/utils.rs`.

## Overview

A training configuration tells the program:

- how to build or restore the model,
- which dataset loader to use,
- which training strategy to run, and
- how long to train and how often to emit progress or snapshots.

The config is read by `load_training_config`, which deserializes the JSON directly into `TrainConfig` using Serde. There is no custom parsing layer, so the JSON shape matches the Rust enums and struct fields exactly.

## MNIST Example

```json
{
  "model_source": {
    "Config": "data/model_config.json"
  },
  "training_dataset": {
    "IdxFormat": {
      "input_idx_file": "data/mnist/train-images-idx3-ubyte",
      "output_idx_file": "data/mnist/train-labels-idx1-ubyte"
    }
  },
  "evaluation_dataset": {
    "IdxFormat": {
      "input_idx_file": "data/mnist/t10k-images-idx3-ubyte",
      "output_idx_file": "data/mnist/t10k-labels-idx1-ubyte"
    }
  },
  "training_strategy": {
    "CpuMiniBatch": {
      "batch_size": 16
    }
  },
  "training_steps": 60000,
  "report_interval": 10,
  "snapshot_interval": 10000
}
```

## Top-Level Schema

Most top-level fields are required:

- `model_source`
- `training_dataset`
- `training_strategy`
- `training_steps`
- `report_interval`
- `snapshot_interval`

The `evaluation_dataset` is optional at training time, but required for evaluation sessions. It can be added or updated later, if needed, though I recommend cloning the model output directory so you don't overwrite the history of the model.

## Enum Encoding

`TrainConfig` contains enums, and Serde is using its default externally-tagged representation.

That means:

- `model_source` is an object with a single variant key such as `{ "Config": "..." }`.
- `training_dataset` is an object with a single variant key such as `{ "IdxFormat": { ... } }`.
- `training_strategy` is either:
  - a string for a unit variant: `"CpuSingleThread"` or `"GpuSingleThread"`,
  - an object for a struct variant: `{ "CpuMiniBatch": { "batch_size": 16 } }` or `{ "GpuMiniBatch": { "batch_size": 16 } }`.
  The GPU variants are only available when the crate is built with `--features gpu`.

## Field-by-Field Reference

### `model_source`

Controls how the initial model is obtained.

Supported forms:

```json
"model_source": {
  "Config": "data/model_config.json"
}
```

Builds a fresh model from a model-config JSON file via `create_from_config`.

See [Model configuration](model-config.md) for the schema of that file.

```json
"model_source": {
  "Snapshot": "data/model_1772887557/model_snapshot_step_50000.json"
}
```

Loads a previously saved model snapshot via `load_model_snapshot`. This is the mechanism for resuming training from an earlier run.

### `training_dataset`

Controls which dataset loader is used for training data.

The only supported dataset source right now is IDX format, e.g. MNIST:

```json
"training_dataset": {
  "IdxFormat": {
    "input_idx_file": "data/mnist/train-images-idx3-ubyte",
    "output_idx_file": "data/mnist/train-labels-idx1-ubyte"
  }
}
```

### `evaluation_dataset`

Controls which dataset loader is used for evaluating a trained model.

The only supported dataset source right now is IDX format, e.g. MNIST:

```json
"evaluation_dataset": {
  "IdxFormat": {
    "input_idx_file": "data/mnist/t10k-images-idx3-ubyte",
    "output_idx_file": "data/mnist/t10k-labels-idx1-ubyte"
  }
}
```

### `training_strategy`

Controls how each optimization step is executed and on which backend.

There are four variants, two CPU-based and two GPU-based. The GPU variants require the `gpu` feature flag at build time.

#### CPU single-threaded

```json
"training_strategy": "CpuSingleThread"
```

- each step selects one random sample from the dataset,
- the model converges on that sample using CPU (`ndarray`) math,
- weights are updated immediately.

This is the simplest strategy and useful for debugging or small experiments.

#### CPU mini-batch

```json
"training_strategy": {
  "CpuMiniBatch": {
    "batch_size": 16
  }
}
```

- each step clones the current model `batch_size` times,
- the clones converge on different randomly selected samples **in parallel** using Rayon,
- the resulting weight updates are collected, summed on CPU, and averaged by `batch_size`,
- the averaged update is applied to the main model.

`batch_size` is required and parsed as a `u32`.

#### GPU single-threaded

```json
"training_strategy": "GpuSingleThread"
```

Requires `--features gpu`.

Follows the same one-sample-per-step loop as `CpuSingleThread`, but every operation (predict, error, convergence, weight update) is dispatched as wgpu compute shaders on the GPU. However, some intermediate data is still moved back to the CPU for decisions on convergence and reporting.

#### GPU mini-batch

```json
"training_strategy": {
  "GpuMiniBatch": {
    "batch_size": 16
  }
}
```

Requires `--features gpu`.

- each step processes `batch_size` samples sequentially on the GPU,
- weight deltas are accumulated **on-device** using a dedicated accumulation buffer, avoiding GPU-to-CPU round-trips between samples,
- the learning rate is pre-scaled by `1/batch_size` so the accumulated deltas are already averaged,
- a single shader pass applies the accumulated deltas to the weight matrices.

`batch_size` is required and parsed as a `u32`.

### `training_steps`

The number of training loop iterations.

Important detail: a step is not an epoch. The current implementation samples random examples with replacement, so `training_steps` is the number of optimizer steps, not the number of full dataset passes.

### `report_interval`

How often the training loop emits progress logs.

Behavior in the current implementation:

- `0` disables reporting,
- any positive value reports when `step % report_interval == 0`,

### `snapshot_interval`

How often the training loop writes model snapshots.

Behavior in the current implementation:

- `0` disables snapshots,
- any positive value saves when `step % snapshot_interval == 0`,

## Path Resolution

Paths inside the config are used as-is by the loaders.

That means relative paths are resolved relative to the process working directory, not relative to the directory that contains the config file.

## Output Files Produced By Training

The `train` binary writes outputs under a timestamped directory like `data/model_<unix_timestamp>/`.

During a run it will save:

- a copy of the training config as `model_training_config.json`,
- periodic snapshots when `snapshot_interval > 0`,
- a final model snapshot at the end of training.

Those saved snapshot files can be fed back into `model_source` using the `Snapshot` variant.

## Minimal Variants

Fresh model, single-threaded:

```json
{
  "model_source": {
    "Config": "data/model_config.json"
  },
  "dataset": {
    "IdxFormat": {
      "input_idx_file": "data/mnist/train-images-idx3-ubyte",
      "output_idx_file": "data/mnist/train-labels-idx1-ubyte"
    }
  },
  "training_strategy": "CpuSingleThread",
  "training_steps": 1000,
  "report_interval": 100,
  "snapshot_interval": 0
}
```

Resume from a saved snapshot:

```json
{
  "model_source": {
    "Snapshot": "data/model_1772887557/model_snapshot_step_50000.json"
  },
  "dataset": {
    "IdxFormat": {
      "input_idx_file": "data/mnist/train-images-idx3-ubyte",
      "output_idx_file": "data/mnist/train-labels-idx1-ubyte"
    }
  },
  "training_strategy": {
    "CpuMiniBatch": {
      "batch_size": 8
    }
  },
  "training_steps": 5000,
  "report_interval": 50,
  "snapshot_interval": 1000
}
```

## Running Training

From the repository root:

```bash
cargo run --release --bin train -- data/training_config.json
```

If the model input size or output size does not match the dataset, the binary will panic during startup rather than silently reshaping anything.
