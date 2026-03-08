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
  "dataset": {
    "IdxFormat": {
      "input_idx_file": "data/mnist/train-images-idx3-ubyte",
      "output_idx_file": "data/mnist/train-labels-idx1-ubyte"
    }
  },
  "training_strategy": {
    "MiniBatch": {
      "batch_size": 16
    }
  },
  "training_steps": 60000,
  "report_interval": 10,
  "snapshot_interval": 10000
}
```

## Top-Level Schema

All top-level fields are required:

- `model_source`
- `dataset`
- `training_strategy`
- `training_steps`
- `report_interval`
- `snapshot_interval`

Although the Rust field comments mention defaults for `report_interval` and `snapshot_interval`, no Serde defaults are implemented. Omitting either field will fail deserialization.

## Enum Encoding

`TrainConfig` contains enums, and Serde is using its default externally-tagged representation.

That means:

- `model_source` is an object with a single variant key such as `{ "Config": "..." }`.
- `dataset` is an object with a single variant key such as `{ "IdxFormat": { ... } }`.
- `training_strategy` is either:
  - the string `"SingleThread"` for the unit variant (no strategy-specific parameters), or
  - an object such as `{ "MiniBatch": { "batch_size": 16 } }` for the struct variant, which does have strategy-specific parametes.

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

### `dataset`

Controls which dataset loader is used.

The only supported dataset source right now is MNIST:

```json
"dataset": {
  "IdxFormat": {
    "input_idx_file": "data/mnist/train-images-idx3-ubyte",
    "output_idx_file": "data/mnist/train-labels-idx1-ubyte"
  }
}
```

Note for MNIST:
- Images are normalized to the range `0..1` when loaded.
- Labels are converted to one-hot vectors with width `10`.

### `training_strategy`

Controls how each optimization step is executed.

#### Single-threaded training

```json
"training_strategy": "SingleThread"
```

- each step selects one random sample from the dataset,
- the model converges on that sample,
- weights are updated immediately.

#### Mini-batch training

```json
"training_strategy": {
  "MiniBatch": {
    "batch_size": 16
  }
}
```

- each step clones the current model `batch_size` times,
- each clone trains on one randomly selected sample,
- the resulting weight updates are averaged,
- the averaged update is then applied to the main model.

`batch_size` is required for the `MiniBatch` variant and is parsed as a `u32`.

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
  "training_strategy": "SingleThread",
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
    "MiniBatch": {
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
