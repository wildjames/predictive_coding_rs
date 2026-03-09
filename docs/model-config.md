# Model Configuration

This document describes the JSON used to build a fresh predictive-coding model. It is consumed by `ModelSource::Config`, parsed in `create_from_config`, and deserialized directly into `PredictiveCodingModelConfig`.

## Overview

A model configuration defines the model architecture and the hyperparameters that control inference-time value updates and training-time weight updates.

It does not contain trained weights or latent state. It only describes how to construct a new model.

## MNIST Example

```json
{
  "gamma": 0.05,
  "alpha": 0.002,
  "convergence_steps": 80,
  "convergence_threshold": 1e-5,
  "layer_sizes": [
    784,
    512,
    10
  ],
  "activation_function": "Relu"
}
```

## Top-Level Schema

All top-level fields are required:

- `layer_sizes`
- `alpha`
- `gamma`
- `convergence_threshold`
- `convergence_steps`
- `activation_function`

There are no defaults for any of these fields. Omitting one will fail deserialization.

## Enum Encoding

`activation_function` is a Rust enum serialized with Serde's default unit-variant representation, so it is written as a plain string.

The currently supported values are:

- `"Relu"`
- `"Sigmoid"`
- `"Tanh"`

## Field-by-Field Reference

### `layer_sizes`

An array of layer widths ordered from input layer to output layer.
/
```json
"layer_sizes": [784, 512, 10]
```

The model constructor creates one `Layer` for each entry in the array, so this produces:

- an input layer of width `784`,
- one hidden layer of width `512`,
- an output layer of width `10`.

### `alpha`

The synaptic learning rate.

```json
"alpha": 0.002
```

This value scales weight updates after the model has converged on a sample. Larger values make weight updates more aggressive, but may cause overshooting. Smaller values make them more conservative, but will lead to slower learning rates.

### `gamma`

The neural learning rate.

```json
"gamma": 0.05
```

This value scales latent-value updates during inference. Larger values make inference dynamics move faster. Smaller values make them move more cautiously.

### `convergence_threshold`

The stopping threshold for inference on a single sample.

```json
"convergence_threshold": 1e-5
```

During `converge_values`, the model computes one timestep of latent-value updates and checks the summed absolute change. If that value is smaller than `convergence_threshold`, the sample is treated as converged.

### `convergence_steps`

The maximum number of inference timesteps allowed for one sample.

```json
"convergence_steps": 80
```

This is a hard upper bound. If the model does not satisfy `convergence_threshold` before this limit, inference still stops after `convergence_steps` iterations.

### `activation_function`

The nonlinearity used throughout the model.

```json
"activation_function": "Relu"
```

Current options:

- `Relu`
- `Sigmoid`
- `Tanh`

The current implementation applies the same activation function to every layer. The config does not support choosing different activations per layer.

The activation is used in two places:

- when predicting the layer below,
- when computing derivatives for value updates and weight updates.

## How Construction Works

When `PredictiveCodingModel::new` builds a model from this config:

- one layer is created per entry in `layer_sizes`,
- layer values are initialized randomly in the range `0..1`,
- predictions and errors start at zero,
- non-input layers get randomly initialized weights,
- all layers receive the same `activation_function`.

Weight initialization uses Xavier-style uniform sampling:

$$
U\left(-\sqrt{\frac{6}{fan_{in}+fan_{out}}},\; \sqrt{\frac{6}{fan_{in}+fan_{out}}}\right)
$$

For a layer of size `current_size` predicting a lower layer of size `lower_size`, the weight matrix has shape `(lower_size, current_size)`.

## Practical Constraints

The parser intentionally performs almost no validation beyond normal JSON deserialization.

That means:

- all fields must be present,
- numeric values are accepted as long as they match the target Rust types,
- the config does not enforce positive `alpha`, positive `gamma`, or minimum layer counts.

In practice, you should use:

- at least an input layer and an output layer,
- positive learning-rate values (unless you want to un-learn your data?),
- dataset-compatible first and last layer sizes.

## Minimal Variants

Direct input-to-output classifier:

```json
{
  "gamma": 0.05,
  "alpha": 0.002,
  "convergence_steps": 80,
  "convergence_threshold": 1e-5,
  "layer_sizes": [784, 10],
  "activation_function": "Relu"
}
```

Two hidden layers with `Tanh`:

```json
{
  "gamma": 0.03,
  "alpha": 0.001,
  "convergence_steps": 100,
  "convergence_threshold": 1e-5,
  "layer_sizes": [784, 256, 128, 10],
  "activation_function": "Tanh"
}
```
