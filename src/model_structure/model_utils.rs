//! Math utilities for predictive coding models.

use crate::{
  data_handling::data_handler::TrainingDataset,
  model_structure::model::{
    PredictiveCodingModel,
    PredictiveCodingModelConfig
  }
};

use std::path::Path;

use ndarray::{Array1, Array2, ArrayBase, Data, Dimension};
use serde::{Deserialize, Serialize};

/// Choose a random index using the rng threadlocal generator, and set the model I/O accordingly.
pub fn set_rand_input_and_output(
  model: &mut PredictiveCodingModel,
  data: &TrainingDataset
) {
  let rand_index: usize = usize::from_ne_bytes(rand::random()) % data.dataset_size;

  // Normalise to the range 0..1
  let input_values: Array1<f32> = data.inputs
    .row(rand_index)
    .to_owned();

  // One-hot output row with label value set to 1.0
  let output_values: Array1<f32> = data.labels
    .row(rand_index)
    .to_owned();

  model.set_input(input_values);
  model.set_output(output_values);

}

pub fn create_from_config(fname: &str) -> PredictiveCodingModel {
  let config: PredictiveCodingModelConfig = serde_json::from_reader(
    std::fs::File::open(fname).unwrap()
  ).unwrap();
  PredictiveCodingModel::new(&config)
}

pub fn save_model_config(
  config: &PredictiveCodingModelConfig,
  filename: &str
) {
  if let Some(parent) = Path::new(filename).parent()
    && !parent.as_os_str().is_empty() {
      std::fs::create_dir_all(parent).unwrap();
    }
  let config_ser = serde_json::to_string(config).unwrap();
  std::fs::write(filename, config_ser).unwrap();
}

pub fn save_model_snapshot(
  model: &PredictiveCodingModel,
  filename: &str
) {
  if let Some(parent) = Path::new(filename).parent()
    && !parent.as_os_str().is_empty() {
      std::fs::create_dir_all(parent).unwrap();
    }
  let model_ser = serde_json::to_string(&model).unwrap();
  std::fs::write(filename, model_ser).unwrap();
}

pub fn load_model_snapshot(filename: &str) -> PredictiveCodingModel {
  let model_ser = std::fs::read_to_string(filename).unwrap();
  serde_json::from_str(&model_ser).unwrap()
}

/// Activation function identifiers for serialization-friendly models.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActivationFunction {
  Relu,
  Sigmoid,
  Tanh
}

impl ActivationFunction {
  /// Apply the activation function.
  pub fn apply(&self, x: f32) -> f32 {
    match self {
      ActivationFunction::Relu => relu(x),
      ActivationFunction::Sigmoid => sigmoid(x),
      ActivationFunction::Tanh => tanh(x),
    }
  }

  /// Apply the activation function derivative.
  pub fn derivative(&self, x: f32) -> f32 {
    match self {
      ActivationFunction::Relu => relu_derivitive(x),
      ActivationFunction::Sigmoid => sigmoid_derivitive(x),
      ActivationFunction::Tanh => tanh_derivative(x),
    }
  }
}


/// Compute the outer product of two arbitrary-dimensional arrays flattened
/// in iteration order.
///
/// Returns a matrix with shape `(a.len(), b.len())` where each element is
/// `a[i] * b[j]`.
pub fn outer_product<SA, DA, SB, DB>(
  a: &ArrayBase<SA, DA>,
  b: &ArrayBase<SB, DB>
) -> Array2<f32>
where
  SA: Data<Elem = f32>,
  SB: Data<Elem = f32>,
  DA: Dimension,
  DB: Dimension,
{
  let a_values: Vec<f32> = a.iter().copied().collect();
  let b_values: Vec<f32> = b.iter().copied().collect();
  let rows = a_values.len();
  let cols = b_values.len();

  Array2::from_shape_fn((rows, cols), |(i, j)| a_values[i] * b_values[j])
}

/// Apply the ReLU activation function.
pub fn relu(x: f32) -> f32 {
  if x > 0.0 {
    x
  } else {
    0.0
  }
}

pub fn relu_derivitive(x: f32) -> f32 {
  if x > 0.0 {
    1.0
  } else {
    0.0
  }
}

/// Apply the sigmoid function
pub fn sigmoid(x:f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivitive(x: f32) -> f32 {
  (-x).exp() / (1.0 + (-x).exp()).powi(2)
}

// Apply tanh
pub fn tanh(x: f32) -> f32 {
  x.tanh()
}

pub fn tanh_derivative(x: f32) -> f32 {
  4.0 / ( (-x).exp() + x.exp()).powi(2)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn assert_within_tol(expected: f32, actual: f32, tol: f32) {
    assert!(
      (expected - actual).abs() < tol,
      "Expected {}, got {}, which is outside the tolerance of {}",
      expected,
      actual,
      tol
    );
  }

  #[test]
  fn test_sigmoid() {
    let x = 0.5;
    let expected = 0.622459;
    let actual = sigmoid(x);
    assert_within_tol(expected, actual, 1e-6);

    let x = -0.5;
    let expected = 0.377541;
    let actual = sigmoid(x);
    assert_within_tol(expected, actual, 1e-6);
  }

  #[test]
  fn test_sigmoid_derivative() {
    let x = 0.5;
    let expected = 0.235004;
    let actual = sigmoid_derivitive(x);
    assert_within_tol(expected, actual, 1e-6);

    let x = -0.5;
    let expected = 0.235004;
    let actual = sigmoid_derivitive(x);
    assert_within_tol(expected, actual, 1e-6);
  }
}
