//! Math utilities for predictive coding models.

use crate::model::model::PredictiveCodingModel;

use std::path::Path;

use ndarray::{Array2, ArrayBase, Data, Dimension};
use serde::{Deserialize, Serialize};

pub fn save_model(
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

/// Activation function identifiers for serialization-friendly models.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActivationFunction {
  Relu,
}

impl ActivationFunction {
  /// Apply the activation function.
  pub fn apply(&self, x: f32) -> f32 {
    match self {
      ActivationFunction::Relu => relu(x),
    }
  }

  /// Apply the activation function derivative.
  pub fn derivative(&self, x: f32) -> f32 {
    match self {
      ActivationFunction::Relu => relu_derivitive(x),
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

