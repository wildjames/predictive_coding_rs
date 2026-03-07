//! Math utilities for predictive coding models.

use crate::{
  data_handling::data_handler::TrainingDataset,
  error::{PredictiveCodingError, Result},
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

fn ensure_parent_dir(filename: &str) -> Result<()> {
  if let Some(parent) = Path::new(filename).parent() && !parent.as_os_str().is_empty() {
      std::fs::create_dir_all(parent)
        .map_err(|source| PredictiveCodingError::io("create directory", parent, source))?;
    }

  Ok(())
}

pub fn load_model_config(fname: &str) -> Result<PredictiveCodingModelConfig> {
  let file = std::fs::File::open(fname)
    .map_err(|source| PredictiveCodingError::io("open model config", fname, source))?;

  serde_json::from_reader(file)
    .map_err(|source| PredictiveCodingError::json_deserialize(fname, source))
}

pub fn create_from_config(fname: &str) -> Result<PredictiveCodingModel> {
  let config = load_model_config(fname)?;
  Ok(PredictiveCodingModel::new(&config))
}

pub fn save_model_config(
  config: &PredictiveCodingModelConfig,
  filename: &str
) -> Result<()> {
  ensure_parent_dir(filename)?;

  let config_ser = serde_json::to_string(config)
    .map_err(|source| PredictiveCodingError::json_serialize(filename, source))?;
  std::fs::write(filename, config_ser)
    .map_err(|source| PredictiveCodingError::io("write model config", filename, source))?;

  Ok(())
}

pub fn save_model_snapshot(
  model: &PredictiveCodingModel,
  filename: &str
) -> Result<()> {
  ensure_parent_dir(filename)?;

  let model_ser = serde_json::to_string(&model)
    .map_err(|source| PredictiveCodingError::json_serialize(filename, source))?;
  std::fs::write(filename, model_ser)
    .map_err(|source| PredictiveCodingError::io("write model snapshot", filename, source))?;

  Ok(())
}

pub fn load_model_snapshot(filename: &str) -> Result<PredictiveCodingModel> {
  let model_ser = std::fs::read_to_string(filename)
    .map_err(|source| PredictiveCodingError::io("read model snapshot", filename, source))?;

  serde_json::from_str(&model_ser)
    .map_err(|source| PredictiveCodingError::json_deserialize(filename, source))
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

  use ndarray::array;
  use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH}
  };

  struct TempDir {
    path: PathBuf,
  }

  impl TempDir {
    fn new(prefix: &str) -> Self {
      let unique_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
      let path = std::env::temp_dir().join(format!(
        "predictive_coding_{prefix}_{}_{}",
        std::process::id(),
        unique_id
      ));
      fs::create_dir_all(&path).unwrap();
      TempDir { path }
    }

    fn join(&self, filename: &str) -> PathBuf {
      self.path.join(filename)
    }
  }

  impl Drop for TempDir {
    fn drop(&mut self) {
      let _ = fs::remove_dir_all(&self.path);
    }
  }

  fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap();
  }

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

  #[test]
  fn load_model_config_parses_expected_json_shape() {
    let temp_dir = TempDir::new("model_config_parse");
    let config_path = temp_dir.join("model_config.json");
    write_file(
      &config_path,
      r#"{
  "layer_sizes": [4, 10],
  "alpha": 0.01,
  "gamma": 0.05,
  "convergence_threshold": 0.0,
  "convergence_steps": 2,
  "activation_function": "Tanh"
}"#
    );

    let actual = load_model_config(config_path.to_str().unwrap()).unwrap();
    let expected = PredictiveCodingModelConfig {
      layer_sizes: vec![4, 10],
      alpha: 0.01,
      gamma: 0.05,
      convergence_threshold: 0.0,
      convergence_steps: 2,
      activation_function: ActivationFunction::Tanh,
    };

    assert_eq!(actual, expected);
  }

  #[test]
  fn load_model_config_rejects_missing_required_fields() {
    let temp_dir = TempDir::new("model_config_missing_field");
    let config_path = temp_dir.join("model_config.json");
    write_file(
      &config_path,
      r#"{
  "layer_sizes": [4, 10],
  "alpha": 0.01,
  "gamma": 0.05,
  "convergence_threshold": 0.0,
  "convergence_steps": 2
}"#
    );

    let result = load_model_config(config_path.to_str().unwrap());

    assert!(result.is_err());
  }

  #[test]
  fn load_model_config_error_includes_path() {
    let temp_dir = TempDir::new("missing_model_config");
    let config_path = temp_dir.join("missing_model_config.json");

    let error = load_model_config(config_path.to_str().unwrap()).unwrap_err();

    assert!(error.to_string().contains(&config_path.display().to_string()));
  }

  #[test]
  fn model_snapshot_round_trips_through_disk() {
    let temp_dir = TempDir::new("snapshot_round_trip");
    let snapshot_path = temp_dir.join("model_snapshot.json");
    let mut model = PredictiveCodingModel::new(&PredictiveCodingModelConfig {
      layer_sizes: vec![4, 10],
      alpha: 0.01,
      gamma: 0.05,
      convergence_threshold: 0.0,
      convergence_steps: 2,
      activation_function: ActivationFunction::Relu,
    });

    model.set_input(array![1.0, 0.0, 0.5, 0.25]);
    model.set_output(array![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    model.compute_predictions_and_errors();

    save_model_snapshot(&model, snapshot_path.to_str().unwrap()).unwrap();
    let loaded_model = load_model_snapshot(snapshot_path.to_str().unwrap()).unwrap();

    let original_json = serde_json::to_value(&model).unwrap();
    let loaded_json = serde_json::to_value(&loaded_model).unwrap();
    assert_eq!(loaded_json, original_json);
  }
}
