use crate::{
  data_handling::{
    data_handler::TrainingDataset,
    mnist::load_mnist
  },
  error::{PredictiveCodingError, Result},
  model_structure::{
    model::PredictiveCodingModel,
    model_utils::{
      create_from_config,
      load_model_snapshot
    }
  }
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum TrainingStrategy {
  SingleThread,
  MiniBatch { batch_size: u32 }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum ModelSource {
  Config(String),
  Snapshot(String)
}

pub fn load_model(model_source: &ModelSource) -> Result<PredictiveCodingModel> {
  match model_source {
    ModelSource::Config(config) => create_from_config(config),
    ModelSource::Snapshot(path) => load_model_snapshot(path)
  }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum DataSetSource {
  MNIST { input_idx_file: String, output_idx_file: String }
}

pub fn load_dataset(dataset_source: &DataSetSource) -> Result<TrainingDataset> {
  match dataset_source {
    DataSetSource::MNIST {
      input_idx_file,
      output_idx_file,
    } => load_mnist(input_idx_file, output_idx_file),
  }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct TrainConfig {
  /// Path to model config file or snapshot
  pub model_source: ModelSource,

  /// The dataset to train on
  pub dataset: DataSetSource,

  /// What training strategy to use
  pub training_strategy: TrainingStrategy,

  /// Training steps
  pub training_steps: u32,
  /// Report interval
  pub report_interval: u32,
  /// Snapshot interval
  pub snapshot_interval: u32,
}

/// Check the model and datased shapes for compatibility.
pub fn validate_model_and_dataset_shapes(
  model: &PredictiveCodingModel,
  data: &TrainingDataset
) -> Result<()> {
  if data.dataset_size == 0 {
    return Err(PredictiveCodingError::validation("Dataset is empty"));
  }

  let layer_sizes: Vec<usize> = model.get_layer_sizes();
  let Some(model_input_size) = layer_sizes.first().copied() else {
    return Err(PredictiveCodingError::validation("Model has no layers"));
  };
  let Some(model_output_size) = layer_sizes.last().copied() else {
    return Err(PredictiveCodingError::validation("Model has no layers"));
  };

  if data.inputs.shape()[1] != model_input_size {
    return Err(PredictiveCodingError::validation(format!(
      "Model input size {} does not match dataset input size {}",
      model_input_size,
      data.inputs.shape()[1]
    )));
  }

  if data.labels.shape()[1] != model_output_size {
    return Err(PredictiveCodingError::validation(format!(
      "Model output size {} does not match dataset output size {}",
      model_output_size,
      data.labels.shape()[1]
    )));
  }

  Ok(())
}

pub fn validate_training_config(training_config: &TrainConfig) -> Result<()> {
  match training_config.training_strategy {
    TrainingStrategy::SingleThread => Ok(()),
    TrainingStrategy::MiniBatch { batch_size } if batch_size > 0 => Ok(()),
    TrainingStrategy::MiniBatch { .. } => Err(PredictiveCodingError::validation(
      "Mini-batch training requires batch_size > 0"
    )),
  }
}

pub fn load_training_config(config_path: &str) -> Result<TrainConfig> {
  let config_str = std::fs::read_to_string(config_path)
    .map_err(|source| PredictiveCodingError::io("read training config", config_path, source))?;
  let training_config: TrainConfig = serde_json::from_str(&config_str)
    .map_err(|source| PredictiveCodingError::json_deserialize(config_path, source))?;

  Ok(training_config)
}

pub fn save_training_config(config: &TrainConfig, output_path: &str) -> Result<()> {
  let file = std::fs::File::create(output_path)
    .map_err(|source| PredictiveCodingError::io("create training config", output_path, source))?;

  serde_json::to_writer_pretty(file, config)
    .map_err(|source| PredictiveCodingError::json_serialize(output_path, source))?;

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  use ndarray::Array2;
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

  fn tiny_model() -> PredictiveCodingModel {
    PredictiveCodingModel::new(&crate::model_structure::model::PredictiveCodingModelConfig {
      layer_sizes: vec![4, 10],
      alpha: 0.01,
      gamma: 0.05,
      convergence_threshold: 0.0,
      convergence_steps: 1,
      activation_function: crate::model_structure::model_utils::ActivationFunction::Relu,
    })
  }

  fn tiny_dataset(input_size: usize, output_size: usize) -> TrainingDataset {
    TrainingDataset {
      dataset_size: 1,
      input_size,
      output_size,
      inputs: Array2::zeros((1, input_size)),
      labels: Array2::zeros((1, output_size)),
    }
  }

  #[test]
  fn load_training_config_parses_expected_json_shape() {
    let temp_dir = TempDir::new("training_config_parse");
    let config_path = temp_dir.join("training_config.json");
    write_file(
      &config_path,
      r#"{
  "model_source": {
    "Snapshot": "test_data/model_snapshot_tiny.json"
  },
  "dataset": {
    "MNIST": {
      "input_idx_file": "test_data/mnist/train-images-idx3-ubyte",
      "output_idx_file": "test_data/mnist/train-labels-idx1-ubyte"
    }
  },
  "training_strategy": {
    "MiniBatch": {
      "batch_size": 4
    }
  },
  "training_steps": 12,
  "report_interval": 3,
  "snapshot_interval": 6
}"#
    );

    let actual: TrainConfig = load_training_config(config_path.to_str().unwrap()).unwrap();
    let expected: TrainConfig = TrainConfig {
      model_source: ModelSource::Snapshot(String::from("test_data/model_snapshot_tiny.json")),
      dataset: DataSetSource::MNIST {
        input_idx_file: String::from("test_data/mnist/train-images-idx3-ubyte"),
        output_idx_file: String::from("test_data/mnist/train-labels-idx1-ubyte"),
      },
      training_strategy: TrainingStrategy::MiniBatch { batch_size: 4 },
      training_steps: 12,
      report_interval: 3,
      snapshot_interval: 6,
    };

    assert_eq!(actual, expected);
  }

  #[test]
  fn load_training_config_rejects_missing_required_fields() {
    let temp_dir: TempDir = TempDir::new("training_config_missing_field");
    let config_path = temp_dir.join("training_config.json");
    write_file(
      &config_path,
      r#"{
  "model_source": {
    "Config": "test_data/model_config_tiny.json"
  },
  "dataset": {
    "MNIST": {
      "input_idx_file": "test_data/mnist/train-images-idx3-ubyte",
      "output_idx_file": "test_data/mnist/train-labels-idx1-ubyte"
    }
  },
  "training_strategy": "SingleThread",
  "training_steps": 4,
  "snapshot_interval": 1
}"#
    );

    // This should fail because the report_interval field is missing.
    let result = load_training_config(config_path.to_str().unwrap());

    assert!(result.is_err());
  }

  #[test]
  fn load_training_config_error_includes_path() {
    let temp_dir = TempDir::new("training_config_missing_file");
    let config_path = temp_dir.join("missing_training_config.json");

    let error = load_training_config(config_path.to_str().unwrap()).unwrap_err();

    assert!(error.to_string().contains(&config_path.display().to_string()));
  }

  #[test]
  fn validate_model_and_dataset_shapes_rejects_input_mismatch() {
    let model: PredictiveCodingModel = tiny_model();
    let dataset: TrainingDataset = tiny_dataset(3, 10);

    let result = validate_model_and_dataset_shapes(&model, &dataset);

    match result {
      Err(PredictiveCodingError::Validation { message }) => {
        assert_eq!(message, "Model input size 4 does not match dataset input size 3");
      }
      other => panic!("expected validation error, got {other:?}"),
    }
  }

  #[test]
  fn validate_model_and_dataset_shapes_rejects_output_mismatch() {
    let model = tiny_model();
    let dataset = tiny_dataset(4, 9);

    let result = validate_model_and_dataset_shapes(&model, &dataset);

    match result {
      Err(PredictiveCodingError::Validation { message }) => {
        assert_eq!(message, "Model output size 10 does not match dataset output size 9");
      }
      other => panic!("expected validation error, got {other:?}"),
    }
  }

  #[test]
  fn validate_training_config_rejects_zero_batch_size() {
    let config = TrainConfig {
      model_source: ModelSource::Config(String::from("unused.json")),
      dataset: DataSetSource::MNIST {
        input_idx_file: String::from("unused-images.idx"),
        output_idx_file: String::from("unused-labels.idx"),
      },
      training_strategy: TrainingStrategy::MiniBatch { batch_size: 0 },
      training_steps: 1,
      report_interval: 0,
      snapshot_interval: 0,
    };

    let result = validate_training_config(&config);

    match result {
      Err(PredictiveCodingError::Validation { message }) => {
        assert_eq!(message, "Mini-batch training requires batch_size > 0");
      }
      other => panic!("expected validation error, got {other:?}"),
    }
  }
}
