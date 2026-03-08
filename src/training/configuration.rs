use crate::{
  data_handling::{
    data_handler::TrainingDataset,
    mnist::load_mnist
  },
  error::{PredictiveCodingError, Result},
  model_structure::{
    model::PredictiveCodingModel,
    configuration::{
      create_from_config,
      load_model_snapshot
    }
  }
};

use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

pub fn load_dataset(dataset_source: &DataSetSource) -> Result<Arc<dyn TrainingDataset>> {
  let data = match dataset_source {
    DataSetSource::MNIST {
      input_idx_file,
      output_idx_file,
    } => load_mnist(input_idx_file, output_idx_file),
  };

  Ok(Arc::new(data?))
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct TrainConfig {
  /// Path to model config file or snapshot
  pub model_source: ModelSource,

  /// The dataset to train on
  pub training_dataset: DataSetSource,
  /// An optional dataset can be given, which will later be used as the
  /// default evaluation dataset to see how well the model performs.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub evaluation_dataset: Option<DataSetSource>,

  /// What training strategy to use
  pub training_strategy: TrainingStrategy,

  /// Training steps
  pub training_steps: u32,
  /// Report interval
  pub report_interval: u32,
  /// Snapshot interval
  pub snapshot_interval: u32,
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
  use crate::test_utils::{TempDir, write_file};

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
  "training_dataset": {
    "MNIST": {
      "input_idx_file": "test_data/mnist/train-images-idx3-ubyte",
      "output_idx_file": "test_data/mnist/train-labels-idx1-ubyte"
    }
  },
  "evaluation_dataset": {
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
      training_dataset: DataSetSource::MNIST {
        input_idx_file: String::from("test_data/mnist/train-images-idx3-ubyte"),
        output_idx_file: String::from("test_data/mnist/train-labels-idx1-ubyte"),
      },
      evaluation_dataset: Some(DataSetSource::MNIST {
        input_idx_file: String::from("test_data/mnist/train-images-idx3-ubyte"),
        output_idx_file: String::from("test_data/mnist/train-labels-idx1-ubyte"),
      }),
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
}
