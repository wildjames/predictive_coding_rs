use crate::model_structure::{
  model::PredictiveCodingModel,
  model_utils::{
    create_from_config,
    load_model_snapshot
  }
};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum TrainingStrategy {
  SingleThread,
  MiniBatch { batch_size: u32 }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum ModelSource {
  Config(String),
  Snapshot(String)
}

pub fn load_model(model_source: &ModelSource) -> PredictiveCodingModel {
  match model_source {
    ModelSource::Config(config) => create_from_config(config),
    ModelSource::Snapshot(path) => load_model_snapshot(path)
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainConfig {
  /// Path to model config file or snapshot
  pub model_source: ModelSource,

  /// What training strategy to use
  pub training_strategy: TrainingStrategy,

  /// Training steps
  pub training_steps: u32,
  /// Report interval (default to 1_000)
  pub report_interval: u32,
  /// Snapshot interval (default to 10_000)
  pub snapshot_interval: u32,
}

pub fn load_training_config(config_path: &str) -> TrainConfig {
  let config_str = std::fs::read_to_string(config_path).expect(
    "Failed to read training config file. Please ensure the file exists and is readable."
  );
  let training_config: TrainConfig = serde_json::from_str(&config_str).expect(
    "Failed to parse training config file. Please ensure it is a valid JSON file with the correct fields."
  );

  training_config
}

pub fn save_training_config(config: &TrainConfig, output_path: &str) {
  serde_json::to_writer_pretty(
    std::fs::File::create(output_path).expect(
      "Failed to create training config output file. Please ensure the path is correct and writable."
    ),
    config
  ).expect(
    "Failed to write training config to file. Please ensure the output path is correct and writable."
  );
}
