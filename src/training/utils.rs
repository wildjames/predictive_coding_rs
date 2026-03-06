use crate::model_structure::{
  model::{PredictiveCodingModel}
};
use crate::data_handling::data_handler;

use serde::{Deserialize, Serialize};
use ndarray::Array1;

pub fn set_rand_input_and_output(
  model: &mut PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset
) {
  let rand_index = usize::from_ne_bytes(rand::random()) % data.num_images;

  // Normalise to the range 0..1
  let input_values: Array1<f32> = data.images
    .row(rand_index)
    .mapv(|x| x as f32 / 255.0)
    .to_owned();

  // One-hot output row with label value set to 1.0
  let output_label: usize = data.labels[rand_index] as usize;

  let output_layer_size: usize = *model.get_layer_sizes().last().unwrap();
  let mut output_values: Array1<f32> = Array1::zeros(output_layer_size);
  output_values[output_label] = 1.0;

  model.set_input(input_values);
  model.set_output(output_values);

}

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainConfig {
  /// Path to model config file
  pub model_config: Option<String>,
  /// Path to model snapshot file
  pub snapshot: Option<String>,

  /// Enable multithreading for training (default: false)
  pub multithread: bool,

  /// Training steps
  pub training_steps: u32,
  /// Mini batch size for multithreaded training (default: 8)
  pub batch_size: u32,
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


  // Assert that we have only config or snapshot
  if training_config.model_config.is_some() == training_config.snapshot.is_some() {
    panic!("Exactly one of --config and --snapshot must be provided");
  }

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
