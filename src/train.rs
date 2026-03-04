//! Training binary for the predictive coding model.

use predictive_coding::{
  data_handling::data_handler,
  model::{
    model::{PredictiveCodingModel},
    model_utils::{create_from_config, save_model_config, save_model}
  },
  training::train_model_handler,
  utils::logging
};

use tracing::info;


/// Entry point for loading data, building the model, and running training.
fn main() {
  logging::setup_tracing(false);

  let data: data_handler::ImagesBWDataset = data_handler::load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST dataset. I have {} images",
    data.num_images
  );

  // Training params
  let report_interval: u32 = 100;
  let snapshot_interval: u32 = 10000;
  let training_steps: u32 = data.num_images as u32; // full dataset
  let convergence_steps: u32 = 50;
  let convergence_threshold: f32 = 1e-5;

  info!(
    "Training hyperparameters:\n\tconvergence steps: {}\n\tconvergence threshold: {}\n\tsnapshot interval: {}",
    convergence_steps,
    convergence_threshold,
    snapshot_interval
  );

  // Build the model
  let fname: &str = "data/model_config.json";
  let mut model: PredictiveCodingModel = create_from_config(fname);

  info!(
    "Model architecture: {:?}\n\tgamma: {}\n\talpha: {}",
    model.get_layer_sizes(),
    model.get_gamma(),
    model.get_alpha()
  );

  let snapshot_output_prefix: String = format!(
    "data/model_snapshots_{}/model",
    chrono::Utc::now().timestamp()
  );
  // Write the config to a file
  save_model_config(&model.get_config(), &format!("{}_config.json", snapshot_output_prefix));

  train_model_handler::train(
    &mut model,
    &data,
    training_steps,
    report_interval,
    snapshot_interval,
    &snapshot_output_prefix,
  );

  info!("Finished training, saving final model");
  save_model(&model, &format!("{}_{}", snapshot_output_prefix, "final.json"));
}
