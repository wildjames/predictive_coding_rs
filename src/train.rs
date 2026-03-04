//! Training binary for the predictive coding model.

use predictive_coding::{
  data_handling::data_handler,
  model::{
    model::{PredictiveCodingModel, PredictiveCodingModelConfig},
    model_utils::{create_from_config, load_model, save_model, save_model_config}
  },
  training::train_model_handler,
  utils::logging
};

use tracing::info;
use clap::Parser;

/// Train a predictive coding model on the MNIST dataset. The model architecture and training hyperparameters are defined in a config file, and the trained model is saved to disk at regular intervals during training. The final trained model is also saved to disk at the end of training. Exactly one of --config and --snapshot should be provided.
#[derive(Parser)]
struct TrainArgs {
  /// Path to model config file
  #[arg(short, long)]
  config: Option<String>,
  /// Path to model snapshot file
  #[arg(short, long)]
  snapshot: Option<String>,

  /// Training steps
  #[arg(long, default_value_t = 100_000)]
  training_steps: u32,
  /// Report interval (default to 1_000)
  #[arg(long, default_value_t = 1_000)]
  report_interval: u32,
  /// Snapshot interval (default to 10_000)
  #[arg(long, default_value_t = 10_000)]
  snapshot_interval: u32,
}

/// Entry point for loading data, building the model, and running training.
fn main() {
  logging::setup_tracing(false);

  let args = TrainArgs::parse();

  // Assert that we have only config or snapshot
  if args.config.is_some() == args.snapshot.is_some() {
    panic!("Exactly one of --config and --snapshot must be provided");
  }

  let data: data_handler::ImagesBWDataset = data_handler::load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST dataset. I have {} images",
    data.num_images
  );

  // Build the model
  let mut model: PredictiveCodingModel = if let Some(config) = args.config {
    create_from_config(&config)
  } else if let Some(snapshot) = args.snapshot {
    load_model(&snapshot)
  } else {
    // Cover compiler error
    panic!("Exactly one of config and snapshot must be provided");
  };

  info!(
    "Training parameters:\n\ttraining steps: {}\n\tsnapshot interval: {}",
    args.training_steps,
    args.snapshot_interval
  );

  let config: PredictiveCodingModelConfig = model.get_config();
  info!(
    "Model architecture:\n\tlayer sizes: {:?}\n\tgamma: {}\n\talpha: {}\n\tactivation function: {:?}\n\tconvergence steps: {}\n\tconvergence threshold: {}",
    config.layer_sizes,
    config.gamma,
    config.alpha,
    config.activation_function,
    config.convergence_steps,
    config.convergence_threshold
  );

  let snapshot_output_prefix: String = format!(
    "data/model_{}/model",
    chrono::Utc::now().timestamp()
  );
  // Write the config and training params to a file
  save_model_config(&model.get_config(), &format!("{}_config.json", snapshot_output_prefix));
  serde_json::to_writer_pretty(
    std::fs::File::create(format!("{}_training_params.json", snapshot_output_prefix)).unwrap(),
    &serde_json::json!({
      "training_steps": args.training_steps,
      "report_interval": args.report_interval,
      "snapshot_interval": args.snapshot_interval,
    })
  ).unwrap();

  train_model_handler::train(
    &mut model,
    &data,
    args.training_steps,
    args.report_interval,
    args.snapshot_interval,
    &snapshot_output_prefix,
  );

  info!("Finished training, saving final model");
  save_model(&model, &format!("{}_{}", snapshot_output_prefix, "final.json"));
}
