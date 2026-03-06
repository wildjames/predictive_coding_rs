//! Train a predictive coding model on the MNIST dataset. The model architecture and training hyperparameters are defined in a config file, and the trained model is saved to disk at regular intervals during training. The final trained model is also saved to disk at the end of training. Exactly one of model_config and model_snapshot should be provided.

use predictive_coding::{
  data_handling::data_handler,
  model_structure::{
    model::{
      PredictiveCodingModel,
      PredictiveCodingModelConfig
    },
    model_utils::{
      create_from_config,
      load_model,
      save_model,
      save_model_config
    }
  },
  training::{
    cpu_train,
    utils::{
      TrainConfig,
      load_training_config,
      save_training_config
    }
  },
  utils::logging
};

use clap::Parser;
use tracing::info;


#[derive(Parser)]
struct TrainArgs {
  // Path to training config file
  #[arg(short, long)]
  config: String
}

/// Entry point for loading data, building the model, and running training.
fn main() {
  logging::setup_tracing(false);

  let args = TrainArgs::parse();
  let training_config: TrainConfig = load_training_config(&args.config);

  let data: data_handler::ImagesBWDataset = data_handler::load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST dataset. I have {} images",
    data.num_images
  );

  // Build the model
  let mut model: PredictiveCodingModel = if let Some(config) = &training_config.model_config {
    create_from_config(config)
  } else if let Some(snapshot) = &training_config.snapshot {
    load_model(snapshot)
  } else {
    // Cover compiler error
    panic!("Exactly one of config and snapshot must be provided");
  };

  info!(
    "Training parameters:\n\ttraining steps: {}\n\treporting interval: {}\n\tsnapshot interval: {}",
    training_config.training_steps,
    training_config.report_interval,
    training_config.snapshot_interval
  );

  let loaded_model_config: PredictiveCodingModelConfig = model.get_config();
  info!(
    "Model architecture:\n\tlayer sizes: {:?}\n\tgamma: {}\n\talpha: {}\n\tactivation function: {:?}\n\tconvergence steps: {}\n\tconvergence threshold: {}",
    loaded_model_config.layer_sizes,
    loaded_model_config.gamma,
    loaded_model_config.alpha,
    loaded_model_config.activation_function,
    loaded_model_config.convergence_steps,
    loaded_model_config.convergence_threshold
  );

  let snapshot_output_prefix: String = format!(
    "data/model_{}/model",
    chrono::Utc::now().timestamp()
  );

  // Write the config and training params to a file
  save_model_config(
    &model.get_config(),
    &format!("{}_config.json", snapshot_output_prefix)
  );
  save_training_config(
    &training_config,
    &format!("{}_training_config.json", snapshot_output_prefix)
  );

  if training_config.mini_batch_enabled {
    info!("Multithreading enabled for training");
    let batch_size: u32 = training_config.batch_size.expect("Batch size must be provided when mini-batch training is enabled");

    cpu_train::mini_batch_train(
      &mut model,
      &data,
      batch_size,
      training_config.training_steps,
      training_config.report_interval,
      training_config.snapshot_interval,
      &snapshot_output_prefix
    );
  } else {
    info!("Multithreading disabled for training");
    cpu_train::train(
      &mut model,
      &data,
      training_config.training_steps,
      training_config.report_interval,
      training_config.snapshot_interval,
      &snapshot_output_prefix,
    );
  }

  info!("Finished training, saving final model");
  save_model(&model, &format!("{}_{}", snapshot_output_prefix, "final.json"));
}
