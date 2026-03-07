//! Train a predictive coding model on the MNIST dataset. The model architecture and training hyperparameters are defined in a config file, and the trained model is saved to disk at regular intervals during training. The final trained model is also saved to disk at the end of training. Exactly one of model_config and model_snapshot should be provided.

use predictive_coding::{
  error::Result,
  training::{
    setup::setup_training_run_handler,
    train_handler::{
      TrainingHandler,
      run_supervised_training_loop
    }
  },
  utils::{logging, timestamp}
};

use clap::Parser;
use tracing::info;

/// The training program accepts a config file as a command line argument, which specifies the model architecture and training hyperparameters. The program then loads the MNIST dataset, builds the model, and runs the training loop, saving the trained model to disk at regular intervals during training and at the end of training.
#[derive(Parser)]
struct TrainArgs {
  /// Path to training config file
  #[arg()]
  config: String,

  /// Optional artifact output prefix. Defaults to `data/model_<timestamp>/model`.
  #[arg(long, default_value_t = format!("data/model_{}/model_", timestamp()))]
  output_prefix: String
}

/// Entry point for loading data, building the model, and running training.
fn main() -> Result<()> {
  logging::setup_tracing(false);

  info!("Starting training run");

  let args: TrainArgs = TrainArgs::parse();
  let mut handler: Box<dyn TrainingHandler> = setup_training_run_handler(args.config, args.output_prefix)?;

  // Execute the loop, training the model
  info!("Beginning training loop");
  run_supervised_training_loop(handler.as_mut())
}
