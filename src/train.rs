//! Train a predictive coding model on the MNIST dataset. The model architecture and training hyperparameters are defined in a config file, and the trained model is saved to disk at regular intervals during training. The final trained model is also saved to disk at the end of training. Exactly one of model_config and model_snapshot should be provided.

use predictive_coding::{
  data_handling::{
    data_handler::TrainingDataset,
    mnist::load_mnist
  },
  model_structure::model::PredictiveCodingModel,
  training::{
    cpu_train,
    train_handler::{
      TrainingHandler,
      run_supervised_training_loop
    },
    utils::{
      TrainConfig,
      TrainingStrategy,
      load_model,
      load_training_config
    }
  },
  utils::logging
};

use clap::Parser;
use tracing::info;


/// The training program accepts a config file as a command line argument, which specifies the model architecture and training hyperparameters. The program then loads the MNIST dataset, builds the model, and runs the training loop, saving the trained model to disk at regular intervals during training and at the end of training.
#[derive(Parser)]
struct TrainArgs {
  /// Path to training config file
  #[arg()]
  config: String
}

/// Entry point for loading data, building the model, and running training.
fn main() {
  logging::setup_tracing(false);

  let args: TrainArgs = TrainArgs::parse();
  let training_config: TrainConfig = load_training_config(&args.config);

  let data: TrainingDataset = load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  info!(
    "Loaded the MNIST dataset. I have {} images",
    data.dataset_size
  );

  // Build the model
  let model: PredictiveCodingModel = load_model(&training_config.model_source);
  info!(
    "Created the model with layer sizes {:?}",
    model.get_layer_sizes()
  );

  let file_output_prefix: String = format!(
    "data/model_{}/model",
    chrono::Utc::now().timestamp()
  );

  let mut handler: Box<dyn TrainingHandler> = match training_config.training_strategy {
    TrainingStrategy::SingleThread => Box::new(cpu_train::SingleThreadTrainHandler::new(
      training_config.clone(),
      model,
      data,
      file_output_prefix.clone()
    )),
    TrainingStrategy::MiniBatch { batch_size } => Box::new(cpu_train::BatchTrainHandler::new(
      training_config.clone(),
      model,
      data,
      file_output_prefix.clone(),
      batch_size,
    ))
  };

  // Execute the loop, training the model
  run_supervised_training_loop(handler.as_mut());
}
