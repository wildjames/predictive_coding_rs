//! Train a predictive coding model on the MNIST dataset. The model architecture and training hyperparameters are defined in a config file, and the trained model is saved to disk at regular intervals during training. The final trained model is also saved to disk at the end of training. Exactly one of model_config and model_snapshot should be provided.

use predictive_coding::{
  data_handling::{
    data_handler::TrainingDataset,
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
      load_dataset,
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

  let data: TrainingDataset = load_dataset(&training_config.dataset);
  info!(
    "Loaded the dataset. I have {} samples",
    data.dataset_size
  );

  // Build the model
  let model: PredictiveCodingModel = load_model(&training_config.model_source);
  info!(
    "Created the model with layer sizes {:?}",
    model.get_layer_sizes()
  );

  // Check that the data and model are compatible
  let layer_sizes = model.get_layer_sizes();
  let model_input_size = layer_sizes.first().unwrap();
  let model_output_size = layer_sizes.last().unwrap();
  if data.inputs.shape()[1] != *model_input_size {
    panic!("Model input size {} does not match dataset input size {}", model_input_size, data.inputs.shape()[1]);
  }
  if data.labels.shape()[1] != *model_output_size {
    panic!("Model output size {} does not match dataset output size {}", model_output_size, data.labels.shape()[1]);
  }

  // Where to put stuff
  let file_output_prefix: String = format!(
    "data/model_{}/model",
    chrono::Utc::now().timestamp()
  );

  // The handler orchestrated the training process by providing hook functions to the training loop.
  // Choose the correct one for this config.
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
