//! Train a predictive coding model on the MNIST dataset. The model architecture and training hyperparameters are defined in a config file, and the trained model is saved to disk at regular intervals during training. The final trained model is also saved to disk at the end of training. Exactly one of model_config and model_snapshot should be provided.

use predictive_coding::{
  data_handling::data_handler,
  model_structure::{
    model::PredictiveCodingModel,
    model_utils::{
      create_from_config,
      load_model,
      save_model
    }
  },
  training::{
    cpu_train,
    train_handler::{
      TrainingHandler,
      run_supervised_training_loop
    },
    utils::{
      TrainConfig,
      TrainingStrategy,
      load_training_config
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
  let model: PredictiveCodingModel = if let Some(config) = &training_config.model_config {
    create_from_config(config)
  } else if let Some(snapshot) = &training_config.snapshot {
    load_model(snapshot)
  } else {
    // Cover compiler error
    panic!("Exactly one of config and snapshot must be provided");
  };

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
    TrainingStrategy::MiniBatch => Box::new(cpu_train::BatchTrainHandler::new(
      training_config.clone(),
      model,
      data,
      file_output_prefix.clone()
    ))
  };

  // Execute the loop, training the model
  run_supervised_training_loop(
    handler.as_mut(),
    &file_output_prefix
  );

  info!("Finished training, saving final model");
  save_model(
    handler.get_model(),
    &format!("{}_{}", file_output_prefix, "final.json")
  );
}
