use crate::{
  data_handling::data_handler::TrainingDataset,
  model_structure::model::PredictiveCodingModel,
  training::{
    cpu_train,
    train_handler::TrainingHandler,
    utils::{
      TrainConfig, TrainingStrategy, load_dataset, load_model, load_training_config, validate_model_and_dataset_shapes
    }
  }
};

use tracing::{info};


fn get_handler(
  training_config: &TrainConfig,
  data: &TrainingDataset,
  file_output_prefix: &str
) -> Box<dyn TrainingHandler> {
  let model: PredictiveCodingModel = load_model(&training_config.model_source);

  match training_config.training_strategy {
    TrainingStrategy::SingleThread => Box::new(cpu_train::SingleThreadTrainHandler::new(
      training_config.clone(),
      model,
      data.clone(),
      file_output_prefix.to_string()
    )),
    TrainingStrategy::MiniBatch { batch_size } => Box::new(cpu_train::BatchTrainHandler::new(
      training_config.clone(),
      model,
      data.clone(),
      file_output_prefix.to_string(),
      batch_size,
    ))
  }
}

/// Sets up a training run handler based on the provided config path and output prefix.
/// The handler will orchestrate the training process by providing hook functions to the training loop.
pub fn setup_training_run_handler(
  config: String,
  output_prefix: String
) -> Box<dyn TrainingHandler> {

  let training_config: TrainConfig = load_training_config(&config);

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

  if let Err(message) = validate_model_and_dataset_shapes(&model, &data) {
    panic!("{}", message);
  }

  // Make sure that the output directory exists
  let output_dir = std::path::Path::new(&output_prefix).parent().unwrap();
  info!("Saving training artifacts to {}", output_dir.display());
  std::fs::create_dir_all(output_dir).expect("Failed to create output directory for training artifacts");

  // The handler orchestrated the training process by providing hook functions to the training loop.
  // Choose the correct one for this config.
  get_handler(
    &training_config,
    &data,
    &output_prefix
  )
}
