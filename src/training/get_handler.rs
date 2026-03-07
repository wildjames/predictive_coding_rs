use crate::{
  data_handling::data_handler::TrainingDataset,
  model_structure::model::PredictiveCodingModel,
  training::{
    cpu_train,
    train_handler::TrainingHandler,
    utils::{
      TrainConfig,
      TrainingStrategy,
      load_model
    }
  }
};


pub fn get_handler(
  training_config: TrainConfig,
  data: TrainingDataset,
  file_output_prefix: String
) -> Box<dyn TrainingHandler> {
  let model: PredictiveCodingModel = load_model(&training_config.model_source);

  match training_config.training_strategy {
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
  }
}
