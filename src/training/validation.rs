use crate::{
  data_handling::data_handler::TrainingDataset,
  error::{
    PredictiveCodingError,
    Result
  },
  model_structure::model::PredictiveCodingModel,
  training::configuration::{
    TrainConfig,
    TrainingStrategy
  }
};

#[cfg(test)]
use std::sync::Arc;

pub fn validate_training_config(training_config: &TrainConfig) -> Result<()> {
  match training_config.training_strategy {
    TrainingStrategy::SingleThread => Ok(()),
    TrainingStrategy::MiniBatch { batch_size } if batch_size > 0 => Ok(()),
    TrainingStrategy::MiniBatch { .. } => Err(PredictiveCodingError::validation(
      "Mini-batch training requires batch_size > 0"
    )),
  }
}

/// Check the model and datased shapes for compatibility.
pub fn validate_model_and_dataset_shapes(
  model: &PredictiveCodingModel,
  data: &dyn TrainingDataset
) -> Result<()> {
  // I could check this before constructing the model, and use a config. But, this is simpler to use.
  if data.get_dataset_size() == 0 {
    return Err(PredictiveCodingError::validation("Dataset is empty"));
  }

  let layer_sizes: Vec<usize> = model.get_layer_sizes();
  if layer_sizes.is_empty() {
    return Err(PredictiveCodingError::validation("Model has no layers"));
  }
  let model_input_size = layer_sizes[0];
  let model_output_size = *layer_sizes.last().unwrap();

  if data.get_input_size() != model_input_size {
    return Err(PredictiveCodingError::validation(format!(
      "Model input size {} does not match dataset input size {}",
      model_input_size,
      data.get_input_size()
    )));
  }

  if data.get_output_size() != model_output_size {
    return Err(PredictiveCodingError::validation(format!(
      "Model output size {} does not match dataset output size {}",
      model_output_size,
      data.get_output_size()
    )));
  }

  Ok(())
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    test_utils::{DummyTrainingDataset, tiny_relu_model},
    training::configuration::{
      DataSetSource,
      ModelSource
    }
  };

  fn tiny_dataset(input_size: usize, output_size: usize) -> Arc<dyn TrainingDataset> {
    Arc::new(DummyTrainingDataset::zeros(1, input_size, output_size))
  }

  #[test]
  fn validate_model_and_dataset_shapes_rejects_input_mismatch() {
    let model: PredictiveCodingModel = tiny_relu_model();
    let dataset: Arc<dyn TrainingDataset> = tiny_dataset(3, 10);

    let result = validate_model_and_dataset_shapes(&model, dataset.as_ref());

    assert!(matches!(
      result,
      Err(PredictiveCodingError::Validation { message })
        if message == "Model input size 4 does not match dataset input size 3"
    ));
  }

  #[test]
  fn validate_model_and_dataset_shapes_rejects_output_mismatch() {
    let model = tiny_relu_model();
    let dataset = tiny_dataset(4, 9);

    let result = validate_model_and_dataset_shapes(&model, dataset.as_ref());

    assert!(matches!(
      result,
      Err(PredictiveCodingError::Validation { message })
        if message == "Model output size 10 does not match dataset output size 9"
    ));
  }

  #[test]
  fn validate_training_config_rejects_zero_batch_size() {
    let config = TrainConfig {
      model_source: ModelSource::Config(String::from("unused.json")),
      dataset: DataSetSource::MNIST {
        input_idx_file: String::from("unused-images.idx"),
        output_idx_file: String::from("unused-labels.idx"),
      },
      training_strategy: TrainingStrategy::MiniBatch { batch_size: 0 },
      training_steps: 1,
      report_interval: 0,
      snapshot_interval: 0,
    };

    let result = validate_training_config(&config);

    assert!(matches!(
      result,
      Err(PredictiveCodingError::Validation { message })
        if message == "Mini-batch training requires batch_size > 0"
    ));
  }

  #[test]
  fn validate_training_config_accepts_supported_strategies() {
    let single_thread = TrainConfig {
      model_source: ModelSource::Config(String::from("unused.json")),
      dataset: DataSetSource::MNIST {
        input_idx_file: String::from("unused-images.idx"),
        output_idx_file: String::from("unused-labels.idx"),
      },
      training_strategy: TrainingStrategy::SingleThread,
      training_steps: 1,
      report_interval: 0,
      snapshot_interval: 0,
    };
    let minibatch = TrainConfig {
      training_strategy: TrainingStrategy::MiniBatch { batch_size: 2 },
      ..single_thread.clone()
    };

    assert!(validate_training_config(&single_thread).is_ok());
    assert!(validate_training_config(&minibatch).is_ok());
  }
}
