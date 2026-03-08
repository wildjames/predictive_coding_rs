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
  data: &TrainingDataset
) -> Result<()> {
  // I could check this before constructing the model, and use a config. But, this is simpler to use.
  if data.dataset_size == 0 {
    return Err(PredictiveCodingError::validation("Dataset is empty"));
  }

  let layer_sizes: Vec<usize> = model.get_layer_sizes();
  let Some(model_input_size) = layer_sizes.first().copied() else {
    return Err(PredictiveCodingError::validation("Model has no layers"));
  };
  let Some(model_output_size) = layer_sizes.last().copied() else {
    return Err(PredictiveCodingError::validation("Model has no layers"));
  };

  if data.inputs.shape()[1] != model_input_size {
    return Err(PredictiveCodingError::validation(format!(
      "Model input size {} does not match dataset input size {}",
      model_input_size,
      data.inputs.shape()[1]
    )));
  }

  if data.labels.shape()[1] != model_output_size {
    return Err(PredictiveCodingError::validation(format!(
      "Model output size {} does not match dataset output size {}",
      model_output_size,
      data.labels.shape()[1]
    )));
  }

  Ok(())
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::training::configuration::{DataSetSource, ModelSource};
  use ndarray::Array2;

  fn tiny_model() -> PredictiveCodingModel {
    PredictiveCodingModel::new(&crate::model_structure::model::PredictiveCodingModelConfig {
      layer_sizes: vec![4, 10],
      alpha: 0.01,
      gamma: 0.05,
      convergence_threshold: 0.0,
      convergence_steps: 1,
      activation_function: crate::model_structure::model_utils::ActivationFunction::Relu,
    })
  }

  fn tiny_dataset(input_size: usize, output_size: usize) -> TrainingDataset {
    TrainingDataset {
      dataset_size: 1,
      input_size,
      output_size,
      inputs: Array2::zeros((1, input_size)),
      labels: Array2::zeros((1, output_size)),
    }
  }

  #[test]
  fn validate_model_and_dataset_shapes_rejects_input_mismatch() {
    let model: PredictiveCodingModel = tiny_model();
    let dataset: TrainingDataset = tiny_dataset(3, 10);

    let result = validate_model_and_dataset_shapes(&model, &dataset);

    match result {
      Err(PredictiveCodingError::Validation { message }) => {
        assert_eq!(message, "Model input size 4 does not match dataset input size 3");
      }
      other => panic!("expected validation error, got {other:?}"),
    }
  }

  #[test]
  fn validate_model_and_dataset_shapes_rejects_output_mismatch() {
    let model = tiny_model();
    let dataset = tiny_dataset(4, 9);

    let result = validate_model_and_dataset_shapes(&model, &dataset);

    match result {
      Err(PredictiveCodingError::Validation { message }) => {
        assert_eq!(message, "Model output size 10 does not match dataset output size 9");
      }
      other => panic!("expected validation error, got {other:?}"),
    }
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

    match result {
      Err(PredictiveCodingError::Validation { message }) => {
        assert_eq!(message, "Mini-batch training requires batch_size > 0");
      }
      other => panic!("expected validation error, got {other:?}"),
    }
  }
}
