
use crate::{
  data_handling::data_handler,
  error::Result,
  model_structure::{
    model::PredictiveCodingModel,
    model_utils::set_rand_input_and_output
  },
  training::{
    train_handler::TrainingHandler,
    configuration::TrainConfig,
  }
};

use ndarray::{Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
use tracing::{info, debug};

pub struct BatchTrainHandler {
  config: TrainConfig,
  model: PredictiveCodingModel,
  data: Arc<dyn data_handler::TrainingDataset>,
  file_output_prefix: String,
  batch_size: u32
}

impl BatchTrainHandler {
  pub fn new(
    config: TrainConfig,
    model: PredictiveCodingModel,
    data: Arc<dyn data_handler::TrainingDataset>,
    file_output_prefix: String,
    batch_size: u32
  ) -> Self {
    BatchTrainHandler {
      config,
      model,
      data,
      file_output_prefix,
      batch_size
    }
  }
}

impl TrainingHandler for BatchTrainHandler {

  fn get_config(&self) -> &TrainConfig {
    &self.config
  }
  fn get_model(&mut self) -> &mut PredictiveCodingModel {
    &mut self.model
  }
  fn get_data(&self) -> &dyn data_handler::TrainingDataset {
    self.data.as_ref()
  }
  fn get_file_output_prefix(&self) -> &String {
    &self.file_output_prefix
  }

  /// Report the training parameters and model config. Also save setup to file for future reference
  fn pre_training_hook(&mut self) -> Result<()> {
    info!("Starting training with mini-batch strategy");
    info!(
      "Mini batch params: batch size = {}",
      self.batch_size
    );
    Ok(())
  }

  /// Train batch_size models in parallel on different data, then compute the average weight update across them and apply it to the model.
  fn train_step(&mut self, _step: u32) -> Result<()> {
    // I'll iterate over this with Rayon to parallelise the batch
    let batch_indexes: Vec<u32> = (0..self.batch_size).collect();

    // Each element of the batch trains on a single sample, and we collect their weight changes as a result.
    // The batch weight changes will be a Vec of length batch_size, where each element is a Vec of length
    // num_layers, and each element of THAT is an array2 of the weight changes for the relevant layer.
    let batch_weight_changes: Vec<Vec<Array2<f32>>> = batch_indexes
      .into_par_iter()
      .map(|b| {
        debug!("Training batch element on a single example: {}", b);

        let mut model_clone: PredictiveCodingModel = self.model.clone();
        set_rand_input_and_output(&mut model_clone, self.data.as_ref());
        model_clone.reinitialise_latents();

        // Train on this example until convergence.
        model_clone.converge_values();
        model_clone.update_weights();

        // Weight updates (Array2) for each layer in the model
        model_clone.compute_weight_updates()
      })
      .collect();

    // Average the weight changes across the batch. Sum outer Vec
    let sum_batch_weight_changes: Vec<Array2<f32>> = batch_weight_changes
      .into_iter()
      .reduce(|acc, weight_changes| {
        acc.into_iter()
          .zip(weight_changes)
          .map(|(sum_weight_change, weight_change)| sum_weight_change + weight_change)
          .collect()
      })
      .unwrap();

    let avg_batch_weight_changes: Vec<Array2<f32>> = sum_batch_weight_changes
      .into_iter()
      .map(|sum_weight_change: Array2<f32>| sum_weight_change / self.batch_size as f32)
      .collect();

    // Apply the average weight changes to the model.
    self.model.apply_weight_updates(avg_batch_weight_changes);

    Ok(())
  }


  fn report_hook(&mut self, step: u32) -> Result<()> {
    // The mini batch model is cloned for each batch element, so the main model never gets inference run on it
    // So, do that in the reporting step to get a sense of how the model is doing on the data.
    self.model.reinitialise_latents();
    set_rand_input_and_output(&mut self.model, self.data.as_ref());
    self.model.converge_values();

    let energy: f32 = self.model.read_total_energy();
    info!("Step {}: Current model state: energy = {:.2}", step, energy);
    Ok(())
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  use crate::model_structure::{
    model::PredictiveCodingModelConfig,
    maths::ActivationFunction
  };
  use crate::training::configuration::TrainingStrategy;
  use crate::training::cpu::singlethreaded::SingleThreadTrainHandler;
  use ndarray::{array, Array1, Array2};

  struct DummyTrainingDataset {
    dataset_size: usize,
    input_size: usize,
    output_size: usize,
    inputs: Array2<f32>,
    labels: Array2<f32>,
  }

  impl data_handler::TrainingDataset for DummyTrainingDataset {
    fn get_dataset_size(&self) -> usize {self.dataset_size}
    fn get_input_size(&self) -> usize {self.input_size}
    fn get_output_size(&self) -> usize {self.output_size}
    fn get_inputs(&self) -> &Array2<f32> {&self.inputs}
    fn get_labels(&self) -> &Array2<f32> {&self.labels}

    fn get_random_input(&self) -> Array1<f32> {
      self.get_input(0)
    }

    fn get_random_input_and_output(&self) -> (Array1<f32>, Array1<f32>) {
      (self.get_input(0), self.get_output(0))
    }

    fn get_input(&self, _index: usize) -> Array1<f32> {
      self.inputs.row(0).to_owned()
    }

    fn get_output(&self, _index: usize) -> Array1<f32> {
      self.labels.row(0).to_owned()
    }
  }

  fn dummy_config() -> TrainConfig {
    TrainConfig {
      model_source: crate::training::configuration::ModelSource::Snapshot(String::from("unused.json")),
      dataset: crate::training::configuration::DataSetSource::MNIST {
        input_idx_file: String::from("unused-images.idx"),
        output_idx_file: String::from("unused-labels.idx"),
      },
      training_strategy: TrainingStrategy::SingleThread,
      training_steps: 1,
      report_interval: 0,
      snapshot_interval: 0,
    }
  }

  fn tiny_model() -> PredictiveCodingModel {
    PredictiveCodingModel::new(&PredictiveCodingModelConfig {
      layer_sizes: vec![4, 10],
      alpha: 0.01,
      gamma: 0.05,
      convergence_threshold: 0.0,
      convergence_steps: 1,
      activation_function: ActivationFunction::Relu,
    })
  }

  fn tiny_dataset() -> Arc<dyn data_handler::TrainingDataset> {
    let mut labels: Array2<f32> = Array2::zeros((1, 10));
    labels.row_mut(0).assign(&array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let data: DummyTrainingDataset = DummyTrainingDataset {
      dataset_size: 1,
      input_size: 4,
      output_size: 10,
      inputs: array![[1.0, 0.0, 0.5, 0.25]],
      labels,
    };

    Arc::new(data)
  }

  fn assert_arrays_close(left: &Array2<f32>, right: &Array2<f32>, tolerance: f32) {
    assert_eq!(left.dim(), right.dim());
    for (left_value, right_value) in left.iter().zip(right.iter()) {
      assert!(
        (left_value - right_value).abs() <= tolerance,
        "expected {left_value} and {right_value} to be within {tolerance}"
      );
    }
  }

  #[test]
  fn minibatch_aggregation_matches_single_sample_update_on_deterministic_fixture() {
    let initial_model: PredictiveCodingModel = tiny_model();
    let dataset: Arc<dyn data_handler::TrainingDataset> = tiny_dataset();
    let config: TrainConfig = dummy_config();

    let mut single_handler: SingleThreadTrainHandler = SingleThreadTrainHandler::new(
      config.clone(),
      initial_model.clone(),
      Arc::clone(&dataset),
      String::from("unused/single"),
    );

    let mut batch_handler: BatchTrainHandler = BatchTrainHandler::new(
      config,
      initial_model,
      dataset,
      String::from("unused/batch"),
      4,
    );

    single_handler.train_step(0).unwrap();
    batch_handler.train_step(0).unwrap();

    let single_weights = &single_handler.get_model().get_layer(1).weights;
    let batch_weights = &batch_handler.get_model().get_layer(1).weights;
    assert_arrays_close(single_weights, batch_weights, 1e-6);
  }
}
