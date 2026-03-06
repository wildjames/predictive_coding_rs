//! Training orchestration for predictive coding models.

use crate::{
  data_handling::data_handler, model_structure::{model::{
    PredictiveCodingModel,
    PredictiveCodingModelConfig
  }, model_utils::save_model_config}, training::{
    train_handler::TrainingHandler,
    utils::{
      TrainConfig, save_training_config, set_rand_input_and_output
    },
  }
};

use ndarray::{Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::info;


pub struct SingleThreadTrainHandler {
  config: TrainConfig,
  model: PredictiveCodingModel,
  data: data_handler::ImagesBWDataset,
  file_output_prefix: String
}

impl TrainingHandler for SingleThreadTrainHandler {
  fn new(
    config: TrainConfig,
    model: PredictiveCodingModel,
    data: data_handler::ImagesBWDataset,
    file_output_prefix: String
  ) -> SingleThreadTrainHandler {
    SingleThreadTrainHandler {
      config,
      model,
      data,
      file_output_prefix
    }
  }

  fn get_config(&self) -> &TrainConfig {
    &self.config
  }
  fn get_model(&mut self) -> &mut PredictiveCodingModel {
    &mut self.model
  }
  fn get_data(&self) -> &data_handler::ImagesBWDataset {
    &self.data
  }
  fn get_file_output_prefix(&self) -> &String {
    &self.file_output_prefix
  }

  fn pre_step_hook(&mut self, _step: u32) {
    // No pre-step hook for single-threaded training
  }

  fn pre_training_hook(&mut self) {
    info!("Starting training with single-threaded strategy");
    info!(
      "Training parameters:\n\ttraining steps: {}\n\treporting interval: {}\n\tsnapshot interval: {}",
      self.config.training_steps,
      self.config.report_interval,
      self.config.snapshot_interval
    );

    let model_config: &PredictiveCodingModelConfig = &self.model.get_config();
    info!(
      "Model architecture:\n\tlayer sizes: {:?}\n\tgamma: {}\n\talpha: {}\n\tactivation function: {:?}\n\tconvergence steps: {}\n\tconvergence threshold: {}",
      model_config.layer_sizes,
      model_config.gamma,
      model_config.alpha,
      model_config.activation_function,
      model_config.convergence_steps,
      model_config.convergence_threshold
    );

    // Write the config and training params to a file
    save_model_config(
      model_config,
      &format!("{}_config.json", &self.file_output_prefix)
    );
    save_training_config(
      &self.config,
      &format!("{}_training_config.json", &self.file_output_prefix)
    );
  }

  fn train(&mut self) {
  set_rand_input_and_output(&mut self.model, &self.data);
  self.model.reinitialise_latents();

  // Train on this example until convergence.
  self.model.converge_values();
  self.model.update_weights();
  }

  fn post_step_hook(&mut self, _step: u32) {
    // No post-step hook for single-threaded training
  }

  fn report_hook(&mut self, step: u32) {
    let energy: f32 = self.model.read_total_energy();
    info!("Step {}: Current model state: energy = {:.2}", step, energy);
  }
}


pub struct BatchTrainHandler {
  config: TrainConfig,
  model: PredictiveCodingModel,
  data: data_handler::ImagesBWDataset,
  file_output_prefix: String
}

impl TrainingHandler for BatchTrainHandler {
  fn new(
    config: TrainConfig,
    model: PredictiveCodingModel,
    data: data_handler::ImagesBWDataset,
    file_output_prefix: String
  ) -> Self {
    BatchTrainHandler {
      config,
      model,
      data,
      file_output_prefix
    }
  }

  fn get_config(&self) -> &TrainConfig {
    &self.config
  }
  fn get_model(&mut self) -> &mut PredictiveCodingModel {
    &mut self.model
  }
  fn get_data(&self) -> &data_handler::ImagesBWDataset {
    &self.data
  }
  fn get_file_output_prefix(&self) -> &String {
    &self.file_output_prefix
  }

  fn pre_step_hook(&mut self, _step: u32) {
    // No pre-step hook for mini-batch training
  }

  fn pre_training_hook(&mut self) {
    info!("Starting training with mini-batch strategy");
    info!(
      "Training parameters:\n\ttraining steps: {}\n\tbatch size: {}\n\treporting interval: {}\n\tsnapshot interval: {}",
      self.config.training_steps,
      self.config.batch_size.expect("Batch size must be set for mini-batch training"),
      self.config.report_interval,
      self.config.snapshot_interval
    );

    let model_config: &PredictiveCodingModelConfig = &self.model.get_config();
    info!(
      "Model architecture:\n\tlayer sizes: {:?}\n\tgamma: {}\n\talpha: {}\n\tactivation function: {:?}\n\tconvergence steps: {}\n\tconvergence threshold: {}",
      model_config.layer_sizes,
      model_config.gamma,
      model_config.alpha,
      model_config.activation_function,
      model_config.convergence_steps,
      model_config.convergence_threshold
    );

    // Write the config and training params to a file
    save_model_config(
      model_config,
      &format!("{}_config.json", &self.file_output_prefix)
    );
    save_training_config(
      &self.config,
      &format!("{}_training_config.json", &self.file_output_prefix)
    );
  }

  /// Train B models in parallel on different data, then compute the average weight update across them and apply it to the model.
  fn train(&mut self) {
    let batch_size: u32 = self.config.batch_size.expect("Batch size must be set for mini-batch training");
    // I'll iterate over this with Rayon to parallelise the batch
    let batch_indexes: Vec<u32> = (0..batch_size).collect();

    // Each element of the batch trains on a single sample, and we collect their weight changes as a result.
    // The batch weight changes will be a Vec of length batch_size, where each element is a Vec of length
    // num_layers, and each element of THAT is an array2 of the weight changes for the relevant layer.
    let batch_weight_changes: Vec<Vec<Array2<f32>>> = batch_indexes
      .into_par_iter()
      .map(|_| {
        let mut model_clone: PredictiveCodingModel = self.model.clone();
        set_rand_input_and_output(&mut model_clone, &self.data);
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
      .map(|sum_weight_change: Array2<f32>| sum_weight_change / batch_size as f32)
      .collect();

    // Apply the average weight changes to the model.
    self.model.apply_weight_updates(avg_batch_weight_changes);
  }


  fn post_step_hook(&mut self, _step: u32) {
    // No post-step hook for mini-batch training
  }


  fn report_hook(&mut self, step: u32) {
    // The mini batch model is cloned for each batch element, so the main model never gets inference run on it
    // So, do that in the reporting step to get a sense of how the model is doing on the data.
    self.model.reinitialise_latents();
    set_rand_input_and_output(&mut self.model, &self.data);
    self.model.converge_values();

    let energy: f32 = self.model.read_total_energy();
    info!("Step {}: Current model state: energy = {:.2}", step, energy);
  }
}
