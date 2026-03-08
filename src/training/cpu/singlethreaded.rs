//! Training orchestration for predictive coding models.

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

use std::sync::Arc;
use tracing::info;


pub struct SingleThreadTrainHandler {
  config: TrainConfig,
  model: PredictiveCodingModel,
  data: Arc<dyn data_handler::TrainingDataset>,
  file_output_prefix: String
}

impl SingleThreadTrainHandler {
  pub fn new(
    config: TrainConfig,
    model: PredictiveCodingModel,
    data: Arc<dyn data_handler::TrainingDataset>,
    file_output_prefix: String
  ) -> Self {
    SingleThreadTrainHandler {
      config,
      model,
      data,
      file_output_prefix
    }
  }
}

impl TrainingHandler for SingleThreadTrainHandler {
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

  fn pre_training_hook(&mut self) -> Result<()> {
    info!("Starting training with single-threaded strategy");
    Ok(())
  }

  fn train_step(&mut self, _step: u32) -> Result<()> {
    set_rand_input_and_output(&mut self.model, self.data.as_ref());
    self.model.reinitialise_latents();

    // Train on this example until convergence.
    self.model.converge_values();
    self.model.update_weights();

    Ok(())
  }

  fn report_hook(&mut self, step: u32) -> Result<()> {
    let energy: f32 = self.model.read_total_energy();
    info!("Step {}: Current model state: energy = {:.2}", step, energy);
    Ok(())
  }
}
