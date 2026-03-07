use tracing::info;

use crate::{
  data_handling::data_handler,
  model_structure::{
    model::PredictiveCodingModel,
    model_utils::save_model_snapshot
  },
  training::utils::{TrainConfig}
};

pub trait TrainingHandler {
  fn get_config(&self) -> &TrainConfig;
  fn get_model(&mut self) -> &mut PredictiveCodingModel;
  fn get_data(&self) -> &data_handler::TrainingDataset;
  fn get_file_output_prefix(&self) -> &String;

  fn pre_training_hook(&mut self);

  fn train_step(&mut self, _step: u32);
  fn report_hook(&mut self, _step: u32);

  /// Once the model has completed training, this will be called. By default, it saves the final model to "{file_output_prefix}_final.json"
  fn post_training_hook(&mut self) {
    let final_output_path: String = format!("{}_{}", self.get_file_output_prefix(), "final_model.json");

    info!("Finished training, saving final model to {}", final_output_path);
    save_model_snapshot(
      self.get_model(),
      &final_output_path
    )
  }

  // Any actions that need to be called with an awareness of the training step can use these hooks. By default, they do nothing.
  // e.g. if you want to anneal the learning rate, that would go here
  fn pre_step_hook(&mut self, _step: u32) {}
  fn post_step_hook(&mut self, _step: u32) {}
}


pub fn run_supervised_training_loop(handler: &mut dyn TrainingHandler) {
  handler.pre_training_hook();

  // Supervised learning
  handler.get_model().pin_input();
  handler.get_model().pin_output();

  let training_config: &TrainConfig = handler.get_config();
  let training_steps: u32 = training_config.training_steps;
  let report_interval: u32 = training_config.report_interval;
  let snapshot_interval: u32 = training_config.snapshot_interval;

  for step in 0..training_steps {
    handler.pre_step_hook(step);
    handler.train_step(step);
    handler.post_step_hook(step);

    if (report_interval > 0) && (step % report_interval == 0) {
      handler.report_hook(step);
    }

    if (snapshot_interval > 0) && (step % snapshot_interval == 0) {
      let oname: String = format!("{}_snapshot_step_{}.json", handler.get_file_output_prefix(), step);
      info!("Saving model snapshot {}", oname);

      save_model_snapshot(handler.get_model(), &oname);
    }
  }

  handler.post_training_hook();
}
