use tracing::info;

use crate::{
  data_handling::data_handler,
  model_structure::{
    model::PredictiveCodingModel,
    model_utils::save_model
  },
  training::utils::{TrainConfig}
};

pub trait TrainingHandler {
  fn new(
    config: TrainConfig,
    model: PredictiveCodingModel,
    data: data_handler::ImagesBWDataset,
    file_output_prefix: String
  ) -> Self where Self: Sized;

  fn get_config(&self) -> &TrainConfig;
  fn get_model(&mut self) -> &mut PredictiveCodingModel;
  fn get_data(&self) -> &data_handler::ImagesBWDataset;
  fn get_file_output_prefix(&self) -> &String;

  fn pre_training_hook(&mut self);
  fn pre_step_hook(&mut self, _step: u32);
  fn train(&mut self);
  fn post_step_hook(&mut self, _step: u32);
  fn report_hook(&mut self, _step: u32);
}

pub fn run_supervised_training_loop(
  handler: &mut dyn TrainingHandler,
  snapshot_output_prefix: &str
) {
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
    handler.train();
    handler.post_step_hook(step);

    if step % report_interval == 0 {
      handler.report_hook(step);
    }

    if step % snapshot_interval == 0 {
      let oname: String = format!("{}_step_{}.json", snapshot_output_prefix, step);
      info!("Saving model snapshot {}", oname);

      save_model(handler.get_model(), &oname);
    }
  }
}
