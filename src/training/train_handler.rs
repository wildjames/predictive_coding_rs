use chrono::{TimeDelta, Utc};
use tracing::info;

use crate::{
  data_handling::TrainingDataset,
  error::Result,
  model::{
    {PredictiveCodingModel, PredictiveCodingModelConfig},
    save_model_config,
    save_model_snapshot
  },
  training::configuration::{
    TrainConfig,
    save_training_config
  }
};

pub trait TrainingHandler {
  fn get_config(&self) -> &TrainConfig;
  fn get_model(&mut self) -> &mut PredictiveCodingModel;
  fn get_data(&self) -> &dyn TrainingDataset;
  fn get_file_output_prefix(&self) -> &String;

  fn pre_training_hook(&mut self) -> Result<()> {
    Ok(())
  }

  fn train_step(&mut self, _step: u32) -> Result<()>;
  fn report_hook(&mut self, _step: u32, _mean_step_time: TimeDelta) -> Result<()> {
    Ok(())
  }

  /// Once the model has completed training, this will be called. By default, it saves the final model to "{file_output_prefix}_final.json"
  fn post_training_hook(&mut self) -> Result<()> {
    let final_output_path: String = format!("{}_{}", self.get_file_output_prefix(), "final_model.json");

    info!("Finished training, saving final model to {}", final_output_path);
    save_model_snapshot(
      self.get_model(),
      &final_output_path
    )
  }

  // Any actions that need to be called with an awareness of the training step can use these hooks. By default, they do nothing.
  // e.g. if you want to anneal the learning rate, that would go here
  fn pre_step_hook(&mut self, _step: u32) -> Result<()> {
    Ok(())
  }
  fn post_step_hook(&mut self, _step: u32) -> Result<()> {
    Ok(())
  }
}


pub fn run_supervised_training_loop(handler: &mut dyn TrainingHandler) -> Result<()> {
  handler.pre_training_hook()?;

  // Supervised learning
  handler.get_model().pin_input();
  handler.get_model().pin_output();

  let training_config: &TrainConfig = handler.get_config();
  let training_steps: u32 = training_config.training_steps;
  let report_interval: u32 = training_config.report_interval;
  let snapshot_interval: u32 = training_config.snapshot_interval;

  info!(
    "Beginning training loop for {} steps with report interval {} and snapshot interval {}",
    training_steps, report_interval, snapshot_interval
  );

  let model_config: &PredictiveCodingModelConfig = &handler.get_model().get_config();
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
    &format!("{}_model_config.json", &handler.get_file_output_prefix())
  )?;
  save_training_config(
    handler.get_config(),
    &format!("{}_training_config.json", &handler.get_file_output_prefix())
  )?;

  // Track the time taken for each step in the current reporting epoch
  let mut step_times: Vec<TimeDelta> = Vec::new();

  // Main loop
  for step in 0..training_steps {
    let start_time = Utc::now();
    handler.pre_step_hook(step)?;
    handler.train_step(step)?;
    handler.post_step_hook(step)?;
    let elapsed: TimeDelta = Utc::now() - start_time;
    step_times.push(elapsed);

    if (report_interval > 0) && (step % report_interval == 0) {
      let mean_step_time: TimeDelta = step_times.iter().sum::<TimeDelta>() / step_times.len() as i32;
      handler.report_hook(step, mean_step_time)?;
      step_times.clear();
    }

    if (snapshot_interval > 0) && (step % snapshot_interval == 0) {
      let oname: String = format!("{}_snapshot_step_{}.json", handler.get_file_output_prefix(), step);
      info!("Saving model snapshot {}", oname);

      save_model_snapshot(handler.get_model(), &oname)?;
    }
  }

  handler.post_training_hook()?;

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_utils::{
    RecordingTrainingHandler,
    TempDir,
    single_thread_train_config,
  };

  #[test]
  fn training_loop_preserves_hook_order_and_report_interval() {
    let temp_dir: TempDir = TempDir::new("training_loop_hooks");
    let output_prefix: String = temp_dir.path().join("model").display().to_string();
    let mut handler: RecordingTrainingHandler = RecordingTrainingHandler::new(
      single_thread_train_config(3, 2, 0),
      output_prefix,
    );

    run_supervised_training_loop(&mut handler).unwrap();

    assert_eq!(
      handler.events,
      vec![
        String::from("pre_training"),
        String::from("pre_step:0"),
        String::from("train_step:0"),
        String::from("post_step:0"),
        String::from("report:0"),
        String::from("pre_step:1"),
        String::from("train_step:1"),
        String::from("post_step:1"),
        String::from("pre_step:2"),
        String::from("train_step:2"),
        String::from("post_step:2"),
        String::from("report:2"),
        String::from("post_training"),
      ]
    );
  }

  #[test]
  fn training_loop_saves_snapshots_at_configured_steps() {
    let temp_dir = TempDir::new("training_loop_snapshots");
    let output_prefix = temp_dir.path().join("model").display().to_string();
    let mut handler: RecordingTrainingHandler = RecordingTrainingHandler::new(
      single_thread_train_config(5, 0, 2),
      output_prefix.clone(),
    );

    run_supervised_training_loop(&mut handler).unwrap();

    assert!(temp_dir.path().join("model_snapshot_step_0.json").exists());
    assert!(!temp_dir.path().join("model_snapshot_step_1.json").exists());
    assert!(temp_dir.path().join("model_snapshot_step_2.json").exists());
    assert!(temp_dir.path().join("model_snapshot_step_4.json").exists());
    assert!(temp_dir.path().join("model_final_model.json").exists());
  }
}
