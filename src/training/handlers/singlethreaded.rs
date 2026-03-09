//! Training orchestration for predictive coding models.

use crate::{
    data_handling::TrainingDataset,
    error::Result,
    model::{PredictiveCodingModel, model_utils::set_rand_input_and_output},
};

use super::{TrainConfig, TrainingHandler};

use chrono::TimeDelta;
use std::sync::Arc;
use tracing::{debug, info};

pub struct SingleThreadTrainHandler {
    config: TrainConfig,
    model: PredictiveCodingModel,
    data: Arc<dyn TrainingDataset>,
    file_output_prefix: String,
}

impl SingleThreadTrainHandler {
    pub fn new(
        config: TrainConfig,
        model: PredictiveCodingModel,
        data: Arc<dyn TrainingDataset>,
        file_output_prefix: String,
    ) -> Self {
        SingleThreadTrainHandler {
            config,
            model,
            data,
            file_output_prefix,
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
    fn get_data(&self) -> &dyn TrainingDataset {
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

    fn report_hook(&mut self, step: u32, mean_step_time: TimeDelta) -> Result<()> {
        debug!(
            "After step {}: mean step duration = {:.2?}",
            step, mean_step_time
        );

        // Estimate how much longer needed to complete the training
        let est_time_to_finish: chrono::Duration =
            mean_step_time * (self.config.training_steps - step) as i32;
        let est_finish_time: chrono::DateTime<chrono::Utc> =
            chrono::Utc::now() + est_time_to_finish;

        let energy: f32 = self.model.read_total_energy();
        info!(
            "Step {}: Current model state: energy = {:.2}\tEstimated finish time: {}",
            step,
            energy,
            est_finish_time.format("%Y-%m-%d %H:%M:%S")
        );
        Ok(())
    }
}
