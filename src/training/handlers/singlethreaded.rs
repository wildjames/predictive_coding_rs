//! Runtime-agnostic single-threaded training handler.

use crate::{
    data_handling::TrainingDataset,
    error::Result,
    model::{ModelSnapshot, PredictiveCodingModelConfig, TrainableModelRuntime},
    training::{TrainConfig, TrainingHandler, log_training_progress},
};

use chrono::TimeDelta;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

use super::super::StepProfile;

pub struct SingleThreadTrainHandler<R: TrainableModelRuntime> {
    config: TrainConfig,
    runtime: R,
    data: Arc<dyn TrainingDataset>,
    file_output_prefix: String,
}

impl<R: TrainableModelRuntime> SingleThreadTrainHandler<R> {
    pub fn new(
        config: TrainConfig,
        runtime: R,
        data: Arc<dyn TrainingDataset>,
        file_output_prefix: String,
    ) -> Self {
        SingleThreadTrainHandler {
            config,
            runtime,
            data,
            file_output_prefix,
        }
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.runtime
    }
}

impl<R: TrainableModelRuntime> TrainingHandler for SingleThreadTrainHandler<R> {
    fn get_config(&self) -> &TrainConfig {
        &self.config
    }
    fn model_snapshot(&mut self) -> Result<ModelSnapshot> {
        self.runtime.snapshot()
    }
    fn model_config(&self) -> PredictiveCodingModelConfig {
        self.runtime.config()
    }
    fn pin_input(&mut self) -> Result<()> {
        self.runtime.pin_input()
    }
    fn pin_output(&mut self) -> Result<()> {
        self.runtime.pin_output()
    }
    fn get_data(&self) -> &dyn TrainingDataset {
        self.data.as_ref()
    }
    fn get_file_output_prefix(&self) -> &String {
        &self.file_output_prefix
    }

    fn pre_training_hook(&mut self) -> Result<()> {
        info!(
            "Starting single-threaded training on {:?} backend",
            self.runtime.backend()
        );
        Ok(())
    }

    fn profiled_train_step(&mut self, _step: u32) -> Result<StepProfile> {
        let mut profile = StepProfile::new();

        let t = Instant::now();
        let (input, output) = self.data.get_random_input_and_output();
        self.runtime
            .set_input(input.as_slice().expect("contiguous input array"))?;
        self.runtime
            .set_output(output.as_slice().expect("contiguous output array"))?;
        profile.record("set_data", t.elapsed());

        let t = Instant::now();
        self.runtime.reinitialise_latents()?;
        profile.record("reinitialise_latents", t.elapsed());

        let t = Instant::now();
        self.runtime.converge_values()?;
        profile.record("converge_values", t.elapsed());

        let t = Instant::now();
        self.runtime.update_weights()?;
        profile.record("update_weights", t.elapsed());

        Ok(profile)
    }

    fn report_hook(&mut self, step: u32, mean_step_time: TimeDelta) -> Result<()> {
        debug!(
            "After step {}: mean step duration = {:.2?}",
            step, mean_step_time
        );
        let energy = self.runtime.total_energy()?;
        log_training_progress(
            step,
            self.config.training_steps - step,
            mean_step_time,
            energy,
        );
        Ok(())
    }
}
