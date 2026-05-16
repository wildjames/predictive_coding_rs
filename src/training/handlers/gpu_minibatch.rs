use crate::{
    data_handling::TrainingDataset,
    error::Result,
    model::{GpuModelRuntime, ModelRuntime, ModelSnapshot, PredictiveCodingModelConfig},
};

use super::{TrainConfig, TrainingHandler};

use chrono::TimeDelta;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

use super::super::StepProfile;

pub struct GpuBatchTrainHandler {
    config: TrainConfig,
    gpu_runtime: GpuModelRuntime,
    data: Arc<dyn TrainingDataset>,
    file_output_prefix: String,
    batch_size: u32,
}

impl GpuBatchTrainHandler {
    pub fn new(
        config: TrainConfig,
        model: crate::model::PredictiveCodingModel,
        data: Arc<dyn TrainingDataset>,
        file_output_prefix: String,
        batch_size: u32,
    ) -> Result<Self> {
        let snapshot = model.to_snapshot();
        let gpu_runtime = GpuModelRuntime::from_snapshot(&snapshot)?;
        Ok(Self {
            config,
            gpu_runtime,
            data,
            file_output_prefix,
            batch_size,
        })
    }
}

impl TrainingHandler for GpuBatchTrainHandler {
    fn get_config(&self) -> &TrainConfig {
        &self.config
    }

    fn model_snapshot(&mut self) -> Result<ModelSnapshot> {
        self.gpu_runtime.snapshot()
    }

    fn model_config(&self) -> PredictiveCodingModelConfig {
        self.gpu_runtime.config()
    }

    fn pin_input(&mut self) -> Result<()> {
        self.gpu_runtime.pin_input()
    }

    fn pin_output(&mut self) -> Result<()> {
        self.gpu_runtime.pin_output()
    }

    fn get_data(&self) -> &dyn TrainingDataset {
        self.data.as_ref()
    }

    fn get_file_output_prefix(&self) -> &String {
        &self.file_output_prefix
    }

    fn pre_training_hook(&mut self) -> Result<()> {
        info!(
            "Starting GPU mini-batch training on {}",
            self.gpu_runtime.gpu_description()
        );
        info!("Mini batch params: batch size = {}", self.batch_size);
        Ok(())
    }

    fn profiled_train_step(&mut self, _step: u32) -> Result<StepProfile> {
        let mut profile = StepProfile::new();

        let t = Instant::now();
        let original_alpha = self.gpu_runtime.config().alpha;
        self.gpu_runtime
            .set_params_alpha(original_alpha / self.batch_size as f32);
        self.gpu_runtime.zero_weight_accumulators();
        profile.record("setup_accumulators", t.elapsed());

        let mut total_set_data = std::time::Duration::ZERO;
        let mut total_reinit = std::time::Duration::ZERO;
        let mut total_converge = std::time::Duration::ZERO;
        let mut total_accumulate = std::time::Duration::ZERO;

        for _ in 0..self.batch_size {
            let t = Instant::now();
            let (input, output) = self.data.get_random_input_and_output();
            self.gpu_runtime
                .set_input(input.as_slice().expect("contiguous input array"))?;
            self.gpu_runtime
                .set_output(output.as_slice().expect("contiguous output array"))?;
            total_set_data += t.elapsed();

            let t = Instant::now();
            self.gpu_runtime.reinitialise_latents()?;
            total_reinit += t.elapsed();

            let t = Instant::now();
            self.gpu_runtime.converge_values()?;
            total_converge += t.elapsed();

            let t = Instant::now();
            self.gpu_runtime.accumulate_weight_deltas_on_device();
            total_accumulate += t.elapsed();
        }

        profile.record("set_data", total_set_data);
        profile.record("reinitialise_latents", total_reinit);
        profile.record("converge_values", total_converge);
        profile.record("accumulate_deltas", total_accumulate);

        let t = Instant::now();
        self.gpu_runtime.apply_accumulated_weight_deltas();
        profile.record("apply_deltas", t.elapsed());

        let t = Instant::now();
        self.gpu_runtime.set_params_alpha(original_alpha);
        profile.record("restore_alpha", t.elapsed());

        Ok(profile)
    }

    fn report_hook(&mut self, step: u32, mean_step_time: TimeDelta) -> Result<()> {
        debug!(
            "After step {}: mean step duration = {:.2?}",
            step, mean_step_time
        );
        let est_time_to_finish = mean_step_time * (self.config.training_steps - step) as i32;
        let est_finish_time = chrono::Utc::now() + est_time_to_finish;

        // Run a quick forward pass to report current energy.
        let (input, output) = self.data.get_random_input_and_output();
        self.gpu_runtime
            .set_input(input.as_slice().expect("contiguous input array"))?;
        self.gpu_runtime
            .set_output(output.as_slice().expect("contiguous output array"))?;
        self.gpu_runtime.reinitialise_latents()?;
        self.gpu_runtime.converge_values()?;

        let energy: f32 = self.gpu_runtime.total_energy()?;
        info!(
            "Step {}: Current model state: energy = {:.2}\tEstimated finish time: {}",
            step,
            energy,
            est_finish_time.format("%Y-%m-%d %H:%M:%S")
        );
        Ok(())
    }
}
