use crate::{
    data_handling::TrainingDataset,
    error::Result,
    model::{GpuModelRuntime, ModelRuntime},
    training::{TrainConfig, log_training_progress},
};

use super::impl_handler_delegation;

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

impl_handler_delegation!(GpuBatchTrainHandler, gpu_runtime, {
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

        // TODO: This executes the batch models serially - this should be parallelised! Needs it's own PR though.
        // FIXME: This also needs to make sure that the alpha scaling is reset on failures.
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

        // Run a quick forward pass to report current energy.
        let (input, output) = self.data.get_random_input_and_output();
        self.gpu_runtime
            .set_input(input.as_slice().expect("contiguous input array"))?;
        self.gpu_runtime
            .set_output(output.as_slice().expect("contiguous output array"))?;
        self.gpu_runtime.reinitialise_latents()?;
        self.gpu_runtime.converge_values()?;

        let energy = self.gpu_runtime.total_energy()?;
        log_training_progress(
            step,
            self.config.training_steps - step,
            mean_step_time,
            energy,
        );
        Ok(())
    }
});
