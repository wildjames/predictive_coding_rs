use crate::{
    data_handling::TrainingDataset,
    error::Result,
    model::{GpuBatchRuntime, GpuModelRuntime, ModelRuntime},
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
    /// Single-sample runtime used for report_hook
    gpu_runtime: GpuModelRuntime,
    /// Batch-parallel runtime used for the actual training steps
    batch_runtime: GpuBatchRuntime,
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
        let batch_runtime = GpuBatchRuntime::from_snapshot(&snapshot, batch_size)?;
        Ok(Self {
            config,
            gpu_runtime,
            batch_runtime,
            data,
            file_output_prefix,
            batch_size,
        })
    }
}

impl_handler_delegation!(GpuBatchTrainHandler, gpu_runtime, {
    fn pre_training_hook(&mut self) -> Result<()> {
        info!(
            "Starting GPU batch-parallel training on {}",
            self.batch_runtime.gpu_description()
        );
        info!("Mini batch params: batch size = {}", self.batch_size);
        Ok(())
    }

    fn profiled_train_step(&mut self, _step: u32) -> Result<StepProfile> {
        let mut profile = StepProfile::new();

        let t = Instant::now();
        let original_alpha = self.batch_runtime.config().alpha;
        self.batch_runtime
            .set_params_alpha(original_alpha / self.batch_size as f32);
        self.batch_runtime.zero_weight_accumulators();
        profile.record("setup_accumulators", t.elapsed());

        // Gather random samples for in/outputs
        let t = Instant::now();
        let samples: Vec<(Vec<f32>, Vec<f32>)> = (0..self.batch_size)
            .map(|_| {
                let (input, output) = self.data.get_random_input_and_output();
                (
                    input.as_slice().expect("contiguous input array").to_vec(),
                    output.as_slice().expect("contiguous output array").to_vec(),
                )
            })
            .collect();
        profile.record("gather_samples", t.elapsed());

        // upload
        let t = Instant::now();
        self.batch_runtime.set_batch_data(&samples)?;
        profile.record("set_data", t.elapsed());

        let t = Instant::now();
        self.batch_runtime.reinitialise_all_latents();
        profile.record("reinitialise_latents", t.elapsed());

        // converge all slots in parallel
        let t = Instant::now();
        self.batch_runtime.converge_all()?;
        profile.record("converge_values", t.elapsed());

        // Update model
        let t = Instant::now();
        self.batch_runtime.accumulate_all_weight_deltas();
        profile.record("accumulate_deltas", t.elapsed());

        let t = Instant::now();
        self.batch_runtime.apply_accumulated_weight_deltas();
        profile.record("apply_deltas", t.elapsed());

        // Restore the original alpha
        let t = Instant::now();
        self.batch_runtime.set_params_alpha(original_alpha);
        profile.record("restore_alpha", t.elapsed());

        Ok(profile)
    }

    fn report_hook(&mut self, step: u32, mean_step_time: TimeDelta) -> Result<()> {
        debug!(
            "After step {}: mean step duration = {:.2?}",
            step, mean_step_time
        );

        // Run a quick forward pass on the single-sample runtime to report current energy.
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
