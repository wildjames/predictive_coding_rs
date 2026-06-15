use crate::{
    data_handling::TrainingDataset,
    error::Result,
    model::{GpuRuntime, ModelSnapshot, PredictiveCodingModelConfig},
    training::{TrainConfig, TrainingHandler, log_training_progress},
};

use chrono::TimeDelta;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

use super::super::StepProfile;

pub struct GpuBatchTrainHandler {
    config: TrainConfig,
    /// Batch-parallel runtime used for the actual training steps
    batch_runtime: GpuRuntime,
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
        let batch_runtime = GpuRuntime::from_snapshot(&snapshot, batch_size)?;
        Ok(Self {
            config,
            batch_runtime,
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
        self.batch_runtime.snapshot()
    }

    fn model_config(&self) -> PredictiveCodingModelConfig {
        self.batch_runtime.config().clone()
    }

    fn pin_input(&mut self) -> Result<()> {
        // Pinning is handled automatically by set_batch_data
        Ok(())
    }

    fn pin_output(&mut self) -> Result<()> {
        // Pinning is handled automatically by set_batch_data
        Ok(())
    }

    fn get_data(&self) -> &dyn TrainingDataset {
        self.data.as_ref()
    }

    fn get_file_output_prefix(&self) -> &String {
        &self.file_output_prefix
    }

    fn pre_training_hook(&mut self) -> Result<()> {
        info!(
            "Starting GPU batch-parallel training on {}",
            self.batch_runtime.gpu_description()
        );
        info!("Mini batch params: batch size = {}", self.batch_size);
        info!("Tip: Watch the 'cpu_idle_waiting_gpu' phase in step profiles. \
               Low values indicate the GPU is bottlenecked and a larger batch size may help.");
        Ok(())
    }

    fn profiled_train_step(&mut self, _step: u32) -> Result<StepProfile> {
        let mut profile = StepProfile::new();
        let mut total_cpu_idle = std::time::Duration::ZERO;

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
        let encode_elapsed = t.elapsed();
        let gpu_idle = self.batch_runtime.poll_gpu();
        total_cpu_idle += gpu_idle;
        profile.record("converge_values", encode_elapsed + gpu_idle);

        // Update model
        let t = Instant::now();
        self.batch_runtime.accumulate_all_weight_deltas();
        let encode_elapsed = t.elapsed();
        let gpu_idle = self.batch_runtime.poll_gpu();
        total_cpu_idle += gpu_idle;
        profile.record("accumulate_deltas", encode_elapsed + gpu_idle);

        let t = Instant::now();
        self.batch_runtime.apply_accumulated_weight_deltas();
        let encode_elapsed = t.elapsed();
        let gpu_idle = self.batch_runtime.poll_gpu();
        total_cpu_idle += gpu_idle;
        profile.record("apply_deltas", encode_elapsed + gpu_idle);

        // Restore the original alpha
        let t = Instant::now();
        self.batch_runtime.set_params_alpha(original_alpha);
        profile.record("restore_alpha", t.elapsed());

        profile.record("cpu_idle_waiting_gpu", total_cpu_idle);

        // Log GPU utilization: fraction of step time where the GPU was active.
        let total_step = profile.total();
        if total_step.as_nanos() > 0 {
            let gpu_util_pct =
                (total_cpu_idle.as_secs_f64() / total_step.as_secs_f64()) * 100.0;
            debug!(
                "GPU utilization: {:.1}% (GPU busy {:.1}ms / step {:.1}ms)",
                gpu_util_pct,
                total_cpu_idle.as_secs_f64() * 1000.0,
                total_step.as_secs_f64() * 1000.0,
            );
        }

        Ok(profile)
    }

    fn report_hook(&mut self, step: u32, mean_step_time: TimeDelta) -> Result<()> {
        debug!(
            "After step {}: mean step duration = {:.2?}",
            step, mean_step_time
        );

        // Run a quick forward pass on the batch runtime to report current energy.
        // Use slot 0's energy after converging a random sample.
        let (input, output) = self.data.get_random_input_and_output();
        let sample = vec![(
            input.as_slice().expect("contiguous input array").to_vec(),
            output.as_slice().expect("contiguous output array").to_vec(),
        )];
        // Fill all batch slots with the same sample (set_batch_data requires batch_size samples)
        let samples: Vec<(Vec<f32>, Vec<f32>)> = sample
            .iter()
            .cycle()
            .take(self.batch_size as usize)
            .cloned()
            .collect();
        self.batch_runtime.set_batch_data(&samples)?;
        self.batch_runtime.reinitialise_all_latents();
        self.batch_runtime.converge_all()?;
        self.batch_runtime.poll_gpu();

        let energy = self.batch_runtime.total_energy()?;
        log_training_progress(
            step,
            self.config.training_steps - step,
            mean_step_time,
            energy,
        );
        Ok(())
    }
}
