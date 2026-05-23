//! Computes the ideal batch size for a given model config and GPU device.
//!
//! The ideal batch size maximises GPU occupancy without exceeding device memory.
//! This module estimates the GPU memory needed per batch slot and divides the
//! available device memory to produce a recommendation.

use std::sync::Arc;

use crate::error::{PredictiveCodingError, Result};
use crate::model::PredictiveCodingModelConfig;

use super::context::GpuContext;

const USABLE_GPU_MEM_PCT: f64 = 0.8;
const ONE_MEGABYTE: f64 = 1_048_576.0;

/// Memory breakdown for a batch-parallel GPU runtime.
#[derive(Debug, Clone)]
pub struct GpuMemoryEstimate {
    /// Bytes used by shared (non-per-slot) buffers: weights, meta, params, accumulators.
    pub shared_bytes: u64,
    /// Bytes used by each additional batch slot: values, predictions, errors, etc.
    pub per_slot_bytes: u64,
    /// Total device memory reported by the adapter (max_buffer_size from limits).
    pub device_memory_bytes: u64,
    /// Recommended max batch size that fits in device memory (with safety margin).
    pub recommended_batch_size: u32,
    /// GPU adapter name.
    pub adapter_name: String,
    /// Graphics backend (e.g. Vulkan, Metal, Dx12).
    pub backend: String,
    /// Device type (e.g. DiscreteGpu, IntegratedGpu, Cpu).
    pub device_type: String,
}

impl std::fmt::Display for GpuMemoryEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "GPU: {} ({}, {})",
            self.adapter_name, self.backend, self.device_type
        )?;
        writeln!(
            f,
            "Device memory: {:.1} MB",
            self.device_memory_bytes as f64 / ONE_MEGABYTE
        )?;
        writeln!(
            f,
            "Shared buffer footprint: {:.2} MB",
            self.shared_bytes as f64 / ONE_MEGABYTE
        )?;
        writeln!(
            f,
            "Per-slot buffer footprint: {:.2} MB",
            self.per_slot_bytes as f64 / ONE_MEGABYTE
        )?;
        write!(
            f,
            "Recommended batch size: {}",
            self.recommended_batch_size
        )
    }
}

/// Estimate GPU memory usage and recommended batch size for the given model config.
///
/// Creates a temporary GPU context to query the adapter's memory limits.
pub fn estimate_batch_size(config: &PredictiveCodingModelConfig) -> Result<GpuMemoryEstimate> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| {
            crate::error::PredictiveCodingError::validation(format!(
                "tokio runtime creation failed: {e}"
            ))
        })?;
    rt.block_on(estimate_batch_size_async(config))
}

/// Async version of [`estimate_batch_size`] using an existing GPU context.
pub async fn estimate_batch_size_with_context(
    config: &PredictiveCodingModelConfig,
    ctx: &Arc<GpuContext>,
) -> Result<GpuMemoryEstimate> {
    let (shared, per_slot) = compute_memory_footprint(config);
    let device_memory = get_device_memory(ctx);
    let max_buffer_size = ctx.adapter_limits.max_buffer_size;

    // Check that the largest single buffer fits into the device's max_buffer_size.
    let largest_buffer = largest_single_buffer_bytes(config);
    if largest_buffer > max_buffer_size {
        return Err(crate::error::PredictiveCodingError::validation(format!(
            "Model requires a single buffer of {:.2} MB (layer weights), \
             but the GPU's max_buffer_size is only {:.2} MB. \
             Reduce model layer sizes to fit this device.",
            largest_buffer as f64 / ONE_MEGABYTE,
            max_buffer_size as f64 / ONE_MEGABYTE,
        )));
    }

    let recommended = compute_recommended_batch_size(shared, per_slot, device_memory)?;

    Ok(GpuMemoryEstimate {
            shared_bytes: shared,
            per_slot_bytes: per_slot,
            device_memory_bytes: device_memory,
            recommended_batch_size: recommended,
            adapter_name: ctx.adapter_info.name.clone(),
            backend: format!("{:?}", ctx.adapter_info.backend),
            device_type: format!("{:?}", ctx.adapter_info.device_type),
    })
}

/// Async version of [`estimate_batch_size`].
async fn estimate_batch_size_async(
    config: &PredictiveCodingModelConfig,
) -> Result<GpuMemoryEstimate> {
    let ctx = GpuContext::new().await?;
    estimate_batch_size_with_context(config, &ctx).await
}

/// Compute the shared and per-slot memory footprints in bytes for a given config.
fn compute_memory_footprint(config: &PredictiveCodingModelConfig) -> (u64, u64) {
    let sizes = &config.layer_sizes;
    let num_layers = sizes.len();
    let f32_size: u64 = std::mem::size_of::<f32>() as u64;

    let mut shared_bytes: u64 = 0;
    let mut per_slot_bytes: u64 = 0;

    for i in 0..num_layers {
        let layer_size = sizes[i] as u64;
        let (weight_rows, weight_cols) = if i == 0 {
            (0u64, 0u64)
        } else {
            (sizes[i - 1] as u64, sizes[i] as u64)
        };
        let weight_count = weight_rows * weight_cols;

        // Shared per layer: weights + weight_deltas_accum + meta
        shared_bytes += weight_count * f32_size; // weights
        shared_bytes += weight_count.max(1) * f32_size; // weight_deltas_accum
        shared_bytes += 7 * std::mem::size_of::<u32>() as u64; // meta (7 u32s)

        // Per-slot per layer: values, predictions, errors, gain_errors, value_changes, weight_deltas
        per_slot_bytes += layer_size * f32_size; // values
        per_slot_bytes += layer_size * f32_size; // predictions
        per_slot_bytes += layer_size * f32_size; // errors
        let gain_error_count = if weight_rows == 0 { 1 } else { weight_rows };
        per_slot_bytes += gain_error_count * f32_size; // gain_errors
        per_slot_bytes += layer_size * f32_size; // value_changes
        per_slot_bytes += weight_count.max(1) * f32_size; // weight_deltas
    }

    // Shared: params buffer (8 f32s)
    shared_bytes += 8 * f32_size;

    // Shared: dummy_lower_errors (layer 0 size)
    if !sizes.is_empty() {
        shared_bytes += sizes[0] as u64 * f32_size;
    }

    // Per-slot: error_sum and value_change_sum buffers
    let total_sums: u64 = sizes
        .iter()
        .map(|&s| (s as u64).div_ceil(64))
        .sum();
    per_slot_bytes += total_sums.max(1) * f32_size; // error_sum
    per_slot_bytes += total_sums.max(1) * f32_size; // value_change_sum

    (shared_bytes, per_slot_bytes)
}

/// Compute the size in bytes of the largest single buffer the model requires.
/// This should be the weight matrix of the layer with the most parameters, I think?
fn largest_single_buffer_bytes(config: &PredictiveCodingModelConfig) -> u64 {
    let sizes = &config.layer_sizes;
    let f32_size: u64 = std::mem::size_of::<f32>() as u64;

    let mut max_bytes: u64 = 0;
    for i in 1..sizes.len() {
        // Weight matrix: rows = previous layer size, cols = current layer size
        let weight_bytes = (sizes[i - 1] as u64) * (sizes[i] as u64) * f32_size;
        max_bytes = max_bytes.max(weight_bytes);
    }
    max_bytes
}

/// Query the adapter's `max_buffer_size` limit.
/// This reflects actual GPU capability, unlike `device.limits()` which
/// returns the (conservative) requested limits.
fn get_device_memory(ctx: &Arc<GpuContext>) -> u64 {
    ctx.adapter_limits.max_buffer_size
}

/// Compute the recommended batch size given memory constraints.
///
/// Uses 80% of device memory as a safety margin (to leave room for
/// driver overhead, bind groups, staging buffers, etc.).
fn compute_recommended_batch_size(
    shared_bytes: u64,
    per_slot_bytes: u64,
    device_memory_bytes: u64,
) -> Result<u32> {
    if device_memory_bytes == 0 || per_slot_bytes == 0 {
        // Can't determine memory, so throw a config error
        return Err(PredictiveCodingError::validation(
            "Could not get device memory byte size, or find the per slot bytes"
        ))
    }

    let usable_memory = (device_memory_bytes as f64 * USABLE_GPU_MEM_PCT) as u64;

    if usable_memory <= shared_bytes {
        return Ok(1);
    }

    let available_for_slots = usable_memory - shared_bytes;
    let max_slots = available_for_slots / per_slot_bytes;

    // Clamp to at least 1, and cap at a reasonable maximum
    Ok(max_slots.clamp(1, 4096) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::maths::ActivationFunction;

    #[test]
    fn memory_footprint_scales_with_layers() {
        let small = PredictiveCodingModelConfig {
            layer_sizes: vec![784, 128, 10],
            alpha: 0.01,
            gamma: 0.05,
            convergence_threshold: 1e-5,
            convergence_steps: 50,
            activation_function: ActivationFunction::Relu,
            weight_clip: 0.1,
        };
        let large = PredictiveCodingModelConfig {
            layer_sizes: vec![784, 1024, 512, 10],
            alpha: 0.01,
            gamma: 0.05,
            convergence_threshold: 1e-5,
            convergence_steps: 50,
            activation_function: ActivationFunction::Relu,
            weight_clip: 0.1,
        };

        let (small_shared, small_per_slot) = compute_memory_footprint(&small);
        let (large_shared, large_per_slot) = compute_memory_footprint(&large);

        assert!(large_shared > small_shared);
        assert!(large_per_slot > small_per_slot);
    }

    #[test]
    fn recommended_batch_size_respects_memory() {
        // 100 MB device, 10 MB shared, 5 MB per slot -> usable = 80 MB, available = 70 MB -> 14 slots
        let rec = compute_recommended_batch_size(10_000_000, 5_000_000, 100_000_000);
        assert_eq!(rec.unwrap(), 14);
    }

    #[test]
    fn recommended_batch_size_zero_memory_gives_error() {
        let rec = compute_recommended_batch_size(1000, 500, 0);
        assert!(rec.is_err());
    }
}
