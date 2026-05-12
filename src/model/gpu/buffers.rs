use std::sync::Arc;

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::error::Result;

use super::context::GpuContext;

/// GPU-side storage for a single predictive-coding layer.
///
/// Each vector (values, predictions, errors) is a 1-D `f32` buffer of length
/// `size`.  Weights are row-major `f32` with shape `(weight_rows, weight_cols)`.
///
/// Every buffer carries both `STORAGE` (for compute shaders) and
/// `COPY_SRC | COPY_DST` (for readback / upload).
pub struct LayerBuffers {
    pub values: wgpu::Buffer,
    pub predictions: wgpu::Buffer,
    pub errors: wgpu::Buffer,
    pub weights: wgpu::Buffer,
    /// Per-node scratch buffer holding `abs(value_change)` after a timestep dispatch.
    pub value_changes: wgpu::Buffer,
    /// Scalar buffer: `[pinned: u32, activation_fn: u32, size: u32, weight_rows: u32, weight_cols: u32, is_top_level: u32]`
    pub meta: wgpu::Buffer,
    pub size: usize,
    pub weight_rows: usize,
    pub weight_cols: usize,
}

/// Activation function IDs that match the constants in the WGSL shaders.
pub const ACTIVATION_RELU: u32 = 0;
pub const ACTIVATION_SIGMOID: u32 = 1;
pub const ACTIVATION_TANH: u32 = 2;

/// Convert from the model enum to the shader constant.
pub fn activation_to_u32(af: crate::model::maths::ActivationFunction) -> u32 {
    use crate::model::maths::ActivationFunction;
    match af {
        ActivationFunction::Relu => ACTIVATION_RELU,
        ActivationFunction::Sigmoid => ACTIVATION_SIGMOID,
        ActivationFunction::Tanh => ACTIVATION_TANH,
    }
}

const BUF_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

impl LayerBuffers {
    /// Upload a single layer's data onto the GPU.
    ///
    /// `is_top_level` must be set by the caller (true for the last layer).
    pub fn from_snapshot(
        ctx: &Arc<GpuContext>,
        layer: &crate::model::snapshot::LayerSnapshot,
        is_top_level: bool,
    ) -> Self {
        let device = &ctx.device;

        let values = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_values"),
            contents: bytemuck::cast_slice(&layer.values),
            usage: BUF_USAGE,
        });

        let predictions = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_predictions"),
            contents: bytemuck::cast_slice(&layer.predictions),
            usage: BUF_USAGE,
        });

        let errors = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_errors"),
            contents: bytemuck::cast_slice(&layer.errors),
            usage: BUF_USAGE,
        });

        let weights = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_weights"),
            // Even if weights is empty (layer 0), wgpu needs a non-zero buffer.
            contents: if layer.weights.is_empty() {
                bytemuck::cast_slice(&[0.0_f32])
            } else {
                bytemuck::cast_slice(&layer.weights)
            },
            usage: BUF_USAGE,
        });

        let zeros = vec![0.0_f32; layer.size];
        let value_changes = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_value_changes"),
            contents: bytemuck::cast_slice(&zeros),
            usage: BUF_USAGE,
        });

        let meta_data: [u32; 6] = [
            layer.pinned as u32,
            activation_to_u32(layer.activation_function),
            layer.size as u32,
            layer.weight_rows as u32,
            layer.weight_cols as u32,
            is_top_level as u32,
        ];
        let meta = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_meta"),
            contents: bytemuck::cast_slice(&meta_data),
            usage: BUF_USAGE,
        });

        Self {
            values,
            predictions,
            errors,
            weights,
            value_changes,
            meta,
            size: layer.size,
            weight_rows: layer.weight_rows,
            weight_cols: layer.weight_cols,
        }
    }
}

/// All layer buffers for the entire model, plus a small uniform buffer for
/// model-level scalars (alpha, gamma, convergence params).
pub struct ModelBuffers {
    pub layers: Vec<LayerBuffers>,
    /// `[alpha: f32, gamma: f32, convergence_threshold: f32, convergence_steps: u32]`
    pub params: wgpu::Buffer,
    /// A small zeroed buffer used as `lower_errors` for layer 0's timestep
    /// bind group. Avoids aliasing layer 0's own errors buffer in two roles.
    pub dummy_lower_errors: wgpu::Buffer,
}

impl ModelBuffers {
    /// Upload a full model snapshot to the GPU.
    pub fn from_snapshot(
        ctx: &Arc<GpuContext>,
        snapshot: &crate::model::snapshot::ModelSnapshot,
    ) -> Self {
        let num_layers = snapshot.layers.len();
        let layers: Vec<LayerBuffers> = snapshot
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| LayerBuffers::from_snapshot(ctx, l, i == num_layers - 1))
            .collect();

        // Pack model-level scalars into a uniform buffer.
        let params_data: [f32; 4] = [
            snapshot.config.alpha,
            snapshot.config.gamma,
            snapshot.config.convergence_threshold,
            snapshot.config.convergence_steps as f32,
        ];
        let params = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("model_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Dummy buffer for layer 0's timestep bind group (lower_errors slot).
        // Layer 0 has weight_rows==0 so the kernel never reads this, but wgpu
        // needs a valid, non-aliased buffer bound to the read-only slot.
        let dummy_size = if !layers.is_empty() {
            layers[0].size
        } else {
            1
        };
        let dummy_zeros = vec![0.0_f32; dummy_size];
        let dummy_lower_errors = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("dummy_lower_errors"),
            contents: bytemuck::cast_slice(&dummy_zeros),
            usage: BUF_USAGE,
        });

        Self {
            layers,
            params,
            dummy_lower_errors,
        }
    }

    /// Read back `values` from a single layer into CPU memory.
    pub async fn download_values(
        ctx: &Arc<GpuContext>,
        layer_buf: &LayerBuffers,
    ) -> Result<Vec<f32>> {
        read_buffer_f32(ctx, &layer_buf.values, layer_buf.size).await
    }

    /// Read back `errors` from a single layer into CPU memory.
    pub async fn download_errors(
        ctx: &Arc<GpuContext>,
        layer_buf: &LayerBuffers,
    ) -> Result<Vec<f32>> {
        read_buffer_f32(ctx, &layer_buf.errors, layer_buf.size).await
    }

    /// Read back `predictions` from a single layer into CPU memory.
    pub async fn download_predictions(
        ctx: &Arc<GpuContext>,
        layer_buf: &LayerBuffers,
    ) -> Result<Vec<f32>> {
        read_buffer_f32(ctx, &layer_buf.predictions, layer_buf.size).await
    }

    /// Read back `weights` from a single layer into CPU memory.
    pub async fn download_weights(
        ctx: &Arc<GpuContext>,
        layer_buf: &LayerBuffers,
    ) -> Result<Vec<f32>> {
        let count = layer_buf.weight_rows * layer_buf.weight_cols;
        read_buffer_f32(ctx, &layer_buf.weights, count).await
    }

    /// Read back `value_changes` (abs value deltas from last timestep) from a single layer.
    pub async fn download_value_changes(
        ctx: &Arc<GpuContext>,
        layer_buf: &LayerBuffers,
    ) -> Result<Vec<f32>> {
        read_buffer_f32(ctx, &layer_buf.value_changes, layer_buf.size).await
    }
}

/// Helper: download `count` f32s from a GPU buffer to the CPU.
async fn read_buffer_f32(
    ctx: &Arc<GpuContext>,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f32>> {
    if count == 0 {
        return Ok(Vec::new());
    }

    let byte_len = (count * std::mem::size_of::<f32>()) as u64;

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_read"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_encoder"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |res| {
        let _ = tx.send(res);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv()
        .map_err(|e| {
            crate::error::PredictiveCodingError::validation(format!("GPU readback failed: {e}"))
        })?
        .map_err(|e| {
            crate::error::PredictiveCodingError::validation(format!("GPU buffer map failed: {e}"))
        })?;

    let view = staging.get_mapped_range(..);
    let data: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(data)
}
