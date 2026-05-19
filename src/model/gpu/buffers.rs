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
    /// Per-weight scratch buffer holding computed deltas before applying to weights.
    pub weight_deltas: wgpu::Buffer,
    /// Per-weight accumulation buffer for minibatch training.
    /// Accumulated deltas are summed here across batch samples, then applied once.
    pub weight_deltas_accum: wgpu::Buffer,
    /// Per-node scratch buffer holding `abs(value_change)` after a timestep dispatch.
    pub value_changes: wgpu::Buffer,
    /// Per-row scratch buffer holding precomputed `f'(W[i,:] · x) * lower_errors[i]`.
    /// Populated by `compute_gain_errors`, consumed by `values_timestep` and `compute_weight_deltas`.
    pub gain_errors: wgpu::Buffer,
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

pub fn activation_function_from_u32(id: u32) -> Result<crate::model::maths::ActivationFunction> {
    use crate::model::maths::ActivationFunction;
    match id {
        ACTIVATION_RELU => Ok(ActivationFunction::Relu),
        ACTIVATION_SIGMOID => Ok(ActivationFunction::Sigmoid),
        ACTIVATION_TANH => Ok(ActivationFunction::Tanh),
        _ => Err(crate::error::PredictiveCodingError::validation(format!(
            "invalid activation function ID in GPU meta buffer: {id}"
        ))),
    }
}

const BUF_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

impl LayerBuffers {
    /// Upload a single layer's data onto the GPU.
    ///
    /// `is_top_level` must be set by the caller (true for the last layer).
    /// `sum_offset` is this layer's starting index in shared buffers
    pub fn from_snapshot(
        ctx: &Arc<GpuContext>,
        layer: &crate::model::snapshot::LayerSnapshot,
        is_top_level: bool,
        sum_offset: u32,
    ) -> Self {
        let device = &ctx.device;

        // Buffer 0
        let values = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_values"),
            contents: bytemuck::cast_slice(&layer.values),
            usage: BUF_USAGE,
        });

        // Buffer 1
        let predictions = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_predictions"),
            contents: bytemuck::cast_slice(&layer.predictions),
            usage: BUF_USAGE,
        });

        // Buffer 2
        let errors = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_errors"),
            contents: bytemuck::cast_slice(&layer.errors),
            usage: BUF_USAGE,
        });

        // Buffer 3
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

        // Buffer 4
        let zeros = vec![0.0_f32; layer.size];
        let value_changes = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_value_changes"),
            contents: bytemuck::cast_slice(&zeros),
            usage: BUF_USAGE,
        });

        // Buffer 5
        let weight_delta_count = if layer.weights.is_empty() {
            1
        } else {
            layer.weights.len()
        };
        let weight_delta_zeros = vec![0.0_f32; weight_delta_count];
        let weight_deltas = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_weight_deltas"),
            contents: bytemuck::cast_slice(&weight_delta_zeros),
            usage: BUF_USAGE,
        });

        let weight_accum_zeros = vec![0.0_f32; weight_delta_count];
        let weight_deltas_accum = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_weight_deltas_accum"),
            contents: bytemuck::cast_slice(&weight_accum_zeros),
            usage: BUF_USAGE,
        });

        // Buffer 6
        let gain_error_count = if layer.weight_rows == 0 {
            1
        } else {
            layer.weight_rows
        };
        let gain_error_zeros = vec![0.0_f32; gain_error_count];
        let gain_errors = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("layer_gain_errors"),
            contents: bytemuck::cast_slice(&gain_error_zeros),
            usage: BUF_USAGE,
        });

        // Buffer 7
        let meta_data: [u32; 7] = [
            layer.pinned as u32,
            activation_to_u32(layer.activation_function),
            layer.size as u32,
            layer.weight_rows as u32,
            layer.weight_cols as u32,
            is_top_level as u32,
            sum_offset,
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
            weight_deltas,
            weight_deltas_accum,
            value_changes,
            gain_errors,
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
    /// Buffer for the error reduction kernel's partial sums (one per workgroup).
    /// Sized to fit all layers' workgroups contiguously (`total_sums` floats).
    pub error_sum: wgpu::Buffer,
    /// Buffer for the value-change reduction kernel's partial sums (one per workgroup).
    /// Same sizing as `error_sum`.
    pub value_change_sum: wgpu::Buffer,
    /// Total number of f32 slots in summing arrays, e.g. `error_sum`, equal to the sum of
    /// `layer_size.div_ceil(64)` across all layers.
    pub total_sums: usize,
}

impl ModelBuffers {
    /// Upload a full model snapshot to the GPU.
    pub fn from_snapshot(
        ctx: &Arc<GpuContext>,
        snapshot: &crate::model::snapshot::ModelSnapshot,
    ) -> Self {
        let num_layers: usize = snapshot.layers.len();

        // Compute prefix-sum offsets into the shared partial_sums buffer.
        // Layer i writes its workgroup partial sums starting at offset[i].
        let mut offsets: Vec<u32> = Vec::with_capacity(num_layers);
        // and running total of all workgroups across all layers, to size the shared buffer.
        let mut running: u32 = 0;
        for l in &snapshot.layers {
            offsets.push(running);
            running += (l.size as u32).div_ceil(64);
        }
        let total_sums = running as usize;

        let layers: Vec<LayerBuffers> = snapshot
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| LayerBuffers::from_snapshot(ctx, l, i == num_layers - 1, offsets[i]))
            .collect();

        // Pack model-level scalars into a uniform buffer.
        // Layout: [alpha, gamma, convergence_threshold, convergence_steps, weight_clip, 0, 0, 0]
        // Needs to be 16-byte aligned, so some padding is needed here
        let params_data: [f32; 8] = [
            snapshot.config.alpha,
            snapshot.config.gamma,
            snapshot.config.convergence_threshold,
            snapshot.config.convergence_steps as f32,
            snapshot.config.weight_clip,
            0.0,
            0.0,
            0.0,
        ];
        let params: wgpu::Buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
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
        let dummy_zeros: Vec<f32> = vec![0.0_f32; dummy_size];
        let dummy_lower_errors: wgpu::Buffer =
            ctx.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("dummy_lower_errors"),
                contents: bytemuck::cast_slice(&dummy_zeros),
                usage: BUF_USAGE,
            });

        // Buffer for error reduction kernel partial sums (one per workgroup).
        // Sized to hold all layers' workgroups contiguously.
        let total_slots: usize = total_sums.max(1);
        let error_sum_zeros: Vec<f32> = vec![0.0_f32; total_slots];
        let error_sum: wgpu::Buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("error_sum"),
            contents: bytemuck::cast_slice(&error_sum_zeros),
            usage: BUF_USAGE,
        });

        // Buffer for value-change reduction kernel partial sums (same sizing).
        let vc_sum_zeros: Vec<f32> = vec![0.0_f32; total_slots];
        let value_change_sum: wgpu::Buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("value_change_sum"),
            contents: bytemuck::cast_slice(&vc_sum_zeros),
            usage: BUF_USAGE,
        });

        Self {
            layers,
            params,
            dummy_lower_errors,
            error_sum,
            value_change_sum,
            total_sums,
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

    /// Read back `weight_deltas` from a single layer into CPU memory.
    pub async fn download_weight_deltas(
        ctx: &Arc<GpuContext>,
        layer_buf: &LayerBuffers,
    ) -> Result<Vec<f32>> {
        let count = layer_buf.weight_rows * layer_buf.weight_cols;
        read_buffer_f32(ctx, &layer_buf.weight_deltas, count).await
    }

    /// Read back `value_changes` (abs value deltas from last timestep) from a single layer.
    pub async fn download_value_changes(
        ctx: &Arc<GpuContext>,
        layer_buf: &LayerBuffers,
    ) -> Result<Vec<f32>> {
        read_buffer_f32(ctx, &layer_buf.value_changes, layer_buf.size).await
    }

    /// Read back the `meta` buffer from a single layer, returning the raw u32 values.
    pub async fn download_meta(
        ctx: &Arc<GpuContext>,
        layer_buf: &LayerBuffers,
    ) -> Result<Vec<u32>> {
        read_buffer_u32(ctx, &layer_buf.meta, 7).await
    }

    /// Read back all partial sums from the error reduction kernel and return
    /// their total.  The buffer holds one slot per workgroup across all layers.
    pub async fn download_error_sum(
        ctx: &Arc<GpuContext>,
        model_bufs: &ModelBuffers,
    ) -> Result<f32> {
        let partial = read_buffer_f32(ctx, &model_bufs.error_sum, model_bufs.total_sums).await?;
        Ok(partial.iter().sum())
    }

    /// Read back all partial sums from the value-change reduction kernel and
    /// return their total.  Same layout as `download_error_sum`.
    pub async fn download_value_change_sum(
        ctx: &Arc<GpuContext>,
        model_bufs: &ModelBuffers,
    ) -> Result<f32> {
        let partial =
            read_buffer_f32(ctx, &model_bufs.value_change_sum, model_bufs.total_sums).await?;
        Ok(partial.iter().sum())
    }
}

async fn read_buffer_t<T>(
    ctx: &Arc<GpuContext>,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>>
where
    T: bytemuck::Pod,
{
    if count == 0 {
        return Ok(Vec::new());
    }

    let byte_len = (count * std::mem::size_of::<T>()) as u64;

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
    let data: Vec<T> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();

    Ok(data)
}

/// Helper: download `count` f32s from a GPU buffer to the CPU.
async fn read_buffer_f32(
    ctx: &Arc<GpuContext>,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f32>> {
    let promise: Result<Vec<f32>> = read_buffer_t(ctx, buffer, count).await;
    promise
}

/// Helper: download `count` u32s from a GPU buffer to the CPU.
async fn read_buffer_u32(
    ctx: &Arc<GpuContext>,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u32>> {
    let promise: Result<Vec<u32>> = read_buffer_t(ctx, buffer, count).await;
    promise
}

// ---------------------------------------------------------------------------
// Batch-parallel buffer set
// ---------------------------------------------------------------------------

/// Per-slot buffers that are independent across samples in a minibatch.
/// Each slot holds its own copy of values/predictions/errors/etc so that
/// all samples can converge on the GPU.
pub struct SlotBuffers {
    pub values: wgpu::Buffer,
    pub predictions: wgpu::Buffer,
    pub errors: wgpu::Buffer,
    pub gain_errors: wgpu::Buffer,
    pub value_changes: wgpu::Buffer,
    pub weight_deltas: wgpu::Buffer,
}

/// GPU-side storage for batch-parallel training
///
/// Shared buffers (weights, weight_deltas_accum, meta) are allocated once.
/// Per-sample state buffers are replicated `batch_size` times so that all
/// samples in a minibatch can converge in parallel.
pub struct BatchLayerBuffers {
    pub batch_size: u32,
    // --- Shared (one singlular buffer, shared) ---
    pub weights: wgpu::Buffer,
    pub weight_deltas_accum: wgpu::Buffer,
    pub meta: wgpu::Buffer,
    // --- Per-slot (batch_size copies) ---
    pub slots: Vec<SlotBuffers>,
    // Layer geometry (for dispatch sizing)
    pub size: usize,
    pub weight_rows: usize,
    pub weight_cols: usize,
}

impl BatchLayerBuffers {
    /// Allocate batch-parallel buffers for a single layer.
    pub fn from_snapshot(
        ctx: &Arc<GpuContext>,
        layer: &crate::model::snapshot::LayerSnapshot,
        is_top_level: bool,
        sum_offset: u32,
        batch_size: u32,
    ) -> Self {
        let device = &ctx.device;

        // Shared
        let weights = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("batch_layer_weights"),
            contents: if layer.weights.is_empty() {
                bytemuck::cast_slice(&[0.0_f32])
            } else {
                bytemuck::cast_slice(&layer.weights)
            },
            usage: BUF_USAGE,
        });

        // Shared
        let weight_delta_count = if layer.weights.is_empty() {
            1
        } else {
            layer.weights.len()
        };
        let weight_accum_zeros = vec![0.0_f32; weight_delta_count];
        let weight_deltas_accum = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("batch_layer_weight_deltas_accum"),
            contents: bytemuck::cast_slice(&weight_accum_zeros),
            usage: BUF_USAGE,
        });

        // Shared
        let meta_data: [u32; 7] = [
            layer.pinned as u32,
            super::buffers::activation_to_u32(layer.activation_function),
            layer.size as u32,
            layer.weight_rows as u32,
            layer.weight_cols as u32,
            is_top_level as u32,
            sum_offset,
        ];
        let meta = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("batch_layer_meta"),
            contents: bytemuck::cast_slice(&meta_data),
            usage: BUF_USAGE,
        });

        // Per-slot buffers
        let mut slots = Vec::with_capacity(batch_size as usize);
        for slot in 0..batch_size {
            let slot_label = |name: &str| format!("batch_s{slot}_{name}");

            let values_data = vec![0.0_f32; layer.size];
            let values = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&slot_label("values")),
                contents: bytemuck::cast_slice(&values_data),
                usage: BUF_USAGE,
            });

            let predictions_data = vec![0.0_f32; layer.size];
            let predictions = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&slot_label("predictions")),
                contents: bytemuck::cast_slice(&predictions_data),
                usage: BUF_USAGE,
            });

            let errors_data = vec![0.0_f32; layer.size];
            let errors = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&slot_label("errors")),
                contents: bytemuck::cast_slice(&errors_data),
                usage: BUF_USAGE,
            });

            let gain_error_count = if layer.weight_rows == 0 {
                1
            } else {
                layer.weight_rows
            };
            let gain_errors_data = vec![0.0_f32; gain_error_count];
            let gain_errors = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&slot_label("gain_errors")),
                contents: bytemuck::cast_slice(&gain_errors_data),
                usage: BUF_USAGE,
            });

            let value_changes_data = vec![0.0_f32; layer.size];
            let value_changes = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&slot_label("value_changes")),
                contents: bytemuck::cast_slice(&value_changes_data),
                usage: BUF_USAGE,
            });

            let weight_delta_zeros = vec![0.0_f32; weight_delta_count];
            let weight_deltas = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&slot_label("weight_deltas")),
                contents: bytemuck::cast_slice(&weight_delta_zeros),
                usage: BUF_USAGE,
            });

            slots.push(SlotBuffers {
                values,
                predictions,
                errors,
                gain_errors,
                value_changes,
                weight_deltas,
            });
        }

        Self {
            batch_size,
            weights,
            weight_deltas_accum,
            meta,
            slots,
            size: layer.size,
            weight_rows: layer.weight_rows,
            weight_cols: layer.weight_cols,
        }
    }
}

/// All batch-parallel layer buffers for the entire model, plus model-level params.
pub struct BatchModelBuffers {
    pub layers: Vec<BatchLayerBuffers>,
    pub batch_size: u32,
    /// `[alpha: f32, gamma: f32, convergence_threshold: f32, convergence_steps: u32, weight_clip: f32, 0, 0, 0]`
    pub params: wgpu::Buffer,
    /// Dummy buffer for layer 0's timestep bind group (lower_errors slot).
    pub dummy_lower_errors: wgpu::Buffer,
    /// Per-slot error_sum buffers for reduction (one per slot).
    pub error_sum: Vec<wgpu::Buffer>,
    /// Per-slot value_change_sum buffers for reduction (one per slot).
    pub value_change_sum: Vec<wgpu::Buffer>,
    /// Total number of f32 slots in each summing array.
    pub total_sums: usize,
}

impl BatchModelBuffers {
    /// Upload a full model snapshot to the GPU with batch-parallel per-slot buffers.
    pub fn from_snapshot(
        ctx: &Arc<GpuContext>,
        snapshot: &crate::model::snapshot::ModelSnapshot,
        batch_size: u32,
    ) -> Self {
        let num_layers = snapshot.layers.len();

        // Compute prefix-sum offsets for partial sum buffers.
        let mut offsets: Vec<u32> = Vec::with_capacity(num_layers);
        let mut running: u32 = 0;
        for l in &snapshot.layers {
            offsets.push(running);
            running += (l.size as u32).div_ceil(64);
        }
        let total_sums = running as usize;

        let layers: Vec<BatchLayerBuffers> = snapshot
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                BatchLayerBuffers::from_snapshot(
                    ctx,
                    l,
                    i == num_layers - 1,
                    offsets[i],
                    batch_size,
                )
            })
            .collect();

        // Model-level params uniform
        let params_data: [f32; 8] = [
            snapshot.config.alpha,
            snapshot.config.gamma,
            snapshot.config.convergence_threshold,
            snapshot.config.convergence_steps as f32,
            snapshot.config.weight_clip,
            0.0,
            0.0,
            0.0,
        ];
        let params = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("batch_model_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Dummy lower_errors for layer 0
        let dummy_size = if !layers.is_empty() {
            layers[0].size
        } else {
            1
        };
        let dummy_zeros = vec![0.0_f32; dummy_size];
        let dummy_lower_errors = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("batch_dummy_lower_errors"),
            contents: bytemuck::cast_slice(&dummy_zeros),
            usage: BUF_USAGE,
        });

        // Per-slot partial sum buffers
        let total_slots = total_sums.max(1);
        let mut error_sum = Vec::with_capacity(batch_size as usize);
        let mut value_change_sum = Vec::with_capacity(batch_size as usize);
        for slot in 0..batch_size {
            let zeros = vec![0.0_f32; total_slots];
            error_sum.push(ctx.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("batch_error_sum_s{slot}")),
                contents: bytemuck::cast_slice(&zeros),
                usage: BUF_USAGE,
            }));
            value_change_sum.push(ctx.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!("batch_vc_sum_s{slot}")),
                contents: bytemuck::cast_slice(&zeros),
                usage: BUF_USAGE,
            }));
        }

        Self {
            layers,
            batch_size,
            params,
            dummy_lower_errors,
            error_sum,
            value_change_sum,
            total_sums,
        }
    }
}
