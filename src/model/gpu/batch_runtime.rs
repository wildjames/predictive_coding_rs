use std::sync::Arc;

use crate::error::{PredictiveCodingError, Result};
use crate::model::{ModelSnapshot, PredictiveCodingModelConfig};

use super::buffers::BatchModelBuffers;
use super::context::GpuContext;
use super::layout::PcBindGroupLayouts;
use super::pipelines::PcPipelines;

/// Batch-parallel GPU runtime
///
/// Holds N copies of per-sample state buffers (one per batch slot) and a single
/// shared set of weights. All N samples converge simultaneously via batched
/// kernel dispatches, which I hope maximises the gpu usage during training.
pub struct GpuBatchRuntime {
    ctx: Arc<GpuContext>,
    config: PredictiveCodingModelConfig,
    buffers: BatchModelBuffers,
    #[allow(dead_code)]
    layouts: PcBindGroupLayouts,
    pipelines: PcPipelines,
    /// pe_bind_groups[slot][pair_idx] - predict/error bind groups per slot.
    pe_bind_groups: Vec<Vec<wgpu::BindGroup>>,
    /// pe_top_bind_groups[slot] - top-layer bind group per slot.
    pe_top_bind_groups: Vec<wgpu::BindGroup>,
    /// tw_bind_groups[slot][layer_idx] - timestep/weight bind groups per slot.
    tw_bind_groups: Vec<Vec<wgpu::BindGroup>>,
    batch_size: u32,
}

impl GpuBatchRuntime {
    /// Build a batch-parallel GPU runtime from a model snapshot.
    pub async fn from_snapshot_async(snapshot: &ModelSnapshot, batch_size: u32) -> Result<Self> {
        let ctx = GpuContext::new().await?;
        Self::from_snapshot_with_context_async(snapshot, ctx, batch_size).await
    }

    /// Like [`from_snapshot_async`] but reuses an existing [`GpuContext`].
    pub async fn from_snapshot_with_context_async(
        snapshot: &ModelSnapshot,
        ctx: Arc<GpuContext>,
        batch_size: u32,
    ) -> Result<Self> {
        let buffers = BatchModelBuffers::from_snapshot(&ctx, snapshot, batch_size);
        let layouts = PcBindGroupLayouts::new(&ctx);
        let pipelines = PcPipelines::new(&ctx, &layouts);

        let (pe_bind_groups, pe_top_bind_groups) =
            Self::build_all_pe_bind_groups(&ctx, &layouts, &buffers);
        let tw_bind_groups = Self::build_all_tw_bind_groups(&ctx, &layouts, &buffers);

        Ok(Self {
            ctx,
            config: snapshot.config.clone(),
            buffers,
            layouts,
            pipelines,
            pe_bind_groups,
            pe_top_bind_groups,
            tw_bind_groups,
            batch_size,
        })
    }

    /// Blocking wrapper around [`from_snapshot_async`].
    pub fn from_snapshot(snapshot: &ModelSnapshot, batch_size: u32) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PredictiveCodingError::validation(format!("tokio runtime creation failed: {e}"))
            })?;
        rt.block_on(Self::from_snapshot_async(snapshot, batch_size))
    }

    /// Blocking wrapper that reuses an existing [`GpuContext`].
    pub fn from_snapshot_with_context(
        snapshot: &ModelSnapshot,
        ctx: Arc<GpuContext>,
        batch_size: u32,
    ) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PredictiveCodingError::validation(format!("tokio runtime creation failed: {e}"))
            })?;
        rt.block_on(Self::from_snapshot_with_context_async(
            snapshot, ctx, batch_size,
        ))
    }

    // -------------------------------------------------------------------
    // Bind group construction
    // -------------------------------------------------------------------

    /// Build predict/error bind groups for all slots.
    fn build_all_pe_bind_groups(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        buffers: &BatchModelBuffers,
    ) -> (Vec<Vec<wgpu::BindGroup>>, Vec<wgpu::BindGroup>) {
        let n = buffers.layers.len();
        let batch_size = buffers.batch_size as usize;

        let mut all_pe: Vec<Vec<wgpu::BindGroup>> = Vec::with_capacity(batch_size);
        let mut all_top: Vec<wgpu::BindGroup> = Vec::with_capacity(batch_size);

        for slot in 0..batch_size {
            let mut slot_groups: Vec<wgpu::BindGroup> = Vec::with_capacity(n.saturating_sub(1));

            // For each adjacent pair (upper=i+1, lower=i), create a bind group
            // that uses slot-specific values/predictions/errors but shared weights/meta.
            for i in 0..n.saturating_sub(1) {
                let upper_layer = &buffers.layers[i + 1];
                let lower_layer = &buffers.layers[i];
                let label = format!("batch_pe_s{slot}_pair_{i}_{}", i + 1);

                // Build a LayerBuffers view for this slot
                let bg = Self::create_pe_bind_group_for_slot(
                    ctx,
                    layouts,
                    &upper_layer.slots[slot].values,
                    &upper_layer.weights,
                    &upper_layer.meta,
                    &lower_layer.slots[slot].values,
                    &lower_layer.slots[slot].predictions,
                    &lower_layer.slots[slot].errors,
                    &lower_layer.meta,
                    &buffers.params,
                    &buffers.error_sum[slot],
                    &label,
                );
                slot_groups.push(bg);
            }
            all_pe.push(slot_groups);

            // Top-layer bind group for this slot
            let top_idx = n - 1;
            let top_layer = &buffers.layers[top_idx];
            let top_label = format!("batch_pe_s{slot}_top");
            let top_bg = Self::create_pe_bind_group_for_slot(
                ctx,
                layouts,
                &top_layer.slots[slot].values,
                &top_layer.weights,
                &top_layer.meta,
                &top_layer.slots[slot].values,
                &top_layer.slots[slot].predictions,
                &top_layer.slots[slot].errors,
                &top_layer.meta,
                &buffers.params,
                &buffers.error_sum[slot],
                &top_label,
            );
            all_top.push(top_bg);
        }

        (all_pe, all_top)
    }

    /// Create a predict/error bind group with explicit buffer references.
    #[allow(clippy::too_many_arguments)]
    fn create_pe_bind_group_for_slot(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        upper_values: &wgpu::Buffer,
        upper_weights: &wgpu::Buffer,
        upper_meta: &wgpu::Buffer,
        lower_values: &wgpu::Buffer,
        lower_predictions: &wgpu::Buffer,
        lower_errors: &wgpu::Buffer,
        lower_meta: &wgpu::Buffer,
        params: &wgpu::Buffer,
        error_sum: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layouts.predict_error,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: upper_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: upper_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: upper_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: lower_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lower_predictions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: lower_errors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: lower_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: error_sum.as_entire_binding(),
                },
            ],
        })
    }

    /// Build timestep/weight bind groups for all slots.
    fn build_all_tw_bind_groups(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        buffers: &BatchModelBuffers,
    ) -> Vec<Vec<wgpu::BindGroup>> {
        let n = buffers.layers.len();
        let batch_size = buffers.batch_size as usize;

        let mut all_tw: Vec<Vec<wgpu::BindGroup>> = Vec::with_capacity(batch_size);

        for slot in 0..batch_size {
            let mut slot_groups: Vec<wgpu::BindGroup> = Vec::with_capacity(n);

            for i in 0..n {
                let layer = &buffers.layers[i];
                let lower_errors_buf = if i == 0 {
                    &buffers.dummy_lower_errors
                } else {
                    &buffers.layers[i - 1].slots[slot].errors
                };
                let label = format!("batch_tw_s{slot}_layer_{i}");

                let bg = Self::create_tw_bind_group_for_slot(
                    ctx,
                    layouts,
                    &buffers.params,
                    &layer.slots[slot].weight_deltas,
                    &layer.slots[slot].gain_errors,
                    &layer.meta,
                    &layer.slots[slot].values,
                    &layer.weights,
                    &layer.slots[slot].errors,
                    &layer.slots[slot].value_changes,
                    lower_errors_buf,
                    &buffers.value_change_sum[slot],
                    &layer.weight_deltas_accum,
                    &label,
                );
                slot_groups.push(bg);
            }
            all_tw.push(slot_groups);
        }

        all_tw
    }

    /// Create a timestep/weight bind group with explicit buffer references.
    #[allow(clippy::too_many_arguments)]
    fn create_tw_bind_group_for_slot(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        params: &wgpu::Buffer,
        weight_deltas: &wgpu::Buffer,
        gain_errors: &wgpu::Buffer,
        upper_meta: &wgpu::Buffer,
        upper_values: &wgpu::Buffer,
        upper_weights: &wgpu::Buffer,
        upper_errors: &wgpu::Buffer,
        upper_value_changes: &wgpu::Buffer,
        lower_errors: &wgpu::Buffer,
        value_change_sum: &wgpu::Buffer,
        weight_deltas_accum: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layouts.timestep_weight,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_deltas.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gain_errors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: upper_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: upper_values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: upper_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: upper_errors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: upper_value_changes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: lower_errors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: value_change_sum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: weight_deltas_accum.as_entire_binding(),
                },
            ],
        })
    }

    // -------------------------------------------------------------------
    // Public batch operations
    // -------------------------------------------------------------------

    /// Upload input/output for all batch slots at once.
    ///
    /// `samples` must have exactly `batch_size` elements, each `(input, output)`.
    pub fn set_batch_data(&self, samples: &[(Vec<f32>, Vec<f32>)]) -> Result<()> {
        let n = self.buffers.layers.len();
        let first_size = self.buffers.layers[0].size;
        let last_size = self.buffers.layers[n - 1].size;

        if !samples.len() == self.batch_size as usize {
            return Err(PredictiveCodingError::validation(format!(
                "expected {} samples, got {}",
                self.batch_size,
                samples.len()
            )));
        }

        for (slot, (input, output)) in samples.iter().enumerate() {
            if input.len() != first_size {
                return Err(PredictiveCodingError::validation(format!(
                    "slot {slot}: input length {} != expected {first_size}",
                    input.len()
                )));
            }
            if output.len() != last_size {
                return Err(PredictiveCodingError::validation(format!(
                    "slot {slot}: output length {} != expected {last_size}",
                    output.len()
                )));
            }

            // Upload input values to slot's layer 0
            self.ctx.queue.write_buffer(
                &self.buffers.layers[0].slots[slot].values,
                0,
                bytemuck::cast_slice(input),
            );

            // Upload output values to slot's last layer
            self.ctx.queue.write_buffer(
                &self.buffers.layers[n - 1].slots[slot].values,
                0,
                bytemuck::cast_slice(output),
            );
        }

        // pin the input and output layers
        self.ctx.queue.write_buffer(
            &self.buffers.layers[0].meta,
            0,
            bytemuck::cast_slice(&[1u32]),
        );
        self.ctx.queue.write_buffer(
            &self.buffers.layers[n - 1].meta,
            0,
            bytemuck::cast_slice(&[1u32]),
        );

        Ok(())
    }

    /// Reinitialise latent values for all batch slots (interior layers).
    pub fn reinitialise_all_latents(&self) {
        let mut rng = rand::rng();
        let n = self.buffers.layers.len();

        for slot in 0..self.batch_size as usize {
            for i in 1..n - 1 {
                let size = self.buffers.layers[i].size;
                let data: Vec<f32> = (0..size)
                    .map(|_| rand::RngExt::random_range(&mut rng, 0.0..1.0))
                    .collect();
                self.ctx.queue.write_buffer(
                    &self.buffers.layers[i].slots[slot].values,
                    0,
                    bytemuck::cast_slice(&data),
                );
            }
        }
    }

    /// Run the convergence loop for all slots simultaneously (fixed-step, no readback).
    ///
    /// Each iteration uses one compute pass per phase (predict, errors, gain_errors,
    /// timestep). Dispatches within a phase target independent per-slot buffers, so
    /// no intra-pass barriers are needed. The implicit barrier between passes
    /// synchronises the phases.
    pub fn converge_all(&self) -> Result<u32> {
        let steps = self.config.convergence_steps;
        let n = self.buffers.layers.len();

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_converge_encoder"),
            });

        for _ in 0..steps {
            // --- Predictions (top-down) ---
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.predict);
                for slot in 0..self.batch_size as usize {
                    for pair_idx in (0..self.pe_bind_groups[slot].len()).rev() {
                        let lower_size = self.buffers.layers[pair_idx].size as u32;
                        let workgroups = lower_size.div_ceil(64);
                        pass.set_bind_group(0, &self.pe_bind_groups[slot][pair_idx], &[]);
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                }
            }

            // --- Errors ---
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.errors);
                for slot in 0..self.batch_size as usize {
                    for (i, bg) in self.pe_bind_groups[slot].iter().enumerate() {
                        let lower_size = self.buffers.layers[i].size as u32;
                        let workgroups = lower_size.div_ceil(64);
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                    // Top layer
                    let top_size = self.buffers.layers[n - 1].size as u32;
                    let workgroups = top_size.div_ceil(64);
                    pass.set_bind_group(0, &self.pe_top_bind_groups[slot], &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }

            // --- Gain errors ---
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.compute_gain_errors);
                for slot in 0..self.batch_size as usize {
                    for i in 0..n {
                        let lb = &self.buffers.layers[i];
                        let total = lb.weight_rows as u32;
                        if total == 0 {
                            continue;
                        }
                        let workgroups = total.div_ceil(64);
                        pass.set_bind_group(0, &self.tw_bind_groups[slot][i], &[]);
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                }
            }

            // --- Timestep (value updates) ---
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.timestep);
                for slot in 0..self.batch_size as usize {
                    for i in 0..n {
                        let lb = &self.buffers.layers[i];
                        let total = lb.size as u32;
                        let workgroups = total.div_ceil(64);
                        pass.set_bind_group(0, &self.tw_bind_groups[slot][i], &[]);
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                }
            }
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(steps)
    }

    /// Zero the shared weight_deltas_accum buffers for all layers.
    pub fn zero_weight_accumulators(&self) {
        for layer in &self.buffers.layers {
            let byte_len =
                (layer.weight_rows * layer.weight_cols).max(1) * std::mem::size_of::<f32>();
            let zeros = vec![0u8; byte_len];
            self.ctx
                .queue
                .write_buffer(&layer.weight_deltas_accum, 0, &zeros);
        }
    }

    /// Accumulate weight deltas from all slots into the shared accum buffer.
    ///
    /// For each slot: dispatch compute_gain_errors -> compute_weight_deltas -> accumulate.
    pub fn accumulate_all_weight_deltas(&self) {
        let n = self.buffers.layers.len();

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_accumulate_encoder"),
            });

        for slot in 0..self.batch_size as usize {
            // Pass 1: compute_gain_errors for layers 1..n
            for i in 1..n {
                let lb = &self.buffers.layers[i];
                let total = lb.weight_rows as u32;
                if total == 0 {
                    continue;
                }
                let workgroups = total.div_ceil(64);

                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.compute_gain_errors);
                pass.set_bind_group(0, &self.tw_bind_groups[slot][i], &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Pass 2: compute_weight_deltas for layers 1..n
            for i in 1..n {
                let lb = &self.buffers.layers[i];
                let total = (lb.weight_rows * lb.weight_cols) as u32;
                if total == 0 {
                    continue;
                }
                let workgroups = total.div_ceil(64);

                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.compute_weight_deltas);
                pass.set_bind_group(0, &self.tw_bind_groups[slot][i], &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Pass 3: accumulate_weight_deltas for layers 1..n
            for i in 1..n {
                let lb = &self.buffers.layers[i];
                let total = (lb.weight_rows * lb.weight_cols) as u32;
                if total == 0 {
                    continue;
                }
                let workgroups = total.div_ceil(64);

                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.accumulate_weight_deltas);
                pass.set_bind_group(0, &self.tw_bind_groups[slot][i], &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Apply the accumulated weight deltas to the shared weight matrices.
    ///
    /// Uses a dedicated set of bind groups that reference the shared weights
    /// and the shared weight_deltas_accum. I reuse slot 0's tw_bind_groups
    /// since the weights and accum buffers are shared across all slots.
    pub fn apply_accumulated_weight_deltas(&self) {
        let n = self.buffers.layers.len();

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_apply_accum_encoder"),
            });

        // Use slot 0's bind groups - they reference the same shared weights/accum
        for i in 1..n {
            let lb = &self.buffers.layers[i];
            let total = (lb.weight_rows * lb.weight_cols) as u32;
            if total == 0 {
                continue;
            }
            let workgroups = total.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.apply_accumulated_weight_deltas);
            pass.set_bind_group(0, &self.tw_bind_groups[0][i], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Overwrite the `alpha` parameter in the GPU params uniform buffer.
    pub fn set_params_alpha(&self, alpha: f32) {
        self.ctx
            .queue
            .write_buffer(&self.buffers.params, 0, bytemuck::cast_slice(&[alpha]));
    }

    /// Block until all submitted GPU work has completed. Returns the wall-clock
    /// duration the CPU spent idle waiting for the GPU.
    pub fn poll_gpu(&self) -> std::time::Duration {
        let t = std::time::Instant::now();
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        t.elapsed()
    }

    /// Human-readable description of the GPU adapter.
    pub fn gpu_description(&self) -> String {
        self.ctx.adapter_description()
    }

    /// Get the model config.
    pub fn config(&self) -> &PredictiveCodingModelConfig {
        &self.config
    }

    /// Download the current weights from the GPU to produce a snapshot.
    ///
    /// Uses a blocking tokio runtime for the async readback.
    pub fn snapshot(&self) -> Result<ModelSnapshot> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PredictiveCodingError::validation(format!("tokio runtime creation failed: {e}"))
            })?;

        let mut layers = Vec::with_capacity(self.buffers.layers.len());
        for lb in self.buffers.layers.iter() {
            // Download shared weights
            let weight_count = lb.weight_rows * lb.weight_cols;
            let weights = if weight_count > 0 {
                rt.block_on(read_buffer_f32_pub(&self.ctx, &lb.weights, weight_count))?
            } else {
                vec![]
            };

            // Use slot 0's values/predictions/errors for the snapshot
            let values =
                rt.block_on(read_buffer_f32_pub(&self.ctx, &lb.slots[0].values, lb.size))?;
            let predictions = rt.block_on(read_buffer_f32_pub(
                &self.ctx,
                &lb.slots[0].predictions,
                lb.size,
            ))?;
            let errors =
                rt.block_on(read_buffer_f32_pub(&self.ctx, &lb.slots[0].errors, lb.size))?;

            // Download meta to get pinned/activation
            let meta = rt.block_on(read_buffer_u32_pub(&self.ctx, &lb.meta, 7))?;
            let pinned = meta[0] != 0;
            let activation_function = super::buffers::activation_function_from_u32(meta[1])?;

            layers.push(crate::model::snapshot::LayerSnapshot {
                values,
                predictions,
                errors,
                weights,
                weight_rows: lb.weight_rows,
                weight_cols: lb.weight_cols,
                pinned,
                activation_function,
                size: lb.size,
            });
        }

        Ok(ModelSnapshot {
            config: self.config.clone(),
            layers,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper functions for buffer readback (used by snapshot)
// ---------------------------------------------------------------------------

async fn read_buffer_f32_pub(
    ctx: &Arc<GpuContext>,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let byte_len = (count * std::mem::size_of::<f32>()) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("batch_staging_read"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_download_encoder"),
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
        .map_err(|e| PredictiveCodingError::validation(format!("GPU readback failed: {e}")))?
        .map_err(|e| PredictiveCodingError::validation(format!("GPU buffer map failed: {e}")))?;

    let view = staging.get_mapped_range(..);
    let data: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();
    Ok(data)
}

async fn read_buffer_u32_pub(
    ctx: &Arc<GpuContext>,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let byte_len = (count * std::mem::size_of::<u32>()) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("batch_staging_read_u32"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch_download_encoder_u32"),
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
        .map_err(|e| PredictiveCodingError::validation(format!("GPU readback failed: {e}")))?
        .map_err(|e| PredictiveCodingError::validation(format!("GPU buffer map failed: {e}")))?;

    let view = staging.get_mapped_range(..);
    let data: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();
    Ok(data)
}
