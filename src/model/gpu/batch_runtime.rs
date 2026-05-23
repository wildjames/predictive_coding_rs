use std::sync::Arc;

use crate::error::{PredictiveCodingError, Result};
use crate::model::{ModelSnapshot, PredictiveCodingModelConfig};

use super::buffers::BatchModelBuffers;
use super::context::GpuContext;
use super::layout::PcBindGroupLayouts;
use super::pipelines::PcPipelines;

/// Batch-parallel GPU runtime (fused dispatch)
///
/// All batch slots' per-sample state is stored in single fused buffers (slot k's
/// data starts at offset `k * layer_size`). Convergence dispatches use the Y
/// dimension for the slot index so the GPU processes all slots in parallel within
/// a single dispatch call per layer.
pub struct GpuRuntime {
    ctx: Arc<GpuContext>,
    config: PredictiveCodingModelConfig,
    buffers: BatchModelBuffers,
    #[allow(dead_code)]
    layouts: PcBindGroupLayouts,
    pipelines: PcPipelines,
    /// pe_bind_groups[pair_idx] - predict/error bind group per layer pair.
    /// Binds fused buffers for upper layer (i+1) and lower layer (i).
    pe_bind_groups: Vec<wgpu::BindGroup>,
    /// Top-layer bind group (top layer as both upper and lower for error computation).
    pe_top_bind_group: wgpu::BindGroup,
    /// tw_bind_groups[layer_idx] - timestep/weight bind group per layer.
    /// Binds fused buffers for the given layer.
    tw_bind_groups: Vec<wgpu::BindGroup>,
    batch_size: u32,
}

impl GpuRuntime {
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

        let pe_bind_groups = Self::build_pe_bind_groups(&ctx, &layouts, &buffers);
        let pe_top_bind_group = Self::build_pe_top_bind_group(&ctx, &layouts, &buffers);
        let tw_bind_groups = Self::build_tw_bind_groups(&ctx, &layouts, &buffers);

        Ok(Self {
            ctx,
            config: snapshot.config.clone(),
            buffers,
            layouts,
            pipelines,
            pe_bind_groups,
            pe_top_bind_group,
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

    /// Build predict/error bind groups - one per adjacent layer pair.
    /// Each bind group references the fused buffers (all slots concatenated).
    fn build_pe_bind_groups(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        buffers: &BatchModelBuffers,
    ) -> Vec<wgpu::BindGroup> {
        let n = buffers.layers.len();
        let mut groups = Vec::with_capacity(n.saturating_sub(1));

        for i in 0..n.saturating_sub(1) {
            let upper_layer = &buffers.layers[i + 1];
            let lower_layer = &buffers.layers[i];
            let label = format!("batch_pe_pair_{i}_{}", i + 1);

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&label),
                layout: &layouts.predict_error,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: upper_layer.values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: upper_layer.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: upper_layer.meta.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lower_layer.values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: lower_layer.predictions.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: lower_layer.errors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: lower_layer.meta.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: buffers.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: buffers.error_sum.as_entire_binding(),
                    },
                ],
            });
            groups.push(bg);
        }

        groups
    }

    /// Build top-layer bind group (top layer as both upper and lower).
    fn build_pe_top_bind_group(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        buffers: &BatchModelBuffers,
    ) -> wgpu::BindGroup {
        let top = &buffers.layers[buffers.layers.len() - 1];
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_pe_top"),
            layout: &layouts.predict_error,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: top.values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: top.weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: top.meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: top.values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: top.predictions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: top.errors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: top.meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.error_sum.as_entire_binding(),
                },
            ],
        })
    }

    /// Build timestep/weight bind groups - one per layer.
    fn build_tw_bind_groups(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        buffers: &BatchModelBuffers,
    ) -> Vec<wgpu::BindGroup> {
        let n = buffers.layers.len();
        let mut groups = Vec::with_capacity(n);

        for i in 0..n {
            let layer = &buffers.layers[i];
            let lower_errors_buf = if i == 0 {
                &buffers.dummy_lower_errors
            } else {
                &buffers.layers[i - 1].errors
            };
            let label = format!("batch_tw_layer_{i}");

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&label),
                layout: &layouts.timestep_weight,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: layer.weight_deltas.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: layer.gain_errors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: layer.meta.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: layer.values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: layer.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: layer.errors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: layer.value_changes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: lower_errors_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: buffers.value_change_sum.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: layer.weight_deltas_accum.as_entire_binding(),
                    },
                ],
            });
            groups.push(bg);
        }

        groups
    }

    // -------------------------------------------------------------------
    // Public batch operations
    // -------------------------------------------------------------------

    /// Upload input/output for all batch slots at once.
    ///
    /// `samples` must have exactly `batch_size` elements, each `(input, output)`.
    /// Data is written into the fused buffers at the appropriate slot offset.
    pub fn set_batch_data(&self, samples: &[(Vec<f32>, Vec<f32>)]) -> Result<()> {
        let n = self.buffers.layers.len();
        let first_size = self.buffers.layers[0].size;
        let last_size = self.buffers.layers[n - 1].size;

        if samples.len() != self.batch_size as usize {
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

            // Write input values at slot offset in fused layer 0 values buffer
            let input_byte_offset = (slot * first_size * std::mem::size_of::<f32>()) as u64;
            self.ctx.queue.write_buffer(
                &self.buffers.layers[0].values,
                input_byte_offset,
                bytemuck::cast_slice(input),
            );

            // Write output values at slot offset in fused last layer values buffer
            let output_byte_offset = (slot * last_size * std::mem::size_of::<f32>()) as u64;
            self.ctx.queue.write_buffer(
                &self.buffers.layers[n - 1].values,
                output_byte_offset,
                bytemuck::cast_slice(output),
            );
        }

        // Pin the input and output layers
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
    /// Data is written into the fused values buffers at appropriate slot offsets.
    pub fn reinitialise_all_latents(&self) {
        let mut rng = rand::rng();
        let n = self.buffers.layers.len();

        for i in 1..n - 1 {
            let size = self.buffers.layers[i].size;
            // Generate random data for ALL slots at once (contiguous)
            let total_elements = size * self.batch_size as usize;
            let data: Vec<f32> = (0..total_elements)
                .map(|_| rand::RngExt::random_range(&mut rng, 0.0..1.0))
                .collect();
            self.ctx.queue.write_buffer(
                &self.buffers.layers[i].values,
                0,
                bytemuck::cast_slice(&data),
            );
        }
    }

    /// Run the convergence loop for all slots simultaneously using fused dispatches.
    ///
    /// Each iteration dispatches one compute pass per phase (predict, errors,
    /// gain_errors, timestep) with Y=batch_size so all slots are processed in
    /// a single GPU dispatch per layer. The implicit barrier between compute
    /// passes synchronises phases.
    ///
    /// Convergence is split into chunks of iterations, each submitted as a
    /// separate command buffer, to avoid GPU timeout (TDR) on long runs.
    pub fn converge_all(&self) -> Result<u32> {
        let steps = self.config.convergence_steps;
        let n = self.buffers.layers.len();
        let bs = self.batch_size;

        // Submit in chunks to avoid TDR on Windows
        const CHUNK_SIZE: u32 = 10;
        let mut remaining = steps;

        while remaining > 0 {
            let chunk = remaining.min(CHUNK_SIZE);
            remaining -= chunk;

            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch_converge_encoder"),
                });

            for _ in 0..chunk {
                // --- Predictions (top-down) ---
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipelines.predict);
                    for pair_idx in (0..self.pe_bind_groups.len()).rev() {
                        let lower_size = self.buffers.layers[pair_idx].size as u32;
                        let workgroups_x = lower_size.div_ceil(64);
                        pass.set_bind_group(0, &self.pe_bind_groups[pair_idx], &[]);
                        pass.dispatch_workgroups(workgroups_x, bs, 1);
                    }
                }

                // --- Errors ---
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipelines.errors);
                    for (i, bg) in self.pe_bind_groups.iter().enumerate() {
                        let lower_size = self.buffers.layers[i].size as u32;
                        let workgroups_x = lower_size.div_ceil(64);
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(workgroups_x, bs, 1);
                    }
                    // Top layer
                    let top_size = self.buffers.layers[n - 1].size as u32;
                    let workgroups_x = top_size.div_ceil(64);
                    pass.set_bind_group(0, &self.pe_top_bind_group, &[]);
                    pass.dispatch_workgroups(workgroups_x, bs, 1);
                }

                // --- Gain errors ---
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipelines.compute_gain_errors);
                    for i in 0..n {
                        let lb = &self.buffers.layers[i];
                        let total = lb.weight_rows as u32;
                        if total == 0 {
                            continue;
                        }
                        let workgroups_x = total.div_ceil(64);
                        pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
                        pass.dispatch_workgroups(workgroups_x, bs, 1);
                    }
                }

                // --- Timestep (value updates) ---
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&self.pipelines.timestep);
                    for i in 0..n {
                        let lb = &self.buffers.layers[i];
                        let total = lb.size as u32;
                        let workgroups_x = total.div_ceil(64);
                        pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
                        pass.dispatch_workgroups(workgroups_x, bs, 1);
                    }
                }
            }

            self.ctx.queue.submit(std::iter::once(encoder.finish()));
        }

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
    /// Three phases:
    ///   1. compute_gain_errors: all slots in parallel (Y=batch_size)
    ///   2. For each slot: set params.current_slot, compute_weight_deltas (Y=1),
    ///      then accumulate_weight_deltas (Y=1)
    pub fn accumulate_all_weight_deltas(&self) {
        let n = self.buffers.layers.len();
        let bs = self.batch_size;

        // Phase 1: compute_gain_errors for all slots in parallel (fused gain_errors buffer)
        {
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch_gain_errors_encoder"),
                });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.compute_gain_errors);
            for i in 1..n {
                let lb = &self.buffers.layers[i];
                let total = lb.weight_rows as u32;
                if total == 0 {
                    continue;
                }
                let workgroups_x = total.div_ceil(64);
                pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
                pass.dispatch_workgroups(workgroups_x, bs, 1);
            }
            drop(pass);
            self.ctx.queue.submit(std::iter::once(encoder.finish()));
        }

        // Phase 2 & 3: For each slot, compute weight deltas then accumulate
        for slot in 0..bs {
            // Write current_slot to params (offset 6 * 4 = 24 bytes)
            self.ctx.queue.write_buffer(
                &self.buffers.params,
                24,
                bytemuck::cast_slice(&[slot as f32]),
            );

            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch_wd_slot_encoder"),
                });

            // compute_weight_deltas (reads slot from params.current_slot)
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.compute_weight_deltas);
                for i in 1..n {
                    let lb = &self.buffers.layers[i];
                    let total = (lb.weight_rows * lb.weight_cols) as u32;
                    if total == 0 {
                        continue;
                    }
                    let workgroups_x = total.div_ceil(64);
                    pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
                    pass.dispatch_workgroups(workgroups_x, 1, 1);
                }
            }

            // accumulate_weight_deltas (adds weight_deltas into accum)
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.accumulate_weight_deltas);
                for i in 1..n {
                    let lb = &self.buffers.layers[i];
                    let total = (lb.weight_rows * lb.weight_cols) as u32;
                    if total == 0 {
                        continue;
                    }
                    let workgroups_x = total.div_ceil(64);
                    pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
                    pass.dispatch_workgroups(workgroups_x, 1, 1);
                }
            }

            self.ctx.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    /// Apply the accumulated weight deltas to the shared weight matrices.
    pub fn apply_accumulated_weight_deltas(&self) {
        let n = self.buffers.layers.len();

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_apply_accum_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.apply_accumulated_weight_deltas);
            for i in 1..n {
                let lb = &self.buffers.layers[i];
                let total = (lb.weight_rows * lb.weight_cols) as u32;
                if total == 0 {
                    continue;
                }
                let workgroups_x = total.div_ceil(64);
                pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
                pass.dispatch_workgroups(workgroups_x, 1, 1);
            }
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Overwrite the `alpha` parameter in the GPU params uniform buffer.
    pub fn set_params_alpha(&self, alpha: f32) {
        self.ctx
            .queue
            .write_buffer(&self.buffers.params, 0, bytemuck::cast_slice(&[alpha]));
    }

    /// Compute total energy (0.5 * sum of squared errors) for batch slot 0.
    ///
    /// Dispatches the `reduce_error_sq` shader with Y=1 so only slot 0 is
    /// reduced, then downloads and sums the partial-sum buffer.
    pub fn total_energy(&self) -> Result<f32> {
        // Dispatch reduce_error_sq over all pe_bind_groups + top
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch_reduce_error_sq"),
            });

        for (i, bg) in self.pe_bind_groups.iter().enumerate() {
            let lower_size = self.buffers.layers[i].size as u32;
            let workgroups_x = lower_size.div_ceil(64);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.sum_error_sq);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        // Top layer
        let top_idx = self.buffers.layers.len() - 1;
        let top_size = self.buffers.layers[top_idx].size as u32;
        let workgroups_x = top_size.div_ceil(64);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.sum_error_sq);
            pass.set_bind_group(0, &self.pe_top_bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        // Download partial sums
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PredictiveCodingError::validation(format!("tokio runtime creation failed: {e}"))
            })?;
        let partial = rt.block_on(read_buffer_f32_pub(
            &self.ctx,
            &self.buffers.error_sum,
            self.buffers.total_sums,
        ))?;
        let sum_sq: f32 = partial.iter().sum();
        Ok(0.5 * sum_sq)
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
    /// Uses slot 0's values/predictions/errors for the snapshot.
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

            // Download slot 0's values/predictions/errors from fused buffers
            let values = rt.block_on(read_buffer_f32_pub(&self.ctx, &lb.values, lb.size))?;
            let predictions =
                rt.block_on(read_buffer_f32_pub(&self.ctx, &lb.predictions, lb.size))?;
            let errors = rt.block_on(read_buffer_f32_pub(&self.ctx, &lb.errors, lb.size))?;

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
