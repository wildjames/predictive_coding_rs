use std::sync::Arc;

use crate::error::{PredictiveCodingError, Result};
use crate::model::gpu::buffers::activation_function_from_u32;
use crate::model::{
    ExecutionBackend, ModelRuntime, ModelSnapshot, PredictiveCodingModelConfig,
    TrainableModelRuntime, WeightUpdateSet, snapshot::LayerSnapshot,
};

use super::buffers::ModelBuffers;
use super::context::GpuContext;
use super::layout::{
    PcBindGroupLayouts, create_predict_error_bind_group, create_timestep_weight_bind_group,
};
use super::pipelines::PcPipelines;

/// GPU runtime for a predictive-coding model.
///
/// Owns the GPU buffers and pre-compiled pipelines.  All heavy compute is
/// dispatched to the GPU and orchestration (convergence loop, etc.)
/// stays on the CPU.
pub struct GpuModelRuntime {
    ctx: Arc<GpuContext>,
    config: PredictiveCodingModelConfig,
    buffers: ModelBuffers,
    #[allow(dead_code)]
    layouts: PcBindGroupLayouts,
    pipelines: PcPipelines,
    /// Predict/error bind groups - one per adjacent layer pair (len = num_layers - 1).
    /// pe_bind_groups[i] binds upper=layer[i+1], lower=layer[i].
    pe_bind_groups: Vec<wgpu::BindGroup>,
    /// Timestep/weight-update bind groups - one per layer (len = num_layers).
    /// tw_bind_groups[i] operates on layer[i] as the "upper" layer.
    /// For layer 0, lower_errors is a dummy buffer (rhs will be 0).
    tw_bind_groups: Vec<wgpu::BindGroup>,
    /// Tokio runtime used to block on async GPU work from synchronous trait methods.
    rt: tokio::runtime::Runtime,
}

/// These are GPU-specific methods not exposed by the ModelRuntime trait.
impl GpuModelRuntime {
    /// Build a new GPU runtime from a model snapshot.
    ///
    /// This is async because device creation is async.  Use
    /// [`GpuModelRuntime::from_snapshot`] if you need a sync entry point.
    pub async fn from_snapshot_async(snapshot: &ModelSnapshot) -> Result<Self> {
        let ctx = GpuContext::new().await?;
        Self::from_snapshot_with_context_async(snapshot, ctx).await
    }

    /// Like [`from_snapshot_async`] but reuses an existing [`GpuContext`].
    pub async fn from_snapshot_with_context_async(
        snapshot: &ModelSnapshot,
        ctx: Arc<GpuContext>,
    ) -> Result<Self> {
        let buffers = ModelBuffers::from_snapshot(&ctx, snapshot);
        let layouts = PcBindGroupLayouts::new(&ctx);
        let pipelines = PcPipelines::new(&ctx, &layouts);

        let pe_bind_groups: Vec<wgpu::BindGroup> =
            Self::build_pe_bind_groups(&ctx, &layouts, &buffers);
        let tw_bind_groups: Vec<wgpu::BindGroup> =
            Self::build_tw_bind_groups(&ctx, &layouts, &buffers);

        // Used for blocking the async readback for sync trait methods
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all() // Enable time and IO drivers, needed for async GPU readback
            .build()
            .map_err(|e| {
                PredictiveCodingError::validation(format!(
                    "failed to create tokio runtime for GPU readback: {e}"
                ))
            })?;

        Ok(Self {
            ctx,
            config: snapshot.config.clone(),
            buffers,
            layouts,
            pipelines,
            pe_bind_groups,
            tw_bind_groups,
            rt,
        })
    }

    /// Blocking wrapper around [`from_snapshot_async`].
    pub fn from_snapshot(snapshot: &ModelSnapshot) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PredictiveCodingError::validation(format!("tokio runtime creation failed: {e}"))
            })?;
        rt.block_on(Self::from_snapshot_async(snapshot))
    }

    /// Blocking wrapper that reuses an existing [`GpuContext`].
    pub fn from_snapshot_with_context(
        snapshot: &ModelSnapshot,
        ctx: Arc<GpuContext>,
    ) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PredictiveCodingError::validation(format!("tokio runtime creation failed: {e}"))
            })?;
        rt.block_on(Self::from_snapshot_with_context_async(snapshot, ctx))
    }

    /// Build predict/error bind groups - one per adjacent layer pair.
    fn build_pe_bind_groups(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        buffers: &ModelBuffers,
    ) -> Vec<wgpu::BindGroup> {
        let n: usize = buffers.layers.len();
        let mut groups: Vec<wgpu::BindGroup> = Vec::with_capacity(n.saturating_sub(1));
        for i in 0..n.saturating_sub(1) {
            let label: String = format!("pe_pair_{i}_{}", i + 1);
            groups.push(create_predict_error_bind_group(
                ctx,
                layouts,
                &buffers.layers[i + 1],
                &buffers.layers[i],
                &buffers.error_sum,
                &buffers.params,
                &label,
            ));
        }
        groups
    }

    /// Build timestep/weight-update bind groups - one per layer.
    ///
    /// `tw_bind_groups[i]` treats `layer[i]` as the "upper" layer.
    /// For layer 0 (no lower layer), we bind its own errors buffer as
    /// `lower_errors`; the kernel will produce rhs=0 because weight_rows==0.
    /// For layers 1..N, `lower_errors` comes from `layer[i-1]`.
    fn build_tw_bind_groups(
        ctx: &Arc<GpuContext>,
        layouts: &PcBindGroupLayouts,
        buffers: &ModelBuffers,
    ) -> Vec<wgpu::BindGroup> {
        let n: usize = buffers.layers.len();
        let mut groups: Vec<wgpu::BindGroup> = Vec::with_capacity(n);
        for i in 0..n {
            // The bottom layer has no layer below it, so I bind a dummy buffer for the error readout
            // Since for that layer, weight_rows == 0, the shade skips it anyway.
            let lower_errors_buf = if i == 0 {
                &buffers.dummy_lower_errors
            } else {
                &buffers.layers[i - 1].errors
            };
            let label: String = format!("tw_layer_{i}");
            groups.push(create_timestep_weight_bind_group(
                ctx,
                layouts,
                &buffers.layers[i],
                lower_errors_buf,
                &buffers.value_change_sum,
                &buffers.params,
                &label,
            ));
        }
        groups
    }

    // -------------------------------------------------------------------
    // Dispatch helpers
    // -------------------------------------------------------------------

    /// Dispatch the prediction kernel for every adjacent layer pair (top-down).
    fn dispatch_predictions(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("predict_encoder"),
            });

        // Iterate top-down: from the highest pair to the lowest.
        for i in (0..self.pe_bind_groups.len()).rev() {
            let lower_size = self.buffers.layers[i].size as u32;
            let workgroups = lower_size.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.predict);
            pass.set_bind_group(0, &self.pe_bind_groups[i], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch the error kernel for every layer.
    ///
    /// Layers 0..N-2 use the pair bind groups on the GPU. The top layer has no
    /// layer above it, so we compute its errors (`values - predictions`) via a
    /// CPU round-trip (download values & predictions, compute, upload).
    fn dispatch_errors(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("errors_encoder"),
            });

        for (i, bg) in self.pe_bind_groups.iter().enumerate() {
            let lower_size = self.buffers.layers[i].size as u32;
            let workgroups = lower_size.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.errors);
            pass.set_bind_group(0, bg, &[]);
            // Note that this does NOT submit immediately, they're being bundled up here
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // The top layer also needs to have its errors computed, but has no pair bind group.
        // Since the error computation only touches the lower layer, we can make a special
        // bind group that uses the top layer values as both upper and lower, and the kernel will
        // only use the lower layer values to compute the errors.
        let top_idx = self.buffers.layers.len() - 1;
        let top_bg = create_predict_error_bind_group(
            &self.ctx,
            &self.layouts,
            &self.buffers.layers[top_idx],
            &self.buffers.layers[top_idx],
            &self.buffers.error_sum,
            &self.buffers.params,
            "pe_top_layer",
        );
        let lower_size = self.buffers.layers[top_idx].size as u32;
        let workgroups = lower_size.div_ceil(64);

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.errors);
        pass.set_bind_group(0, &top_bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

        // pass borrowed the encoder, so drop it before submitting the command buffer
        drop(pass);

        // And actually moved to the GPU here, in a single move.
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch `reduce_error_sq` for every layer in a single encoder submit.
    ///
    /// Each layer writes to its own offset in the shared `partial_sums` buffer, so all layers can be
    /// batched without overwriting each other.
    fn dispatch_reduce_error_sq(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reduce_error_sq_encoder"),
            });

        // Layers 0..N-2 via the existing pe_bind_groups.
        for (i, bg) in self.pe_bind_groups.iter().enumerate() {
            let lower_size = self.buffers.layers[i].size as u32;
            let workgroups = lower_size.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.sum_error_sq);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Top layer has no pe_bind_group, create a temporary one with the
        // top layer as both upper and lower (the shader only reads
        // lower_errors / lower_meta).
        let top_idx = self.buffers.layers.len() - 1;
        let top_bg = create_predict_error_bind_group(
            &self.ctx,
            &self.layouts,
            &self.buffers.layers[top_idx],
            &self.buffers.layers[top_idx],
            &self.buffers.error_sum,
            &self.buffers.params,
            "pe_top_layer_energy",
        );
        let top_size = self.buffers.layers[top_idx].size as u32;
        let workgroups = top_size.div_ceil(64);

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.sum_error_sq);
        pass.set_bind_group(0, &top_bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch `reduce_value_change` for every layer in a single encoder submit.
    ///
    /// Each layer writes to its own offset in the shared `value_change_sum` buffer,
    /// so all layers can be batched without overwriting each other.
    /// Unlike `dispatch_reduce_error_sq`, the tw_bind_groups cover all layers
    /// (including the top layer), so no special case is needed.
    fn dispatch_reduce_value_change(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reduce_value_change_encoder"),
            });

        for (i, bg) in self.tw_bind_groups.iter().enumerate() {
            let size = self.buffers.layers[i].size as u32;
            let workgroups = size.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.sum_value_change);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Dispatch the timestep kernel for every layer.
    ///
    /// Layer 0 uses the bottom self-bind-group.  Layers 1..N use `bind_groups[i-1]`
    /// which has upper=layer[i], lower=layer[i-1].
    ///
    /// Two passes:
    ///   1. compute_gain_errors: precompute f'(W·x) ⊙ lower_errors per row
    ///   2. values_timestep: use precomputed gain_errors to update values
    ///
    /// All layers are independent within each pass, so they are batched in a
    /// single command encoder per pass.
    fn dispatch_timestep(&self) {
        // Pass 1: compute gain_errors for all layers
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gain_errors_encoder"),
            });

        for (i, bg) in self.tw_bind_groups.iter().enumerate() {
            let weight_rows = self.buffers.layers[i].weight_rows as u32;
            if weight_rows == 0 {
                continue;
            }
            let workgroups = weight_rows.div_ceil(64);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.compute_gain_errors);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Pass 2: values_timestep using precomputed gain_errors
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestep_encoder"),
            });

        for (i, bg) in self.tw_bind_groups.iter().enumerate() {
            let size = self.buffers.layers[i].size as u32;
            let workgroups = size.div_ceil(64);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.timestep);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Download `value_changes` from all layers and return the mean absolute
    /// value change (same metric as the CPU backend).
    fn read_total_value_change(&self) -> Result<f32> {
        self.dispatch_reduce_value_change();

        let total: f32 = self.rt.block_on(ModelBuffers::download_value_change_sum(
            &self.ctx,
            &self.buffers,
        ))?;

        let total_nodes: usize = self.buffers.layers.iter().map(|lb| lb.size).sum();
        Ok(total / total_nodes as f32)
    }

    /// Dispatch the weight-update kernel for every layer pair (three-pass).
    ///
    /// Pass 1: compute gain_errors (preactivation derivatives * lower errors)
    /// Pass 2: compute deltas into a separate buffer
    /// Pass 3: apply deltas to the weight matrix
    fn dispatch_weight_updates(&self) {
        self.dispatch_compute_weight_deltas();
        self.dispatch_apply_weight_deltas();
    }

    /// Dispatch passes 1+2 of the weight update: compute gain_errors then
    /// compute weight deltas into the `weight_deltas` buffer (does NOT apply).
    fn dispatch_compute_weight_deltas(&self) {
        // Pass 1: compute_gain_errors
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("weight_gain_errors_encoder"),
            });

        for i in 1..self.tw_bind_groups.len() {
            let lb = &self.buffers.layers[i];
            let weight_rows = lb.weight_rows as u32;
            if weight_rows == 0 {
                continue;
            }
            let workgroups: u32 = weight_rows.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.compute_gain_errors);
            pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Pass 2: compute_weight_deltas
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("weight_deltas_encoder"),
            });

        for i in 1..self.tw_bind_groups.len() {
            let lb = &self.buffers.layers[i];
            let total_weights = (lb.weight_rows * lb.weight_cols) as u32;
            if total_weights == 0 {
                continue;
            }
            let workgroups = total_weights.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.compute_weight_deltas);
            pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    fn dispatch_apply_weight_deltas(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apply_weight_deltas_encoder"),
            });

        for i in 1..self.tw_bind_groups.len() {
            let lb = &self.buffers.layers[i];
            let total_weights = (lb.weight_rows * lb.weight_cols) as u32;
            if total_weights == 0 {
                continue;
            }
            let workgroups = total_weights.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.apply_weight_deltas);
            pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    // -------------------------------------------------------------------
    // GPU-native minibatch helpers
    // -------------------------------------------------------------------

    /// Zero the `weight_deltas_accum` buffers for all layers.
    /// Call at the start of each minibatch.
    pub fn zero_weight_accumulators(&self) {
        for lb in &self.buffers.layers {
            let byte_len: usize =
                (lb.weight_rows * lb.weight_cols).max(1) * std::mem::size_of::<f32>();
            let zeros: Vec<u8> = vec![0u8; byte_len];
            self.ctx
                .queue
                .write_buffer(&lb.weight_deltas_accum, 0, &zeros);
        }
    }

    /// Dispatch gain_errors + weight_deltas + accumulate for all layers.
    /// This computes the weight deltas for the current sample and adds
    /// them to the accumulation buffer without applying to the weights.
    pub fn accumulate_weight_deltas_on_device(&self) {
        // Pass 1 and 2: compute gain_errors and weight_deltas
        self.dispatch_compute_weight_deltas();

        // Pass 3: accumulate into weight_deltas_accum
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("accumulate_weight_deltas_encoder"),
            });

        for i in 1..self.tw_bind_groups.len() {
            let lb = &self.buffers.layers[i];
            let total_weights: u32 = (lb.weight_rows * lb.weight_cols) as u32;
            if total_weights == 0 {
                continue;
            }
            let workgroups: u32 = total_weights.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.accumulate_weight_deltas);
            pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Apply the accumulated weight deltas to the weight matrices.
    /// Call once at the end of a minibatch.
    pub fn apply_accumulated_weight_deltas(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apply_accumulated_weight_deltas_encoder"),
            });

        for i in 1..self.tw_bind_groups.len() {
            let lb = &self.buffers.layers[i];
            let total_weights = (lb.weight_rows * lb.weight_cols) as u32;
            if total_weights == 0 {
                continue;
            }
            let workgroups = total_weights.div_ceil(64);

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.apply_accumulated_weight_deltas);
            pass.set_bind_group(0, &self.tw_bind_groups[i], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Overwrite the `alpha` parameter in the GPU params uniform buffer.
    /// Used by the minibatch handler to scale deltas by `1/batch_size`.
    pub fn set_params_alpha(&self, alpha: f32) {
        self.ctx
            .queue
            .write_buffer(&self.buffers.params, 0, bytemuck::cast_slice(&[alpha]));
    }

    /// Human-readable description of the GPU adapter.
    pub fn gpu_description(&self) -> String {
        self.ctx.adapter_description()
    }

    // -------------------------------------------------------------------
    // Buffer upload / download helpers
    // -------------------------------------------------------------------

    /// Upload a flat f32 slice into a layer's values buffer and update the
    /// pinned flag in the meta buffer.
    fn upload_values(&self, layer_idx: usize, data: &[f32], pinned: bool) {
        let lb = &self.buffers.layers[layer_idx];
        self.ctx
            .queue
            .write_buffer(&lb.values, 0, bytemuck::cast_slice(data));
        // Overwrite just the pinned flag (first u32 in meta).
        self.ctx
            .queue
            .write_buffer(&lb.meta, 0, bytemuck::cast_slice(&[pinned as u32]));
    }

    /// Updates the pinned flag for a layer in the meta buffer.
    fn set_pinned(&self, layer_idx: usize, pinned: bool) {
        let lb = &self.buffers.layers[layer_idx];
        self.ctx
            .queue
            .write_buffer(&lb.meta, 0, bytemuck::cast_slice(&[pinned as u32]));
    }

    fn download_layer_values(&self, layer_idx: usize) -> Result<Vec<f32>> {
        let lb = &self.buffers.layers[layer_idx];
        self.rt
            .block_on(ModelBuffers::download_values(&self.ctx, lb))
    }
}

// ---------------------------------------------------------------------------
// ModelRuntime trait implementation
// ---------------------------------------------------------------------------

impl ModelRuntime for GpuModelRuntime {
    fn backend(&self) -> ExecutionBackend {
        ExecutionBackend::Gpu
    }

    fn config(&self) -> PredictiveCodingModelConfig {
        self.config.clone()
    }

    fn layer_sizes(&self) -> Vec<usize> {
        self.config.layer_sizes.clone()
    }

    /// Download the current model snapshot from the GPU.
    /// This involves a full readback of all layer buffers, so can be slow.
    fn snapshot(&mut self) -> Result<ModelSnapshot> {
        let mut layers = Vec::with_capacity(self.buffers.layers.len());

        for lb in self.buffers.layers.iter() {
            let values = self
                .rt
                .block_on(ModelBuffers::download_values(&self.ctx, lb))?;
            let errors = self
                .rt
                .block_on(ModelBuffers::download_errors(&self.ctx, lb))?;
            let weights = self
                .rt
                .block_on(ModelBuffers::download_weights(&self.ctx, lb))?;
            let predictions = self
                .rt
                .block_on(ModelBuffers::download_predictions(&self.ctx, lb))?;
            let meta = self
                .rt
                .block_on(ModelBuffers::download_meta(&self.ctx, lb))?;

            let pinned: bool = meta[0] != 0;
            let activation_function: crate::model::maths::ActivationFunction =
                activation_function_from_u32(meta[1])?;

            layers.push(LayerSnapshot {
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

    fn set_input(&mut self, input_values: &[f32]) -> Result<()> {
        let expected = self.config.layer_sizes[0];
        if input_values.len() != expected {
            return Err(PredictiveCodingError::validation(format!(
                "input length {} does not match expected size {expected}",
                input_values.len()
            )));
        }
        self.upload_values(0, input_values, true);
        Ok(())
    }

    fn set_output(&mut self, output_values: &[f32]) -> Result<()> {
        let last = self.config.layer_sizes.len() - 1;
        let expected = self.config.layer_sizes[last];
        if output_values.len() != expected {
            return Err(PredictiveCodingError::validation(format!(
                "output length {} does not match expected size {expected}",
                output_values.len()
            )));
        }
        self.upload_values(last, output_values, true);
        Ok(())
    }

    fn pin_input(&mut self) -> Result<()> {
        self.set_pinned(0, true);
        Ok(())
    }

    fn unpin_input(&mut self) -> Result<()> {
        self.set_pinned(0, false);
        Ok(())
    }

    fn pin_output(&mut self) -> Result<()> {
        let last = self.buffers.layers.len() - 1;
        self.set_pinned(last, true);
        Ok(())
    }

    fn unpin_output(&mut self) -> Result<()> {
        let last = self.buffers.layers.len() - 1;
        self.set_pinned(last, false);
        Ok(())
    }

    fn reinitialise_latents(&mut self) -> Result<()> {
        // Re-upload random values for interior layers (skip first and last).
        // TODO: Can we dispatch a kernel to do this, so we avoid a data upload?
        let mut rng = rand::rng();
        for i in 1..self.buffers.layers.len() - 1 {
            let size = self.buffers.layers[i].size;
            let data: Vec<f32> = (0..size)
                .map(|_| rand::RngExt::random_range(&mut rng, 0.0..1.0))
                .collect();
            self.upload_values(i, &data, false);
        }
        Ok(())
    }

    fn compute_predictions_and_errors(&mut self) -> Result<()> {
        self.dispatch_predictions();
        self.dispatch_errors();
        Ok(())
    }

    fn timestep(&mut self) -> Result<f32> {
        self.dispatch_timestep();
        self.read_total_value_change()
    }

    fn converge_values(&mut self) -> Result<u32> {
        let mut convergence_count: u32 = 0;

        while convergence_count < self.config.convergence_steps {
            self.dispatch_predictions();
            self.dispatch_errors();
            self.dispatch_timestep();

            let mean_change = self.read_total_value_change()?;
            convergence_count += 1;

            if mean_change.abs() < self.config.convergence_threshold {
                break;
            }
        }

        Ok(convergence_count)
    }

    fn total_error(&mut self) -> Result<f32> {
        let mut total: f32 = 0.0;
        for lb in &self.buffers.layers {
            let errs = self
                .rt
                .block_on(ModelBuffers::download_errors(&self.ctx, lb))?;
            total += errs.iter().sum::<f32>();
        }
        Ok(total)
    }

    fn total_energy(&mut self) -> Result<f32> {
        // Tell the GPU to gather the errors
        self.dispatch_reduce_error_sq();

        let sum_sq: f32 = self
            .rt
            .block_on(ModelBuffers::download_error_sum(&self.ctx, &self.buffers))?;

        Ok(0.5 * sum_sq)
    }

    fn input_values(&mut self) -> Result<Vec<f32>> {
        self.download_layer_values(0)
    }

    fn output_values(&mut self) -> Result<Vec<f32>> {
        let last = self.buffers.layers.len() - 1;
        self.download_layer_values(last)
    }
}

// ---------------------------------------------------------------------------
// TrainableModelRuntime trait implementation
// ---------------------------------------------------------------------------

impl TrainableModelRuntime for GpuModelRuntime {
    fn compute_weight_updates(&mut self) -> Result<WeightUpdateSet> {
        // Dispatch gain_errors and weight_deltas on the GPU, then download
        // only the resulting weight_deltas buffers
        self.dispatch_compute_weight_deltas();

        let n = self.buffers.layers.len();
        let mut updates: Vec<Vec<f32>> = Vec::with_capacity(n.saturating_sub(1));
        let mut shapes: Vec<(usize, usize)> = Vec::with_capacity(n.saturating_sub(1));

        for i in 1..n {
            let lb = &self.buffers.layers[i];
            let deltas = self
                .rt
                .block_on(ModelBuffers::download_weight_deltas(&self.ctx, lb))?;
            shapes.push((lb.weight_rows, lb.weight_cols));
            updates.push(deltas);
        }

        Ok(WeightUpdateSet { updates, shapes })
    }

    fn apply_weight_updates(&mut self, updates: &WeightUpdateSet) -> Result<()> {
        // Upload the delta vectors into each layer's weight_deltas buffer,
        // then dispatch the apply kernel to add them to weights in-place.
        for (i, delta) in updates.updates.iter().enumerate() {
            let layer_idx = i + 1;
            let lb = &self.buffers.layers[layer_idx];
            self.ctx
                .queue
                .write_buffer(&lb.weight_deltas, 0, bytemuck::cast_slice(delta));
        }
        self.dispatch_apply_weight_deltas();
        Ok(())
    }

    /// Override to run the full weight update on the GPU without a round-trip.
    fn update_weights(&mut self) -> Result<()> {
        self.dispatch_weight_updates();
        Ok(())
    }
}
