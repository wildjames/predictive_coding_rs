use std::sync::Arc;

use super::context::GpuContext;

/// The way that data is laid out in the GPU buffers must be consistent between the Rust
/// code and the WGSL shaders. This module defines that layout, and offers helper functions.
/// Any changes to this layout MUST be reflected in the WGSL shaders, and vice versa!
///
/// Bind-group layouts shared by all predictive-coding compute pipelines.
///
/// Two layouts are used to stay within the 8 storage-buffers-per-stage limit:
///
/// - `predict_error`: used by `compute_predictions` and `compute_errors`.
/// - `timestep_weight`: used by `values_timestep`, `compute_weight_deltas`, and `apply_weight_deltas`.
pub struct PcBindGroupLayouts {
    /// Layout for the predict / error kernels.
    ///
    /// Binding 0: upper_values     (storage, read)
    /// Binding 1: upper_weights    (storage, read)
    /// Binding 2: upper_meta       (storage, read)
    /// Binding 3: lower_values     (storage, read)
    /// Binding 4: lower_preds      (storage, read_write)
    /// Binding 5: lower_errors     (storage, read_write)
    /// Binding 6: lower_meta       (storage, read)
    /// Binding 7: params           (uniform)
    /// Binding 8: shared_sum       (storage, read_write)
    pub predict_error: wgpu::BindGroupLayout,

    /// Layout for the timestep / weight-update kernels.
    ///
    /// Binding 0: params               (uniform)
    /// Binding 1: weight_deltas        (storage, read_write)
    /// Binding 2: gain_errors          (storage, read_write)
    /// Binding 3: upper_meta           (storage, read)
    /// Binding 4: upper_values         (storage, read_write)
    /// Binding 5: upper_weights        (storage, read_write)
    /// Binding 6: upper_errors         (storage, read_write)
    /// Binding 7: upper_value_changes  (storage, read_write)
    /// Binding 8: lower_errors         (storage, read)
    pub timestep_weight: wgpu::BindGroupLayout,
}

impl PcBindGroupLayouts {
    pub fn new(ctx: &Arc<GpuContext>) -> Self {
        let predict_error = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pc_predict_error_layout"),
                entries: &[
                    storage_entry(0, true),  // upper values
                    storage_entry(1, true),  // upper weights
                    storage_entry(2, true),  // upper meta
                    storage_entry(3, true),  // lower values
                    storage_entry(4, false), // lower predictions
                    storage_entry(5, false), // lower errors
                    storage_entry(6, true),  // lower meta
                    uniform_entry(7),        // params
                    storage_entry(8, false), // shared_sum
                ],
            });

        let timestep_weight =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pc_timestep_weight_layout"),
                    entries: &[
                        uniform_entry(0),        // params
                        storage_entry(1, false), // weight_deltas (rw)
                        storage_entry(2, false), // gain_errors (rw)
                        storage_entry(3, true),  // upper meta
                        storage_entry(4, false), // upper values (rw for timestep)
                        storage_entry(5, false), // upper weights (rw for weight update)
                        storage_entry(6, false), // upper errors (rw)
                        storage_entry(7, false), // upper value_changes (rw)
                        storage_entry(8, true),  // lower errors
                    ],
                });

        Self {
            predict_error,
            timestep_weight,
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create the predict/error bind group for one adjacent layer pair.
pub fn create_predict_error_bind_group(
    ctx: &Arc<GpuContext>,
    layouts: &PcBindGroupLayouts,
    upper: &super::buffers::LayerBuffers,
    lower: &super::buffers::LayerBuffers,
    error_sum_buffer: &wgpu::Buffer,
    params: &wgpu::Buffer,
    label: &str,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &layouts.predict_error,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: upper.values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: upper.weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: upper.meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: lower.values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: lower.predictions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: lower.errors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: lower.meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: error_sum_buffer.as_entire_binding(),
            },
        ],
    })
}

/// Create the timestep/weight-update bind group for one layer pair.
///
/// The "upper" layer is the one whose values/weights get updated.
/// `lower_errors` come from the layer below.
pub fn create_timestep_weight_bind_group(
    ctx: &Arc<GpuContext>,
    layouts: &PcBindGroupLayouts,
    upper: &super::buffers::LayerBuffers,
    lower_errors: &wgpu::Buffer,
    params: &wgpu::Buffer,
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
                resource: upper.weight_deltas.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: upper.gain_errors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: upper.meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: upper.values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: upper.weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: upper.errors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: upper.value_changes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: lower_errors.as_entire_binding(),
            },
        ],
    })
}
