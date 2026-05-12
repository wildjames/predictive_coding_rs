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
/// - `timestep_weight`: used by `values_timestep` and `compute_weight_updates`.
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
    pub predict_error: wgpu::BindGroupLayout,

    /// Layout for the timestep / weight-update kernels.
    ///
    /// Binding 0: upper_values         (storage, read_write)
    /// Binding 1: upper_weights        (storage, read_write)
    /// Binding 2: upper_meta           (storage, read)
    /// Binding 3: upper_errors         (storage, read_write)
    /// Binding 4: upper_value_changes  (storage, read_write)
    /// Binding 5: lower_errors         (storage, read)
    /// Binding 6: params               (uniform)
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
                ],
            });

        let timestep_weight =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pc_timestep_weight_layout"),
                    entries: &[
                        storage_entry(0, false), // upper values (rw for timestep)
                        storage_entry(1, false), // upper weights (rw for weight update)
                        storage_entry(2, true),  // upper meta
                        storage_entry(3, false), // upper errors (rw)
                        storage_entry(4, false), // upper value_changes (rw)
                        storage_entry(5, true),  // lower errors
                        uniform_entry(6),        // params
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
                resource: upper.errors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: upper.value_changes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: lower_errors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params.as_entire_binding(),
            },
        ],
    })
}
