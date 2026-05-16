use std::sync::Arc;

use super::{context::GpuContext, layout::PcBindGroupLayouts};

/// Precompiled compute pipelines for every kernel in the predictive-coding
/// inference / training loop.
pub struct PcPipelines {
    pub predict: wgpu::ComputePipeline,
    pub errors: wgpu::ComputePipeline,
    pub sum_error_sq: wgpu::ComputePipeline,
    pub compute_gain_errors: wgpu::ComputePipeline,
    pub timestep: wgpu::ComputePipeline,
    pub compute_weight_deltas: wgpu::ComputePipeline,
    pub apply_weight_deltas: wgpu::ComputePipeline,
    pub sum_value_change: wgpu::ComputePipeline,
}

impl PcPipelines {
    /// Compile all shaders and build pipelines.
    pub fn new(ctx: &Arc<GpuContext>, layouts: &PcBindGroupLayouts) -> Self {
        let device = &ctx.device;

        // Shader for predict / error (uses predict_error layout)
        let pe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pc_predict_error_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/predictive_coding.wgsl").into()),
        });

        let pe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pc_predict_error_pipeline_layout"),
            bind_group_layouts: &[&layouts.predict_error],
            immediate_size: 0,
        });

        // Shader for timestep / weight-update (uses timestep_weight layout)
        let tw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pc_timestep_weight_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/timestep_weight.wgsl").into()),
        });

        let tw_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pc_timestep_weight_pipeline_layout"),
            bind_group_layouts: &[&layouts.timestep_weight],
            immediate_size: 0,
        });

        let make_pe = |entry: &str, label: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pe_layout),
                module: &pe_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: Default::default(),
            })
        };

        let make_tw = |entry: &str, label: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&tw_layout),
                module: &tw_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: Default::default(),
            })
        };

        Self {
            predict: make_pe("compute_predictions", "predict_pipeline"),
            errors: make_pe("compute_errors", "errors_pipeline"),
            sum_error_sq: make_pe("reduce_error_sq", "sum_error_sq_pipeline"),
            compute_gain_errors: make_tw("compute_gain_errors", "compute_gain_errors_pipeline"),
            timestep: make_tw("values_timestep", "timestep_pipeline"),
            compute_weight_deltas: make_tw(
                "compute_weight_deltas",
                "compute_weight_deltas_pipeline",
            ),
            apply_weight_deltas: make_tw("apply_weight_deltas", "apply_weight_deltas_pipeline"),
            sum_value_change: make_tw("reduce_value_change", "sum_value_change_pipeline"),
        }
    }
}
