use std::sync::Arc;

use crate::error::{PredictiveCodingError, Result};

/// Holds the wgpu device and queue, shared across all GPU operations.
///
/// Create once at startup via [`GpuContext::new`] and pass by `Arc` to
/// buffers, pipelines, and the runtime.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    /// The adapter's hardware limits (as opposed to the device's requested limits).
    pub adapter_limits: wgpu::Limits,
}

impl GpuContext {
    /// Request a GPU device with default limits.
    ///
    /// This is async because wgpu adapter/device negotiation is async.
    pub async fn new() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                PredictiveCodingError::validation(format!("no suitable GPU adapter found: {e}"))
            })?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("predictive_coding_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    // The timestep_weight bind group layout uses 10 storage buffers
                    // (plus 1 uniform), exceeding the default limit of 8.
                    max_storage_buffers_per_shader_stage: 10,
                    // TODO: If models are ever in a place where they exceed the default limits, I'll need to handle it here. For now, my test models are like 10MB tops, so it's not an issue.
                    ..wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .map_err(|e| {
                PredictiveCodingError::validation(format!("failed to create GPU device: {e}"))
            })?;

        Ok(Arc::new(Self {
            device,
            queue,
            adapter_info,
            adapter_limits,
        }))
    }

    /// Human-readable description of the adapter (name, backend, device type).
    pub fn adapter_description(&self) -> String {
        format!(
            "{} ({:?}, {:?})",
            self.adapter_info.name, self.adapter_info.backend, self.adapter_info.device_type,
        )
    }
}
