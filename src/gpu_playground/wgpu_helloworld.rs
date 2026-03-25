use std::sync::mpsc::channel;

use wgpu::{
    Instance,
    util::{BufferInitDescriptor, DeviceExt},
};

// https://sotrh.github.io/learn-wgpu/compute/introduction

#[tokio::main]
pub async fn main() {
    let instance = Instance::new(&Default::default());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    let shader: wgpu::ShaderModule =
        device.create_shader_module(wgpu::include_wgsl!("introduction.wgsl"));

    let pipeline: wgpu::ComputePipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Introduction Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

    let input_data: Vec<u32> = (0..64).collect::<Vec<u32>>();

    let input_buffer: wgpu::Buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let output_buffer: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: input_buffer.size(),
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let temp_buffer: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("temp"),
        size: input_buffer.size(),
        // Note that the temp buffer needs MAP_READ and COPY_DST, whereas the output buffer
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder: wgpu::CommandEncoder = device.create_command_encoder(&Default::default());

    {
        let num_dispatches: u32 = input_data.len().div_ceil(64) as u32;

        let mut pass: wgpu::ComputePass<'_> = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_dispatches, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &temp_buffer, 0, output_buffer.size());
    queue.submit([encoder.finish()]);

    let output_data: Vec<u32> = {
        let (tx, rx) = channel();
        temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();

        let binding: wgpu::BufferView = temp_buffer.get_mapped_range(..);
        let output_data: Vec<u32> = bytemuck::cast_slice(&binding).to_owned();

        output_data
    };

    println!("Input data: {:?}", input_data);
    println!("Output data: {:?}", output_data);
}
