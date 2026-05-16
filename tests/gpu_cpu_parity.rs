//! Integration tests that verify GPU and CPU backends produce identical results.
//!
//! Requires the `gpu` feature and a working GPU adapter (or software fallback).
//! All tests share a single [`GpuContext`] to avoid intermittent SIGSEGV from
//! concurrent driver teardown of multiple devices.

#![cfg(feature = "gpu")]

use std::sync::{Arc, OnceLock};

use predictive_coding::model::{
    CpuModelRuntime, GpuModelRuntime, ModelSnapshot, PredictiveCodingModel,
    PredictiveCodingModelConfig, TrainableModelRuntime, gpu::GpuContext, maths::ActivationFunction,
};

/// Absolute tolerance for floating-point comparisons.
/// GPU f32 may differ slightly due to different reduction order.
const TOL: f32 = 1e-4;

fn assert_vecs_close(label: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{label}: length mismatch (cpu={}, gpu={})",
        cpu.len(),
        gpu.len()
    );
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            (c - g).abs() < TOL,
            "{label}[{i}]: cpu={c} gpu={g} diff={}",
            (c - g).abs()
        );
    }
}

/// Build a snapshot that both backends can start from.
///
/// Weights and values are randomly initialised (not seeded), so results vary
/// between runs.  Within a single run both the CPU and GPU runtimes receive
/// the same byte-identical snapshot.
fn make_test_snapshot(layer_sizes: &[usize], activation: ActivationFunction) -> ModelSnapshot {
    let config = PredictiveCodingModelConfig {
        layer_sizes: layer_sizes.to_vec(),
        alpha: 0.01,
        gamma: 0.05,
        convergence_threshold: 0.001,
        convergence_steps: 50,
        activation_function: activation,
    };
    // Build a randomly-initialised model and immediately snapshot it so
    // both backends start from byte-identical state.
    PredictiveCodingModel::new(&config).to_snapshot()
}

type BoxedRuntime = Box<dyn TrainableModelRuntime>;

/// Single shared GPU context, initialised once and reused by every test.
fn shared_gpu_context() -> Arc<GpuContext> {
    static CTX: OnceLock<Arc<GpuContext>> = OnceLock::new();
    CTX.get_or_init(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime for shared GpuContext");
        rt.block_on(GpuContext::new()).expect("GpuContext::new")
    })
    .clone()
}

fn make_runtimes(
    layer_sizes: &[usize],
    activation: ActivationFunction,
) -> (BoxedRuntime, BoxedRuntime, Vec<f32>, Vec<f32>) {
    let snapshot: ModelSnapshot = make_test_snapshot(layer_sizes, activation);

    let cpu: CpuModelRuntime =
        CpuModelRuntime::from_snapshot(&snapshot).expect("cpu from snapshot");
    let gpu: GpuModelRuntime =
        GpuModelRuntime::from_snapshot_with_context(&snapshot, shared_gpu_context())
            .expect("gpu from snapshot");

    let input_size = layer_sizes[0];
    let output_size = *layer_sizes.last().unwrap();

    // Deterministic I/O data
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 + 1.0) / input_size as f32)
        .collect();
    let output: Vec<f32> = (0..output_size)
        .map(|i| (i as f32) / output_size as f32)
        .collect();

    (Box::new(cpu), Box::new(gpu), input, output)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[test]
fn predictions_and_errors_match() {
    for activation in [
        ActivationFunction::Relu,
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
    ] {
        let (mut cpu, mut gpu, input, output) = make_runtimes(&[4, 8, 5, 3], activation);

        cpu.set_input(&input).unwrap();
        cpu.set_output(&output).unwrap();

        gpu.set_input(&input).unwrap();
        gpu.set_output(&output).unwrap();

        cpu.compute_predictions_and_errors().unwrap();
        gpu.compute_predictions_and_errors().unwrap();

        let cpu_snap = cpu.snapshot().unwrap();
        let gpu_snap = gpu.snapshot().unwrap();

        // Compare layer by layer to check for mismatches
        for (i, (cl, gl)) in cpu_snap
            .layers
            .iter()
            .zip(gpu_snap.layers.iter())
            .enumerate()
        {
            let label = format!("{activation:?} layer {i}");
            assert_vecs_close(&format!("{label} values"), &cl.values, &gl.values);
            assert_vecs_close(
                &format!("{label} predictions"),
                &cl.predictions,
                &gl.predictions,
            );
            assert_vecs_close(&format!("{label} errors"), &cl.errors, &gl.errors);
        }
    }
}

#[test]
fn total_error_and_energy_match() {
    let (mut cpu, mut gpu, input, output) =
        make_runtimes(&[4, 8, 5, 3], ActivationFunction::Sigmoid);

    cpu.set_input(&input).unwrap();
    cpu.set_output(&output).unwrap();

    gpu.set_input(&input).unwrap();
    gpu.set_output(&output).unwrap();

    cpu.compute_predictions_and_errors().unwrap();
    gpu.compute_predictions_and_errors().unwrap();

    // The total error and energy may differ slightly due to maths being different on GPU and CPU, so allow some tolerance.
    let cpu_err = cpu.total_error().unwrap();
    let gpu_err = gpu.total_error().unwrap();
    assert!(
        (cpu_err - gpu_err).abs() < TOL,
        "total_error: cpu={cpu_err} gpu={gpu_err}"
    );

    let cpu_energy = cpu.total_energy().unwrap();
    let gpu_energy = gpu.total_energy().unwrap();
    assert!(
        (cpu_energy - gpu_energy).abs() < TOL,
        "total_energy: cpu={cpu_energy} gpu={gpu_energy}"
    );
}

#[test]
fn timestep_values_match() {
    let (mut cpu, mut gpu, input, output) = make_runtimes(&[4, 8, 5, 3], ActivationFunction::Tanh);

    cpu.set_input(&input).unwrap();
    cpu.set_output(&output).unwrap();

    gpu.set_input(&input).unwrap();
    gpu.set_output(&output).unwrap();

    // Run prediction + error + timestep
    cpu.compute_predictions_and_errors().unwrap();
    gpu.compute_predictions_and_errors().unwrap();

    let cpu_change = cpu.timestep().unwrap();
    let gpu_change = gpu.timestep().unwrap();

    assert!(
        (cpu_change - gpu_change).abs() < TOL,
        "timestep mean change: cpu={cpu_change} gpu={gpu_change}"
    );

    // Compare resulting values after the timestep
    let cpu_snap = cpu.snapshot().unwrap();
    let gpu_snap = gpu.snapshot().unwrap();

    for (i, (cl, gl)) in cpu_snap
        .layers
        .iter()
        .zip(gpu_snap.layers.iter())
        .enumerate()
    {
        assert_vecs_close(
            &format!("post-timestep layer {i} values"),
            &cl.values,
            &gl.values,
        );
    }
}

#[test]
fn converge_values_match() {
    let (mut cpu, mut gpu, input, output) =
        make_runtimes(&[3, 5, 8, 2], ActivationFunction::Sigmoid);

    cpu.set_input(&input).unwrap();
    cpu.set_output(&output).unwrap();

    gpu.set_input(&input).unwrap();
    gpu.set_output(&output).unwrap();

    let cpu_steps = cpu.converge_values().unwrap();
    let gpu_steps = gpu.converge_values().unwrap();

    assert_eq!(
        cpu_steps, gpu_steps,
        "convergence steps: cpu={cpu_steps} gpu={gpu_steps}"
    );

    let cpu_snap: ModelSnapshot = cpu.snapshot().unwrap();
    let gpu_snap: ModelSnapshot = gpu.snapshot().unwrap();

    for (i, (cl, gl)) in cpu_snap
        .layers
        .iter()
        .zip(gpu_snap.layers.iter())
        .enumerate()
    {
        assert_vecs_close(
            &format!("post-converge layer {i} values"),
            &cl.values,
            &gl.values,
        );
        assert_vecs_close(
            &format!("post-converge layer {i} errors"),
            &cl.errors,
            &gl.errors,
        );
    }
}

#[test]
fn weight_updates_match() {
    let (mut cpu, mut gpu, input, output) = make_runtimes(&[4, 6, 2, 3], ActivationFunction::Relu);

    cpu.set_input(&input).unwrap();
    cpu.set_output(&output).unwrap();

    gpu.set_input(&input).unwrap();
    gpu.set_output(&output).unwrap();

    // Converge first so errors are meaningful
    cpu.compute_predictions_and_errors().unwrap();
    gpu.compute_predictions_and_errors().unwrap();

    let cpu_updates = cpu.compute_weight_updates().unwrap();
    let gpu_updates = gpu.compute_weight_updates().unwrap();

    assert_eq!(
        cpu_updates.shapes, gpu_updates.shapes,
        "weight update shapes mismatch"
    );

    for (i, (cu, gu)) in cpu_updates
        .updates
        .iter()
        .zip(gpu_updates.updates.iter())
        .enumerate()
    {
        assert_vecs_close(&format!("weight_update layer {}", i + 1), cu, gu);
    }
}

#[test]
fn full_training_step_parity() {
    // Run a full train step: set I/O, converge, update weights, then compare.
    let (mut cpu, mut gpu, input, output) =
        make_runtimes(&[4, 8, 4, 3], ActivationFunction::Sigmoid);

    cpu.set_input(&input).unwrap();
    cpu.set_output(&output).unwrap();

    gpu.set_input(&input).unwrap();
    gpu.set_output(&output).unwrap();

    cpu.converge_values().unwrap();
    gpu.converge_values().unwrap();

    // Use compute + apply (not update_weights) so both do the same path.
    let cpu_updates = cpu.compute_weight_updates().unwrap();
    let gpu_updates = gpu.compute_weight_updates().unwrap();

    cpu.apply_weight_updates(&cpu_updates).unwrap();
    gpu.apply_weight_updates(&gpu_updates).unwrap();

    let cpu_snap = cpu.snapshot().unwrap();
    let gpu_snap = gpu.snapshot().unwrap();

    for (i, (cl, gl)) in cpu_snap
        .layers
        .iter()
        .zip(gpu_snap.layers.iter())
        .enumerate()
    {
        let label = format!("full-step layer {i}");
        assert_vecs_close(&format!("{label} values"), &cl.values, &gl.values);
        assert_vecs_close(&format!("{label} weights"), &cl.weights, &gl.weights);
    }
}
