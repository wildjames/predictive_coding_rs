//! Integration tests that verify GPU batch runtime and CPU backend produce identical results.
//!
//! Requires the `gpu` feature and a working GPU adapter (or software fallback).
//! All tests share a single [`GpuContext`] to avoid intermittent SIGSEGV from
//! concurrent driver teardown of multiple devices.

#![cfg(feature = "gpu")]

use std::sync::{Arc, OnceLock};

use predictive_coding::model::{
    CpuModelRuntime, GpuRuntime, ModelRuntime, ModelSnapshot, PredictiveCodingModel,
    PredictiveCodingModelConfig, TrainableModelRuntime, gpu::GpuContext, maths::ActivationFunction,
};

/// Absolute tolerance for floating-point comparisons.
/// GPU f32 may differ slightly due to different reduction order.
const TOL: f32 = 1e-4;

/// Slightly higher tolerance for full training step comparisons where
/// FP drift accumulates across convergence iterations.
const TRAIN_TOL: f32 = 5e-4;

fn assert_vecs_close(label: &str, cpu: &[f32], gpu: &[f32], tol: f32) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{label}: length mismatch (cpu={}, gpu={})",
        cpu.len(),
        gpu.len()
    );
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            (c - g).abs() < tol,
            "{label}[{i}]: cpu={c} gpu={g} diff={}",
            (c - g).abs()
        );
    }
}

/// Build a snapshot that both backends can start from.
fn make_test_snapshot(layer_sizes: &[usize], activation: ActivationFunction) -> ModelSnapshot {
    let config = PredictiveCodingModelConfig {
        layer_sizes: layer_sizes.to_vec(),
        alpha: 0.01,
        gamma: 0.05,
        convergence_threshold: 0.0, // Fixed steps to ensure identical iteration count
        convergence_steps: 50,
        activation_function: activation,
        weight_clip: 0.0,
    };
    PredictiveCodingModel::new(&config).to_snapshot()
}

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

/// Create a CPU runtime and a batch GPU runtime (batch_size=1) from the same snapshot,
/// along with deterministic input/output data.
fn make_cpu_and_batch(
    layer_sizes: &[usize],
    activation: ActivationFunction,
) -> (CpuModelRuntime, GpuRuntime, Vec<f32>, Vec<f32>) {
    let snapshot = make_test_snapshot(layer_sizes, activation);

    let cpu = CpuModelRuntime::from_snapshot(&snapshot).expect("cpu from snapshot");
    let gpu = GpuRuntime::from_snapshot_with_context(&snapshot, shared_gpu_context(), 1)
        .expect("batch gpu from snapshot");

    let input_size = layer_sizes[0];
    let output_size = *layer_sizes.last().unwrap();

    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 + 1.0) / input_size as f32)
        .collect();
    let output: Vec<f32> = (0..output_size)
        .map(|i| (i as f32) / output_size as f32)
        .collect();

    (cpu, gpu, input, output)
}

// -----------------------------------------------------------------------
// Tests: batch_size=1 GPU vs CPU parity
// -----------------------------------------------------------------------

/// After convergence, the latent values and errors should match CPU.
#[test]
fn converge_values_match_cpu() {
    for activation in [
        ActivationFunction::Relu,
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
    ] {
        let (mut cpu, gpu, input, output) = make_cpu_and_batch(&[4, 8, 5, 3], activation);

        // CPU path
        cpu.set_input(&input).unwrap();
        cpu.set_output(&output).unwrap();
        let cpu_steps = cpu.converge_values().unwrap();

        // GPU batch path (batch_size=1)
        gpu.set_batch_data(&[(input.clone(), output.clone())])
            .unwrap();
        let gpu_steps = gpu.converge_all().unwrap();

        assert_eq!(
            cpu_steps, gpu_steps,
            "{activation:?}: convergence steps differ (cpu={cpu_steps} gpu={gpu_steps})"
        );

        let cpu_snap = cpu.snapshot().unwrap();
        let gpu_snap = gpu.snapshot().unwrap();

        for (i, (cl, gl)) in cpu_snap
            .layers
            .iter()
            .zip(gpu_snap.layers.iter())
            .enumerate()
        {
            let label = format!("{activation:?} layer {i}");
            assert_vecs_close(&format!("{label} values"), &cl.values, &gl.values, TOL);
            assert_vecs_close(&format!("{label} errors"), &cl.errors, &gl.errors, TOL);
        }
    }
}

/// After a full training step (converge + weight update), weights should match CPU.
#[test]
fn full_training_step_parity() {
    for activation in [
        ActivationFunction::Relu,
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
    ] {
        let (mut cpu, gpu, input, output) = make_cpu_and_batch(&[4, 8, 4, 3], activation);

        // --- CPU path ---
        cpu.set_input(&input).unwrap();
        cpu.set_output(&output).unwrap();
        cpu.converge_values().unwrap();
        let cpu_updates = cpu.compute_weight_updates().unwrap();
        cpu.apply_weight_updates(&cpu_updates).unwrap();

        // --- GPU batch path (batch_size=1) ---
        gpu.set_params_alpha(gpu.config().alpha);
        gpu.zero_weight_accumulators();
        gpu.set_batch_data(&[(input.clone(), output.clone())])
            .unwrap();
        gpu.converge_all().unwrap();
        gpu.accumulate_all_weight_deltas();
        gpu.apply_accumulated_weight_deltas();

        let cpu_snap = cpu.snapshot().unwrap();
        let gpu_snap = gpu.snapshot().unwrap();

        for (i, (cl, gl)) in cpu_snap
            .layers
            .iter()
            .zip(gpu_snap.layers.iter())
            .enumerate()
        {
            let label = format!("{activation:?} full-step layer {i}");
            assert_vecs_close(&format!("{label} values"), &cl.values, &gl.values, TRAIN_TOL);
            assert_vecs_close(
                &format!("{label} weights"),
                &cl.weights,
                &gl.weights,
                TRAIN_TOL,
            );
        }
    }
}

/// Verify that multiple training steps accumulate correctly (batch_size=1).
#[test]
fn multi_step_weight_accumulation() {
    let (mut cpu, gpu, input, output) =
        make_cpu_and_batch(&[4, 6, 3], ActivationFunction::Sigmoid);

    let steps = 3;

    for _ in 0..steps {
        // CPU step
        cpu.set_input(&input).unwrap();
        cpu.set_output(&output).unwrap();
        cpu.converge_values().unwrap();
        let updates = cpu.compute_weight_updates().unwrap();
        cpu.apply_weight_updates(&updates).unwrap();

        // GPU batch step
        gpu.set_params_alpha(gpu.config().alpha);
        gpu.zero_weight_accumulators();
        gpu.set_batch_data(&[(input.clone(), output.clone())])
            .unwrap();
        gpu.converge_all().unwrap();
        gpu.accumulate_all_weight_deltas();
        gpu.apply_accumulated_weight_deltas();
    }

    let cpu_snap = cpu.snapshot().unwrap();
    let gpu_snap = gpu.snapshot().unwrap();

    for (i, (cl, gl)) in cpu_snap
        .layers
        .iter()
        .zip(gpu_snap.layers.iter())
        .enumerate()
    {
        // After multiple steps, tolerance grows slightly
        assert_vecs_close(
            &format!("multi-step layer {i} weights"),
            &cl.weights,
            &gl.weights,
            TRAIN_TOL * steps as f32,
        );
    }
}

// -----------------------------------------------------------------------
// Tests: batch_size > 1 correctness
// -----------------------------------------------------------------------

/// Verify that batch_size=N produces the same weights as N sequential
/// single-sample updates on CPU (each accumulating into a shared delta buffer).
#[test]
fn batch_matches_sequential_cpu_updates() {
    let batch_size = 4u32;
    let snapshot = make_test_snapshot(&[4, 6, 3], ActivationFunction::Sigmoid);

    let input_size = snapshot.config.layer_sizes[0];
    let output_size = *snapshot.config.layer_sizes.last().unwrap();

    // Generate deterministic samples
    let samples: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
        .map(|s| {
            let input: Vec<f32> = (0..input_size)
                .map(|i| ((i + s as usize) as f32 + 1.0) / (input_size as f32 * 2.0))
                .collect();
            let output: Vec<f32> = (0..output_size)
                .map(|i| ((i + s as usize) as f32) / (output_size as f32 * 2.0))
                .collect();
            (input, output)
        })
        .collect();

    // --- CPU sequential path: accumulate weight deltas from N samples ---
    let alpha = snapshot.config.alpha;
    let mut accumulated_updates: Option<Vec<Vec<f32>>> = None;

    for (input, output) in &samples {
        let mut cpu = CpuModelRuntime::from_snapshot(&snapshot).expect("cpu from snapshot");
        cpu.set_input(input).unwrap();
        cpu.set_output(output).unwrap();
        cpu.converge_values().unwrap();
        let updates = cpu.compute_weight_updates().unwrap();

        match &mut accumulated_updates {
            None => {
                accumulated_updates = Some(updates.updates.clone());
            }
            Some(accum) => {
                for (a, u) in accum.iter_mut().zip(updates.updates.iter()) {
                    for (av, uv) in a.iter_mut().zip(u.iter()) {
                        *av += *uv;
                    }
                }
            }
        }
    }

    // Apply accumulated updates to a fresh snapshot
    let mut modified_snap = snapshot.clone();
    if let Some(accum) = &accumulated_updates {
        for (layer_idx, layer_deltas) in accum.iter().enumerate() {
            let l = &mut modified_snap.layers[layer_idx + 1];
            for (w, d) in l.weights.iter_mut().zip(layer_deltas.iter()) {
                *w += *d;
            }
        }
    }

    // --- GPU batch path ---
    let gpu =
        GpuRuntime::from_snapshot_with_context(&snapshot, shared_gpu_context(), batch_size)
            .expect("batch gpu runtime");

    gpu.set_params_alpha(alpha);
    gpu.zero_weight_accumulators();
    gpu.set_batch_data(&samples).unwrap();
    gpu.converge_all().unwrap();
    gpu.accumulate_all_weight_deltas();
    gpu.apply_accumulated_weight_deltas();

    let gpu_snap = gpu.snapshot().unwrap();

    for (i, (cl, gl)) in modified_snap
        .layers
        .iter()
        .zip(gpu_snap.layers.iter())
        .enumerate()
    {
        if cl.weights.is_empty() {
            continue;
        }
        assert_vecs_close(
            &format!("batch_vs_cpu layer {i} weights"),
            &cl.weights,
            &gl.weights,
            TRAIN_TOL,
        );
    }
}

/// Smoke test: batch_size > 1 runs without panicking and produces changed weights.
#[test]
fn batch_runtime_multi_sample_smoke() {
    let snapshot = make_test_snapshot(&[4, 6, 3], ActivationFunction::Relu);
    let batch_size = 4u32;

    let batch =
        GpuRuntime::from_snapshot_with_context(&snapshot, shared_gpu_context(), batch_size)
            .expect("batch gpu runtime");

    let input_size = snapshot.config.layer_sizes[0];
    let output_size = *snapshot.config.layer_sizes.last().unwrap();

    let samples: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
        .map(|s| {
            let input: Vec<f32> = (0..input_size)
                .map(|i| ((i + s as usize) as f32 + 1.0) / input_size as f32)
                .collect();
            let output: Vec<f32> = (0..output_size)
                .map(|i| ((i + s as usize) as f32) / output_size as f32)
                .collect();
            (input, output)
        })
        .collect();

    batch.set_params_alpha(snapshot.config.alpha / batch_size as f32);
    batch.zero_weight_accumulators();
    batch.set_batch_data(&samples).unwrap();
    batch.reinitialise_all_latents();
    batch.converge_all().unwrap();
    batch.accumulate_all_weight_deltas();
    batch.apply_accumulated_weight_deltas();

    let result_snap = batch.snapshot().unwrap();
    assert_eq!(result_snap.layers.len(), snapshot.layers.len());

    // Weights should have changed
    let initial_weights = &snapshot.layers[1].weights;
    let final_weights = &result_snap.layers[1].weights;
    let any_changed = initial_weights
        .iter()
        .zip(final_weights.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(any_changed, "weights should change after a training step");
}

/// Verify snapshot returns valid layer geometry.
#[test]
fn batch_snapshot_geometry() {
    let sizes = &[4, 8, 5, 3];
    let snapshot = make_test_snapshot(sizes, ActivationFunction::Tanh);
    let gpu =
        GpuRuntime::from_snapshot_with_context(&snapshot, shared_gpu_context(), 2)
            .expect("batch gpu runtime");

    let result = gpu.snapshot().unwrap();
    assert_eq!(result.layers.len(), sizes.len());

    for (i, layer) in result.layers.iter().enumerate() {
        assert_eq!(layer.size, sizes[i]);
        assert_eq!(layer.values.len(), sizes[i]);
        assert_eq!(layer.predictions.len(), sizes[i]);
        assert_eq!(layer.errors.len(), sizes[i]);

        if i > 0 {
            assert_eq!(layer.weight_rows, sizes[i]);
            assert_eq!(layer.weight_cols, sizes[i - 1]);
            assert_eq!(layer.weights.len(), sizes[i] * sizes[i - 1]);
        }
    }
}
