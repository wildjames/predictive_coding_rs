// ---------------------------------------------------------------------------
// Predictive-coding compute shaders — timestep & weight-update kernels
//
// Buffer layout (matches layout.rs PcBindGroupLayouts::timestep_weight):
//   @group(0) @binding(0)  upper_values         : array<f32>  (rw)
//   @group(0) @binding(1)  upper_weights        : array<f32>  (rw)
//   @group(0) @binding(2)  upper_meta           : array<u32>  (read)  [pinned, activation_fn, size, weight_rows, weight_cols, is_top_level]
//   @group(0) @binding(3)  upper_errors         : array<f32>  (rw)
//   @group(0) @binding(4)  upper_value_changes  : array<f32>  (rw)
//   @group(0) @binding(5)  lower_errors         : array<f32>  (read)
//   @group(0) @binding(6)  params               : vec4<f32>   (uniform) [alpha, gamma, conv_thresh, conv_steps]
//   @group(0) @binding(7)  weight_deltas        : array<f32>  (rw)
// ---------------------------------------------------------------------------

// Activation function IDs (must match buffers::ACTIVATION_* constants)
const ACTIVATION_RELU: u32    = 0u;
const ACTIVATION_SIGMOID: u32 = 1u;
const ACTIVATION_TANH: u32    = 2u;

// ---- bindings -------------------------------------------------------------

@group(0) @binding(0) var<storage, read_write> upper_values         : array<f32>;
@group(0) @binding(1) var<storage, read_write> upper_weights        : array<f32>;
@group(0) @binding(2) var<storage, read>       upper_meta           : array<u32>;
@group(0) @binding(3) var<storage, read_write> upper_errors         : array<f32>;
@group(0) @binding(4) var<storage, read_write> upper_value_changes  : array<f32>;
@group(0) @binding(5) var<storage, read>       lower_errors         : array<f32>;
@group(0) @binding(6) var<uniform>             params               : vec4<f32>;
@group(0) @binding(7) var<storage, read_write> weight_deltas        : array<f32>;

// ---- helpers --------------------------------------------------------------

fn activation_derivative(x: f32, fn_id: u32) -> f32 {
    switch fn_id {
        case ACTIVATION_RELU {
            if x > 0.0 { return 1.0; } else { return 0.0; }
        }
        case ACTIVATION_SIGMOID {
            let em = exp(-x);
            return em / pow(1.0 + em, 2.0);
        }
        case ACTIVATION_TANH {
            let ep = exp(x) + exp(-x);
            return 4.0 / (ep * ep);
        }
        default {
            return 1.0;
        }
    }
}

fn weight_index(row: u32, col: u32, num_cols: u32) -> u32 {
    return row * num_cols + col;
}

// ---- kernels --------------------------------------------------------------

/// Update the **upper** layer's node values.
///
/// For non-top layers:
///   value_change[j] = ( -upper_errors[j] + rhs[j] ) * gamma
/// For the top layer:
///   value_change[j] = rhs[j] * gamma
///
/// where rhs[j] = sum_i  W[i][j] * f'(preact[i]) * lower_errors[i]
///       preact[i] = sum_k W[i][k] * upper_values[k]
///
/// If the layer has no weights (layer 0 bottom self-group, weight_rows==0),
/// rhs is zero.
///
/// Writes abs(value_change) to upper_value_changes for convergence detection.
///
/// One thread per upper-layer node.
@compute @workgroup_size(64)
fn values_timestep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let upper_size  = upper_meta[2];
    let idx         = gid.x;

    if idx >= upper_size {
        return;
    }

    // Pinned layers don't update.
    let pinned = upper_meta[0];
    if pinned != 0u {
        upper_value_changes[idx] = 0.0;
        return;
    }

    let gamma       = params.y;
    let is_top      = upper_meta[5];
    let weight_rows = upper_meta[3]; // == lower_size
    let weight_cols = upper_meta[4]; // == upper_size
    let act_fn      = upper_meta[1];

    // rhs[j] = W^T · (f'(W·x) ⊙ lower_errors)  evaluated at column j
    //        = sum_i W[i][j] * f'(dot(W[i,:], x)) * lower_errors[i]
    var rhs: f32 = 0.0;
    if weight_rows > 0u {
        for (var i: u32 = 0u; i < weight_rows; i = i + 1u) {
            var preact: f32 = 0.0;
            for (var k: u32 = 0u; k < weight_cols; k = k + 1u) {
                preact += upper_weights[weight_index(i, k, weight_cols)] * upper_values[k];
            }
            let gain_error = activation_derivative(preact, act_fn) * lower_errors[i];
            rhs += upper_weights[weight_index(i, idx, weight_cols)] * gain_error;
        }
    }

    var value_change: f32;
    if is_top != 0u {
        value_change = rhs * gamma;
    } else {
        value_change = (-upper_errors[idx] + rhs) * gamma;
    }

    upper_values[idx] = upper_values[idx] + value_change;
    upper_value_changes[idx] = abs(value_change);
}

/// Compute weight deltas into a separate buffer (no race on upper_weights).
///
/// delta_W[i][j] = alpha * f'(preact[i]) * lower_errors[i] * upper_values[j]
///
/// One thread per weight element.
@compute @workgroup_size(64)
fn compute_weight_deltas(@builtin(global_invocation_id) gid: vec3<u32>) {
    let weight_rows = upper_meta[3]; // lower_size
    let weight_cols = upper_meta[4]; // upper_size
    let total       = weight_rows * weight_cols;
    let flat_idx    = gid.x;

    if flat_idx >= total {
        return;
    }

    let i = flat_idx / weight_cols; // row (lower node index)
    let j = flat_idx % weight_cols; // col (upper node index)

    let alpha  = params.x;
    let act_fn = upper_meta[1];

    var preact: f32 = 0.0;
    for (var k: u32 = 0u; k < weight_cols; k = k + 1u) {
        preact += upper_weights[weight_index(i, k, weight_cols)] * upper_values[k];
    }

    let gain_error = activation_derivative(preact, act_fn) * lower_errors[i];
    let delta = alpha * gain_error * upper_values[j];

    weight_deltas[flat_idx] = delta;
}

/// Apply precomputed weight deltas to the weight matrix.
///
/// W[i][j] += weight_deltas[i * cols + j]
///
/// One thread per weight element. Must be dispatched AFTER compute_weight_deltas.
@compute @workgroup_size(64)
fn apply_weight_deltas(@builtin(global_invocation_id) gid: vec3<u32>) {
    let weight_rows = upper_meta[3];
    let weight_cols = upper_meta[4];
    let total       = weight_rows * weight_cols;
    let flat_idx    = gid.x;

    if flat_idx >= total {
        return;
    }

    upper_weights[flat_idx] = upper_weights[flat_idx] + weight_deltas[flat_idx];
}
