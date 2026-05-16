// ---------------------------------------------------------------------------
// Predictive-coding compute shaders — timestep & weight-update kernels
//
// Buffer layout (matches layout.rs PcBindGroupLayouts::timestep_weight):
//   @group(0) @binding(0)  params               : vec4<f32>   (uniform) [alpha, gamma, conv_thresh, conv_steps]
//   @group(0) @binding(1)  weight_deltas        : array<f32>  (rw)
//   @group(0) @binding(2)  gain_errors          : array<f32>  (rw)
//   @group(0) @binding(3)  upper_meta           : array<u32>  (read)  [pinned, activation_fn, size, weight_rows, weight_cols, is_top_level]
//   @group(0) @binding(4)  upper_values         : array<f32>  (rw)
//   @group(0) @binding(5)  upper_weights        : array<f32>  (rw)
//   @group(0) @binding(6)  upper_errors         : array<f32>  (rw)
//   @group(0) @binding(7)  upper_value_changes  : array<f32>  (rw)
//   @group(0) @binding(8)  lower_errors         : array<f32>  (read)
//   @group(0) @binding(9)  vc_partial_sums      : array<f32>  (rw)     per-workgroup partial sums, offset per layer
// ---------------------------------------------------------------------------

// Activation function IDs (must match buffers::ACTIVATION_* constants)
const ACTIVATION_RELU: u32    = 0u;
const ACTIVATION_SIGMOID: u32 = 1u;
const ACTIVATION_TANH: u32    = 2u;

// ---- bindings -------------------------------------------------------------

@group(0) @binding(0) var<uniform>             params               : vec4<f32>;
@group(0) @binding(1) var<storage, read_write> weight_deltas        : array<f32>;
@group(0) @binding(2) var<storage, read_write> gain_errors          : array<f32>;
@group(0) @binding(3) var<storage, read>       upper_meta           : array<u32>;
@group(0) @binding(4) var<storage, read_write> upper_values         : array<f32>;
@group(0) @binding(5) var<storage, read_write> upper_weights        : array<f32>;
@group(0) @binding(6) var<storage, read_write> upper_errors         : array<f32>;
@group(0) @binding(7) var<storage, read_write> upper_value_changes  : array<f32>;
@group(0) @binding(8) var<storage, read>       lower_errors         : array<f32>;
@group(0) @binding(9) var<storage, read_write> vc_partial_sums      : array<f32>;

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

/// Precompute gain_error[i] = f'(W[i,:] · upper_values) * lower_errors[i]
/// for each row i of the weight matrix.
///
/// Must be dispatched BEFORE values_timestep and compute_weight_deltas.
/// One thread per weight row (i.e., per lower-layer node).
@compute @workgroup_size(64)
fn compute_gain_errors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let weight_rows = upper_meta[3];
    let weight_cols = upper_meta[4];
    let i = gid.x;

    if i >= weight_rows {
        return;
    }

    let act_fn = upper_meta[1];

    var preact: f32 = 0.0;
    for (var k: u32 = 0u; k < weight_cols; k = k + 1u) {
        preact += upper_weights[weight_index(i, k, weight_cols)] * upper_values[k];
    }

    gain_errors[i] = activation_derivative(preact, act_fn) * lower_errors[i];
}

/// Update the **upper** layer's node values.
///
/// For non-top layers:
///   value_change[j] = ( -upper_errors[j] + rhs[j] ) * gamma
/// For the top layer:
///   value_change[j] = rhs[j] * gamma
///
/// where rhs[j] = sum_i  W[i][j] * gain_errors[i]
///
/// Requires compute_gain_errors to have been dispatched first.
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

    // rhs[j] = W^T · gain_errors  evaluated at column j
    //        = sum_i W[i][j] * gain_errors[i]
    var rhs: f32 = 0.0;
    if weight_rows > 0u {
        for (var i: u32 = 0u; i < weight_rows; i = i + 1u) {
            rhs += upper_weights[weight_index(i, idx, weight_cols)] * gain_errors[i];
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
/// delta_W[i][j] = alpha * gain_errors[i] * upper_values[j]
///
/// Requires compute_gain_errors to have been dispatched first.
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

    let alpha = params.x;
    let delta = alpha * gain_errors[i] * upper_values[j];

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

// workgroup so that I can sync the threads
var<workgroup> shared_vc_sum: array<f32, 64>;

/// Gather the absolute value changes for this layer into partial sums.
///
/// Each workgroup reduces its 64 value_changes into one value and writes it to
/// `vc_partial_sums[offset + workgroup_id]`, where `offset` comes from
/// `upper_meta[6]`.  This lets every layer write to its own region of the
/// shared vc_partial_sums buffer in a single dispatch.
@compute @workgroup_size(64)
fn reduce_value_change(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let upper_size = upper_meta[2];
    let offset     = upper_meta[6]; // per-layer offset into vc_partial_sums
    let idx        = gid.x;
    let local_idx  = lid.x;

    if idx < upper_size {
        shared_vc_sum[local_idx] = upper_value_changes[idx];
    } else {
        shared_vc_sum[local_idx] = 0.0;
    }
    workgroupBarrier();

    for (var stride: u32 = 32u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_vc_sum[local_idx] += shared_vc_sum[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        vc_partial_sums[offset + gid.x / 64u] = shared_vc_sum[0];
    }
}
