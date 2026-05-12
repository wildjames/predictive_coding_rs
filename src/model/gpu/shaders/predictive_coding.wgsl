// ---------------------------------------------------------------------------
// Predictive-coding compute shaders — predict & error kernels
//
// Buffer layout (matches layout.rs PcBindGroupLayouts::predict_error):
//   @group(0) @binding(0)  upper_values     : array<f32>  (read)
//   @group(0) @binding(1)  upper_weights    : array<f32>  (read)   row-major (lower_size × upper_size)
//   @group(0) @binding(2)  upper_meta       : array<u32>  (read)   [pinned, activation_fn, size, weight_rows, weight_cols, is_top_level]
//   @group(0) @binding(3)  lower_values     : array<f32>  (read)
//   @group(0) @binding(4)  lower_preds      : array<f32>  (rw)
//   @group(0) @binding(5)  lower_errors     : array<f32>  (rw)
//   @group(0) @binding(6)  lower_meta       : array<u32>  (read)   [pinned, activation_fn, size, weight_rows, weight_cols, is_top_level]
//   @group(0) @binding(7)  params           : vec4<f32>   (uniform) [alpha, gamma, conv_thresh, conv_steps]
// ---------------------------------------------------------------------------

// Activation function IDs (must match buffers::ACTIVATION_* constants)
const ACTIVATION_RELU: u32    = 0u;
const ACTIVATION_SIGMOID: u32 = 1u;
const ACTIVATION_TANH: u32    = 2u;

// ---- bindings -------------------------------------------------------------

@group(0) @binding(0) var<storage, read>       upper_values  : array<f32>;
@group(0) @binding(1) var<storage, read>       upper_weights : array<f32>;
@group(0) @binding(2) var<storage, read>       upper_meta    : array<u32>;
@group(0) @binding(3) var<storage, read>       lower_values  : array<f32>;
@group(0) @binding(4) var<storage, read_write> lower_preds   : array<f32>;
@group(0) @binding(5) var<storage, read_write> lower_errors  : array<f32>;
@group(0) @binding(6) var<storage, read>       lower_meta    : array<u32>;
@group(0) @binding(7) var<uniform>             params        : vec4<f32>;

// ---- helpers --------------------------------------------------------------

fn activation(x: f32, fn_id: u32) -> f32 {
    switch fn_id {
        case ACTIVATION_RELU {
            return max(x, 0.0);
        }
        case ACTIVATION_SIGMOID {
            return 1.0 / (1.0 + exp(-x));
        }
        case ACTIVATION_TANH {
            return tanh(x);
        }
        default {
            return x;
        }
    }
}

// Weights are stored row-major with shape (lower_size, upper_size).
// W[row][col] = upper_weights[row * upper_size + col]
fn weight_index(row: u32, col: u32, num_cols: u32) -> u32 {
    return row * num_cols + col;
}

// ---- kernels --------------------------------------------------------------

/// Compute predictions of the lower layer from the upper layer.
///
/// pred_lower[i] = activation( sum_j W[i][j] * upper_values[j] )
///
/// One thread per lower-layer node.
@compute @workgroup_size(64)
fn compute_predictions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lower_size = lower_meta[2];
    let upper_size = upper_meta[2];
    let act_fn     = upper_meta[1];
    let idx        = gid.x;

    if idx >= lower_size {
        return;
    }

    var dot: f32 = 0.0;
    for (var j: u32 = 0u; j < upper_size; j = j + 1u) {
        dot += upper_weights[weight_index(idx, j, upper_size)] * upper_values[j];
    }

    lower_preds[idx] = activation(dot, act_fn);
}

/// Compute prediction errors for the lower layer.
///
/// error[i] = value[i] - prediction[i]
///
/// One thread per lower-layer node.
@compute @workgroup_size(64)
fn compute_errors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lower_size = lower_meta[2];
    let idx        = gid.x;

    if idx >= lower_size {
        return;
    }

    lower_errors[idx] = lower_values[idx] - lower_preds[idx];
}
