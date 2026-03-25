use serde::{Deserialize, Serialize};

use super::{maths::ActivationFunction, model_structure::PredictiveCodingModelConfig};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LayerSnapshot {
    pub values: Vec<f32>,
    pub predictions: Vec<f32>,
    pub errors: Vec<f32>,
    pub weights: Vec<f32>,
    pub weight_rows: usize,
    pub weight_cols: usize,
    pub pinned: bool,
    pub activation_function: ActivationFunction,
    pub size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelSnapshot {
    pub config: PredictiveCodingModelConfig,
    pub layers: Vec<LayerSnapshot>,
}
