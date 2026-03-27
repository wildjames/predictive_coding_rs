use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::{ModelSnapshot, PredictiveCodingModelConfig};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionBackend {
    Cpu,
    Gpu,
    Auto,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightUpdateSet {
    pub updates: Vec<Vec<f32>>,
    pub shapes: Vec<(usize, usize)>,
}

pub trait ModelRuntime: Send {
    fn backend(&self) -> ExecutionBackend;
    fn config(&self) -> PredictiveCodingModelConfig;
    fn layer_sizes(&self) -> Vec<usize>;
    fn snapshot(&mut self) -> Result<ModelSnapshot>;

    fn set_input(&mut self, input_values: &[f32]) -> Result<()>;
    fn set_output(&mut self, output_values: &[f32]) -> Result<()>;

    fn pin_input(&mut self) -> Result<()>;
    fn unpin_input(&mut self) -> Result<()>;
    fn pin_output(&mut self) -> Result<()>;
    fn unpin_output(&mut self) -> Result<()>;

    fn reinitialise_latents(&mut self) -> Result<()>;
    fn compute_predictions_and_errors(&mut self) -> Result<()>;
    fn timestep(&mut self) -> Result<f32>;
    fn converge_values(&mut self) -> Result<u32>;

    fn total_error(&mut self) -> Result<f32>;
    fn total_energy(&mut self) -> Result<f32>;

    fn input_values(&mut self) -> Result<Vec<f32>>;
    fn output_values(&mut self) -> Result<Vec<f32>>;
}

/// The trainiable model runtime exposes additional methods used only during training.
pub trait TrainableModelRuntime: ModelRuntime {
    fn compute_weight_updates(&mut self) -> Result<WeightUpdateSet>;
    fn apply_weight_updates(&mut self, updates: &WeightUpdateSet) -> Result<()>;

    fn update_weights(&mut self) -> Result<()> {
        let updates = self.compute_weight_updates()?;
        self.apply_weight_updates(&updates)
    }
}
