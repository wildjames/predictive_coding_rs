use ndarray::{Array1, Array2};

use crate::error::{PredictiveCodingError, Result};

use super::{
    ExecutionBackend, Layer, ModelRuntime, ModelSnapshot, PredictiveCodingModel,
    PredictiveCodingModelConfig, TrainableModelRuntime, WeightUpdateSet, maths::outer_product,
};

pub struct CpuModelRuntime {
    model: PredictiveCodingModel,
}

impl CpuModelRuntime {
    pub fn new(config: &PredictiveCodingModelConfig) -> Self {
        CpuModelRuntime {
            model: PredictiveCodingModel::new(config),
        }
    }

    pub fn from_model(model: PredictiveCodingModel) -> Self {
        CpuModelRuntime { model }
    }

    pub fn from_snapshot(snapshot: &ModelSnapshot) -> Result<Self> {
        Ok(CpuModelRuntime {
            model: PredictiveCodingModel::from_snapshot(snapshot)?,
        })
    }

    pub fn model(&self) -> &PredictiveCodingModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut PredictiveCodingModel {
        &mut self.model
    }

    pub fn into_model(self) -> PredictiveCodingModel {
        self.model
    }

    fn validate_layer_width(actual: usize, expected: usize, label: &str) -> Result<()> {
        if actual != expected {
            return Err(PredictiveCodingError::validation(format!(
                "{label} length {actual} does not match expected size {expected}"
            )));
        }

        Ok(())
    }

    fn compute_predictions_for_layer(lower_layer: &mut Layer, upper_layer: &Layer) {
        let preactivation: Array1<f32> = upper_layer.weights.dot(&upper_layer.values);
        lower_layer.predictions = preactivation.mapv(|a| upper_layer.activation_function.apply(a));
    }

    fn compute_errors_for_layer(layer: &mut Layer) {
        layer.errors = &layer.values - &layer.predictions;
    }

    fn total_error_for_layer(layer: &Layer) -> f32 {
        layer.errors.iter().copied().sum::<f32>()
    }

    fn total_energy_for_layer(layer: &Layer) -> f32 {
        layer.errors.mapv(|x| x.powi(2)).sum()
    }

    fn values_timestep(
        layer: &mut Layer,
        is_top_level: bool,
        gamma: f32,
        lower_layer: Option<&Layer>,
    ) -> f32 {
        if layer.pinned {
            return 0.0;
        }

        let rhs: Array1<f32> = if let Some(lower_layer) = lower_layer {
            let preactivation: Array1<f32> = layer.weights.dot(&layer.values);
            let activation_function_derivative: Array1<f32> =
                preactivation.mapv(|a| layer.activation_function.derivative(a));
            let gain_modulated_errors: Array1<f32> =
                activation_function_derivative * &lower_layer.errors;

            layer.weights.t().dot(&gain_modulated_errors)
        } else {
            Array1::zeros(layer.values.len())
        };

        let value_changes: Array1<f32> = if is_top_level {
            rhs * gamma
        } else {
            (-&layer.errors + rhs) * gamma
        };

        layer.values += &value_changes;
        value_changes.mapv(|x| x.abs()).sum()
    }

    fn compute_weight_updates_for_layer(
        alpha: f32,
        upper_layer: &Layer,
        lower_layer: &Layer,
    ) -> Array2<f32> {
        let preactivation: Array1<f32> = upper_layer.weights.dot(&upper_layer.values);
        let activation_function_derivative: Array1<f32> =
            preactivation.mapv(|a| upper_layer.activation_function.derivative(a));
        let gain_modulated_errors: Array1<f32> =
            &activation_function_derivative * &lower_layer.errors;

        alpha * outer_product(&gain_modulated_errors, &upper_layer.values)
    }

    fn compute_predictions_internal(&mut self) {
        let num_layers = self.model.layers.len();
        if num_layers < 2 {
            return;
        }

        for index in (0..num_layers - 1).rev() {
            let (lower, upper) = self.model.layers.split_at_mut(index + 1);
            let lower_layer = &mut lower[index];
            let upper_layer = &upper[0];

            Self::compute_predictions_for_layer(lower_layer, upper_layer);
        }
    }

    fn compute_errors_internal(&mut self) {
        for layer in &mut self.model.layers {
            Self::compute_errors_for_layer(layer);
        }
    }

    fn timestep_internal(&mut self) -> f32 {
        if self.model.layers.is_empty() {
            return 0.0;
        }

        let mut total_value_changes =
            Self::values_timestep(&mut self.model.layers[0], false, self.model.gamma, None);

        let num_layers = self.model.layers.len();
        for index in 1..num_layers {
            let (lower, upper) = self.model.layers.split_at_mut(index);
            let lower_layer = &lower[index - 1];
            let upper_layer = &mut upper[0];
            let is_top_level = index == num_layers - 1;

            total_value_changes += Self::values_timestep(
                upper_layer,
                is_top_level,
                self.model.gamma,
                Some(lower_layer),
            );
        }

        let total_num_nodes = self
            .model
            .layers
            .iter()
            .map(|layer| layer.values.len())
            .sum::<usize>() as f32;

        total_value_changes / total_num_nodes
    }

    fn converge_values_internal(&mut self) -> u32 {
        let mut converged = false;
        let mut convergence_count = 0;

        while !converged && convergence_count < self.model.convergence_steps {
            self.compute_predictions_internal();
            self.compute_errors_internal();

            if self.timestep_internal().abs() < self.model.convergence_threshold {
                converged = true;
            }

            convergence_count += 1;
        }

        convergence_count
    }

    fn total_error_internal(&self) -> f32 {
        self.model
            .layers
            .iter()
            .map(Self::total_error_for_layer)
            .sum()
    }

    fn total_energy_internal(&self) -> f32 {
        0.5 * self
            .model
            .layers
            .iter()
            .map(Self::total_energy_for_layer)
            .sum::<f32>()
    }

    fn compute_weight_updates_internal(&self) -> Vec<Array2<f32>> {
        let num_layers = self.model.layers.len();
        let mut weight_updates = Vec::with_capacity(num_layers.saturating_sub(1));

        for index in 0..num_layers.saturating_sub(1) {
            weight_updates.push(Self::compute_weight_updates_for_layer(
                self.model.alpha,
                &self.model.layers[index + 1],
                &self.model.layers[index],
            ));
        }

        weight_updates
    }

    fn apply_weight_updates_internal(&mut self, weight_updates: &[Array2<f32>]) {
        for (index, weights) in weight_updates.iter().enumerate() {
            self.model.layers[index + 1].weights += weights;
        }
    }
}

// The surface that the rest of the codebase will interact with
impl ModelRuntime for CpuModelRuntime {
    fn backend(&self) -> ExecutionBackend {
        ExecutionBackend::Cpu
    }

    fn config(&self) -> PredictiveCodingModelConfig {
        self.model.get_config()
    }

    fn layer_sizes(&self) -> Vec<usize> {
        self.model.get_layer_sizes()
    }

    fn snapshot(&mut self) -> Result<ModelSnapshot> {
        Ok(self.model.to_snapshot())
    }

    fn set_input(&mut self, input_values: &[f32]) -> Result<()> {
        Self::validate_layer_width(input_values.len(), self.model.get_input().len(), "input")?;
        self.model
            .set_input(Array1::from_vec(input_values.to_vec()));
        Ok(())
    }

    fn set_output(&mut self, output_values: &[f32]) -> Result<()> {
        Self::validate_layer_width(output_values.len(), self.model.get_output().len(), "output")?;
        self.model
            .set_output(Array1::from_vec(output_values.to_vec()));
        Ok(())
    }

    fn pin_input(&mut self) -> Result<()> {
        self.model.pin_input();
        Ok(())
    }

    fn unpin_input(&mut self) -> Result<()> {
        self.model.unpin_input();
        Ok(())
    }

    fn pin_output(&mut self) -> Result<()> {
        self.model.pin_output();
        Ok(())
    }

    fn unpin_output(&mut self) -> Result<()> {
        self.model.unpin_output();
        Ok(())
    }

    fn reinitialise_latents(&mut self) -> Result<()> {
        self.model.reinitialise_latents();
        Ok(())
    }

    fn compute_predictions_and_errors(&mut self) -> Result<()> {
        self.compute_predictions_internal();
        self.compute_errors_internal();
        Ok(())
    }

    fn timestep(&mut self) -> Result<f32> {
        Ok(self.timestep_internal())
    }

    fn converge_values(&mut self) -> Result<u32> {
        Ok(self.converge_values_internal())
    }

    fn total_error(&mut self) -> Result<f32> {
        Ok(self.total_error_internal())
    }

    fn total_energy(&mut self) -> Result<f32> {
        Ok(self.total_energy_internal())
    }

    fn input_values(&mut self) -> Result<Vec<f32>> {
        Ok(self.model.get_input().to_vec())
    }

    fn output_values(&mut self) -> Result<Vec<f32>> {
        Ok(self.model.get_output().to_vec())
    }
}

impl TrainableModelRuntime for CpuModelRuntime {
    fn compute_weight_updates(&mut self) -> Result<WeightUpdateSet> {
        let arrays: Vec<Array2<f32>> = self.compute_weight_updates_internal();

        Ok(WeightUpdateSet {
            shapes: arrays.iter().map(|array| array.dim()).collect(),
            updates: arrays
                .into_iter()
                .map(|array| array.iter().copied().collect())
                .collect(),
        })
    }

    fn apply_weight_updates(&mut self, updates: &WeightUpdateSet) -> Result<()> {
        if updates.updates.len() != updates.shapes.len() {
            return Err(PredictiveCodingError::validation(
                "weight update payload has mismatched update and shape counts",
            ));
        }

        let expected_layer_count: usize = self.model.get_layers().len().saturating_sub(1);
        if updates.updates.len() != expected_layer_count {
            return Err(PredictiveCodingError::validation(format!(
                "weight update payload contains {} layers but model expects {}",
                updates.updates.len(),
                expected_layer_count
            )));
        }

        let mut arrays: Vec<Array2<f32>> = Vec::with_capacity(updates.updates.len());
        for (index, (update_values, (rows, cols))) in updates
            .updates
            .iter()
            .zip(updates.shapes.iter().copied())
            .enumerate()
        {
            let expected_shape = self.model.get_layer(index + 1).weights.dim();
            if expected_shape != (rows, cols) {
                return Err(PredictiveCodingError::validation(format!(
                    "weight update shape {:?} does not match model layer {} shape {:?}",
                    (rows, cols),
                    index + 1,
                    expected_shape
                )));
            }

            let array: Array2<f32> = Array2::from_shape_vec((rows, cols), update_values.clone())
                .map_err(|_| {
                    PredictiveCodingError::validation(format!(
                        "weight update layer {} contains {} values but expected {}",
                        index + 1,
                        update_values.len(),
                        rows * cols
                    ))
                })?;
            arrays.push(array);
        }

        self.apply_weight_updates_internal(&arrays);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::{ModelRuntime, TrainableModelRuntime, maths::ActivationFunction};
    use crate::test_utils::tiny_relu_model;
    use ndarray::array;

    #[test]
    fn cpu_runtime_snapshot_round_trips_through_model() {
        let model = tiny_relu_model();
        let mut runtime = CpuModelRuntime::from_model(model.clone());

        let snapshot = runtime.snapshot().unwrap();
        let restored = CpuModelRuntime::from_snapshot(&snapshot).unwrap();

        assert_eq!(restored.model().get_config(), model.get_config());
        assert_eq!(restored.model().get_layer_sizes(), model.get_layer_sizes());
    }

    #[test]
    fn cpu_runtime_validates_input_size() {
        let mut runtime = CpuModelRuntime::from_model(tiny_relu_model());

        let error = runtime.set_input(&[1.0, 2.0]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "validation error: input length 2 does not match expected size 4"
        );
    }

    #[test]
    fn cpu_runtime_weight_updates_match_model_layer_count() {
        let mut runtime = CpuModelRuntime::from_model(tiny_relu_model());
        runtime.compute_predictions_and_errors().unwrap();

        let updates = runtime.compute_weight_updates().unwrap();

        assert_eq!(updates.updates.len(), 1);
        assert_eq!(updates.shapes, vec![(4, 10)]);
    }

    #[test]
    fn cpu_runtime_timestep_uses_hidden_layer_error_term_for_non_top_layers() {
        let mut runtime = CpuModelRuntime::new(&PredictiveCodingModelConfig {
            layer_sizes: vec![1, 1, 1],
            alpha: 0.05,
            gamma: 0.5,
            convergence_threshold: 0.0,
            convergence_steps: 1,
            activation_function: ActivationFunction::Relu,
        });

        runtime.model_mut().layers[0].pinned = true;
        runtime.model_mut().layers[0].errors = array![0.0];
        runtime.model_mut().layers[1].values = array![1.0];
        runtime.model_mut().layers[1].errors = array![0.25];
        runtime.model_mut().layers[1].weights = array![[1.0]];
        runtime.model_mut().layers[2].pinned = true;

        runtime.timestep().unwrap();

        assert_eq!(runtime.model().get_layer(1).values, array![0.875]);
    }
}
