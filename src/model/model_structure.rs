//! Predictive coding model implementation.
//!
//! Defines a layered model with local prediction errors and weight updates.

use super::{
    maths::ActivationFunction,
    snapshot::{LayerSnapshot, ModelSnapshot},
};
use crate::error::{PredictiveCodingError, Result};

use ndarray::{Array1, Array2};
use rand::{RngExt, rngs::ThreadRng};
use serde::{Deserialize, Serialize};

/// A single predictive coding layer with values, predictions, errors, and weights.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub values: Array1<f32>,
    /// node activation values for this layer, x^l
    pub predictions: Array1<f32>, //Predictions for the value of nodes in this layer, according to the layer above. u^l = f(x^{l+1}, w^{l+1})
    pub errors: Array1<f32>,  // Errors for this layer, e^l
    pub weights: Array2<f32>, // weights to predict the layer below, w^l
    pub pinned: bool, // If a layer is pinned, its values are not updated during time evolution (e.g. input layers in unsupervised learning, or input and output layers in supervised learning)
    pub activation_function: ActivationFunction,
    pub size: usize, // The number of nodes in this layer, for easy reference. Should be the same as values.len(), predictions.len(), and /errors.len()
    pub xavier_limit: f32,
}

impl Layer {
    /// Initialises a layer of the given size.
    /// Populates the values if given, and pins the layer against changing the values during compute iterations if specified.
    /// If values are not given, they're set to random vlaues between 0 and 1
    /// Weights are randomly initialised, and predictions and errors are initialised to 0.0
    /// Takes ownership of the given values, if they are given, so that we can updated them in place later.
    pub fn new(
        size: usize,
        lower_size: Option<usize>,
        activation_function: ActivationFunction,
        values: Option<Array1<f32>>,
        pinned: Option<bool>,
    ) -> Self {
        let mut rng = rand::rng();

        // Use provided values if we have them, otherwise random data 0..1
        let values: Array1<f32> = match values {
            Some(v) => v,
            None => Array1::from_shape_fn(size, |_| rng.random_range(0.0..1.0)),
        };

        // Generate random weights for a blank model layer.
        // Shape is (lower_size, size) to map from this layer to the one below.
        let weights_shape = match lower_size {
            Some(lower) => (lower, size),
            None => (0, size),
        };
        // Xavier initialization: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
        let xavier_limit: f32 = if weights_shape.0 + weights_shape.1 > 0 {
            (6.0_f32 / (weights_shape.0 + weights_shape.1) as f32).sqrt()
        } else {
            1.0
        };
        let weights = Array2::from_shape_fn(weights_shape, |_| {
            rng.random_range(-xavier_limit..xavier_limit)
        });

        Layer {
            values,
            predictions: Array1::zeros(size),
            errors: Array1::zeros(size),
            weights,
            pinned: pinned.unwrap_or(false),
            activation_function,
            size,
            xavier_limit,
        }
    }

    /// Randomise weights between -xavier_limit and xavier_limit for all nodes in this layer.
    pub fn randomise_weights(&mut self) {
        let mut rng = rand::rng();
        self.weights = Array2::from_shape_fn(self.weights.dim(), |_| {
            rng.random_range(-self.xavier_limit..self.xavier_limit)
        });
    }

    /// Randomise values between 0..1 for all nodes in this layer.
    pub fn randomise_values(&mut self, rng: &mut rand::prelude::ThreadRng) {
        self.values = Array1::from_shape_fn(self.values.len(), |_| rng.random_range(0.0..1.0));
    }

    /// Replace the layer values and pin them to avoid updates during inference.
    pub fn pin_values(&mut self, values: Array1<f32>) {
        self.values = values;
        self.pinned = true;
    }

    /// Unpin the layer values to allow updates during inference.
    pub fn unpin_values(&mut self) {
        self.pinned = false;
    }
}

/// A multi-layer predictive coding model with value and weight updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictiveCodingModel {
    pub layers: Vec<Layer>,
    pub alpha: f32, // synaptic learning rate
    pub gamma: f32, // neural learning rate
    pub convergence_threshold: f32,
    pub convergence_steps: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct PredictiveCodingModelConfig {
    pub layer_sizes: Vec<usize>,
    pub alpha: f32,
    pub gamma: f32,
    pub convergence_threshold: f32,
    pub convergence_steps: u32,
    pub activation_function: ActivationFunction,
}

impl PredictiveCodingModel {
    /// Construct a model with the given layer sizes and learning rates.
    ///
    /// alpha is the synaptic learning rate, which controls how much the weights are updated after each inference step.
    /// gamma is the neural learning rate, which controls how much the node values are updated during inference.
    /// activation_function is applied to the node values when computing predictions for the layer below.
    pub fn new(config: &PredictiveCodingModelConfig) -> Self {
        let mut layers = Vec::new();
        for (index, layer_size) in config.layer_sizes.iter().enumerate() {
            let lower_size = if index == 0 {
                None
            } else {
                Some(config.layer_sizes[index - 1])
            };

            layers.push(Layer::new(
                *layer_size,
                lower_size,
                config.activation_function,
                None,
                None,
            ));
        }

        PredictiveCodingModel {
            layers,
            alpha: config.alpha,
            gamma: config.gamma,
            convergence_threshold: config.convergence_threshold,
            convergence_steps: config.convergence_steps,
        }
    }

    // Getters for model properties, so I don't have to expose the model fields directly
    pub fn get_config(&self) -> PredictiveCodingModelConfig {
        PredictiveCodingModelConfig {
            layer_sizes: self.layers.iter().map(|l| l.size).collect(),
            alpha: self.alpha,
            gamma: self.gamma,
            convergence_steps: self.convergence_steps,
            convergence_threshold: self.convergence_threshold,
            // I only allow that all layers have the same activation function
            activation_function: self.layers.first().unwrap().activation_function,
        }
    }

    pub fn from_snapshot(snapshot: &ModelSnapshot) -> Result<Self> {
        if snapshot.layers.len() != snapshot.config.layer_sizes.len() {
            return Err(PredictiveCodingError::validation(format!(
                "snapshot contains {} layers but config declares {}",
                snapshot.layers.len(),
                snapshot.config.layer_sizes.len()
            )));
        }

        let mut layers: Vec<Layer> = Vec::with_capacity(snapshot.layers.len());
        for (index, layer_snapshot) in snapshot.layers.iter().enumerate() {
            let expected_size: usize = snapshot.config.layer_sizes[index];
            if layer_snapshot.size != expected_size {
                return Err(PredictiveCodingError::validation(format!(
                    "snapshot layer {} has size {} but config declares {}",
                    index, layer_snapshot.size, expected_size
                )));
            }

            if layer_snapshot.values.len() != layer_snapshot.size
                || layer_snapshot.predictions.len() != layer_snapshot.size
                || layer_snapshot.errors.len() != layer_snapshot.size
            {
                return Err(PredictiveCodingError::validation(format!(
                    "snapshot layer {} has inconsistent vector lengths for declared size {}",
                    index, layer_snapshot.size
                )));
            }

            let expected_weight_shape: (usize, usize) = if index == 0 {
                (0, layer_snapshot.size)
            } else {
                (snapshot.config.layer_sizes[index - 1], layer_snapshot.size)
            };
            if (layer_snapshot.weight_rows, layer_snapshot.weight_cols) != expected_weight_shape {
                return Err(PredictiveCodingError::validation(format!(
                    "snapshot layer {} has weight shape {:?} but expected {:?}",
                    index,
                    (layer_snapshot.weight_rows, layer_snapshot.weight_cols),
                    expected_weight_shape
                )));
            }

            let weights: Array2<f32> = Array2::from_shape_vec(
                (layer_snapshot.weight_rows, layer_snapshot.weight_cols),
                layer_snapshot.weights.clone(),
            )
            .map_err(|_| {
                PredictiveCodingError::validation(format!(
                    "snapshot layer {} weight payload has {} values but expected {}",
                    index,
                    layer_snapshot.weights.len(),
                    layer_snapshot.weight_rows * layer_snapshot.weight_cols
                ))
            })?;

            let xavier_limit: f32 = if expected_weight_shape.0 + expected_weight_shape.1 > 0 {
                (6.0_f32 / (expected_weight_shape.0 + expected_weight_shape.1) as f32).sqrt()
            } else {
                1.0
            };

            layers.push(Layer {
                values: Array1::from_vec(layer_snapshot.values.clone()),
                predictions: Array1::from_vec(layer_snapshot.predictions.clone()),
                errors: Array1::from_vec(layer_snapshot.errors.clone()),
                weights,
                pinned: layer_snapshot.pinned,
                activation_function: layer_snapshot.activation_function,
                size: layer_snapshot.size,
                xavier_limit,
            });
        }

        Ok(PredictiveCodingModel {
            layers,
            alpha: snapshot.config.alpha,
            gamma: snapshot.config.gamma,
            convergence_threshold: snapshot.config.convergence_threshold,
            convergence_steps: snapshot.config.convergence_steps,
        })
    }

    pub fn to_snapshot(&self) -> ModelSnapshot {
        ModelSnapshot {
            config: self.get_config(),
            layers: self
                .layers
                .iter()
                .map(|layer| LayerSnapshot {
                    values: layer.values.to_vec(),
                    predictions: layer.predictions.to_vec(),
                    errors: layer.errors.to_vec(),
                    weights: layer.weights.iter().copied().collect(),
                    weight_rows: layer.weights.nrows(),
                    weight_cols: layer.weights.ncols(),
                    pinned: layer.pinned,
                    activation_function: layer.activation_function,
                    size: layer.size,
                })
                .collect(),
        }
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }
    pub fn get_layer(&self, index: usize) -> &Layer {
        &self.layers[index]
    }
    pub fn get_layer_sizes(&self) -> Vec<usize> {
        self.layers.iter().map(|l| l.size).collect()
    }
    pub fn get_alpha(&self) -> f32 {
        self.alpha
    }
    pub fn get_gamma(&self) -> f32 {
        self.gamma
    }
    pub fn get_activation_function(&self) -> ActivationFunction {
        self.layers.first().unwrap().activation_function
    }

    /// Set the values of the input layer to the given input values, and pin the input layer.
    pub fn get_input(&self) -> &Array1<f32> {
        &self.layers[0].values
    }

    /// Sets the values of the input layer to the given input values, and pin the input layer.
    pub fn set_input(&mut self, input_values: Array1<f32>) {
        self.layers[0].pin_values(input_values);
    }

    /// Prevent the input layer values from being updated during inference, by pinning the input layer.
    pub fn pin_input(&mut self) {
        self.layers[0].pinned = true;
    }

    /// Allow the input layer values to be updated during inference, by unpinning the input layer.
    pub fn unpin_input(&mut self) {
        self.layers[0].unpin_values();
    }

    /// Randomise the input layer values between 0..1 and unpin the input layer to allow updates during inference.
    pub fn randomise_input(&mut self) {
        let input_layer = &mut self.layers[0];
        let mut rng = rand::rng();
        input_layer.randomise_values(&mut rng);
        input_layer.unpin_values();
    }

    /// Set the values of the output layer to the given output values, and pins the output layer.
    pub fn get_output(&self) -> &Array1<f32> {
        &self.layers.last().unwrap().values
    }

    /// Sets the values of the output layer to the given output values, and pins the output layer.
    pub fn set_output(&mut self, output_values: Array1<f32>) {
        self.layers.last_mut().unwrap().pin_values(output_values);
    }

    /// Prevent the output layer values from being updated during inference, by pinning the output layer.
    pub fn pin_output(&mut self) {
        self.layers.last_mut().unwrap().pinned = true;
    }

    /// Allow the output layer values to be updated during inference, by unpinning the output layer.
    pub fn unpin_output(&mut self) {
        self.layers.last_mut().unwrap().unpin_values();
    }

    /// Randomise the output layer values between 0..1 and unpin the output layer to allow updates during inference.
    pub fn randomise_output(&mut self) {
        let output_layer = self.layers.last_mut().unwrap();
        let mut rng = rand::rng();
        output_layer.randomise_values(&mut rng);
        output_layer.unpin_values();
    }

    /// Reinitialise all unpinned (latent) layer values to small random values.
    /// Should be called before each new training sample to avoid carrying over
    /// converged state from a previous sample.
    pub fn reinitialise_latents(&mut self) {
        let mut rng: ThreadRng = rand::rng();
        for layer in &mut self.layers {
            if !layer.pinned {
                layer.randomise_values(&mut rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::PredictiveCodingModelConfig;
    use ndarray::array;

    #[test]
    fn layer_construction_uses_provided_values_and_default_xavier_limit() {
        let provided_values = array![0.25, 0.75];
        let layer = Layer::new(
            2,
            None,
            ActivationFunction::Relu,
            Some(provided_values.clone()),
            Some(true),
        );

        assert_eq!(layer.values, provided_values);
        assert_eq!(layer.weights.dim(), (0, 2));
        assert!(layer.pinned);

        let zero_sized_layer = Layer::new(
            0,
            None,
            ActivationFunction::Relu,
            Some(Array1::zeros(0)),
            None,
        );
        assert_eq!(zero_sized_layer.xavier_limit, 1.0);
    }

    #[test]
    fn randomise_weights_preserves_shape_and_xavier_bound() {
        let mut layer = Layer::new(3, Some(2), ActivationFunction::Relu, None, None);
        let xavier_limit = layer.xavier_limit;

        layer.randomise_weights();

        assert_eq!(layer.weights.dim(), (2, 3));
        assert!(
            layer
                .weights
                .iter()
                .all(|weight| weight.abs() <= xavier_limit),
            "weights should stay within the Xavier initialisation bounds"
        );
    }

    #[test]
    fn model_snapshot_round_trips_through_portable_snapshot() {
        let model = PredictiveCodingModel::new(&PredictiveCodingModelConfig {
            layer_sizes: vec![4, 10],
            alpha: 0.01,
            gamma: 0.05,
            convergence_threshold: 0.0,
            convergence_steps: 2,
            activation_function: ActivationFunction::Relu,
        });

        let snapshot = model.to_snapshot();
        let restored = PredictiveCodingModel::from_snapshot(&snapshot).unwrap();

        assert_eq!(restored.get_config(), model.get_config());
        assert_eq!(restored.get_layer_sizes(), model.get_layer_sizes());
    }
}
