//! Predictive coding model implementation.
//!
//! Defines a layered model with local prediction errors and weight updates.

use crate::model::model_utils::{ActivationFunction, outer_product};

use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use rand::RngExt;

/// A single predictive coding layer with values, predictions, errors, and weights.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
  pub values: Array1<f32>, /// node activation values for this layer, x^l
  pub predictions: Array1<f32>, //Predictions for the value of nodes in this layer, according to the layer above. u^l = f(x^{l+1}, w^{l+1})
  pub errors: Array1<f32>, // Errors for this layer, e^l
  pub weights: Array2<f32>, // weights to predict the layer below, w^l
  pub pinned: bool, // If a layer is pinned, its values are not updated during time evolution (e.g. input layers in unsupervised learning, or input and output layers in supervised learning)
  pub activation_function: ActivationFunction,
  pub size: usize, // The number of nodes in this layer, for easy reference. Should be the same as values.len(), predictions.len(), and errors.len()
  xavier_limit: f32,
}

impl Layer {
  /// Initialises a layer of the given size.
  /// Populates the values if given, and pins the layer against changing the values during compute iterations if specified.
  /// If values are not given, they're set to random vlaues between 0 and 1
  /// Weights are randomly initialised, and predictions and errors are initialised to 0.0
  /// Takes ownership of the given values, if they are given, so that we can updated them in place later.
  fn new(
    size: usize,
    lower_size: Option<usize>,
    activation_function: ActivationFunction,
    values: Option<Array1<f32>>,
    pinned: Option<bool>
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
    let weights = Array2::from_shape_fn(weights_shape, |_| rng.random_range(-xavier_limit..xavier_limit));

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

  pub fn randomise_weights(&mut self) {
    let mut rng = rand::rng();
    self.weights = Array2::from_shape_fn(self.weights.dim(), |_| rng.random_range(-self.xavier_limit..self.xavier_limit));
  }

  pub fn randomise_values(&mut self, mut rng: rand::prelude::ThreadRng) {
    self.values = Array1::from_shape_fn(self.values.len(), |_| rng.random_range(0.0..1.0));
  }

  /// Replace the layer values and pin them to avoid updates during inference.
  fn pin_values(&mut self, values: Array1<f32>) {
    self.values = values;
    self.pinned = true;
  }

  fn unpin_values(&mut self) {
    self.pinned = false;
  }

  /// Update the predictions for this layer based on the values of the layer above it.
  fn compute_predictions(&mut self, upper_layer: &Layer) {
    // Note that the prediction computation should *never* be run for an output layer, but making sure of this is the responsibility of the model, not the layer.
    // Besides, since an output layer has no upper layer to pass in, this function would not be callable

    // u^l = phi(W^{l+1} * x^{l+1})
    // Preactivation first, then apply nonlinearity
    let preactivation: Array1<f32> = upper_layer.weights.dot(&upper_layer.values);
    self.predictions = preactivation.mapv(|a| upper_layer.activation_function.apply(a));
  }

  /// Update the errors for this layer based on the predictions and values of this layer.
  fn compute_errors(&mut self) {
    // error is the difference between the predicted and actual value of each node
    self.errors = &self.values - &self.predictions;
  }

  /// Sum the signed error values for all nodes in this layer.
  fn read_total_error(&self) -> f32 {
    // Sum the errors of all nodes in this layer
    self.errors.iter().sum()
  }

  /// Sum the squared error values for all nodes in this layer.
  fn read_total_energy(&self) -> f32 {
    // Energy is the sum of the squared errors of all nodes in this layer
    self.errors.mapv(|x| x.powi(2)).iter().sum()
  }

  /// Compute the change in node values under a single timestep of PC.
  /// Returns the summed absolute change in node values across this layer.
  /// For the input layer, there is no lower layer and None should be passed in instead.
  fn _values_timestep(&mut self, is_top_level: bool, gamma: f32, lower_layer: Option<&Layer>) -> f32 {
    if self.pinned {
      return 0.0
    }

    let rhs: Array1<f32> = if let Some(lower_layer) = lower_layer {
      // RHS: W^{l,T} * (phi'(a^{l-1}) (hammard) e^{l-1})
      // where a^{l-1} = W^l * x^l is the preactivation for the layer below (see 2506.06332)

      // a^{l-1} = W^l * x^l
      let preactivation: Array1<f32> = self.weights.dot(&self.values);

      // phi'(a^{l-1})
      let activation_function_eval_derivitive: Array1<f32> = preactivation.mapv(|a| self.activation_function.derivative(a));

      // phi'(a^{l-1}) (hammard) e^{l-1}
      let gain_modulated_errors: Array1<f32> = activation_function_eval_derivitive * &lower_layer.errors;

      // W^{l,T} * (phi'(a^{l-1}) (hammard) e^{l-1})
      self.weights.t().dot(&gain_modulated_errors)
    } else {
       Array1::zeros(self.values.len())
    };

    // Note that in the output layer, errors are always 0 so the first term of the parentheses is ignored.
    let value_changes: Array1<f32> = if is_top_level {
      gamma * rhs
    } else {
      gamma * (-&self.errors + rhs)
    };

    // Update my values and sum the changes to return
    self.values += &value_changes;
    value_changes.mapv(|x| x.abs()).sum()
  }

  fn values_timestep(&mut self, gamma: f32, lower_layer: Option<&Layer>) -> f32 {
    self._values_timestep(false, gamma, lower_layer)
  }


  fn values_timestep_top_level(&mut self, gamma: f32, lower_layer: &Layer) -> f32 {
    self._values_timestep(true, gamma, Some(lower_layer))
  }

  /// Update prediction weights after convergence based on lower-layer errors.
  fn update_weights(&mut self, alpha: f32, lower_layer: &Layer) {
    // W^{l+1} += alpha * (phi'(a^l) (hammard) e^l) * x^{l+1,T}
    // where a^l = W^{l+1} * x^{l+1} is the preactivation for the layer below
    let preactivation: Array1<f32> = self.weights.dot(&self.values);
    let activation_function_result: Array1<f32> = preactivation.mapv(|a| self.activation_function.derivative(a));
    let gain_modulated_errors: Array1<f32> = &activation_function_result * &lower_layer.errors;

    // outer product yields (lower_size, upper_size)
    let weight_changes: Array2<f32> = alpha * outer_product(&gain_modulated_errors, &self.values);

    self.weights += &weight_changes;
  }
}

/// A multi-layer predictive coding model with value and weight updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictiveCodingModel {
  pub layers: Vec<Layer>,
  pub gamma: f32, // neural learning rate
  pub alpha: f32, // synaptic learning rate
}

impl PredictiveCodingModel {
  /// Construct a model with the given layer sizes and learning rates.
  ///
  /// alpha is the synaptic learning rate, which controls how much the weights are updated after each inference step.
  /// gamma is the neural learning rate, which controls how much the node values are updated during inference.
  /// activation_function is applied to the node values when computing predictions for the layer below.
  pub fn new(
    layer_sizes: &[usize],
    gamma: f32,
    alpha: f32,
    activation_function: ActivationFunction
) -> Self {
    let mut layers = Vec::new();
    for (index, layer_size) in layer_sizes.iter().enumerate() {
      let lower_size = if index == 0 { None } else { Some(layer_sizes[index - 1]) };
      layers.push(Layer::new(
        *layer_size,
        lower_size,
        activation_function,
        None,
        None
      ));
    }

    PredictiveCodingModel {
      layers,
      gamma,
      alpha,
    }
  }

  /// Set the values of the input layer to the given input values, and pin the input layer.
  pub fn set_input(&mut self, input_values: Array1<f32>) {
    self.layers[0].pin_values(input_values);
  }

  pub fn pin_input(&mut self) {
    self.layers[0].pinned = true;
  }

  pub fn unpin_input(&mut self) {
    self.layers[0].unpin_values();
  }

  pub fn randomise_input(&mut self) {
    let input_layer = &mut self.layers[0];
    let mut rng = rand::rng();
    input_layer.values = Array1::from_shape_fn(input_layer.size, |_| rng.random_range(0.0..1.0));
  }

  /// Set the values of the output layer to the given output values, and pins the output layer.
  pub fn set_output(&mut self, output_values: Array1<f32>) {
    self.layers.last_mut().unwrap().pin_values(output_values);
  }

  pub fn pin_output(&mut self) {
    self.layers.last_mut().unwrap().pinned = true;
  }

  pub fn unpin_output(&mut self) {
    self.layers.last_mut().unwrap().unpin_values();
  }

  pub fn randomise_output(&mut self) {
    let output_layer = self.layers.last_mut().unwrap();
    let mut rng = rand::rng();
    output_layer.values = Array1::from_shape_fn(output_layer.size, |_| rng.random_range(0.0..1.0));
  }

  /// Reinitialise all unpinned (latent) layer values to small random values.
  /// Should be called before each new training sample to avoid carrying over
  /// converged state from a previous sample.
  pub fn reinitialise_latents(&mut self) {
    let rng = rand::rng();
    for layer in &mut self.layers {
      if !layer.pinned {
        layer.randomise_values(rng.clone());
      }
    }
  }

  /// Evolves node values until convergence, recomputing predictions and errors each step.
  /// Returns the number of steps taken and the per-step delta values.
  pub fn converge_values_with_updates(
    &mut self,
    convergence_threshold: f32,
    convergence_steps: u32
  ) -> (u32, Vec::<f32>) {
    let mut converged: bool = false;
    let mut convergence_count: u32 = 0;

    let mut value_changes: Vec<f32> = vec![];
    while !converged && (convergence_count < convergence_steps) {
      self.compute_predictions_and_errors();
      value_changes.push(self.timestep());

      if value_changes.last().unwrap().abs() < convergence_threshold {
        converged = true;
      }
      convergence_count += 1;
    };

    (convergence_count, value_changes)
  }

  /// Compute predictions for each layer and then update errors.
  pub fn compute_predictions_and_errors(&mut self) {
    self.compute_predictions();
    self.compute_errors();
  }

  /// Compute predictions for all layers from top to bottom.
  pub fn compute_predictions(&mut self) {
    let num_layers = self.layers.len();
    for i in (0..num_layers - 1).rev() { // iterate backwards through the layers
      // Since the target layer needs to be mutable to update the predictions, I need to split the vector
      // Luckily, this is not a transformative operation, so split_at_mut is still fast
      let (lower, upper) = self.layers.split_at_mut(i+1);
      let lower_layer = &mut lower[i];
      let upper_layer = &upper[0];

      lower_layer.compute_predictions(upper_layer);
    }
  }

  /// Compute prediction errors for all layers.
  pub fn compute_errors(&mut self) {
    for i in 0..self.layers.len() {
      self.layers[i].compute_errors();
    }
  }

  /// Sum signed errors across all layers.
  pub fn read_total_error(&self) -> f32 {
    // Sum the errors of all nodes in all layers
    let mut total_error = 0.0;
    for layer in &self.layers {
      total_error += layer.read_total_error();
    }
    total_error
  }

  /// Sum squared errors across all layers.
  pub fn read_total_energy(&self) -> f32 {
    // Sum the energy of all nodes in all layers
    let mut total_energy = 0.0;
    for layer in &self.layers {
      total_energy += layer.read_total_energy();
    }

    0.5 * total_energy
  }

  /// Compute the change in node values under a single timestep of PC.
  /// Returns the mean change in node values across all layers.
  pub fn timestep(&mut self) -> f32 {
    let mut total_value_changes = 0.0;

    // update the input layer, which has no lower layer
    total_value_changes += self.layers[0].values_timestep(self.gamma, None);

    // the update of a node value depends on the errors of the layer below it.
    let num_layers: usize = self.layers.len();
    for i in 0..num_layers - 1 { // in rust, the range is exclusive of the upper bound
      let (lower, upper) = self.layers.split_at_mut(i + 1);
      let lower_layer: &Layer = &lower[i];
      let upper_layer: &mut Layer = &mut upper[0];

      // The last layer is handled differently.
      if i == num_layers - 1 { // the last i to be processed will be the second to last layer
        total_value_changes += upper_layer.values_timestep_top_level(self.gamma, lower_layer);
      } else {
        total_value_changes += upper_layer.values_timestep(self.gamma, Some(lower_layer));
      }
    }

    // Mean
    let total_num_nodes = self.layers.iter().map(|layer| layer.values.len()).sum::<usize>() as f32;
    total_value_changes / total_num_nodes
  }

  /// Update prediction weights for all layers after inference.
  pub fn update_weights(&mut self) {
    let num_layers = self.layers.len();
    for i in 0..num_layers - 1 {
      let (lower, upper) = self.layers.split_at_mut(i + 1);
      let lower_layer = &lower[i];
      let upper_layer = &mut upper[0];

      upper_layer.update_weights(self.alpha, lower_layer);
    }
  }
}
