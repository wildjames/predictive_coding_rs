use crate::model_utils;

use ndarray::{Array1, Array2};
use rand::RngExt;
use tracing::debug;

pub struct Layer {
  values: Array1<f32>, /// node activation values for this layer, x^l
  predictions: Array1<f32>, //Predictions for the value of nodes in this layer, according to the layer above. u^l = f(x^{l+1}, w^{l+1})
  errors: Array1<f32>, // Errors for this layer, e^l
  weights: Array2<f32>, // weights to predict the layer below, w^l
  pinned: bool, // If a layer is pinned, its values are not updated during time evolution (e.g. input layers in unsupervised learning, or input and output layers in supervised learning)
  activation_function: fn(f32) -> f32
}

impl Layer {
  /// Initialises a layer of the given size.
  /// Populates the values if given, and pins the layer against changing the values during compute iterations if specified.
  /// Weights are randomly initialised, and predictions and errors are initialised to 0.0
  /// Takes ownership of the given values, if they are given, so that we can updated them in place later.
  fn new(
    size: usize,
    lower_size: Option<usize>,
    activation_function: fn(f32) -> f32,
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
    let weights = Array2::from_shape_fn(weights_shape, |_| rng.random::<f32>());

    Layer {
      values,
      predictions: Array1::zeros(size),
      errors: Array1::zeros(size),
      weights,
      pinned: pinned.unwrap_or(false),
      activation_function,
    }
  }

  /// Update the predictions for this layer based on the values of the layer above it
  fn compute_predictions(&mut self, upper_layer: &Layer) {
    // Note that the prediction computation should *never* be run for an output layer, but making sure of this is the responsibility of the model, not the layer.
    // Besides, since an output layer has no upper layer to pass in, this function would not be callable

    // \bf{u}^l = \bf{W}^{l+1} \phi(\bf{x}^{l+1})
    let activation_values: Array1<f32> = upper_layer.values.mapv(|x| (self.activation_function)(x));
    self.predictions = upper_layer.weights.dot(&activation_values);
  }

  /// Update the errors for this layer based on the predictions and values of this layer
  fn compute_errors(&mut self) {
    // error is the difference between the predicted and actual value of each node
    self.errors = &self.values - &self.predictions;
  }

  fn read_total_error(&self) -> f32 {
    // Sum the errors of all nodes in this layer
    self.errors.iter().sum()
  }

  fn read_total_energy(&self) -> f32 {
    // Energy is the sum of the squared errors of all nodes in this layer
    self.errors.mapv(|x| x.powi(2)).iter().sum()
  }

  /// Compute the change in node values under a single timestep of PC.
  /// Returns the summed absolute change in node values across this layer
  /// For the input layer, there is no lower layer and None should be passed in instead
  fn values_timestep(&mut self, gamma: f32, lower_layer: Option<&Layer>) -> f32 {
    if self.pinned {
      return 0.0
    }

    let mut rhs: Array1<f32> = Array1::zeros(self.values.len());
    if let Some(lower_layer) = lower_layer {
      // RHS: \phi(x^l) \odot ((W^l)^T \cdot e^{l-1})
      // where odot is the Hadamard product and ^T is the transpose
      let activation_values: Array1<f32> = self.values.mapv(|x| (self.activation_function)(x)); // phi(x^l)
      let weighted_errors: Array1<f32> = self.weights.t().dot(&lower_layer.errors); // (W^l)^T \cdot e^{l-1}
      rhs = activation_values * weighted_errors; // phi(x^l) \odot ((W^l)^T \cdot e^{l-1})
    }

    // Note that in the output layer, errors are always 0 so the first term of the parentheses is ignored.
    let value_changes: Array1<f32> = gamma * (-&self.errors + rhs);

    // Update my values and sum the changes to return
    self.values += &value_changes;
    value_changes.mapv(|x| x.abs()).sum()
  }

  /// Done after convergence and error calcualations to update the prediction network weights
  fn update_weights(&mut self, alpha: f32, lower_layer: &Layer) {
    let activation_values: Array1<f32> = self.values.mapv(|x| (self.activation_function) (x));
    // outer product yields (lower_size, upper_size)
    let weight_changes: Array2<f32> = alpha * model_utils::outer_product(&lower_layer.errors, &activation_values);

    self.weights += &weight_changes;
  }
}

pub struct PredictiveCodingModel {
  pub layers: Vec<Layer>,
  pub gamma: f32, // neural learning rate
  pub alpha: f32, // synaptic learning rate
}

impl PredictiveCodingModel {
  pub fn new(layer_sizes: &[usize], gamma: f32, alpha: f32, activation_function: fn(f32) -> f32) -> Self {
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

  // TODO
  // pub fn save_to_idx<P: AsRef<Path>>(&self, _path: P) {}
  // pub fn load_from_idx<P: AsRef<Path>>(_path: P) -> Self {Self::new(&[0], 0.0, 0.0, model_utils::relu)}

  pub fn set_input(&mut self, input_values: Array1<f32>) {
    // Set the values of the input layer to the given input values, and pin the input layer.
    // Normalise input values to 0..1
    let normed_input = input_values.mapv(|x| x / 255.0);
    self.layers[0].values = normed_input;
    self.layers[0].pinned = true;
  }

  pub fn compute_predictions(&mut self) {
    debug!("I have {} layers", self.layers.len());
    for i in (0..self.layers.len() - 1).rev() { // iterate backwards through the layers
      debug!("Computing predictions for layer {}", i);
      // Since the target layer needs to be mutable to update the predictions, I need to split the vector
      // Luckily, this is not a transformative operation, so split_at_mut is still fast
      let (lower, upper) = self.layers.split_at_mut(i+1);
      debug!("lower has len {}, upper has len {}", lower.len(), upper.len());
      let lower_layer = &mut lower[i];
      let upper_layer = &upper[0];
      debug!("Got layers");

      lower_layer.compute_predictions(upper_layer);
    }
  }

  pub fn compute_errors(&mut self) {
    for i in 0..self.layers.len() {
      self.layers[i].compute_errors();
    }
  }

  pub fn read_total_error(&self) -> f32 {
    // Sum the errors of all nodes in all layers
    let mut total_error = 0.0;
    for layer in &self.layers {
      total_error += layer.read_total_error();
    }
    total_error
  }

  pub fn read_total_energy(&self) -> f32 {
    // Sum the energy of all nodes in all layers
    let mut total_energy = 0.0;
    for layer in &self.layers {
      total_energy += layer.read_total_energy();
    }

    0.5 * total_energy
  }

  /// Compute the change in node values under a single timestep of PC.
  /// Returns the mean change in node values across all layers
  pub fn timestep(&mut self) -> f32 {
    let mut total_value_changes = 0.0;

    // update the input layer, which has no lower layer
    total_value_changes += self.layers[0].values_timestep(self.gamma, None);

    // the update of a node value depends on the errors of the layer below it
    for i in 0..self.layers.len() - 1 {
      let (lower, upper) = self.layers.split_at_mut(i + 1);
      let lower_layer = &lower[i];
      let upper_layer = &mut upper[0];

      total_value_changes += upper_layer.values_timestep(self.gamma, Some(lower_layer));
    }

    // Mean
    let total_num_nodes = self.layers.iter().map(|layer| layer.values.len()).sum::<usize>() as f32;
    total_value_changes / total_num_nodes
  }

  pub fn update_weights(&mut self) {
    for i in 0..self.layers.len() - 1 {
      let (lower, upper) = self.layers.split_at_mut(i + 1);
      let lower_layer = &lower[i];
      let upper_layer = &mut upper[0];

      upper_layer.update_weights(self.alpha, lower_layer);
    }
  }
}
