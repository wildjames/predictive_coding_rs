use ndarray::{Array1, Array2};
use tracing::debug;

pub struct Layer {
  values: Array1<f32>, /// node activation values for this layer, x^l
  predictions: Array1<f32>, //Predictions for the value of nodes in this layer, according to the layer above. u^l = f(x^{l+1}, w^{l+1})
  errors: Array1<f32>, // Errors for this layer, e_l
  weights: Array2<f32>, // weights to predict the layer below, w^l
  pinned: bool, // If a layer is pinned, its values are not updated during time evolution (e.g. input layers in unsupervised learning, or input and output layers in supervised learning)
  activation_function: fn(f32) -> f32,
}

impl Layer {
  /// Initialises a layer of the given size.
  /// Populates the values if given, and pins the layer against changing the values during compute iterations if specified.
  /// Weights are randomly initialised, and predictions and errors are initialised to 0.0
  /// Takes ownership of the given values, if they are given, so that we can updated them in place later.
  fn new(
    size: usize,
    activation_function: fn(f32) -> f32,
    values: Option<Array1<f32>>,
    pinned: Option<bool>
  ) -> Self {

    // Use provided values if we have them
    let values = match values {
      Some(v) => v,
      None => Array1::zeros(size)
    };

    // Generate random weights for a blank model layer.
    let weights = Array2::from_shape_fn(
      (size, size), |_| rand::random::<f32>() - 0.5
    );

    Layer {
      values,
      predictions: Array1::zeros(size),
      errors: Array1::zeros(size),
      weights,
      pinned: pinned.unwrap_or(false),
      activation_function
    }
  }

  /// Update the predictions for this layer based on the values of the layer above it
  fn compute_predictions(&mut self, upper_layer: &Layer) {
    // \bf{u}^l_{i,l} = \sum^{J_l}_{j=1}\bf{W}^{l+1}_{i,j}\phi(\bf{x}^{l+1}_{j,t})
    let activation_values: Array1<f32> = upper_layer.values.mapv(|x| (self.activation_function)(x));

    // Then, construct a 2D array of weights times the relevant activation value
    let predictions2d = self.weights.outer_iter()
        .zip(activation_values.iter())
        .map(|(weight_row, activation_value)| weight_row.mapv(|w| w * *activation_value))
        .collect::<Vec<_>>();

    // Finally, sum across the rows to get the final predictions for this layer
    self.predictions = predictions2d
      .iter()
      .fold(
        Array1::zeros(self.values.len()),
        |acc, row| acc + row);
  }

  /// Update the errors for this layer based on the predictions and values of this layer
  fn compute_errors(&mut self) {
    // error is the difference between the predicted and actual value of each node
    self.errors = &self.predictions - &self.values;
  }

  fn read_total_error(&self) -> f32 {
    // Sum the errors of all nodes in this layer
    self.errors.iter().sum()
  }

  /// Compute the change in node values under a single timestep of PC.
  /// Returns the summed change in node values across this layer
  fn values_timestep(&mut self, _gamma: f32, _lower_layer: &Layer) -> f32 {
    //TODO
    0.0
  }

  //TODO
  fn update_weights(&mut self, _alpha: f32, _lower_layer: &Layer) {}
}

pub struct PredictiveCodingModel {
  pub layers: Vec<Layer>,
  pub gamma: f32, // neural learning rate
  pub alpha: f32, // synaptic learning rate
}

impl PredictiveCodingModel {
  pub fn new(layer_sizes: &[usize], gamma: f32, alpha: f32, activation_function: fn(f32) -> f32) -> Self {
    let mut layers = Vec::new();
    for layer_size in layer_sizes {
      layers.push(Layer::new(
        *layer_size,
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

  /// Compute the change in node values under a single timestep of PC.
  /// Returns the mean change in node values across all layers
  pub fn timestep(&mut self) -> f32 {
    let mut total_value_changes = 0.0;
    // the update of a node value depends on the errors of the layer below it
    for i in 0..self.layers.len() - 1 {
      let (lower, upper) = self.layers.split_at_mut(i + 1);
      let lower_layer = &lower[i];
      let upper_layer = &mut upper[0]; // will be updated

      total_value_changes += upper_layer.values_timestep(self.gamma, lower_layer);
    }

    let total_num_nodes = self.layers.iter().map(|layer| layer.values.len()).sum::<usize>() as f32;

    // Mean
    total_value_changes / total_num_nodes
  }

  pub fn update_weights(&mut self) {
    for i in 0..self.layers.len() - 1 {
      let (lower, upper) = self.layers.split_at_mut(i + 1);
      let lower_layer = &lower[i];
      let upper_layer = &mut upper[0]; // will be updated

      upper_layer.update_weights(self.alpha, lower_layer);
    }
  }
}
