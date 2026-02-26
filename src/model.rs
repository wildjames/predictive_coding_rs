pub struct Layer {
  values: Vec<f32>, /// node activation values for this layer, x^l
  predictions: Vec<f32>, //Predictions for the value of nodes in this layer, according to the layer above. u^l = f(x^{l+1}, w^{l+1})
  errors: Vec<f32>, // Errors for this layer, e_l
  weights: Vec<f32>, // weights to predict the layer below, w^l
  pinned: bool, // If a layer is pinned, its values are not updated during time evolution (e.g. input layers in unsupervised learning, or input and output layers in supervised learning)
  activation_function: fn(f32) -> f32
}

impl Layer {
  /// Initialises a layer of the given size.
  /// Populates the values if given, and pins the layer against changing the values during compute iterations if specified.
  /// Weights are randomly initialised, and predictions and errors are initialised to 0.0
  pub fn new(size: usize, activation_function: fn(f32) -> f32 , pinned: Option<bool>, values: Option<Vec<f32>>) -> Self {

    // Use provided values if we have them
    let values = match values {
      Some(v) => v,
      None => vec![0.0; size]
    };

    // Generate random weights for a blank model layer.
    let weights = (0..size).map(|_| rand::random::<f32>()).collect();

    Layer {
      values,
      predictions: vec![0.0; size],
      errors: vec![0.0; size],
      weights,
      pinned: pinned.unwrap_or(false),
      activation_function
    }
  }

  /// Update the predictions for this layer based on the values of the layer above it
  pub fn compute_predictions(&mut self, upper_layer: &Layer) {}

  pub fn compute_errors(&mut self) {
    // Update the errors for this layer based on the predictions and values of this layer
    self.errors = self.values.iter().zip(self.predictions.iter())
        .map(|(v, p)| v - p)
        .collect();
  }

  pub fn compute_total_error(&self) -> f32 {
    // Sum the errors of all nodes in this layer
    self.errors.iter().sum()
  }

  pub fn values_timestep(&mut self, lower_layer: &Layer) {}
}

pub struct PredictiveCodingModel {
  pub layers: Vec<Layer>,
  pub gamma: f32, // neural learning rate
  pub alpha: f32, // synaptic learning rate
}

impl PredictiveCodingModel {
  pub fn new(layer_sizes: &[usize], gamma: f32, alpha: f32, activation_function: fn(f32) -> f32) -> Self {
    let mut layers = Vec::new();
    for i in 0..layer_sizes.len() {
      layers.push(Layer::new(
        layer_sizes[i],
        activation_function,
        None,
        None,
      ));
    }

    PredictiveCodingModel {
      layers,
      gamma,
      alpha,
    }
  }

  pub fn set_input(&mut self, input_values: Vec<f32>) {
    // Set the values of the input layer to the given input values, and pin the input layer
    self.layers[0].values = input_values;
    self.layers[0].pinned = true;
  }

  pub fn compute_predictions(&mut self) {
    for i in (0..self.layers.len() - 1).rev() { // iterate backwards through the layers
      let (lower, upper) = self.layers.split_at_mut(i);
      let upper_layer = &upper[0];
      lower[i].compute_predictions(upper_layer);
    }
  }

  pub fn compute_errors(&mut self) {
    for i in 0..self.layers.len() {
      self.layers[i].compute_errors();
    }
  }

  pub fn compute_total_error(&self) -> f32 {
    // Sum the errors of all nodes in all layers
    let mut total_error = 0.0;
    for layer in &self.layers {
      total_error += layer.compute_total_error();
    }
    total_error
  }

  pub fn timestep(&mut self) {}

  pub fn update_weights(&mut self) {}
}
