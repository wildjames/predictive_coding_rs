use std::env;

use image::GrayImage;
use tracing::{error, info};
use ndarray::Array1;

use predictive_coding::{
  model::model_utils::{load_model},
  utils::logging
};


fn main() {
  logging::setup_tracing(false);

  let args: Vec<String> = env::args().collect();
  if args.len() != 3 {
    error!("Usage: generate <model_file>");
    std::process::exit(1);
  }

  let model_fname = args[1].clone();
  let mut model = load_model(&model_fname);
  info!("Loaded model from {}", model_fname);

  model.unpin_input();
  model.randomise_input();

  let output_label: u8 = args[2]
    .parse()
    .map(|x| if x < 10 {x} else {panic!("Output label must be a number between 0 and 9")})
    .expect("Output label must be a number between 0 and 9");

  let output_values: Array1<f32> = (0..10)
    .map(|i| if i == output_label { 1.0 } else { 0.0 })
    .collect();

  model.set_output(output_values);
  model.pin_output();

  info!("Is the input pinned? {}", model.get_layers().first().unwrap().pinned);
  info!("Is the output pinned? {}", model.get_layers().last().unwrap().pinned);

  model.converge_values_with_updates();

  let output_path = format!("{}_gen_{}.png", model_fname, output_label);
  println!("Saving generated image to {}", output_path);

  let image_data = model.get_layer(0).values.map(|&x| (x * 255.0) as u8).to_vec();

  let img = GrayImage::from_raw(
    28,
    28,
    image_data
  )
    .expect("Failed to create image");

  img.save(output_path)
    .expect("Failed to save generated image");
}
