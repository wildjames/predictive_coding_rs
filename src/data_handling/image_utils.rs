use crate::data_handling::data_handler::ImagesBWDataset;

use std::path::Path;
use image::{GrayImage};

/// Write a single dataset image to disk as a grayscale PNG.
#[allow(dead_code)]
pub fn output_image<P: AsRef<Path>>(
  data: &ImagesBWDataset,
  index: usize,
  output_path: P
) -> image::ImageResult<GrayImage> {

  let image_data = data.images.row(index).to_vec();
  let width = data.image_width;
  let height = data.image_height;
  let img = GrayImage::from_raw(width, height, image_data.to_vec())
    .expect("Failed to create image");

  // Ensure that the output directory exists
  if let Some(parent) = output_path.as_ref().parent() {
    std::fs::create_dir_all(parent)?;
  }

  img.save(output_path)?;
  Ok(img)
}
