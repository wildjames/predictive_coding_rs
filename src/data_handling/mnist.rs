//! Dataset loading and preprocessing utilities.

use crate::{
  data_handling::data_handler::TrainingDataset,
  error::{PredictiveCodingError, Result},
};

use tracing::{debug, info};

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use ndarray::{Array1, Array2, s};


// Homebrew IDX reader, since it's a simple format and I don't want to add a dependency.
// IDX reader based on documentation here: https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html

/// Generic IDX data struct
struct IdxData {
  data_type: u8,
  num_dimensions: u8, // e.g. 3 for images (num_images, px_x, px_y)
  dimensions: Vec<u32>, // dimensions of the data, e.g. [60000, 28, 28] for the images
  data: Array1<u8>, // byte data
}

/// Load an IDX file into a flattened byte array with metadata.
fn load_idx<P: AsRef<Path>>(path: P) -> Result<IdxData> {
  let path = path.as_ref();
  let file = File::open(path)
    .map_err(|source| PredictiveCodingError::io("open IDX file", path, source))?;
  let mut reader = BufReader::new(file);

  // Read the magic number (first 4 bytes)
  let mut magic_number_buf = [0u8; 4];
  reader
    .read_exact(&mut magic_number_buf)
    .map_err(|source| PredictiveCodingError::io("read IDX header", path, source))?;

  // The first two bytes are always 0, the third byte is the data type, and the fourth byte is the number of dimensions
  let data_type: u8 = magic_number_buf[2];
  let num_dimensions: u8 = magic_number_buf[3];

  debug!("Data type: 0x{:02x}", data_type);
  debug!("Number of dimensions: {}", num_dimensions);

  // Read the dimensions (next num_dimensions * 4 bytes)
  let mut dimensions = Vec::new();
  for _ in 0..num_dimensions {
    let mut dimension_buf = [0u8; 4];
    reader
      .read_exact(&mut dimension_buf)
      .map_err(|source| PredictiveCodingError::io("read IDX dimensions", path, source))?;
    let dimension_size = u32::from_be_bytes(dimension_buf);
    dimensions.push(dimension_size);
    debug!("Dimension size: {}", dimension_size);
  }

  // Read the data (the rest of the file)
  let mut vector_data = Vec::new();
  reader
    .read_to_end(&mut vector_data)
    .map_err(|source| PredictiveCodingError::io("read IDX payload", path, source))?;
  // Convert the data to an Array1<u8> for easier handling later
  let data: Array1<u8> = Array1::from(vector_data);

  Ok(IdxData {
    data_type,
    num_dimensions,
    dimensions,
    data,
  })
}


pub struct MnistDataset {
  dataset_size: usize,
  input_size: usize,
  output_size: usize,
  inputs: Array2<f32>,
  labels: Array2<f32>,
}

impl TrainingDataset for MnistDataset {
  fn get_dataset_size(&self) -> usize {self.dataset_size}
  fn get_input_size(&self) -> usize {self.input_size}
  fn get_output_size(&self) -> usize {self.output_size}

  fn get_random_input(&self) -> Array1<f32> {
    let rand_index: usize = usize::from_ne_bytes(rand::random()) % self.dataset_size;
    self.get_input(rand_index)
  }

  fn get_random_input_and_output(&self) -> (Array1<f32>, Array1<f32>) {
    let rand_index: usize = usize::from_ne_bytes(rand::random()) % self.dataset_size;
    (self.get_input(rand_index), self.get_output(rand_index))
  }

  fn get_input(&self, index: usize) -> Array1<f32> {self.inputs.row(index).to_owned()}
  fn get_output(&self, index: usize) -> Array1<f32> {self.labels.row(index).to_owned()}
}


/// Load MNIST images and labels from IDX files.
pub fn load_mnist<P: AsRef<Path>>(images_path: P, labels_path: P) -> Result<MnistDataset> {
  let images_path = images_path.as_ref();
  let labels_path = labels_path.as_ref();
  info!(
    "Loading MNIST dataset from {} and {}",
    images_path.display(),
    labels_path.display()
  );

  let images_idx = load_idx(images_path)?;
  let labels_idx = load_idx(labels_path)?;

  // Check that the images are the correct data format. 0x08 is unsigned byte, don't support anything else yet
  if images_idx.data_type != 0x08 || images_idx.num_dimensions != 3 {
    return Err(PredictiveCodingError::invalid_data(format!(
      "invalid image IDX format in {}",
      images_path.display()
    )));
  }
  // Check that the labels are the correct data format
  if labels_idx.data_type != 0x08 || labels_idx.num_dimensions != 1 {
    return Err(PredictiveCodingError::invalid_data(format!(
      "invalid label IDX format in {}",
      labels_path.display()
    )));
  }

  let num_images = images_idx.dimensions[0] as usize;
  let num_labels = labels_idx.dimensions[0] as usize;
  if num_labels != num_images {
    return Err(PredictiveCodingError::invalid_data(format!(
      "mismatch between number of images in {} ({num_images}) and labels in {} ({num_labels})",
      images_path.display(),
      labels_path.display()
    )));
  }
  debug!("Number of images and labels: {}", num_images);
  debug!("Image dimensions: {}x{}", images_idx.dimensions[1], images_idx.dimensions[2]);

  // Then parse them to appropriate vectors
  let input_size = (images_idx.dimensions[1] * images_idx.dimensions[2]) as usize; // will be read in these chunks
  debug!("Input size: {}", input_size);

  let mut images: Array2<f32> = Array2::zeros((num_images, input_size));
  debug!("Parsing image data into array...");
  for i in 0..num_images {
    let start = i * input_size;
    let end = start + input_size;

    let data: Array1<f32> = images_idx.data
      .slice(s![start..end])
      .mapv(|x| x as f32 / 255.0); // normalize to [0, 1]

    images.row_mut(i).assign(&data)
  }

  // Create label array, one-hot on the label index.
  let output_size: usize = 10; // MNIST has 10 classes (digits 0-9)
  let mut labels: Array2<f32> = Array2::zeros((num_labels, output_size));
  debug!("Parsing label data into array...");
  for i in 0..num_labels {
    let label_value = labels_idx.data[i] as usize;
    labels[[i, label_value]] = 1.0;
  }

  debug!("Finished parsing MNIST dataset.");
  Ok(MnistDataset {
    dataset_size: num_images,
    inputs: images,
    labels,
    input_size,
    output_size,
  })
}
