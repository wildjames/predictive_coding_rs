//! Dataset loading and preprocessing utilities.

use tracing::debug;

use std::fs::File;
use std::io::{self, Read, BufReader};
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
fn load_idx<P: AsRef<Path>>(path: P) -> io::Result<IdxData> {
  let file = File::open(path)?;
  let mut reader = BufReader::new(file);

  // Read the magic number (first 4 bytes)
  let mut magic_number_buf = [0u8; 4];
  reader.read_exact(&mut magic_number_buf)?;

  // The first two bytes are always 0, the third byte is the data type, and the fourth byte is the number of dimensions
  let data_type: u8 = magic_number_buf[2];
  let num_dimensions: u8 = magic_number_buf[3];

  debug!("Data type: 0x{:02x}", data_type);
  debug!("Number of dimensions: {}", num_dimensions);

  // Read the dimensions (next num_dimensions * 4 bytes)
  let mut dimensions = Vec::new();
  for _ in 0..num_dimensions {
    let mut dimension_buf = [0u8; 4];
    reader.read_exact(&mut dimension_buf)?;
    let dimension_size = u32::from_be_bytes(dimension_buf);
    dimensions.push(dimension_size);
    debug!("Dimension size: {}", dimension_size);
  }

  // Read the data (the rest of the file)
  let mut vector_data = Vec::new();
  reader.read_to_end(&mut vector_data)?;
  // Convert the data to an Array1<u8> for easier handling later
  let data: Array1<u8> = Array1::from(vector_data);

  Ok(IdxData {
    data_type,
    num_dimensions,
    dimensions,
    data,
  })
}


/// Black and white, single channel images with labels. e.g. MNIST dataset
#[derive(Clone)]
pub struct TrainingDataset {
  pub dataset_size: usize,
  pub inputs: Array2<u8>,
  pub labels: Array1<u8>,
  pub image_width: u32,
  pub image_height: u32,
}

/// Load MNIST images and labels from IDX files.
pub fn load_mnist<P: AsRef<Path>>(images_path: P, labels_path: P) -> io::Result<TrainingDataset> {

  let images_idx = load_idx(images_path)?;
  let labels_idx = load_idx(labels_path)?;

  // Check that the images are the correct data format. 0x08 is unsigned byte, don't support anything else yet
  if images_idx.data_type != 0x08 || images_idx.num_dimensions != 3 {
    return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid image IDX format"));
  }
  // Check that the labels are the correct data format
  if labels_idx.data_type != 0x08 || labels_idx.num_dimensions != 1 {
    return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid label IDX format"));
  }

  let num_images = images_idx.dimensions[0] as usize;
  let num_labels = labels_idx.dimensions[0] as usize;
  if num_labels != num_images {
    return Err(io::Error::new(io::ErrorKind::InvalidData, "Mismatch between number of images and labels"));
  }

  // Then parse them to appropriate vectors
  let image_size = (images_idx.dimensions[1] * images_idx.dimensions[2]) as usize; // will be read in these chunks

  let mut images: Array2<u8> = Array2::zeros((num_images, image_size));
  for i in 0..num_images {
    let start = i * image_size;
    let end = start + image_size;

    images.row_mut(i).assign(&images_idx.data.slice(s![start..end]));
  }

  Ok(
    TrainingDataset {
      dataset_size: num_images,
      inputs: images,
      labels: labels_idx.data,
      image_height: images_idx.dimensions[1],
      image_width: images_idx.dimensions[2],
    }
  )
}
