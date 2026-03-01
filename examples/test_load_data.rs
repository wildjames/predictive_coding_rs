use predictive_coding::train_data_handler::{load_mnist, output_image, ImagesBWDataset};

fn main() {
  let data: ImagesBWDataset = load_mnist(
      "data/mnist/train-images-idx3-ubyte",
      "data/mnist/train-labels-idx1-ubyte")
    .unwrap();
  println!(
    "Loaded the MNIST dataset. I have {} images",
    data.num_images
  );

  let rand_index: usize = usize::from_ne_bytes(rand::random()) % data.num_images;
  output_image(&data, rand_index, format!("data/images/mnist_{}.png", rand_index)).unwrap();
}
