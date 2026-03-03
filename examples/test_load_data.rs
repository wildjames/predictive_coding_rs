use predictive_coding::data_handling::{
  data_handler::{
    ImagesBWDataset, load_mnist
  },
  image_utils::output_image
};

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
