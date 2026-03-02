use predictive_coding::{train_model_handler, utils};
use tonic::transport::Endpoint;

#[tokio::main]
async fn main() {
  utils::setup_tracing();

  let addr: Endpoint = "http://127.0.0.1:50051".parse().unwrap();
  train_model_handler::start_grpc_client(addr).await.unwrap();
}
