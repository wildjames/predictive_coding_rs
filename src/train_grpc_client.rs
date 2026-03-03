use predictive_coding::utils::logging;
use predictive_coding::training::train_model_handler;

use tonic::transport::Endpoint;

#[tokio::main]
async fn main() {
  logging::setup_tracing();

  let addr: Endpoint = std::env::var("SERVER_ADDR")
    .unwrap_or("http://192.168.3.111:50051".into())
    .parse()
    .unwrap();
  train_model_handler::start_grpc_client(addr).await.unwrap();
}
