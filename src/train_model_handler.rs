//! Training orchestration for predictive coding models.

use crate::model;
use crate::data_handler;

use std::sync::{Arc, mpsc};
use std::pin::Pin;

use ndarray::Array1;
use tonic::transport::Endpoint;
use tracing::{info, debug, error};

use liveplot::{
  data::x_formatter::{XFormatter, DecimalFormatter},
  LivePlotConfig,
  AutoFitConfig,
  PlotPoint,
  PlotCommand,
  channel_plot,
  run_liveplot
};

use async_stream::try_stream;
use tokio::sync::Mutex;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use futures_core::Stream;

fn save_model(
  model: &model::PredictiveCodingModel,
  filename: &str
) {
  let model_ser = serde_json::to_string(&model).unwrap();
  std::fs::write(filename, model_ser).unwrap();
}

fn set_input_and_output(
  model: &mut model::PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset
) {
  let rand_index = usize::from_ne_bytes(rand::random()) % data.num_images;

  // Normalise to the range 0..1
  let input_values: Array1<f32> = data.images
    .row(rand_index)
    .mapv(|x| x as f32 / 255.0)
    .to_owned();

  // One-hot output row with label value set to 1.0
  let output_label: usize = data.labels[rand_index] as usize;

  let output_layer_size = model.layers.last().unwrap().size;
  let mut output_values: Array1<f32> = Array1::zeros(output_layer_size);
  output_values[output_label] = 1.0;

  model.set_input(input_values);
  model.set_output(output_values);

}

/// Run inference to convergence on a single sample and update weights.
fn train_sample(
  model: &mut model::PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset,
  convergence_threshold: f32,
  convergence_steps: u32
) {
  set_input_and_output(model, data);
  // Train on this example until convergence.
  model.converge_values_with_updates(convergence_threshold, convergence_steps);
  model.update_weights();
}


/// Train the model for a number of steps using randomly sampled data.
pub fn train(
  model: &mut model::PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset,
  training_steps: u32,
  convergence_steps: u32,
  snapshot_interval: u32,
  convergence_threshold: f32
) {
  // Current timestamp
  let fname_base: String = format!("model_{}", chrono::Utc::now().timestamp());

  for step in 0..training_steps {
    train_sample(
      model,
      data,
      convergence_threshold,
      convergence_steps
    );

    let error = model.read_total_error();
    let energy = model.read_total_energy();
    debug!(
      "Step {}, error {}, energy {}",
      step, error, energy,
    );

    if step % snapshot_interval == 0 {
      info!(
        "Step {}, error {}, energy {}",
        step, error, energy,
      );

      save_model(model, &format!("{}_step_{}.json", fname_base, step));
    }
  }

  let model_error = model.read_total_error();
  info!(
    "Final error of the model is {}",
    model_error
  );

  let model_energy = model.read_total_energy();
  info!(
    "Final energy of the model is {}",
    model_energy
  );
}

/// Train with a local live plot of model energy.
pub fn train_plotting_local(
  model: &mut model::PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset,
  training_steps: u32,
  convergence_steps: u32,
  snapshot_interval: u32,
  convergence_threshold: f32) {

  let (sink, rx) = channel_plot();
  let trace = sink.create_trace("Model Energy", Some("Model Energy"));

  // Run the dataset within a worker thread, since the plotter wants the main one.
  std::thread::scope(|s| {
    s.spawn(move || {
      let fname_base: String = format!("model_{}", chrono::Utc::now().timestamp());

      for step in 0..training_steps {
        train_sample(
          model,
          data,
          convergence_threshold,
          convergence_steps
        );

        let error = model.read_total_error();
        let energy = model.read_total_energy();
        let step_x = step as f64;
        // skip the first sample, since it will have massive energy
        if step > 0 {
          let _ = sink.send_point(&trace, PlotPoint { x: step_x, y: energy as f64 });
        }

        debug!(
          "Step {}, error {}, energy {}",
          step, error, energy,
        );

        if step % snapshot_interval == 0 {
          info!(
            "Step {}, error {}, energy {}",
            step, error, energy,
          );

            save_model(model, &format!("{}_{}.json", fname_base, step));
        }
      }
    });

    // Run the plotting in the main thread
    let auto_fit = AutoFitConfig {
      auto_fit_to_view: true,
      keep_max_fit: true
    };

    let x_formatter = XFormatter::Decimal(DecimalFormatter {
      decimal_places: Some(0),
      unit: None
    });

    let cfg = LivePlotConfig {
      time_window_secs: training_steps as f64,
      max_points: training_steps as usize,
      x_formatter,
      auto_fit,
      ..Default::default()
    };

    run_liveplot(rx, cfg)
      .expect("Failed to create the plot window. If you are on SSH, verify your VcXsrv configuration and DISPLAY settings.");
  });
}

// gRPC server and client for live plotting during training.

#[derive(Clone)]
struct TrainModelGrpcSvc {
  model: Arc<Mutex<model::PredictiveCodingModel>>,
  data: Arc<data_handler::ImagesBWDataset>,
  training_steps: u32,
  snapshot_interval: u32,
  convergence_steps: u32,
  convergence_threshold: f32,
}

impl TrainModelGrpcSvc {
  fn new(
    model: model::PredictiveCodingModel,
    data: data_handler::ImagesBWDataset,
    training_steps: u32,
    snapshot_interval: u32,
    convergence_steps: u32,
    convergence_threshold: f32,
  ) -> Self {
    Self {
      model: Arc::new(Mutex::new(model)),
      data: Arc::new(data),
      training_steps,
      snapshot_interval,
      convergence_steps,
      convergence_threshold,
    }
  }
}

// Include the generated proto just for the example
pub mod train_model {
  pub mod v1 {
    tonic::include_proto!("train_model.v1");
  }
}
use train_model::v1::{
    Sample,
    SubscribeEnergyRequest,
  train_model_grpc_client::TrainModelGrpcClient,
  train_model_grpc_server::{TrainModelGrpc, TrainModelGrpcServer}
};


#[tonic::async_trait]
impl TrainModelGrpc for TrainModelGrpcSvc {
  type SubscribeEnergyStreamStream = Pin<Box<dyn Stream<Item = Result<Sample, Status>> + Send + 'static>>;

  async fn subscribe_energy_stream(
    &self,
    _request: Request<SubscribeEnergyRequest>,
  ) -> Result<Response<Self::SubscribeEnergyStreamStream>, Status> {
    let model: Arc<Mutex<model::PredictiveCodingModel>> = Arc::clone(&self.model);
    let data: Arc<data_handler::ImagesBWDataset> = Arc::clone(&self.data);
    let snapshot_interval = self.snapshot_interval;
    let training_steps = self.training_steps;
    let convergence_steps = self.convergence_steps;
    let convergence_threshold = self.convergence_threshold;

    let out = try_stream! {
      let fname_base: String = format!("data/model_snapshots/{}/model_snap", chrono::Utc::now().timestamp());
      for step in 0..training_steps {
        let (error, energy) = {
          let mut model_guard = model.lock().await;

          train_sample(
            &mut model_guard,
            &data,
            convergence_threshold,
            convergence_steps
          );

          let error = model_guard.read_total_error();
          let energy = model_guard.read_total_energy();

          if step % snapshot_interval == 0 {
            info!(
              "Step {}, error {}, energy {}",
              step, error, energy,
            );

            save_model(&model_guard, &format!("{}_{}.json", fname_base, step));
          }

          (error, energy)
        };

        debug!(
          "Step {}, error {}, energy {}",
          step, error, energy,
        );
        let sample = Sample {
          step,
          error,
          energy
        };
        yield sample;
      }
    };

    Ok(Response::new(Box::pin(out) as Self::SubscribeEnergyStreamStream))
  }
}


pub fn start_grpc_server_sync(
  model: &mut model::PredictiveCodingModel,
  data: &data_handler::ImagesBWDataset,
  training_steps: u32,
  snapshot_interval: u32,
  convergence_steps: u32,
  convergence_threshold: f32
) {
  let rt = tokio::runtime::Runtime::new().unwrap();
  rt.block_on(start_grpc_server(
    model.clone(),
    data.clone(),
    training_steps,
    snapshot_interval,
    convergence_steps,
    convergence_threshold
  )).unwrap();
}


pub async fn start_grpc_server(
  model: model::PredictiveCodingModel,
  data: data_handler::ImagesBWDataset,
  training_steps: u32,
  snapshot_interval: u32,
  convergence_steps: u32,
  convergence_threshold: f32
) -> Result<(), Box<dyn std::error::Error>> {
  let addr: std::net::SocketAddr = "127.0.0.1:50051".parse()?;
  let svc = TrainModelGrpcSvc::new(
    model,
    data,
    training_steps,
    snapshot_interval,
    convergence_steps,
    convergence_threshold,
  );

  info!("Starting gRPC server on {}", addr);
  Server::builder()
    .add_service(TrainModelGrpcServer::new(svc))
    .serve(addr)
    .await?;

  Ok(())
}

pub async fn start_grpc_client(dst: Endpoint) -> Result<(), Box<dyn std::error::Error>> {
  let (tx, rx) = mpsc::channel::<PlotCommand>();

  let ui_handle = std::thread::spawn(move || {
    if let Err(e) = run_liveplot(rx, LivePlotConfig::default()) {
      eprintln!("Failed to run live plot: {}", e);
    }
  });

  let mut client = TrainModelGrpcClient::connect(dst).await?;
  let mut stream = client.subscribe_energy_stream(Request::new(SubscribeEnergyRequest {}))
    .await?
    .into_inner();

  let _ = tx.send(PlotCommand::RegisterTrace {
    id: 1,
    name: "Model Energy".into(),
    info: None,
  });

  while let Some(sample) = stream.message().await? {
    let step = sample.step as f64;
    let energy = sample.energy as f64;
    let cmd = PlotCommand::Point{
      trace_id: 1,
      point: PlotPoint { x: step, y: energy }
    };
    if tx.send(cmd).is_err() {
      error!("Failed to send plot command, UI thread may have exited");
      break;
    }
  }

  let _ = ui_handle.join();
  Ok(())
}
