use tracing::Level;

/// Configure global tracing subscriber for logging.
pub fn setup_tracing(debug: bool) {
  let subscriber = tracing_subscriber::FmtSubscriber::builder()
    .with_max_level(if debug { Level::DEBUG } else { Level::INFO })
    .finish();

  tracing::subscriber::set_global_default(subscriber)
    .expect("Setting default subscriber failed");
}
