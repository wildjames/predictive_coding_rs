use tracing::Level;

/// Configure global tracing subscriber for logging.
pub fn setup_tracing() {
  // a builder for `FmtSubscriber`.
  let subscriber = tracing_subscriber::FmtSubscriber::builder()
    .with_max_level(Level::TRACE)
    .finish();

  tracing::subscriber::set_global_default(subscriber)
    .expect("Setting default subscriber failed");
}
