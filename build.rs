fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Compile the protobuf definitions and generate Rust code
  tonic_prost_build::configure()
      .build_server(true)
      .compile_protos(&["proto/train_model.proto"], &["proto"])?;
  println!("cargo:rerun-if-changed=proto/train_model.proto");
  Ok(())
}
