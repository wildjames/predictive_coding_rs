//! Library entry point for predictive coding model training and utilities.
//!
//! This crate exposes model components, training helpers, and data loaders.

pub mod data_handling;
pub mod error;
pub mod inference;
pub mod model;
pub mod training;
pub mod utils;

#[cfg(test)]
extern crate self as predictive_coding;

#[cfg(test)]
mod test_utils;
