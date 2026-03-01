//! Library entry point for predictive coding model training and utilities.
//!
//! This crate exposes model components, training helpers, and data loaders.

/// Predictive coding model definitions and training logic.
pub mod model;
/// Math utilities used by model components.
pub mod model_utils;
/// Dataset loading and preprocessing helpers.
pub mod train_data_handler;
/// Training orchestration utilities for the predictive coding model.
pub mod train_model_handler;
