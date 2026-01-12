//! Regression Module
//!
//! ## Purpose
//!
//! This module provides the core functionality for local regression fitting,
//! including the `RegressionContext` struct and the `SolverLinalg` trait.
//!
//! ## Features
//!
//! - Local regression fitting with support for different polynomial degrees.
//! - Support for different solver backends.
//! - Support for different weight functions.

/// Regression Context
mod context;

/// Generic Regression
mod generic;

/// Specialized Regression
mod specialized;

/// Regression Types
mod types;

/// Re-exports
pub use context::RegressionContext;
pub use specialized::SolverLinalg;
pub use types::{PolynomialDegree, ZeroWeightFallback};
