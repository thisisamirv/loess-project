//! Input abstractions for LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides a unified abstraction for LOESS inputs, allowing the
//! `fit` method to process multiple data formats (slices, vectors, ndarray)
//! through a single interface.
//!
//! ## Design notes
//!
//! * **Zero-copy where possible**: Provides direct slice access to underlying data buffers.
//! * **Interoperability**: Bridges standard Rust collections with specialized numerical libraries.
//! * **Fail-fast validation**: Ensures memory continuity for multi-dimensional types before processing.
//!
//! ## Key concepts
//!
//! * **LoessInput Trait**: The core abstraction that requires types to provide a contiguous slice view.
//! * **Memory Continuity**: Essential for efficient LOESS kernel processing.
//!
//! ## Invariants
//!
//! * Returned slices must represent all elements in the input container.
//! * Inputs must be contiguous in memory; non-contiguous inputs return an error.
//!
//! ## Non-goals
//!
//! * This module does not perform data cleaning or imputation.
//! * This module does not handle data reshaping or dimensionality reduction.

// Feature-gated imports
#[cfg(feature = "cpu")]
use ndarray::{ArrayBase, Data, Ix1};

// External dependencies
use num_traits::Float;

// Export dependencies from loess-rs crate
use loess_rs::internals::primitives::errors::LoessError;

/// Trait for types that can be used as input for LOESS smoothing.
pub trait LoessInput<T: Float> {
    /// Convert the input to a contiguous slice.
    fn as_loess_slice(&self) -> Result<&[T], LoessError>;
}

impl<T: Float> LoessInput<T> for [T] {
    fn as_loess_slice(&self) -> Result<&[T], LoessError> {
        Ok(self)
    }
}

impl<T: Float> LoessInput<T> for Vec<T> {
    fn as_loess_slice(&self) -> Result<&[T], LoessError> {
        Ok(self.as_slice())
    }
}

#[cfg(feature = "cpu")]
impl<T: Float, S> LoessInput<T> for ArrayBase<S, Ix1>
where
    S: Data<Elem = T>,
{
    fn as_loess_slice(&self) -> Result<&[T], LoessError> {
        self.as_slice().ok_or_else(|| {
            LoessError::InvalidInput("ndarray input must be contiguous in memory".to_string())
        })
    }
}
