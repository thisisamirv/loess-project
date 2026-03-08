//! Input abstractions for LOESS smoothing.
//!
//! This module provides a unified abstraction for LOESS inputs, allowing the
//! `fit` method to process multiple data formats (slices, vectors, ndarray)
//! through a single interface.

// External dependencies
#[cfg(feature = "cpu")]
use ndarray::{ArrayBase, Data, Ix1};
use num_traits::Float;

// Export dependencies from loess crate
use loess_rs::internals::primitives::errors::LoessError;

// Trait for types that can be used as input for LOESS smoothing.
pub trait LoessInput<T: Float> {
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

impl<T: Float, I: LoessInput<T> + ?Sized> LoessInput<T> for &I {
    fn as_loess_slice(&self) -> Result<&[T], LoessError> {
        (**self).as_loess_slice()
    }
}
