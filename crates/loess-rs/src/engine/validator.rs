//! Input validation for LOESS configuration and data.
//!
//! This module provides comprehensive validation functions for LOESS
//! configuration parameters and input data. It checks requirements
//! such as input lengths, finite values, and parameter bounds.
//!
//! ## srrstats Compliance
//!
//! @srrstats {G2.0} Input validation: non-empty arrays, matching lengths, finite values.
//! @srrstats {G2.1} Edge case handling: minimum points, parameter bounds, duplicates.
//! @srrstats {G2.3} Informative error messages for invalid configuration.

// External dependencies
#[cfg(not(feature = "std"))]
use alloc::format;
use num_traits::Float;

// Internal dependencies
use crate::primitives::errors::LoessError;

// Validation utility for LOESS configuration and input data.
pub struct Validator;

impl Validator {
    // Validate input arrays for LOESS smoothing.
    pub fn validate_inputs<T: Float>(x: &[T], y: &[T]) -> Result<(), LoessError> {
        // Check 1: Non-empty arrays
        if x.is_empty() || y.is_empty() {
            return Err(LoessError::EmptyInput);
        }

        // Check 2: Matching lengths
        let n = x.len();
        if n != y.len() {
            return Err(LoessError::MismatchedInputs {
                x_len: n,
                y_len: y.len(),
            });
        }

        // Check 3: Sufficient points for regression
        if n < 2 {
            return Err(LoessError::TooFewPoints { got: n, min: 2 });
        }

        // Check 4: All values finite (combined loop for cache locality)
        for i in 0..n {
            if !x[i].is_finite() {
                return Err(LoessError::InvalidNumericValue(format!(
                    "x[{}]={}",
                    i,
                    x[i].to_f64().unwrap_or(f64::NAN)
                )));
            }
            if !y[i].is_finite() {
                return Err(LoessError::InvalidNumericValue(format!(
                    "y[{}]={}",
                    i,
                    y[i].to_f64().unwrap_or(f64::NAN)
                )));
            }
        }

        Ok(())
    }

    // Validate a single numeric value for finiteness.
    pub fn validate_scalar<T: Float>(val: T, name: &str) -> Result<(), LoessError> {
        if !val.is_finite() {
            return Err(LoessError::InvalidNumericValue(format!(
                "{}={}",
                name,
                val.to_f64().unwrap_or(f64::NAN)
            )));
        }
        Ok(())
    }

    // Validate the smoothing fraction (bandwidth) parameter.
    pub fn validate_fraction<T: Float>(fraction: T) -> Result<(), LoessError> {
        if !fraction.is_finite() || fraction <= T::zero() || fraction > T::one() {
            return Err(LoessError::InvalidFraction(
                fraction.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // Validate the number of robustness iterations.
    pub fn validate_iterations(iterations: usize) -> Result<(), LoessError> {
        const MAX_ITERATIONS: usize = 1000;
        if iterations > MAX_ITERATIONS {
            return Err(LoessError::InvalidIterations(iterations));
        }
        Ok(())
    }

    // Validate the confidence/prediction interval level.
    pub fn validate_interval_level<T: Float>(level: T) -> Result<(), LoessError> {
        if !level.is_finite() || level <= T::zero() || level >= T::one() {
            return Err(LoessError::InvalidIntervals(
                level.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // Validate a collection of candidate fractions for cross-validation.
    pub fn validate_cv_fractions<T: Float>(fracs: &[T]) -> Result<(), LoessError> {
        if fracs.is_empty() {
            return Err(LoessError::InvalidFraction(0.0));
        }

        for &f in fracs {
            Self::validate_fraction(f)?;
        }

        Ok(())
    }

    // Validate the number of folds for k-fold cross-validation.
    pub fn validate_kfold(k: usize) -> Result<(), LoessError> {
        if k < 2 {
            return Err(LoessError::InvalidNumericValue(format!(
                "k-fold must be at least 2, got {}",
                k
            )));
        }
        Ok(())
    }

    // Validate the auto-convergence tolerance.
    pub fn validate_tolerance<T: Float>(tol: T) -> Result<(), LoessError> {
        if !tol.is_finite() || tol <= T::zero() {
            return Err(LoessError::InvalidTolerance(
                tol.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // Validate delta optimization parameter (equivalent to cell size in some contexts).
    pub fn validate_delta<T: Float>(delta: T) -> Result<(), LoessError> {
        if !delta.is_finite() || delta < T::zero() {
            return Err(LoessError::InvalidDelta(delta.to_f64().unwrap_or(f64::NAN)));
        }
        Ok(())
    }

    // Validate the chunk size for shared processing in streaming mode.
    pub fn validate_chunk_size(chunk_size: usize, min: usize) -> Result<(), LoessError> {
        if chunk_size < min {
            return Err(LoessError::InvalidChunkSize {
                got: chunk_size,
                min,
            });
        }
        Ok(())
    }

    // Validate the overlap between consecutive chunks in streaming mode.
    pub fn validate_overlap(overlap: usize, chunk_size: usize) -> Result<(), LoessError> {
        if overlap >= chunk_size {
            return Err(LoessError::InvalidOverlap {
                overlap,
                chunk_size,
            });
        }
        Ok(())
    }

    // Validate the maximum capacity of the sliding window in online mode.
    pub fn validate_window_capacity(window_capacity: usize, min: usize) -> Result<(), LoessError> {
        if window_capacity < min {
            return Err(LoessError::InvalidWindowCapacity {
                got: window_capacity,
                min,
            });
        }
        Ok(())
    }

    // Validate the activation threshold for online smoothing.
    pub fn validate_min_points(
        min_points: usize,
        window_capacity: usize,
    ) -> Result<(), LoessError> {
        if min_points < 2 || min_points > window_capacity {
            return Err(LoessError::InvalidMinPoints {
                got: min_points,
                window_capacity,
            });
        }
        Ok(())
    }

    // Validate that no parameters were set multiple times in the builder.
    pub fn validate_no_duplicates(duplicate_param: Option<&'static str>) -> Result<(), LoessError> {
        if let Some(param) = duplicate_param {
            return Err(LoessError::DuplicateParameter { parameter: param });
        }
        Ok(())
    }
}
