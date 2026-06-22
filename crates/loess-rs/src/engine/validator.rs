//! Input validation for LOESS configuration and data.
//!
//! ## Purpose
//!
//! This module provides comprehensive validation functions for LOESS
//! configuration parameters and input data. It checks requirements
//! such as input lengths, finite values, and parameter bounds.
//!
//! ## Design notes
//!
//! * **Fail-Fast**: Validation stops at the first error encountered.
//! * **Efficiency**: Checks are ordered from cheap to expensive.
//! * **Generics**: Validation is generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Parameter Bounds**: Enforces constraints like fraction in (0, 1].
//! * **Finite Checks**: Ensures all inputs are finite (no NaN/Inf).
//! * **Regression Requirements**: Ensures at least 2 points for linear regression.
//!
//! ## Invariants
//!
//! * All validated inputs satisfy their respective mathematical constraints.
//! * Validation logic is deterministic and side-effect free.
//!
//! ## Non-goals
//!
//! * This module does not sort, transform, or filter input data.
//! * This module does not provide automatic correction of invalid inputs.
//! * This module does not perform the smoothing or optimization itself.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::format;

// External dependencies
use num_traits::Float;

// Internal dependencies
use crate::primitives::errors::LoessError;

// ============================================================================
// Validator
// ============================================================================

/// Validation utility for LOESS configuration and input data.
///
/// Provides static methods for validating various LOESS parameters and
/// input data. All methods return `Result<(), LoessError>` and fail fast
/// upon identifying the first violation.
pub struct Validator;

impl Validator {
    // ========================================================================
    // Core Input Validation
    // ========================================================================

    /// Validate input arrays for LOESS smoothing.
    pub fn validate_inputs<T: Float>(
        x: &[T],
        y: &[T],
        dimensions: usize,
    ) -> Result<(), LoessError> {
        // Check 1: Non-empty arrays
        if x.is_empty() || y.is_empty() {
            return Err(LoessError::EmptyInput);
        }

        // Check 2: Matching lengths (x.len() should be y.len() * dimensions)
        let n_y = y.len();
        if x.len() != n_y * dimensions {
            return Err(LoessError::MismatchedInputs {
                x_len: x.len(),
                y_len: n_y,
            });
        }

        // Check 3: Sufficient points for regression
        if n_y < 2 {
            return Err(LoessError::TooFewPoints { got: n_y, min: 2 });
        }

        // Check 4: All values finite (combined loop for cache locality)
        for (i, &val) in x.iter().enumerate() {
            if !val.is_finite() {
                return Err(LoessError::InvalidNumericValue(format!(
                    "x[{}]={}",
                    i,
                    val.to_f64().unwrap_or(f64::NAN)
                )));
            }
        }
        for (i, &val) in y.iter().enumerate() {
            if !val.is_finite() {
                return Err(LoessError::InvalidNumericValue(format!(
                    "y[{}]={}",
                    i,
                    val.to_f64().unwrap_or(f64::NAN)
                )));
            }
        }

        Ok(())
    }

    /// Validate a single numeric value for finiteness.
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

    // ========================================================================
    // Parameter Validation
    // ========================================================================

    /// Validate the smoothing fraction (bandwidth) parameter.
    pub fn validate_fraction<T: Float>(fraction: T) -> Result<(), LoessError> {
        if !fraction.is_finite() || fraction <= T::zero() || fraction > T::one() {
            return Err(LoessError::InvalidFraction(
                fraction.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate the number of robustness iterations.
    ///
    /// # Notes
    ///
    /// * 0 iterations means initial fit only (no robustness).
    /// * Maximum of 1000 iterations to prevent excessive computation.
    pub fn validate_iterations(iterations: usize) -> Result<(), LoessError> {
        const MAX_ITERATIONS: usize = 1000;
        if iterations > MAX_ITERATIONS {
            return Err(LoessError::InvalidIterations(iterations));
        }
        Ok(())
    }

    /// Validate the confidence/prediction interval level.
    pub fn validate_interval_level<T: Float>(level: T) -> Result<(), LoessError> {
        if !level.is_finite() || level <= T::zero() || level >= T::one() {
            return Err(LoessError::InvalidIntervals(
                level.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate a collection of candidate fractions for cross-validation.
    pub fn validate_cv_fractions<T: Float>(fracs: &[T]) -> Result<(), LoessError> {
        if fracs.is_empty() {
            return Err(LoessError::InvalidFraction(0.0));
        }

        for &f in fracs {
            Self::validate_fraction(f)?;
        }

        Ok(())
    }

    /// Validate the number of folds for k-fold cross-validation.
    pub fn validate_kfold(k: usize) -> Result<(), LoessError> {
        if k < 2 {
            return Err(LoessError::InvalidNumericValue(format!(
                "k-fold must be at least 2, got {}",
                k
            )));
        }
        Ok(())
    }

    /// Validate the auto-convergence tolerance.
    pub fn validate_tolerance<T: Float>(tol: T) -> Result<(), LoessError> {
        if !tol.is_finite() || tol <= T::zero() {
            return Err(LoessError::InvalidTolerance(
                tol.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate interpolation grid resolution against vertex limits.
    ///
    /// **Resolution First:**
    /// - `cell` must be in the range (0, 1]
    /// - Consistency check only when BOTH `cell` AND `interpolation_vertices` are
    ///   explicitly provided by the user
    ///
    /// When both are provided, estimates the number of vertices required by the
    /// cell size and checks if it exceeds the user-provided limit.
    pub fn validate_interpolation_grid<T: Float>(
        cell: T,
        fraction: T,
        dimensions: usize,
        max_vertices: usize,
        cell_provided: bool,
        limit_provided: bool,
    ) -> Result<(), LoessError> {
        let cell_f64 = cell.to_f64().unwrap_or(0.2);

        // Rule 1: Validate cell range (0, 1]
        if cell_f64 <= 0.0 || cell_f64 > 1.0 {
            return Err(LoessError::InvalidCell(cell_f64));
        }

        // Rule 2: Only check consistency when BOTH parameters are user-provided
        // This follows "Resolution First" philosophy where the grid size is purely
        // resolution-driven via `cell` unless explicitly limited.
        if cell_provided && limit_provided {
            let one = T::one();

            // Effective cell width fraction relative to domain: fraction * cell
            let width_fraction = fraction * cell;

            if width_fraction <= T::zero() {
                return Err(LoessError::InsufficientVertices {
                    required: usize::MAX,
                    limit: max_vertices,
                    cell: cell_f64,
                    cell_provided,
                    limit_provided,
                });
            }

            // Estimated cells per dimension: 1 / width_fraction
            let cells_per_dim = one / width_fraction;
            let cells_per_dim_f64 = cells_per_dim.to_f64().unwrap_or(f64::INFINITY);

            // Vertices per dimension is cells + 1. Ceiling to be safe.
            let vertices_per_dim = cells_per_dim_f64.ceil() + 1.0;

            // Calculate total estimated vertices
            let estimated_vertices = vertices_per_dim.powi(dimensions as i32);

            if estimated_vertices > max_vertices as f64 {
                let required = if estimated_vertices > (usize::MAX as f64) {
                    usize::MAX
                } else {
                    estimated_vertices as usize
                };

                return Err(LoessError::InsufficientVertices {
                    required,
                    limit: max_vertices,
                    cell: cell_f64,
                    cell_provided,
                    limit_provided,
                });
            }
        }

        Ok(())
    }

    // ========================================================================
    // Adapter-Specific Validation
    // ========================================================================

    /// Validate the chunk size for shared processing in streaming mode.
    pub fn validate_chunk_size(chunk_size: usize, min: usize) -> Result<(), LoessError> {
        if chunk_size < min {
            return Err(LoessError::InvalidChunkSize {
                got: chunk_size,
                min,
            });
        }
        Ok(())
    }

    /// Validate the overlap between consecutive chunks in streaming mode.
    pub fn validate_overlap(overlap: usize, chunk_size: usize) -> Result<(), LoessError> {
        if overlap >= chunk_size {
            return Err(LoessError::InvalidOverlap {
                overlap,
                chunk_size,
            });
        }
        Ok(())
    }

    /// Validate the maximum capacity of the sliding window in online mode.
    pub fn validate_window_capacity(window_capacity: usize, min: usize) -> Result<(), LoessError> {
        if window_capacity < min {
            return Err(LoessError::InvalidWindowCapacity {
                got: window_capacity,
                min,
            });
        }
        Ok(())
    }

    /// Validate the activation threshold for online smoothing.
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

    /// Validate that no parameters were set multiple times in the builder.
    pub fn validate_no_duplicates(duplicate_param: Option<&'static str>) -> Result<(), LoessError> {
        if let Some(param) = duplicate_param {
            return Err(LoessError::DuplicateParameter { parameter: param });
        }
        Ok(())
    }
}
