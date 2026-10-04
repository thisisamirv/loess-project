//! Error types for LOESS operations.
//!
//! This module defines error conditions that can occur during LOESS smoothing,
//! including input validation, parameter constraints, and adapter limitations.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::error::Error;
#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::{Display, Formatter, Result};

// Error type for LOESS operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LoessError {
    // Input arrays are empty; LOESS requires at least 2 points.
    EmptyInput,

    // Generic invalid input error with a descriptive message.
    InvalidInput(String),

    // `x` and `y` arrays must have the same number of elements.
    MismatchedInputs {
        // Number of elements in the `x` array.
        x_len: usize,
        // Number of elements in the `y` array.
        y_len: usize,
        // Number of predictor dimensions the model was built for.
        dimensions: usize,
    },

    // Input data contains NaN or infinite values.
    InvalidNumericValue(String),

    // Number of points is below the minimum requirement for the selected parameters.
    TooFewPoints {
        // Number of points provided.
        got: usize,
        // Minimum required points.
        min: usize,
    },

    // Smoothing fraction must be in the range (0, 1].
    InvalidFraction(f64),

    // Robustness iterations (0 means initial fit only).
    InvalidIterations(usize),

    // Interval coverage level must be strictly between 0 and 1.
    InvalidIntervals(f64),

    InvalidBootstrapSamples(usize),

    // Convergence tolerance must be positive and finite.
    InvalidTolerance(f64),

    // Chunk size must be large enough to accommodate the minimum window.
    InvalidChunkSize {
        // The chunk size provided.
        got: usize,
        // Minimum required chunk size.
        min: usize,
    },

    // Overlap must be strictly less than the chunk size to ensure progress.
    InvalidOverlap {
        // The overlap provided.
        overlap: usize,
        // The chunk size.
        chunk_size: usize,
    },

    // Window capacity must be large enough for the requested smoothing parameters.
    InvalidWindowCapacity {
        // The window capacity provided.
        got: usize,
        // Minimum required window capacity.
        min: usize,
    },

    // Minimum points must be at least 2 and at most the window capacity.
    InvalidMinPoints {
        // The min_points provided.
        got: usize,
        // The window capacity.
        window_capacity: usize,
    },

    // Selected adapter does not support the requested feature (e.g., cross-validation).
    UnsupportedFeature {
        // Name of the adapter (e.g., "Streaming", "Online").
        adapter: &'static str,
        // Name of the unsupported feature.
        feature: &'static str,
    },

    // Parameter was set multiple times in the builder.
    DuplicateParameter {
        // Name of the parameter that was set multiple times.
        parameter: &'static str,
    },

    // Runtime execution error.
    RuntimeError(String),

    // Cell size must be in the range (0, 1].
    InvalidCell(f64),

    // Interpolation cell size requires more vertices than allowed limit.
    InsufficientVertices {
        // Estimated number of vertices required.
        required: usize,
        // Maximum number of vertices allowed.
        limit: usize,
        // The cell size that caused the overflow.
        cell: f64,
        // Whether the cell size was explicitly provided by the user.
        cell_provided: bool,
        // Whether the limit was explicitly provided by the user.
        limit_provided: bool,
    },

    // An invalid string value was passed for a configuration option.
    InvalidOption {
        // The name of the configuration option.
        option: &'static str,
        // The invalid value that was provided.
        value: String,
        // Comma-separated list of valid values for the option.
        valid: &'static str,
    },

    // Multiple invalid string option values were passed to the builder.
    //
    // Collects all parse errors from string builder methods and reports them together at `build()`.
    ParseErrors(Vec<LoessError>),

    // `Predict::call()` was called without `.retain_model(true)` on the builder
    // (Batch adapter only), so no fitted-model state was retained to evaluate against.
    PredictionUnavailable,

    // A `predict()` query point fell outside the training data's per-dimension range
    // while `ExtrapolationPolicy::Error` was in effect.
    PredictOutOfRange {
        // Index of the out-of-range predictor dimension.
        dimension: usize,
        // The out-of-range query value.
        query: f64,
        // Minimum of the training range for this dimension.
        min: f64,
        // Maximum of the training range for this dimension.
        max: f64,
    },

    // A `predict()` query point under `ExtrapolationPolicy::Linear` fell farther beyond
    // the training range than `Predict::max_extrapolation_distance` allows. The
    // first-order Taylor extension has no inherent cap, so an unbounded distance can
    // produce arbitrarily extreme values; this guard is opt-in (the option defaults to
    // `None`, preserving the original unbounded behavior).
    ExtrapolationTooFar {
        // Index of the out-of-range predictor dimension.
        dimension: usize,
        // Distance beyond the training boundary on this dimension.
        distance: f64,
        // The configured maximum allowed distance.
        max_distance: f64,
    },

    // A `predict()` query point passed the per-dimension bounding-box range check (so it
    // wasn't caught by `ExtrapolationPolicy`) but its actual nearest-neighbor window is
    // farther away than `Predict::max_neighbor_distance` allows. An axis-aligned
    // bounding box isn't a convex hull: a point can sit inside every dimension's range
    // yet fall in an empty "corner" far from any real training data (e.g. diagonally or
    // non-rectangularly distributed data). This guard is opt-in (the option defaults to
    // `None`, preserving the original silent-extrapolation behavior).
    SparseNeighborhood {
        // Raw (metric-independent) Euclidean distance to the farthest point in the
        // query's k-nearest-neighbor window.
        distance: f64,
        // The configured maximum allowed distance, in the same raw-coordinate units.
        max_distance: f64,
    },

    // `.return_gradient()` was requested but `surface_mode` isn't `"direct"` (the default
    // `"interpolation"` mode only stores value+gradient at a sparse grid of vertices, not
    // enough to reconstruct an exact per-point gradient). Previously this combination
    // silently left `gradient` as `None`; surfaced as an error instead, since it's easy to
    // set `.return_gradient()`, forget `.surface_mode("direct")`, and not notice the
    // silently-empty result.
    GradientRequiresDirectSurfaceMode,

    // `.return_se()`/`.confidence_intervals()`/`.prediction_intervals()` was requested on
    // `OnlineLoess` but `update_mode` isn't `"full"` (the default `"incremental"` mode
    // bypasses the full executor pipeline for speed, so standard errors are never
    // computed there). Previously this combination silently left `standard_error` as
    // `None`; surfaced as an error instead, since it's easy to set `.return_se()`, forget
    // `.update_mode("full")`, and not notice the silently-empty result.
    StandardErrorRequiresFullUpdateMode,

    // `.iterations(n)` with `n > 0` was requested on `OnlineLoess` but
    // `update_mode` isn't `"full"`. The default `"incremental"` mode fits
    // only the latest point and never runs robustness iterations.
    RobustnessIterationsRequireFullUpdateMode,
}

impl Display for LoessError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::EmptyInput => write!(f, "Input arrays are empty"),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::MismatchedInputs {
                x_len,
                y_len,
                dimensions,
            } => {
                let pts = if *y_len == 1 { "point" } else { "points" };
                let expected = y_len * dimensions;
                write!(
                    f,
                    "Length mismatch: x has {x_len} elements, y has {y_len} {pts} \
                     (dimensions={dimensions}, expected x length = {expected})"
                )
            }
            Self::InvalidNumericValue(s) => write!(f, "Invalid numeric value: {s}"),
            Self::TooFewPoints { got, min } => {
                write!(f, "Too few points: got {got}, need at least {min}")
            }
            Self::InvalidFraction(frac) => {
                write!(f, "Invalid fraction: {frac} (must be > 0 and <= 1)")
            }
            Self::InvalidIterations(iter) => {
                write!(f, "Invalid iterations: {iter} (must be in [0, 1000])")
            }
            Self::InvalidIntervals(level) => {
                write!(f, "Invalid interval level: {level} (must be > 0 and < 1)")
            }
            Self::InvalidBootstrapSamples(samples) => {
                write!(
                    f,
                    "Invalid bootstrap samples: {samples} (must be at least 2)"
                )
            }
            Self::InvalidTolerance(tol) => {
                write!(f, "Invalid tolerance: {tol} (must be > 0 and finite)")
            }
            Self::InvalidChunkSize { got, min } => {
                write!(f, "Invalid chunk_size: {got} (must be at least {min})")
            }
            Self::InvalidOverlap {
                overlap,
                chunk_size,
            } => {
                write!(
                    f,
                    "Invalid overlap: {overlap} (must be less than chunk_size {chunk_size})"
                )
            }
            Self::InvalidWindowCapacity { got, min } => {
                write!(f, "Invalid window_capacity: {got} (must be at least {min})")
            }
            Self::InvalidMinPoints {
                got,
                window_capacity,
            } => {
                write!(
                    f,
                    "Invalid min_points: {got} (must be between 2 and window_capacity {window_capacity})"
                )
            }
            Self::UnsupportedFeature { adapter, feature } => {
                write!(f, "Adapter '{adapter}' does not support feature: {feature}")
            }
            Self::DuplicateParameter { parameter } => {
                write!(
                    f,
                    "Parameter '{parameter}' was set multiple times. Each parameter can only be configured once."
                )
            }
            Self::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            Self::InvalidCell(cell) => {
                write!(f, "Invalid cell size: {cell} (must be in range (0, 1])")
            }
            Self::InsufficientVertices {
                required,
                limit,
                cell,
                cell_provided,
                limit_provided,
            } => {
                let cell_desc = if *cell_provided {
                    format!("user-provided cell size {cell}")
                } else {
                    format!("default cell size {cell}")
                };
                let limit_desc = if *limit_provided {
                    format!("user-provided limit {limit}")
                } else {
                    format!("default limit (N = {limit})")
                };

                if !*cell_provided && *limit_provided {
                    write!(
                        f,
                        "Insufficient vertices: {cell_desc} does not work with {limit_desc}. Try passing a larger cell size manually."
                    )
                } else {
                    write!(
                        f,
                        "Insufficient vertices: {cell_desc} requires ~{required} vertices, but {limit_desc} is too small"
                    )
                }
            }
            Self::InvalidOption {
                option,
                value,
                valid,
            } => {
                write!(
                    f,
                    "Invalid value '{value}' for '{option}'. Valid options: {valid}"
                )
            }
            Self::ParseErrors(errors) => {
                write!(f, "Multiple configuration errors ({} total):", errors.len())?;
                for (i, e) in errors.iter().enumerate() {
                    write!(f, " [{i}] {e}")?;
                }
                Ok(())
            }
            Self::PredictionUnavailable => write!(
                f,
                "predict() requires .retain_model(true) on the Batch builder before build()"
            ),
            Self::PredictOutOfRange {
                dimension,
                query,
                min,
                max,
            } => write!(
                f,
                "predict() query point is out of range on dimension {dimension}: {query} \
                 not in [{min}, {max}] (ExtrapolationPolicy::Error)"
            ),
            Self::ExtrapolationTooFar {
                dimension,
                distance,
                max_distance,
            } => write!(
                f,
                "predict() query point on dimension {dimension} is {distance} past the \
                 training boundary, exceeding max_extrapolation_distance ({max_distance}) \
                 (ExtrapolationPolicy::Linear)"
            ),
            Self::SparseNeighborhood {
                distance,
                max_distance,
            } => write!(
                f,
                "predict() query point is within the training bounding box, but its \
                 nearest-neighbor window extends {distance}, exceeding \
                 max_neighbor_distance ({max_distance}); the point likely falls in a \
                 sparse region far from real training data"
            ),
            Self::GradientRequiresDirectSurfaceMode => write!(
                f,
                "return_gradient() requires surface_mode(\"direct\"); the default \
                 \"interpolation\" mode cannot reconstruct an exact per-point gradient"
            ),
            Self::StandardErrorRequiresFullUpdateMode => write!(
                f,
                "return_se()/confidence_intervals()/prediction_intervals() requires \
                 update_mode(\"full\") on OnlineLoess; the default \"incremental\" mode \
                 never computes standard errors"
            ),
            Self::RobustnessIterationsRequireFullUpdateMode => write!(
                f,
                "iterations > 0 requires update_mode(\"full\") on OnlineLoess"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl Error for LoessError {}
