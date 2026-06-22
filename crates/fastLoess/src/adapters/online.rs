//! Online adapter for incremental LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the online (incremental) execution adapter for LOESS
//! smoothing. It wraps the loess-rs OnlineLoess with parallel execution support.
//!
//! ## Design notes
//!
//! * **Storage**: Uses a fixed-size circular buffer for the sliding window.
//! * **Processing**: Performs smoothing on the current window for each new point.
//! * **Parallelism**: Optional parallel execution (defaults to false for latency).
//!
//! ## Key concepts
//!
//! * **Sliding Window**: Maintains recent history up to `capacity`.
//! * **Incremental Processing**: Validates, adds, evicts, and smooths.
//! * **Initialization Phase**: Returns `None` until `min_points` are accumulated.
//! * **Update Modes**: Supports `Incremental` (fast) and `Full` (accurate) modes.
//!
//! ## Invariants
//!
//! * Window size never exceeds capacity.
//! * All values in window are finite.
//! * At least `min_points` are required before smoothing.
//! * Window maintains insertion order (oldest to newest).
//!
//! ## Non-goals
//!
//! * This adapter does not support confidence/prediction intervals.
//! * This adapter does not compute diagnostic statistics.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle out-of-order points.

// Feature-gated imports
#[cfg(feature = "cpu")]
use crate::engine::executor::{smooth_pass_parallel, vertex_pass_parallel};
#[cfg(feature = "cpu")]
use crate::evaluation::cv::cv_pass_parallel;
#[cfg(feature = "cpu")]
use crate::evaluation::intervals::interval_pass_parallel;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Internal dependencies
use crate::math::neighborhood::build_kdtree_parallel;

// Export dependencies from loess-rs crate
use loess_rs::internals::adapters::online::{OnlineLoessBuilder, OnlineOutput, UpdateMode};
use loess_rs::internals::algorithms::regression::PolynomialDegree;
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::algorithms::regression::ZeroWeightFallback;
use loess_rs::internals::algorithms::robustness::RobustnessMethod;
use loess_rs::internals::engine::executor::SurfaceMode;
use loess_rs::internals::math::boundary::BoundaryPolicy;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::math::scaling::ScalingMethod;
use loess_rs::internals::primitives::backend::Backend;
use loess_rs::internals::primitives::errors::LoessError;

// ============================================================================
// Extended Online LOESS Builder
// ============================================================================

/// Builder for online LOESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelOnlineLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    /// Base builder from the loess-rs crate
    pub base: OnlineLoessBuilder<T>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> Default
    for ParallelOnlineLoessBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelOnlineLoessBuilder<T>
{
    /// Create a new online LOESS builder with default parameters.
    fn new() -> Self {
        let mut base = OnlineLoessBuilder::default();
        // Default to false for online (latency-sensitive)
        base.parallel = Some(false);
        Self { base }
    }

    /// Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    /// Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // ========================================================================
    // Shared Setters
    // ========================================================================

    /// Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.base.weight_function = wf;
        self
    }

    /// Set the robustness method for outlier handling.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.base.robustness_method = method;
        self
    }

    /// Set the residual scaling method (MAR/MAD).
    pub fn scaling_method(mut self, method: ScalingMethod) -> Self {
        self.base.scaling_method = method;
        self
    }

    /// Set the zero-weight fallback policy.
    pub fn zero_weight_fallback(mut self, fallback: ZeroWeightFallback) -> Self {
        self.base.zero_weight_fallback = fallback;
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.base.boundary_policy = policy;
        self
    }

    /// Set the polynomial degree.
    pub fn polynomial_degree(mut self, degree: PolynomialDegree) -> Self {
        self.base.polynomial_degree = degree;
        self
    }

    /// Set the number of dimensions explicitly (though usually inferred from input).
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.base.dimensions = dims;
        self
    }

    /// Set the distance metric.
    pub fn distance_metric(mut self, metric: DistanceMetric<T>) -> Self {
        self.base.distance_metric = metric;
        self
    }

    /// Set the surface evaluation mode (Direct or Interpolation).
    pub fn surface_mode(mut self, mode: SurfaceMode) -> Self {
        self.base.surface_mode = mode;
        self
    }

    /// Set the cell size for interpolation mode.
    pub fn cell(mut self, cell: f64) -> Self {
        self.base.cell = Some(cell);
        self
    }

    /// Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.base.interpolation_vertices = Some(vertices);
        self
    }

    /// Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.base = self.base.boundary_degree_fallback(enabled);
        self
    }

    /// Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    /// Set whether to compute residuals.
    pub fn compute_residuals(mut self, compute: bool) -> Self {
        self.base.compute_residuals = compute;
        self
    }

    /// Set whether to return robustness weights.
    pub fn return_robustness_weights(mut self, ret: bool) -> Self {
        self.base.return_robustness_weights = ret;
        self
    }

    // ========================================================================
    // Online-Specific Setters
    // ========================================================================

    /// Set the window capacity.
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.base.window_capacity = capacity;
        self
    }

    /// Set the minimum points required before smoothing.
    pub fn min_points(mut self, min: usize) -> Self {
        self.base.min_points = min;
        self
    }

    /// Set the update mode (Incremental/Full).
    pub fn update_mode(mut self, mode: UpdateMode) -> Self {
        self.base.update_mode = mode;
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the online processor.
    pub fn build(self) -> Result<ParallelOnlineLoess<T>, LoessError> {
        // Check for deferred errors from adapter conversion
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Configure parallel callbacks before building
        let mut builder = self.base;

        #[cfg(feature = "cpu")]
        {
            if builder.parallel.unwrap_or(true) {
                builder.custom_smooth_pass = Some(smooth_pass_parallel);
                builder.custom_cv_pass = Some(cv_pass_parallel);
                builder.custom_interval_pass = Some(interval_pass_parallel);
                builder.custom_vertex_pass = Some(vertex_pass_parallel);
                builder.custom_kdtree_builder = Some(build_kdtree_parallel);
            }
        }

        let processor = builder.build()?;
        Ok(ParallelOnlineLoess { processor })
    }
}

// ============================================================================
// Extended Online LOESS Processor
// ============================================================================

/// Online LOESS processor with parallel support.
pub struct ParallelOnlineLoess<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    processor: loess_rs::internals::adapters::online::OnlineLoess<T>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static>
    ParallelOnlineLoess<T>
{
    /// Add a new point and get its smoothed value.
    pub fn add_point(&mut self, x: &[T], y: T) -> Result<Option<OnlineOutput<T>>, LoessError> {
        self.processor.add_point(x, y)
    }

    /// Get the current window size.
    pub fn window_size(&self) -> usize {
        self.processor.window_size()
    }

    /// Clear the window.
    pub fn reset(&mut self) {
        self.processor.reset();
    }
}
