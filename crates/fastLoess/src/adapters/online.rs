//! Online adapter for incremental LOESS smoothing.
//!
//! This module provides the online (incremental) execution adapter for LOESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive.
//!
//! ## srrstats Compliance
//!
//! @srrstats {G1.6} Sliding window with optional parallel re-smoothing.
//! @srrstats {G2.1} Configurable min_points threshold before smoothing starts.

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Internal dependencies
use crate::engine::executor::{smooth_pass_parallel, vertex_pass_parallel};
use crate::evaluation::cv::cv_pass_parallel;
use crate::evaluation::intervals::interval_pass_parallel;
use crate::math::neighborhood::build_kdtree_parallel;
#[cfg(feature = "dev")]
use crate::parse::IntoEnum;

// Export dependencies from loess-rs crate
#[cfg(feature = "dev")]
use loess_rs::internals::adapters::online::UpdateMode;
use loess_rs::internals::adapters::online::{OnlineLoessBuilder, OnlineOutput};
#[cfg(feature = "dev")]
use loess_rs::internals::algorithms::regression::PolynomialDegree;
use loess_rs::internals::algorithms::regression::SolverLinalg;
#[cfg(feature = "dev")]
use loess_rs::internals::algorithms::regression::ZeroWeightFallback;
#[cfg(feature = "dev")]
use loess_rs::internals::algorithms::robustness::RobustnessMethod;
#[cfg(feature = "dev")]
use loess_rs::internals::engine::executor::SurfaceMode;
#[cfg(feature = "dev")]
use loess_rs::internals::math::boundary::BoundaryPolicy;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::distance::DistanceMetric;
#[cfg(feature = "dev")]
use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::linalg::FloatLinalg;
#[cfg(feature = "dev")]
use loess_rs::internals::math::scaling::ScalingMethod;
#[cfg(feature = "dev")]
use loess_rs::internals::primitives::backend::Backend;
use loess_rs::internals::primitives::errors::LoessError;

// Builder for online LOESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelOnlineLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    // Base builder from the loess-rs crate
    pub base: OnlineLoessBuilder<T>,
    // Parse errors from string-accepting builder methods; reported together by `build()`.
    pub(crate) parse_errors: Vec<LoessError>,
    // Pending weighted distance metric weights (applied at build time).
    pub(crate) weighted_metric_weights: Option<Vec<T>>,
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
    // Create a new online LOESS builder with default parameters.
    fn new() -> Self {
        let mut base = OnlineLoessBuilder::default();
        // Default to false for online (latency-sensitive)
        base.parallel = Some(false);
        Self {
            base,
            parse_errors: Vec::new(),
            weighted_metric_weights: None,
        }
    }
}

#[cfg(feature = "dev")]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelOnlineLoessBuilder<T>
{
    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the kernel weight function.
    #[allow(private_bounds)]
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness method for outlier handling.
    #[allow(private_bounds)]
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the residual scaling method (MAR/MAD).
    #[allow(private_bounds)]
    pub fn scaling_method(mut self, method: impl IntoEnum<ScalingMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.scaling_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the zero-weight fallback policy.
    #[allow(private_bounds)]
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    #[allow(private_bounds)]
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the polynomial degree.
    #[allow(private_bounds)]
    pub fn polynomial_degree(mut self, degree: impl IntoEnum<PolynomialDegree>) -> Self {
        match degree.into_enum() {
            Ok(d) => self.base.polynomial_degree = d,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the number of dimensions explicitly (though usually inferred from input).
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.base.dimensions = dims;
        self
    }

    // Set the distance metric.
    #[allow(private_bounds)]
    pub fn distance_metric(mut self, metric: impl IntoEnum<DistanceMetric<T>>) -> Self {
        match metric.into_enum() {
            Ok(m) => self.base.distance_metric = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the surface evaluation mode (Direct or Interpolation).
    #[allow(private_bounds)]
    pub fn surface_mode(mut self, mode: impl IntoEnum<SurfaceMode>) -> Self {
        match mode.into_enum() {
            Ok(m) => self.base.surface_mode = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the cell size for interpolation mode.
    pub fn cell(mut self, cell: f64) -> Self {
        self.base.cell = Some(cell);
        self
    }

    // Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.base.interpolation_vertices = Some(vertices);
        self
    }

    // Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.base.boundary_degree_fallback = enabled;
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Set whether to compute residuals.
    pub fn compute_residuals(mut self, compute: bool) -> Self {
        self.base.compute_residuals = compute;
        self
    }

    // Set whether to return robustness weights.
    pub fn return_robustness_weights(mut self, ret: bool) -> Self {
        self.base.return_robustness_weights = ret;
        self
    }

    // Set the window capacity.
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.base.window_capacity = capacity;
        self
    }

    // Set the minimum points required before smoothing.
    pub fn min_points(mut self, min: usize) -> Self {
        self.base.min_points = min;
        self
    }

    // Set the update mode (Incremental/Full).
    #[allow(private_bounds)]
    pub fn update_mode(mut self, mode: impl IntoEnum<UpdateMode>) -> Self {
        match mode.into_enum() {
            Ok(m) => self.base.update_mode = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set per-dimension weights for the `"weighted"` distance metric.
    pub fn weighted_metric_weights(mut self, weights: Vec<T>) -> Self {
        self.weighted_metric_weights = Some(weights);
        self
    }
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelOnlineLoessBuilder<T>
{
    // Build the online processor.
    pub fn build(mut self) -> Result<ParallelOnlineLoess<T>, LoessError> {
        // Check for parse errors from string builder methods
        if !self.parse_errors.is_empty() {
            return Err(LoessError::ParseErrors(self.parse_errors));
        }

        // Apply weighted_metric_weights
        if let Some(weights) = self.weighted_metric_weights.take() {
            self.base.distance_metric = DistanceMetric::Weighted(weights);
        } else if let DistanceMetric::Weighted(ref w) = self.base.distance_metric
            && w.is_empty()
        {
            return Err(LoessError::InvalidOption {
                option: "distance_metric",
                value: "weighted".to_string(),
                valid: "use .weighted_metric_weights(vec![...]) to supply per-dimension weights",
            });
        }

        // Check for deferred errors from adapter conversion
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Configure parallel callbacks before building
        let mut builder = self.base;

        if builder.parallel.unwrap_or(true) {
            builder.custom_smooth_pass = Some(smooth_pass_parallel);
            builder.custom_cv_pass = Some(cv_pass_parallel);
            builder.custom_interval_pass = Some(interval_pass_parallel);
            builder.custom_vertex_pass = Some(vertex_pass_parallel);
            builder.custom_kdtree_builder = Some(build_kdtree_parallel);
        }

        let processor = builder.build()?;
        Ok(ParallelOnlineLoess { processor })
    }
}

// Online LOESS processor with parallel support.
pub struct ParallelOnlineLoess<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    processor: loess_rs::internals::adapters::online::OnlineLoess<T>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static>
    ParallelOnlineLoess<T>
{
    // Add a new point and get its smoothed value.
    pub fn add_point(&mut self, x: &[T], y: T) -> Result<Option<OnlineOutput<T>>, LoessError> {
        self.processor.add_point(x, y)
    }

    // Get the current window size.
    pub fn window_size(&self) -> usize {
        self.processor.window_size()
    }

    // Clear the window.
    pub fn reset(&mut self) {
        self.processor.reset();
    }
}
