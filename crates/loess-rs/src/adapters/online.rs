//! Online adapter for incremental LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the online (incremental) execution adapter for LOESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive.
//!
//! ## Design notes
//!
//! * **Storage**: Uses a fixed-size circular buffer (VecDeque) for the sliding window.
//! * **Eviction**: Automatically evicts oldest points when capacity is reached.
//! * **Processing**: Performs smoothing on the current window for each new point.
//! * **Generics**: Generic over `Float` types.
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
//!

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::{collections::VecDeque, vec::Vec};
#[cfg(feature = "std")]
use std::{collections::VecDeque, vec::Vec};

// External dependencies
use core::fmt::Debug;

// Internal dependencies
use crate::algorithms::regression::{PolynomialDegree, SolverLinalg, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{
    CVPassFn, FitPassFn, IntervalPassFn, KDTreeBuilderFn, LoessConfig, LoessExecutor, SmoothPassFn,
    SurfaceMode, VertexPassFn,
};
use crate::engine::validator::Validator;
use crate::math::boundary::BoundaryPolicy;
use crate::math::distance::{DistanceLinalg, DistanceMetric};
use crate::math::kernel::WeightFunction;
use crate::math::linalg::FloatLinalg;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::errors::LoessError;

/// Update mode for online LOESS processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    /// Recompute all points in the window from scratch.
    Full,

    /// Optimized incremental update.
    #[default]
    Incremental,
}

// ============================================================================
// Online LOESS Builder
// ============================================================================

/// Builder for online LOESS processor.
#[derive(Debug, Clone)]
pub struct OnlineLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    /// Window capacity (maximum number of points to retain)
    pub window_capacity: usize,

    /// Minimum points before smoothing starts
    pub min_points: usize,

    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Convergence tolerance for early stopping (None = disabled)
    pub auto_converge: Option<T>,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Update mode for incremental processing
    pub update_mode: UpdateMode,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Residual scaling method
    pub scaling_method: ScalingMethod,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Policy for handling data boundaries
    pub boundary_policy: BoundaryPolicy,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Whether to return robustness weights
    pub return_robustness_weights: bool,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LoessError>,

    /// Polynomial degree for local regression
    pub polynomial_degree: PolynomialDegree,

    /// Number of predictor dimensions (default: 1).
    pub dimensions: usize,

    /// Distance metric for nD neighborhood computation.
    pub distance_metric: DistanceMetric<T>,

    /// Cell size for interpolation subdivision (default: 0.2).
    pub cell: Option<f64>,

    /// Maximum number of vertices for interpolation.
    pub interpolation_vertices: Option<usize>,

    /// Evaluation mode (default: Interpolation)
    pub surface_mode: SurfaceMode,

    /// Whether to reduce polynomial degree at boundary vertices during interpolation.
    pub boundary_degree_fallback: bool,

    /// Tracks if any parameter was set multiple times (for validation)
    #[doc(hidden)]
    pub(crate) duplicate_param: Option<&'static str>,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    /// Custom smooth pass function.
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    /// Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    /// Custom fit pass function.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    /// Custom vertex pass function.
    #[doc(hidden)]
    pub custom_vertex_pass: Option<VertexPassFn<T>>,

    /// Custom KD-tree builder function.
    #[doc(hidden)]
    pub custom_kdtree_builder: Option<KDTreeBuilderFn<T>>,

    /// Execution backend hint.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    /// Parallel execution hint.
    #[doc(hidden)]
    pub parallel: Option<bool>,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + SolverLinalg> Default
    for OnlineLoessBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + SolverLinalg> OnlineLoessBuilder<T> {
    /// Create a new online LOESS builder with default parameters.
    fn new() -> Self {
        Self {
            window_capacity: 1000,
            min_points: 3,
            fraction: T::from(0.2).unwrap(),
            iterations: 1,
            weight_function: WeightFunction::default(),
            update_mode: UpdateMode::default(),
            robustness_method: RobustnessMethod::default(),
            scaling_method: ScalingMethod::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            boundary_policy: BoundaryPolicy::default(),
            compute_residuals: false,
            return_robustness_weights: false,
            auto_converge: None,
            deferred_error: None,
            polynomial_degree: PolynomialDegree::default(),
            dimensions: 1,
            distance_metric: DistanceMetric::default(),
            cell: None,
            interpolation_vertices: None,
            surface_mode: SurfaceMode::default(),
            boundary_degree_fallback: true,
            duplicate_param: None,
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_fit_pass: None,
            custom_vertex_pass: None,
            custom_kdtree_builder: None,
            backend: None,
            parallel: None,
        }
    }

    // ========================================================================
    // Shared Setters
    // ========================================================================

    /// Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.fraction = fraction;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set the robustness method for outlier handling.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    /// Set the residual scaling method (MAR/MAD).
    pub fn scaling_method(mut self, method: ScalingMethod) -> Self {
        self.scaling_method = method;
        self
    }

    /// Set the zero-weight fallback policy.
    pub fn zero_weight_fallback(mut self, fallback: ZeroWeightFallback) -> Self {
        self.zero_weight_fallback = fallback;
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    /// Set the polynomial degree.
    pub fn polynomial_degree(mut self, degree: PolynomialDegree) -> Self {
        self.polynomial_degree = degree;
        self
    }

    /// Set the number of dimensions explicitly (though usually inferred from input).
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    /// Set the distance metric.
    pub fn distance_metric(mut self, metric: DistanceMetric<T>) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the evaluation mode (Interpolation or Direct).
    pub fn surface_mode(mut self, mode: SurfaceMode) -> Self {
        self.surface_mode = mode;
        self
    }

    /// Set the interpolation cell size (default: 0.2).
    pub fn cell(mut self, cell: f64) -> Self {
        self.cell = Some(cell);
        self
    }

    /// Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.interpolation_vertices = Some(vertices);
        self
    }

    /// Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.boundary_degree_fallback = enabled;
        self
    }

    /// Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.auto_converge = Some(tolerance);
        self
    }

    /// Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.compute_residuals = enabled;
        self
    }

    /// Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.return_robustness_weights = enabled;
        self
    }

    // ========================================================================
    // Online-Specific Setters
    // ========================================================================

    /// Set window capacity (maximum number of points to retain).
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.window_capacity = capacity;
        self
    }

    /// Set minimum points before smoothing starts.
    pub fn min_points(mut self, min: usize) -> Self {
        self.min_points = min;
        self
    }

    /// Set the update mode for incremental processing.
    pub fn update_mode(mut self, mode: UpdateMode) -> Self {
        self.update_mode = mode;
        self
    }

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

    /// Set a custom smooth pass function.
    #[doc(hidden)]
    pub fn custom_smooth_pass(mut self, pass: SmoothPassFn<T>) -> Self {
        self.custom_smooth_pass = Some(pass);
        self
    }

    /// Set a custom cross-validation pass function.
    #[doc(hidden)]
    pub fn custom_cv_pass(mut self, pass: CVPassFn<T>) -> Self {
        self.custom_cv_pass = Some(pass);
        self
    }

    /// Set a custom interval estimation pass function.
    #[doc(hidden)]
    pub fn custom_interval_pass(mut self, pass: IntervalPassFn<T>) -> Self {
        self.custom_interval_pass = Some(pass);
        self
    }

    /// Set the execution backend hint.
    #[doc(hidden)]
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set a custom KD-tree builder function.
    #[doc(hidden)]
    pub fn custom_kdtree_builder(mut self, kdtree_builder_fn: Option<KDTreeBuilderFn<T>>) -> Self {
        self.custom_kdtree_builder = kdtree_builder_fn;
        self
    }

    /// Set whether to use parallel execution.
    #[doc(hidden)]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the online processor.
    pub fn build(self) -> Result<OnlineLoess<T>, LoessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Check for duplicate parameter configuration
        Validator::validate_no_duplicates(self.duplicate_param)?;

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate iterations
        Validator::validate_iterations(self.iterations)?;

        // Validate configuration early
        Validator::validate_window_capacity(self.window_capacity, 3)?;
        Validator::validate_min_points(self.min_points, self.window_capacity)?;

        let capacity = self.window_capacity;
        Ok(OnlineLoess {
            config: self,
            window_x: VecDeque::with_capacity(capacity),
            window_y: VecDeque::with_capacity(capacity),
            scratch_x: Vec::with_capacity(capacity),
            scratch_y: Vec::with_capacity(capacity),
        })
    }
}

// ============================================================================
// Online LOESS Output
// ============================================================================

/// Result of a single online update.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineOutput<T> {
    /// Smoothed value for the latest point
    pub smoothed: T,

    /// Standard error (if computed)
    pub std_error: Option<T>,

    /// Residual (y - smoothed)
    pub residual: Option<T>,

    /// Robustness weight for the latest point (if computed)
    pub robustness_weight: Option<T>,

    /// Number of robustness iterations actually performed
    pub iterations_used: Option<usize>,
}

// ============================================================================
// Online LOESS Processor
// ============================================================================

/// Online LOESS processor for streaming data.
pub struct OnlineLoess<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    config: OnlineLoessBuilder<T>,
    window_x: VecDeque<T>,
    window_y: VecDeque<T>,
    /// Pre-allocated scratch buffer for x values during smoothing
    scratch_x: Vec<T>,
    /// Pre-allocated scratch buffer for y values during smoothing
    scratch_y: Vec<T>,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + 'static + SolverLinalg>
    OnlineLoess<T>
{
    /// Add a new point and get its smoothed value.
    pub fn add_point(&mut self, x: &[T], y: T) -> Result<Option<OnlineOutput<T>>, LoessError> {
        // Validate new point
        let dimensions = self.config.dimensions;
        if x.len() != dimensions {
            return Err(LoessError::MismatchedInputs {
                x_len: x.len(),
                y_len: 1,
            });
        }
        for &xi in x {
            Validator::validate_scalar(xi, "x")?;
        }
        Validator::validate_scalar(y, "y")?;

        // Add to window
        for &xi in x {
            self.window_x.push_back(xi);
        }
        self.window_y.push_back(y);

        // Evict oldest if over capacity
        if self.window_y.len() > self.config.window_capacity {
            for _ in 0..dimensions {
                self.window_x.pop_front();
            }
            self.window_y.pop_front();
        }

        // Check if we have enough points
        if self.window_y.len() < self.config.min_points {
            return Ok(None);
        }

        // Convert window to vectors for smoothing using scratch buffers
        self.scratch_x.clear();
        self.scratch_y.clear();
        self.scratch_x.extend(self.window_x.iter().copied());
        self.scratch_y.extend(self.window_y.iter().copied());

        let x_vec = &self.scratch_x;
        let y_vec = &self.scratch_y;

        // Special case: exactly two points, use exact linear fit (1D only)
        if y_vec.len() == 2 && dimensions == 1 {
            let x0 = x_vec[0];
            let x1 = x_vec[1];
            let y0 = y_vec[0];
            let y1 = y_vec[1];

            let smoothed = if x1 != x0 {
                let last_x = x[0];
                let slope = (y1 - y0) / (x1 - x0);
                y0 + slope * (last_x - x0)
            } else {
                // Identical x: use mean for stability
                (y0 + y1) / T::from(2.0).unwrap()
            };

            let residual = y - smoothed;

            return Ok(Some(OnlineOutput {
                smoothed,
                std_error: None,
                residual: Some(residual),
                robustness_weight: Some(T::one()),
                iterations_used: Some(0),
            }));
        }

        // Smooth using LOESS for windows of size >= 3

        // Choose update strategy based on configuration
        let (smoothed, std_err, rob_weight, iterations) = match self.config.update_mode {
            UpdateMode::Incremental => {
                // Incremental mode: single-pass fit (no robustness) for maximum performance.
                let n = x_vec.len() / self.config.dimensions;
                let cell_to_use = self.config.cell.unwrap_or(0.2);
                let limit = self.config.interpolation_vertices.unwrap_or(n);
                let cell_provided = self.config.cell.is_some();
                let limit_provided = self.config.interpolation_vertices.is_some();

                if self.config.surface_mode == SurfaceMode::Interpolation {
                    Validator::validate_interpolation_grid(
                        T::from(cell_to_use).unwrap_or_else(|| T::from(0.2).unwrap()),
                        self.config.fraction,
                        self.config.dimensions,
                        limit,
                        cell_provided,
                        limit_provided,
                    )?;
                }

                let config = LoessConfig {
                    fraction: Some(self.config.fraction),
                    iterations: 0, // No robustness for incremental mode (speed)
                    weight_function: self.config.weight_function,
                    robustness_method: self.config.robustness_method,
                    scaling_method: self.config.scaling_method,
                    zero_weight_fallback: self.config.zero_weight_fallback,
                    boundary_policy: self.config.boundary_policy,
                    polynomial_degree: self.config.polynomial_degree,
                    dimensions: self.config.dimensions,
                    distance_metric: self.config.distance_metric.clone(),
                    auto_converge: None,
                    cv_fractions: None,
                    cv_kind: None,
                    return_variance: None,
                    cv_seed: None,
                    surface_mode: self.config.surface_mode,
                    interpolation_vertices: self.config.interpolation_vertices,
                    cell: self.config.cell,
                    boundary_degree_fallback: self.config.boundary_degree_fallback,
                    // ++++++++++++++++++++++++++++++++++++++
                    // +               DEV                  +
                    // ++++++++++++++++++++++++++++++++++++++
                    custom_smooth_pass: self.config.custom_smooth_pass,
                    custom_cv_pass: self.config.custom_cv_pass,
                    custom_interval_pass: self.config.custom_interval_pass,
                    custom_fit_pass: self.config.custom_fit_pass,
                    custom_vertex_pass: self.config.custom_vertex_pass,
                    custom_kdtree_builder: self.config.custom_kdtree_builder,
                    parallel: self.config.parallel.unwrap_or(false),
                    backend: self.config.backend,
                };

                let result = LoessExecutor::run_with_config(x_vec, y_vec, config);
                let smoothed_val = result.smoothed.last().copied().ok_or_else(|| {
                    LoessError::InvalidNumericValue("No smoothed output produced".into())
                })?;

                (smoothed_val, None, Some(T::one()), result.iterations)
            }
            UpdateMode::Full => {
                // Validate grid resolution
                let n = x_vec.len() / self.config.dimensions;
                let cell_to_use = self.config.cell.unwrap_or(0.2);
                let limit = self.config.interpolation_vertices.unwrap_or(n);
                let cell_provided = self.config.cell.is_some();
                let limit_provided = self.config.interpolation_vertices.is_some();

                if self.config.surface_mode == SurfaceMode::Interpolation {
                    Validator::validate_interpolation_grid(
                        T::from(cell_to_use).unwrap_or_else(|| T::from(0.2).unwrap()),
                        self.config.fraction,
                        self.config.dimensions,
                        limit,
                        cell_provided,
                        limit_provided,
                    )?;
                }

                // Full mode: re-smooth entire window
                let config = LoessConfig {
                    fraction: Some(self.config.fraction),
                    iterations: self.config.iterations,
                    weight_function: self.config.weight_function,
                    robustness_method: self.config.robustness_method,
                    scaling_method: self.config.scaling_method,
                    zero_weight_fallback: self.config.zero_weight_fallback,
                    boundary_policy: self.config.boundary_policy,
                    polynomial_degree: self.config.polynomial_degree,
                    dimensions: self.config.dimensions,
                    distance_metric: self.config.distance_metric.clone(),
                    auto_converge: self.config.auto_converge,
                    cv_fractions: None,
                    cv_kind: None,
                    return_variance: None,
                    cv_seed: None,
                    surface_mode: self.config.surface_mode,
                    interpolation_vertices: self.config.interpolation_vertices,
                    cell: self.config.cell,
                    boundary_degree_fallback: self.config.boundary_degree_fallback,
                    // ++++++++++++++++++++++++++++++++++++++
                    // +               DEV                  +
                    // ++++++++++++++++++++++++++++++++++++++
                    custom_smooth_pass: self.config.custom_smooth_pass,
                    custom_cv_pass: self.config.custom_cv_pass,
                    custom_interval_pass: self.config.custom_interval_pass,
                    custom_fit_pass: self.config.custom_fit_pass,
                    custom_vertex_pass: self.config.custom_vertex_pass,
                    custom_kdtree_builder: self.config.custom_kdtree_builder,
                    parallel: self.config.parallel.unwrap_or(false),
                    backend: self.config.backend,
                };

                let result = LoessExecutor::run_with_config(x_vec, y_vec, config.clone());
                let smoothed_vec = result.smoothed;
                let se_vec = result.std_errors;

                let smoothed_val = smoothed_vec.last().copied().ok_or_else(|| {
                    LoessError::InvalidNumericValue("No smoothed output produced".into())
                })?;
                let std_err = se_vec.as_ref().and_then(|v| v.last().copied());
                let rob_weight = if self.config.return_robustness_weights {
                    result.robustness_weights.last().copied()
                } else {
                    None
                };

                (smoothed_val, std_err, rob_weight, result.iterations)
            }
        };

        let residual = y - smoothed;

        Ok(Some(OnlineOutput {
            smoothed,
            std_error: std_err,
            residual: Some(residual),
            robustness_weight: rob_weight,
            iterations_used: iterations,
        }))
    }

    /// Get the current window size.
    pub fn window_size(&self) -> usize {
        self.window_x.len()
    }

    /// Clear the window.
    pub fn reset(&mut self) {
        self.window_x.clear();
        self.window_y.clear();
    }
}
