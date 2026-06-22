//! Batch adapter for standard LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the batch execution adapter for LOESS smoothing.
//! It handles complete datasets in memory with sequential processing, making
//! it suitable for small to medium-sized datasets.
//!
//! ## Design notes
//!
//! * **Processing**: Processes entire dataset in a single pass.
//! * **Delegation**: Delegates computation to the execution engine.
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Batch Processing**: Validates, filters, and executes in a single pass.
//!
//! ## Invariants
//!
//! * Input arrays x and y must have the same length.
//! * All values must be finite.
//! * At least 2 data points are required.
//! * Output order matches input order.
//!
//! ## Non-goals
//!
//! * This adapter does not handle streaming data (use streaming adapter).
//! * This adapter does not handle incremental updates (use online adapter).
//! * This adapter does not handle missing values.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;

// Internal dependencies
use crate::algorithms::regression::{PolynomialDegree, SolverLinalg, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{
    CVPassFn, FitPassFn, IntervalPassFn, KDTreeBuilderFn, LoessConfig, LoessExecutor, SmoothPassFn,
    SurfaceMode, VertexPassFn,
};
use crate::engine::output::LoessResult;
use crate::engine::validator::Validator;
use crate::evaluation::cv::CVKind;
use crate::evaluation::diagnostics::Diagnostics;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::distance::{DistanceLinalg, DistanceMetric};
use crate::math::hat_matrix::HatMatrixStats;
use crate::math::kernel::WeightFunction;
use crate::math::linalg::FloatLinalg;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::errors::LoessError;

// ============================================================================
// Batch LOESS Builder
// ============================================================================

/// Builder for batch LOESS processor.
#[derive(Debug, Clone)]
pub struct BatchLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Residual scaling method
    pub scaling_method: ScalingMethod,

    /// Confidence/Prediction interval configuration
    pub interval_type: Option<IntervalMethod<T>>,

    /// Fractions for cross-validation
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation method kind
    pub cv_kind: Option<CVKind>,

    /// Cross-validation seed
    pub cv_seed: Option<u64>,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LoessError>,

    /// Tolerance for auto-convergence
    pub auto_converge: Option<T>,

    /// Whether to compute diagnostic statistics
    pub return_diagnostics: bool,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Whether to return robustness weights
    pub return_robustness_weights: bool,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Policy for handling data boundaries
    pub boundary_policy: BoundaryPolicy,

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

    /// Whether to reduce polynomial degree to Linear at boundary vertices during interpolation.
    /// When `true` (default), vertices outside the tight data bounds use Linear fits.
    /// Set to `false` to match R's loess behavior exactly.
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
    for BatchLoessBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + SolverLinalg> BatchLoessBuilder<T> {
    /// Create a new batch LOESS builder with default parameters.
    fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap(),
            iterations: 3,
            weight_function: WeightFunction::default(),
            robustness_method: RobustnessMethod::default(),
            scaling_method: ScalingMethod::default(),
            interval_type: None,
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            deferred_error: None,
            auto_converge: None,
            return_diagnostics: false,
            compute_residuals: false,
            return_robustness_weights: false,
            zero_weight_fallback: ZeroWeightFallback::default(),
            boundary_policy: BoundaryPolicy::default(),
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
            parallel: None,
            backend: None,
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

    /// Set the number of dimensions explicitly.
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    /// Set the distance metric.
    pub fn distance_metric(mut self, metric: DistanceMetric<T>) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set surface evaluation mode (Interpolation or Direct).
    pub fn surface_mode(mut self, mode: SurfaceMode) -> Self {
        self.surface_mode = mode;
        self
    }

    /// Set cell size for interpolation.
    pub fn cell(mut self, cell: f64) -> Self {
        self.cell = Some(cell);
        self
    }

    /// Set maximum number of interpolation vertices.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.interpolation_vertices = Some(vertices);
        self
    }

    /// Set whether to reduce polynomial degree at boundary vertices during interpolation.
    /// When `true` (default), vertices outside the tight data bounds use Linear fits.
    /// Set to `false` to match R's loess behavior exactly.
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
    // Batch-Specific Setters
    // ========================================================================

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.return_diagnostics = enabled;
        self
    }

    /// Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.interval_type = Some(IntervalMethod::confidence(level));
        self
    }

    /// Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.interval_type = Some(IntervalMethod::prediction(level));
        self
    }

    /// Enable cross-validation with the specified fractions.
    pub fn cross_validate(mut self, fractions: Vec<T>) -> Self {
        self.cv_fractions = Some(fractions);
        self
    }

    /// Set the cross-validation method.
    pub fn cv_kind(mut self, kind: CVKind) -> Self {
        self.cv_kind = Some(kind);
        self
    }

    /// Set the random seed for reproducible cross-validation.
    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.cv_seed = Some(seed);
        self
    }

    /// Enable returning standard errors in the result.
    pub fn return_se(mut self, enabled: bool) -> Self {
        if enabled {
            self.interval_type = Some(IntervalMethod::se());
        } else if let Some(method) = self.interval_type {
            if method.se && !method.confidence && !method.prediction {
                self.interval_type = None;
            }
        }
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

    /// Build the batch processor.
    pub fn build(self) -> Result<BatchLoess<T>, LoessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Check for duplicate parameter configuration
        Validator::validate_no_duplicates(self.duplicate_param)?;

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate iterations
        Validator::validate_iterations(self.iterations)?;

        // Validate interval type
        if let Some(ref method) = self.interval_type {
            Validator::validate_interval_level(method.level)?;
        }

        // Validate CV fractions and method
        if let Some(ref fracs) = self.cv_fractions {
            Validator::validate_cv_fractions(fracs)?;
        }
        if let Some(CVKind::KFold(k)) = self.cv_kind {
            Validator::validate_kfold(k)?;
        }

        // Validate auto convergence tolerance
        if let Some(tol) = self.auto_converge {
            Validator::validate_tolerance(tol)?;
        }

        Ok(BatchLoess { config: self })
    }
}

// ============================================================================
// Batch LOESS Processor
// ============================================================================

/// Batch LOESS processor.
#[derive(Clone)]
pub struct BatchLoess<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    config: BatchLoessBuilder<T>,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + 'static + SolverLinalg> BatchLoess<T> {
    /// Perform LOESS smoothing on the provided data.
    pub fn fit(self, x: &[T], y: &[T]) -> Result<LoessResult<T>, LoessError> {
        Validator::validate_inputs(x, y, self.config.dimensions)?;

        // KD-Tree handles unsorted data natively - no need to sort

        // Check grid resolution only for interpolation mode
        if self.config.surface_mode == SurfaceMode::Interpolation {
            let n = y.len();
            let cell_to_use = self.config.cell.unwrap_or(0.2);
            let limit = self.config.interpolation_vertices.unwrap_or(n);
            let cell_provided = self.config.cell.is_some();
            let limit_provided = self.config.interpolation_vertices.is_some();

            Validator::validate_interpolation_grid(
                T::from(cell_to_use).unwrap_or_else(|| T::from(0.2).unwrap()),
                self.config.fraction,
                self.config.dimensions,
                limit,
                cell_provided,
                limit_provided,
            )?;
        }

        // Configure batch execution
        let config = LoessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            weight_function: self.config.weight_function,
            zero_weight_fallback: self.config.zero_weight_fallback,
            robustness_method: self.config.robustness_method,
            scaling_method: self.config.scaling_method,
            cv_fractions: self.config.cv_fractions,
            cv_kind: self.config.cv_kind,
            auto_converge: self.config.auto_converge,
            return_variance: self.config.interval_type,
            boundary_policy: self.config.boundary_policy,
            polynomial_degree: self.config.polynomial_degree,
            dimensions: self.config.dimensions,
            distance_metric: self.config.distance_metric.clone(),
            cv_seed: self.config.cv_seed,
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

        // Execute unified LOESS (KD-Tree handles unsorted data)
        let result = LoessExecutor::run_with_config(x, y, config);

        let y_smooth = result.smoothed;
        let std_errors = result.std_errors;
        let iterations_used = result.iterations;
        let fraction_used = result.used_fraction;
        let cv_scores = result.cv_scores;

        // Calculate residuals (data is in original order, no unsorting needed)
        let residuals: Vec<T> = y
            .iter()
            .zip(y_smooth.iter())
            .map(|(&orig, &smoothed_val)| orig - smoothed_val)
            .collect();

        // Get robustness weights from executor result (final iteration weights)
        let rob_weights = if self.config.return_robustness_weights {
            result.robustness_weights
        } else {
            Vec::new()
        };

        // Compute diagnostic statistics if requested
        let diagnostics = if self.config.return_diagnostics {
            Some(Diagnostics::compute(
                y,
                &y_smooth,
                &residuals,
                std_errors.as_deref(),
            ))
        } else {
            None
        };

        // Compute hat matrix statistics from leverage if available
        // (Must happen before residuals is moved into residuals_out)
        let (enp, trace_hat, delta1, delta2, residual_scale, leverage_out) =
            if let Some(lev) = result.leverage {
                let stats = HatMatrixStats::from_leverage(lev);
                // Compute RSS (residual sum of squares)
                let rss = residuals.iter().fold(T::zero(), |acc, &r| acc + r * r);
                let res_scale = stats.compute_residual_scale(rss);
                (
                    Some(stats.trace),
                    Some(stats.trace),
                    Some(stats.delta1),
                    Some(stats.delta2),
                    Some(res_scale),
                    Some(stats.leverage),
                )
            } else {
                (None, None, None, None, None, None)
            };

        // Compute intervals
        let (conf_lower, conf_upper, pred_lower, pred_upper) =
            if let Some(method) = &self.config.interval_type {
                if let Some(se) = &std_errors {
                    method.compute_intervals(&y_smooth, se, &residuals, delta1, delta2)?
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            };

        // Results are already in original order (no sorting/unsorting needed with KD-Tree)
        let residuals_out = if self.config.compute_residuals {
            Some(residuals)
        } else {
            None
        };
        let rob_weights_out = if self.config.return_robustness_weights {
            Some(rob_weights)
        } else {
            None
        };

        Ok(LoessResult {
            x: x.to_vec(),
            dimensions: self.config.dimensions,
            distance_metric: self.config.distance_metric.clone(),
            polynomial_degree: self.config.polynomial_degree,
            y: y_smooth,
            standard_errors: std_errors,
            confidence_lower: conf_lower,
            confidence_upper: conf_upper,
            prediction_lower: pred_lower,
            prediction_upper: pred_upper,
            residuals: residuals_out,
            robustness_weights: rob_weights_out,
            fraction_used,
            iterations_used,
            cv_scores,
            diagnostics,
            enp,
            trace_hat,
            delta1,
            delta2,
            residual_scale,
            leverage: leverage_out,
        })
    }
}
