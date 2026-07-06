//! Batch adapter for standard LOESS smoothing.
//!
//! This module provides the batch execution adapter for LOESS smoothing.
//! It handles complete datasets in memory with optional parallel processing,
//! making it suitable for small to medium-sized datasets.
//!
//! ## srrstats Compliance
//!
//! @srrstats {G3.0} Rayon-based parallel execution for CPU-bound workloads.

// Imports
use crate::engine::executor::{smooth_pass_parallel, vertex_pass_parallel};
use crate::evaluation::cv::cv_pass_parallel;
use crate::evaluation::intervals::interval_pass_parallel;

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from loess-rs crate
use loess_rs::internals::adapters::batch::BatchLoessBuilder;
use loess_rs::internals::algorithms::regression::PolynomialDegree;
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::algorithms::regression::ZeroWeightFallback;
use loess_rs::internals::algorithms::robustness::RobustnessMethod;
use loess_rs::internals::api::SurfaceMode;
use loess_rs::internals::engine::output::LoessResult;
use loess_rs::internals::evaluation::cv::CVKind;
use loess_rs::internals::math::boundary::BoundaryPolicy;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::math::scaling::ScalingMethod;
use loess_rs::internals::primitives::backend::Backend;
use loess_rs::internals::primitives::errors::LoessError;

// Internal dependencies
use crate::input::LoessInput;
use crate::math::neighborhood::build_kdtree_parallel;
use crate::parse::IntoEnum;

// Builder for batch LOESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelBatchLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    // Base builder from the loess-rs crate
    pub base: BatchLoessBuilder<T>,
    // Parse errors from string-accepting builder methods; reported together by `build()`.
    pub(crate) parse_errors: Vec<LoessError>,
    // Pending weighted distance metric weights (applied at build time).
    pub(crate) weighted_metric_weights: Option<Vec<T>>,
    // CV method string ("kfold" or "loocv"), applied at build time.
    pub(crate) cv_method_str: Option<String>,
    // K value for K-fold CV (default 5).
    pub(crate) cv_k_val: usize,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> Default
    for ParallelBatchLoessBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelBatchLoessBuilder<T>
{
    // Create a new batch LOESS builder with default parameters.
    //
    // # Defaults
    //
    // * All base parameters from loess-rs BatchLoessBuilder
    // * parallel: true (fastLoess extension)
    fn new() -> Self {
        let mut base = BatchLoessBuilder::default();
        base.parallel = Some(true); // Default to parallel in fastLoess
        Self {
            base,
            parse_errors: Vec::new(),
            weighted_metric_weights: None,
            cv_method_str: None,
            cv_k_val: 5,
        }
    }

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

    // Set the number of dimensions explicitly.
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

    // Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.base = self.base.boundary_degree_fallback(enabled);
        self
    }

    // Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.base.interpolation_vertices = Some(vertices);
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }

    // Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.base = self.base.confidence_intervals(level);
        self
    }

    // Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.base = self.base.prediction_intervals(level);
        self
    }

    // Set the cross-validation method: `"kfold"` or `"loocv"`.
    pub fn cv_method(mut self, method: &str) -> Self {
        self.cv_method_str = Some(method.to_string());
        self
    }

    // Set the number of folds for K-fold cross-validation (default: 5).
    pub fn cv_k(mut self, k: usize) -> Self {
        self.cv_k_val = k;
        self
    }

    // Set the candidate fractions to evaluate during cross-validation.
    pub fn cv_fractions(mut self, fractions: Vec<T>) -> Self {
        self.base.cv_fractions = Some(fractions);
        self
    }

    // Set the random seed for reproducible cross-validation fold splitting.
    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.base.cv_seed = Some(seed);
        self
    }

    // Set per-dimension weights for the `"weighted"` distance metric.
    //
    // Calling this method selects the weighted Euclidean metric and supplies
    // the weight vector (one entry per predictor dimension). Must be combined
    // with `.distance_metric("weighted")` or called on its own.
    pub fn weighted_metric_weights(mut self, weights: Vec<T>) -> Self {
        self.weighted_metric_weights = Some(weights);
        self
    }

    // Enable returning standard errors in the result.
    pub fn return_se(mut self, enabled: bool) -> Self {
        self.base = self.base.return_se(enabled);
        self
    }

    // Set user-defined case weights (one per observation).
    //
    // Weights multiply the local kernel weight at each neighborhood point:
    // `w_ij = custom_weights[j] * K(d_ij / h) * robustness_j`.
    //
    // Analogous to `weights` in R's `stats::loess`. Must have the same length as `y`.
    pub fn custom_weights(mut self, weights: Vec<T>) -> Self {
        self.base = self.base.custom_weights(weights);
        self
    }

    // Build the batch processor.
    pub fn build(mut self) -> Result<ParallelBatchLoess<T>, LoessError> {
        // Check for parse errors from string builder methods
        if !self.parse_errors.is_empty() {
            return Err(LoessError::ParseErrors(self.parse_errors));
        }

        // Apply weighted_metric_weights: override distance_metric with Weighted(weights).
        // If distance_metric("weighted") was called without weighted_metric_weights(), error.
        if let Some(weights) = self.weighted_metric_weights.take() {
            self.base.distance_metric = DistanceMetric::Weighted(weights);
        } else if let DistanceMetric::Weighted(ref w) = self.base.distance_metric {
            if w.is_empty() {
                return Err(LoessError::InvalidOption {
                    option: "distance_metric",
                    value: "weighted".to_string(),
                    valid: "use .weighted_metric_weights(vec![...]) to supply per-dimension weights",
                });
            }
        }

        // Apply CV method string to base.cv_kind.
        if let Some(ref method_str) = self.cv_method_str {
            let lower = method_str.to_lowercase();
            let kind = match lower.as_str() {
                "kfold" | "k_fold" | "k-fold" => Ok(CVKind::KFold(self.cv_k_val)),
                "loocv" | "loo_cv" | "loo-cv" => Ok(CVKind::LOOCV),
                _ => Err(LoessError::InvalidOption {
                    option: "cv_method",
                    value: method_str.clone(),
                    valid: "kfold, loocv",
                }),
            };
            match kind {
                Ok(k) => self.base.cv_kind = Some(k),
                Err(e) => return Err(e),
            }
        }

        // Check for deferred errors from adapter conversion
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Validate by attempting to build the base processor
        // This reuses the validation logic centralized in the loess-rs crate
        let _ = self.base.clone().build()?;

        Ok(ParallelBatchLoess { config: self })
    }
}

// Batch LOESS processor with parallel support.
#[derive(Clone)]
pub struct ParallelBatchLoess<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    config: ParallelBatchLoessBuilder<T>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static>
    ParallelBatchLoess<T>
{
    // Perform LOESS smoothing on the provided data.
    pub fn fit<I1, I2>(self, x: &I1, y: &I2) -> Result<LoessResult<T>, LoessError>
    where
        I1: LoessInput<T> + ?Sized,
        I2: LoessInput<T> + ?Sized,
    {
        let x_slice = x.as_loess_slice()?;
        let y_slice = y.as_loess_slice()?;

        // Configure the base builder with parallel callback if enabled
        let mut builder = self.config.base;

        match builder.backend.unwrap_or(Backend::CPU) {
            Backend::CPU => {
                if builder.parallel.unwrap_or(true) {
                    builder.custom_smooth_pass = Some(smooth_pass_parallel);
                    builder.custom_cv_pass = Some(cv_pass_parallel);
                    builder.custom_interval_pass = Some(interval_pass_parallel);
                    builder.custom_vertex_pass = Some(vertex_pass_parallel);
                    builder.custom_kdtree_builder = Some(build_kdtree_parallel);
                }
            }
        }

        // Delegate execution to the base implementation
        let processor = builder.build()?;
        processor.fit(x_slice, y_slice)
    }
}
