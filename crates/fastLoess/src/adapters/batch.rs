//! Batch adapter for standard LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the batch execution adapter for LOESS smoothing.
//! It handles complete datasets in memory with optional parallel processing,
//! making it suitable for small to medium-sized datasets.
//!
//! ## Design notes
//!
//! * **Processing**: Processes entire dataset in a single pass.
//! * **Delegation**: Delegates computation to the execution engine.
//! * **Parallelism**: Adds parallel execution via `rayon` (fastLoess extension).
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Batch Processing**: Validates, executes, and returns results.
//! * **Builder Pattern**: Fluent API for configuration with sensible defaults.
//! * **Parallel Execution**: Uses Rayon for multi-threaded processing.
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

// Export dependencies from loess-rs crate
use loess_rs::internals::adapters::batch::BatchLoessBuilder;
use loess_rs::internals::algorithms::regression::PolynomialDegree;
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::algorithms::regression::ZeroWeightFallback;
use loess_rs::internals::algorithms::robustness::RobustnessMethod;
use loess_rs::internals::api::SurfaceMode;
use loess_rs::internals::engine::output::LoessResult;
use loess_rs::internals::evaluation::cv::{CVConfig, CVKind};
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

// ============================================================================
// Extended Batch LOESS Builder
// ============================================================================

/// Builder for batch LOESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelBatchLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    /// Base builder from the loess-rs crate
    pub base: BatchLoessBuilder<T>,
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
    /// Create a new batch LOESS builder with default parameters.
    ///
    /// # Defaults
    ///
    /// * All base parameters from loess-rs BatchLoessBuilder
    /// * parallel: true (fastLoess extension)
    fn new() -> Self {
        let mut base = BatchLoessBuilder::default();
        base.parallel = Some(true); // Default to parallel in fastLoess
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

    /// Set the number of dimensions explicitly.
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

    /// Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.base = self.base.boundary_degree_fallback(enabled);
        self
    }

    /// Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.base.interpolation_vertices = Some(vertices);
        self
    }

    /// Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    /// Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // ========================================================================
    // Batch-Specific Setters
    // ========================================================================

    /// Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }

    /// Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.base = self.base.confidence_intervals(level);
        self
    }

    /// Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.base = self.base.prediction_intervals(level);
        self
    }

    /// Enable cross-validation using the specified configuration.
    pub fn cross_validate(mut self, config: CVConfig<'_, T>) -> Self {
        self.base.cv_fractions = Some(config.fractions().to_vec());
        self.base.cv_kind = Some(config.kind());
        self.base.cv_seed = config.get_seed();
        self
    }

    /// Set the random seed for reproducible cross-validation.
    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.base.cv_seed = Some(seed);
        self
    }

    /// Set the cross-validation method.
    pub fn cv_kind(mut self, method: CVKind) -> Self {
        self.base.cv_kind = Some(method);
        self
    }

    /// Enable returning standard errors in the result.
    pub fn return_se(mut self, enabled: bool) -> Self {
        self.base = self.base.return_se(enabled);
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the batch processor.
    pub fn build(self) -> Result<ParallelBatchLoess<T>, LoessError> {
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

// ============================================================================
// Extended Batch LOESS Processor
// ============================================================================

/// Batch LOESS processor with parallel support.
pub struct ParallelBatchLoess<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    config: ParallelBatchLoessBuilder<T>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static>
    ParallelBatchLoess<T>
{
    /// Perform LOESS smoothing on the provided data.
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
                #[cfg(not(feature = "cpu"))]
                {
                    // Fallback to sequential if cpu feature is disabled
                    builder.custom_smooth_pass = None;
                    builder.custom_cv_pass = None;
                    builder.custom_interval_pass = None;
                    builder.custom_vertex_pass = None;
                }
            }
            Backend::GPU => {
                // GPU backend not yet supported for LOESS
                return Err(LoessError::UnsupportedFeature {
                    adapter: "Batch",
                    feature: "GPU backend (not yet implemented)",
                });
            }
        }

        // Delegate execution to the base implementation
        let processor = builder.build()?;
        processor.fit(x_slice, y_slice)
    }
}
