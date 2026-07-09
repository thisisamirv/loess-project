//! Streaming adapter for large-scale LOESS smoothing.
//!
//! This module provides the streaming execution adapter for LOESS smoothing
//! on datasets too large to fit in memory. It divides the data into overlapping
//! chunks, processes each chunk independently, and merges the results while
//! handling boundary effects.
//!
//! ## srrstats Compliance
//!
//! @srrstats {G1.6} Chunk-based streaming with parallel execution per chunk.
//! @srrstats {G3.0} Rayon parallelization injected for chunk processing.

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from loess-rs crate
#[cfg(feature = "dev")]
use loess_rs::internals::adapters::streaming::MergeStrategy;
use loess_rs::internals::adapters::streaming::StreamingLoessBuilder;
#[cfg(feature = "dev")]
use loess_rs::internals::algorithms::regression::PolynomialDegree;
use loess_rs::internals::algorithms::regression::SolverLinalg;
#[cfg(feature = "dev")]
use loess_rs::internals::algorithms::regression::ZeroWeightFallback;
#[cfg(feature = "dev")]
use loess_rs::internals::algorithms::robustness::RobustnessMethod;
#[cfg(feature = "dev")]
use loess_rs::internals::engine::executor::SurfaceMode;
use loess_rs::internals::engine::output::LoessResult;
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

// Internal dependencies
use crate::engine::executor::{smooth_pass_parallel, vertex_pass_parallel};
use crate::evaluation::cv::cv_pass_parallel;
use crate::evaluation::intervals::interval_pass_parallel;
use crate::input::LoessInput;
use crate::math::neighborhood::build_kdtree_parallel;
#[cfg(feature = "dev")]
use crate::parse::IntoEnum;

// Builder for streaming LOESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelStreamingLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    // Base builder from the loess-rs crate
    pub base: StreamingLoessBuilder<T>,
    // Parse errors from string-accepting builder methods; reported together by `build()`.
    pub(crate) parse_errors: Vec<LoessError>,
    // Pending weighted distance metric weights (applied at build time).
    pub(crate) weighted_metric_weights: Option<Vec<T>>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> Default
    for ParallelStreamingLoessBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelStreamingLoessBuilder<T>
{
    // Create a new streaming LOESS builder with default parameters.
    fn new() -> Self {
        let mut base = StreamingLoessBuilder::default();
        base.parallel = Some(true); // Default to parallel in fastLoess
        Self {
            base,
            parse_errors: Vec::new(),
            weighted_metric_weights: None,
        }
    }
}

#[cfg(feature = "dev")]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelStreamingLoessBuilder<T>
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

    // Set chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base.chunk_size = size;
        self
    }

    // Set overlap between chunks.
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.base.overlap = overlap;
        self
    }

    // Set the merge strategy for overlapping chunks.
    #[allow(private_bounds)]
    pub fn merge_strategy(mut self, strategy: impl IntoEnum<MergeStrategy>) -> Self {
        match strategy.into_enum() {
            Ok(s) => self.base.merge_strategy = s,
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
    ParallelStreamingLoessBuilder<T>
{
    // Build the streaming processor.
    pub fn build(mut self) -> Result<ParallelStreamingLoess<T>, LoessError> {
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

        Ok(ParallelStreamingLoess {
            config: self,
            processor: None,
        })
    }
}

// Streaming LOESS processor with parallel support.
pub struct ParallelStreamingLoess<
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync,
> {
    config: ParallelStreamingLoessBuilder<T>,
    processor: Option<loess_rs::internals::adapters::streaming::StreamingLoess<T>>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static>
    ParallelStreamingLoess<T>
{
    // Process a chunk of data.
    pub fn process_chunk<I1, I2>(&mut self, x: &I1, y: &I2) -> Result<LoessResult<T>, LoessError>
    where
        I1: LoessInput<T> + ?Sized,
        I2: LoessInput<T> + ?Sized,
    {
        let x_slice = x.as_loess_slice()?;
        let y_slice = y.as_loess_slice()?;

        // Lazily initialize the processor with parallel callbacks
        if self.processor.is_none() {
            let mut builder = self.config.base.clone();

            if builder.parallel.unwrap_or(true) {
                builder.custom_smooth_pass = Some(smooth_pass_parallel);
                builder.custom_cv_pass = Some(cv_pass_parallel);
                builder.custom_interval_pass = Some(interval_pass_parallel);
                builder.custom_vertex_pass = Some(vertex_pass_parallel);
                builder.custom_kdtree_builder = Some(build_kdtree_parallel);
            }

            self.processor = Some(builder.build()?);
        }

        self.processor
            .as_mut()
            .unwrap()
            .process_chunk(x_slice, y_slice)
    }

    // Finalize processing and get any remaining buffered data.
    pub fn finalize(&mut self) -> Result<LoessResult<T>, LoessError> {
        if let Some(ref mut proc) = self.processor {
            proc.finalize()
        } else {
            // No data processed yet
            Ok(LoessResult {
                x: Vec::new(),
                dimensions: self.config.base.dimensions,
                distance_metric: self.config.base.distance_metric.clone(),
                polynomial_degree: self.config.base.polynomial_degree,
                y: Vec::new(),
                standard_errors: None,
                confidence_lower: None,
                confidence_upper: None,
                prediction_lower: None,
                prediction_upper: None,
                residuals: None,
                robustness_weights: None,
                diagnostics: None,
                iterations_used: None,
                fraction_used: self.config.base.fraction,
                cv_scores: None,
                enp: None,
                trace_hat: None,
                delta1: None,
                delta2: None,
                residual_scale: None,
                leverage: None,
            })
        }
    }

    // Reset the processor state.
    pub fn reset(&mut self) {
        if let Some(ref mut proc) = self.processor {
            proc.reset();
        }
    }
}
