//! Streaming adapter for large-scale LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the streaming execution adapter for LOESS smoothing
//! on datasets too large to fit in memory. It wraps the loess-rs StreamingLoess
//! with parallel execution support.
//!
//! ## Design notes
//!
//! * **Strategy**: Processes data in fixed-size chunks with configurable overlap.
//! * **Parallelism**: Adds parallel execution via `rayon` (fastLoess extension).
//! * **Generics**: Generic over `Float` types.
//!
//! ## Key concepts
//!
//! * **Chunked Processing**: Divides stream into `chunk_size` pieces.
//! * **Overlap**: Ensures smooth transitions, typically 2x window size.
//! * **Merging**: Handles value conflicts in overlapping regions.
//! * **Boundary Policies**: Handles edge effects at stream start/end.
//!
//! ## Invariants
//!
//! * Chunk size must be larger than overlap.
//! * Overlap must be sufficient for local smoothing window.
//! * values must be finite.
//! * At least 2 points per chunk.
//!
//! ## Non-goals
//!
//! * This adapter does not support confidence/prediction intervals.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle batch processing.
//! * This adapter does not handle incremental updates.
//! * This adapter requires chunks to be provided in stream order.

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
use loess_rs::internals::adapters::streaming::{MergeStrategy, StreamingLoessBuilder};
use loess_rs::internals::algorithms::regression::PolynomialDegree;
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::algorithms::regression::ZeroWeightFallback;
use loess_rs::internals::algorithms::robustness::RobustnessMethod;
use loess_rs::internals::engine::executor::SurfaceMode;
use loess_rs::internals::engine::output::LoessResult;
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
// Extended Streaming LOESS Builder
// ============================================================================

/// Builder for streaming LOESS processor with parallel support.
#[derive(Debug, Clone)]
pub struct ParallelStreamingLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    /// Base builder from the loess-rs crate
    pub base: StreamingLoessBuilder<T>,
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
    /// Create a new streaming LOESS builder with default parameters.
    fn new() -> Self {
        let mut base = StreamingLoessBuilder::default();
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

    /// Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    /// Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // ========================================================================
    // Streaming-Specific Setters
    // ========================================================================

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }

    /// Set chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base.chunk_size = size;
        self
    }

    /// Set overlap between chunks.
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.base.overlap = overlap;
        self
    }

    /// Set the merge strategy for overlapping chunks.
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.base.merge_strategy = strategy;
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the streaming processor.
    pub fn build(self) -> Result<ParallelStreamingLoess<T>, LoessError> {
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

// ============================================================================
// Extended Streaming LOESS Processor
// ============================================================================

/// Streaming LOESS processor with parallel support.
pub struct ParallelStreamingLoess<
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync,
> {
    config: ParallelStreamingLoessBuilder<T>,
    processor: Option<loess_rs::internals::adapters::streaming::StreamingLoess<T>>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Float + Debug + Send + Sync + 'static>
    ParallelStreamingLoess<T>
{
    /// Process a chunk of data.
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

            self.processor = Some(builder.build()?);
        }

        self.processor
            .as_mut()
            .unwrap()
            .process_chunk(x_slice, y_slice)
    }

    /// Finalize processing and get any remaining buffered data.
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

    /// Reset the processor state.
    pub fn reset(&mut self) {
        if let Some(ref mut proc) = self.processor {
            proc.reset();
        }
    }
}
