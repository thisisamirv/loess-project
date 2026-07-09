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
use loess_rs::internals::adapters::streaming::StreamingLoessBuilder;
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::engine::output::LoessResult;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::primitives::errors::LoessError;

// Internal dependencies
use crate::engine::executor::{smooth_pass_parallel, vertex_pass_parallel};
use crate::evaluation::cv::cv_pass_parallel;
use crate::evaluation::intervals::interval_pass_parallel;
use crate::input::LoessInput;
use crate::math::neighborhood::build_kdtree_parallel;

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
}impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
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
