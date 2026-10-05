//! Online adapter for incremental LOESS smoothing.
//!
//! This module provides the online (incremental) execution adapter for LOESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive.
// ## srrstats Compliance
//
// @srrstats {G1.6} Sliding window for real-time incremental updates (always sequential).
// @srrstats {G2.1} Configurable min_points threshold before smoothing starts.

// External dependencies
use num_traits::Float;
use std::fmt::Debug;
use std::result::Result;

// Export dependencies from loess-rs crate
use crate::adapters::apply_weighted_metric_weights;
use loess_rs::PredictOutput;
use loess_rs::internals::adapters::online::{OnlineLoessBuilder, OnlineOutput};
use loess_rs::internals::algorithms::regression::specialized::SolverLinalg;
use loess_rs::internals::engine::executor::PredictQuery;
use loess_rs::internals::evaluation::diagnostics::Diagnostics;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::linalg::FloatLinalg;
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
        let base = OnlineLoessBuilder::default();
        Self {
            base,
            parse_errors: Vec::new(),
            weighted_metric_weights: None,
        }
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

        apply_weighted_metric_weights(
            &mut self.base.distance_metric,
            self.weighted_metric_weights.take(),
        )?;

        // Check for deferred errors from adapter conversion
        if let Some(ref err) = self.base.deferred_error {
            return Err(err.clone());
        }

        // Configure parallel callbacks before building
        let builder = self.base;

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

    /// Add a point with an observation weight.
    pub fn add_point_weighted(
        &mut self,
        x: &[T],
        y: T,
        weight: T,
    ) -> Result<Option<OnlineOutput<T>>, LoessError> {
        self.processor.add_point_weighted(x, y, weight)
    }

    /// Compute diagnostics for the current sliding window on demand.
    pub fn window_diagnostics(&self) -> Result<Option<Diagnostics<T>>, LoessError> {
        self.processor.window_diagnostics()
    }

    /// Predict query points using the current sliding window.
    pub fn predict_window(
        &self,
        new_x: &[T],
        options: &PredictQuery<T>,
    ) -> Result<PredictOutput<T>, LoessError> {
        self.processor.predict_window(new_x, options)
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
