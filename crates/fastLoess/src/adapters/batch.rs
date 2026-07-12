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
use loess_rs::internals::algorithms::regression::SolverLinalg;
use loess_rs::internals::engine::output::LoessResult;
use loess_rs::internals::evaluation::cv::CVKind;
use loess_rs::internals::evaluation::defaults::DEFAULT_CV_K_FOLDS;
use loess_rs::internals::math::distance::DistanceLinalg;
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::linalg::FloatLinalg;
use loess_rs::internals::primitives::backend::Backend;
use loess_rs::internals::primitives::errors::LoessError;

// Internal dependencies
use crate::input::LoessInput;
use crate::math::neighborhood::build_kdtree_parallel;

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
            cv_k_val: DEFAULT_CV_K_FOLDS,
        }
    }
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelBatchLoessBuilder<T>
{
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
        } else if let DistanceMetric::Weighted(ref w) = self.base.distance_metric
            && w.is_empty()
        {
            return Err(LoessError::InvalidOption {
                option: "distance_metric",
                value: "weighted".to_string(),
                valid: "use .weighted_metric_weights(vec![...]) to supply per-dimension weights",
            });
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
            {
                let k = kind?;
                self.base.cv_kind = Some(k)
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
