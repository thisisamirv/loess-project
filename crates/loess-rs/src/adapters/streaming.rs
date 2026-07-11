//! Streaming adapter for large-scale LOESS smoothing.
//!
//! This module provides the streaming execution adapter for LOESS smoothing
//! on datasets too large to fit in memory. It divides the data into overlapping
//! chunks, processes each chunk independently, and merges the results while
//! handling boundary effects.
//!
//! ## srrstats Compliance
//!
//! @srrstats {G1.6} Memory-efficient streaming for large datasets via chunking.
//! Configurable merge strategies (Average, WeightedAverage, TakeFirst, TakeLast).

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use core::mem;

// Internal dependencies
use crate::adapters::defaults::*;
use crate::algorithms::defaults::*;
use crate::algorithms::regression::{PolynomialDegree, SolverLinalg, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::defaults::*;
use crate::engine::executor::{
    CVPassFn, FitPassFn, IntervalPassFn, KDTreeBuilderFn, LoessConfig, LoessExecutor, SmoothPassFn,
    SurfaceMode, VertexPassFn,
};
use crate::engine::output::LoessResult;
use crate::engine::validator::Validator;
use crate::evaluation::diagnostics::DiagnosticsState;
use crate::math::boundary::BoundaryPolicy;
use crate::math::defaults::*;
use crate::math::distance::{DistanceLinalg, DistanceMetric};
use crate::math::kernel::WeightFunction;
use crate::math::linalg::FloatLinalg;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;
use crate::primitives::errors::LoessError;

// Strategy for merging overlapping regions between streaming chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    // Arithmetic mean of overlapping smoothed values: `(v1 + v2) / 2`.
    Average,

    // Distance-based weights that favor values from the center of each chunk:
    // v1 * (1 - alpha) + v2 * alpha where `alpha` is the relative position within the overlap.
    #[default]
    WeightedAverage,

    // Use the value from the first chunk in processing order.
    TakeFirst,

    // Use the value from the last chunk in processing order.
    TakeLast,
}

// Builder for streaming LOESS processor.
#[derive(Debug, Clone)]
pub struct StreamingLoessBuilder<T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    // Chunk size for processing
    pub chunk_size: usize,

    // Overlap between chunks
    pub overlap: usize,

    // Smoothing fraction (span)
    pub fraction: T,

    // Number of robustness iterations
    pub iterations: usize,

    // Convergence tolerance for early stopping (None = disabled)
    pub auto_converge: Option<T>,

    // Kernel weight function
    pub weight_function: WeightFunction,

    // Boundary handling policy
    pub boundary_policy: BoundaryPolicy,

    // Robustness method
    pub robustness_method: RobustnessMethod,

    // Residual scaling method
    pub scaling_method: ScalingMethod,

    // Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    // Merging strategy for overlapping chunks
    pub merge_strategy: MergeStrategy,

    // Whether to return residuals
    pub compute_residuals: bool,

    // Whether to return diagnostics
    pub return_diagnostics: bool,

    // Whether to return robustness weights
    pub return_robustness_weights: bool,

    // Deferred error from adapter conversion
    pub deferred_error: Option<LoessError>,

    // Polynomial degree for local regression
    pub polynomial_degree: PolynomialDegree,

    // Number of predictor dimensions (default: 1).
    pub dimensions: usize,

    // Distance metric for nD neighborhood computation.
    pub distance_metric: DistanceMetric<T>,

    // Cell size for interpolation subdivision (default: 0.2).
    pub cell: Option<f64>,

    // Maximum number of vertices for interpolation.
    pub interpolation_vertices: Option<usize>,

    // Evaluation mode (default: Interpolation)
    pub surface_mode: SurfaceMode,

    // Whether to reduce polynomial degree at boundary vertices during interpolation.
    // When `true` (default), Linear fits are used outside data bounds.
    pub boundary_degree_fallback: bool,

    // Tracks if any parameter was set multiple times (for validation)
    #[doc(hidden)]
    pub(crate) duplicate_param: Option<&'static str>,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    // Custom smooth pass function.
    #[doc(hidden)]
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    // Custom cross-validation pass function.
    #[doc(hidden)]
    pub custom_cv_pass: Option<CVPassFn<T>>,

    // Custom interval estimation pass function.
    #[doc(hidden)]
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    // Custom fit pass function.
    #[doc(hidden)]
    pub custom_fit_pass: Option<FitPassFn<T>>,

    // Custom vertex pass function.
    #[doc(hidden)]
    pub custom_vertex_pass: Option<VertexPassFn<T>>,

    // Custom KD-tree builder function.
    #[doc(hidden)]
    pub custom_kdtree_builder: Option<KDTreeBuilderFn<T>>,

    // Execution backend hint.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    // Parallel execution hint.
    #[doc(hidden)]
    pub parallel: Option<bool>,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + SolverLinalg> Default
    for StreamingLoessBuilder<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + SolverLinalg>
    StreamingLoessBuilder<T>
{
    // Create a new streaming LOESS builder with default parameters.
    fn new() -> Self {
        Self {
            chunk_size: DEFAULT_STREAMING_CHUNK_SIZE,
            overlap: DEFAULT_STREAMING_OVERLAP,
            fraction: T::from(DEFAULT_FRACTION).unwrap(),
            iterations: DEFAULT_STREAMING_ITERATIONS,
            weight_function: DEFAULT_WEIGHT_FUNCTION_ENUM,
            boundary_policy: DEFAULT_BOUNDARY_POLICY_ENUM,
            robustness_method: DEFAULT_ROBUSTNESS_METHOD_ENUM,
            scaling_method: DEFAULT_SCALING_METHOD_ENUM,
            zero_weight_fallback: DEFAULT_ZERO_WEIGHT_FALLBACK_ENUM,
            merge_strategy: DEFAULT_STREAMING_MERGE_STRATEGY_ENUM,
            compute_residuals: DEFAULT_RETURN_RESIDUALS,
            return_diagnostics: DEFAULT_RETURN_DIAGNOSTICS,
            return_robustness_weights: DEFAULT_RETURN_ROBUSTNESS_WEIGHTS,
            auto_converge: default_auto_converge(),
            deferred_error: None,
            polynomial_degree: DEFAULT_POLYNOMIAL_DEGREE_ENUM,
            dimensions: DEFAULT_DIMENSIONS,
            distance_metric: default_distance_metric(),
            cell: None,
            interpolation_vertices: None,
            surface_mode: DEFAULT_SURFACE_MODE_ENUM,
            boundary_degree_fallback: DEFAULT_BOUNDARY_DEGREE_FALLBACK,
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

    // Build the streaming processor.
    pub fn build(self) -> Result<StreamingLoess<T>, LoessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Check for duplicate parameter configuration
        Validator::validate_no_duplicates(self.duplicate_param)?;

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate iterations
        Validator::validate_iterations(self.iterations)?;

        // Validate chunk size
        Validator::validate_chunk_size(self.chunk_size, 10)?;

        // Validate overlap
        Validator::validate_overlap(self.overlap, self.chunk_size)?;

        let has_diag = self.return_diagnostics;
        Ok(StreamingLoess {
            config: self,
            overlap_buffer_x: Vec::new(),
            overlap_buffer_y: Vec::new(),
            overlap_buffer_smoothed: Vec::new(),
            overlap_buffer_robustness_weights: Vec::new(),
            diagnostics_state: if has_diag {
                Some(DiagnosticsState::new())
            } else {
                None
            },
        })
    }
}

// Streaming LOESS processor for large datasets.
pub struct StreamingLoess<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> {
    config: StreamingLoessBuilder<T>,
    overlap_buffer_x: Vec<T>,
    overlap_buffer_y: Vec<T>,
    overlap_buffer_smoothed: Vec<T>,
    overlap_buffer_robustness_weights: Vec<T>,
    diagnostics_state: Option<DiagnosticsState<T>>,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + 'static + SolverLinalg>
    StreamingLoess<T>
{
    // Process a chunk of data.
    pub fn process_chunk(&mut self, x: &[T], y: &[T]) -> Result<LoessResult<T>, LoessError> {
        // Validate inputs using standard validator
        Validator::validate_inputs(x, y, self.config.dimensions)?;

        // Combine with overlap from previous chunk
        let prev_overlap_len = self.overlap_buffer_smoothed.len();
        let (combined_x, combined_y) = if self.overlap_buffer_x.is_empty() {
            // No overlap: copy data directly
            (x.to_vec(), y.to_vec())
        } else {
            let mut cx = mem::take(&mut self.overlap_buffer_x);
            cx.extend_from_slice(x);
            let mut cy = mem::take(&mut self.overlap_buffer_y);
            cy.extend_from_slice(y);
            (cx, cy)
        };

        // Check grid resolution (max_vertices defaults to N = chunk_size)
        // Note: For streaming, validation should be against chunk_size since we fit on chunks.
        let n = combined_y.len() / self.config.dimensions;
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

        // Execute LOESS on combined data (KD-Tree handles unsorted data)
        let config = LoessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            weight_function: self.config.weight_function,
            zero_weight_fallback: self.config.zero_weight_fallback,
            robustness_method: self.config.robustness_method,
            scaling_method: self.config.scaling_method,
            boundary_policy: self.config.boundary_policy,
            polynomial_degree: self.config.polynomial_degree,
            dimensions: self.config.dimensions,
            distance_metric: self.config.distance_metric.clone(),
            cv_fractions: None,
            cv_kind: None,
            auto_converge: self.config.auto_converge,
            return_variance: None,
            cv_seed: None,
            surface_mode: self.config.surface_mode,
            interpolation_vertices: self.config.interpolation_vertices,
            cell: self.config.cell,
            boundary_degree_fallback: self.config.boundary_degree_fallback,
            custom_weights: None,
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
        // Execute LOESS on combined data
        let result = LoessExecutor::run_with_config(&combined_x, &combined_y, config);
        let smoothed = result.smoothed;

        // Determine how much to return vs buffer
        let combined_points = combined_y.len();
        let overlap_start = combined_points.saturating_sub(self.config.overlap);
        let return_start = prev_overlap_len;
        let dimensions = self.config.dimensions;

        // Build output: merged overlap (if any) + new data
        let mut y_smooth_out = Vec::new();
        if prev_overlap_len > 0 {
            // Merge the overlap region
            let prev_smooth = mem::take(&mut self.overlap_buffer_smoothed);
            for (i, (&prev_val, &curr_val)) in prev_smooth
                .iter()
                .zip(smoothed.iter())
                .take(prev_overlap_len)
                .enumerate()
            {
                let merged = match self.config.merge_strategy {
                    MergeStrategy::Average => (prev_val + curr_val) / T::from(2.0).unwrap(),
                    MergeStrategy::WeightedAverage => {
                        let weight = T::from(i as f64 / prev_overlap_len as f64).unwrap();
                        prev_val * (T::one() - weight) + curr_val * weight
                    }
                    MergeStrategy::TakeFirst => prev_val,
                    MergeStrategy::TakeLast => curr_val,
                };
                y_smooth_out.push(merged);
            }
        }

        // Merge robustness weights if requested
        let mut rob_weights_out = if self.config.return_robustness_weights {
            Some(Vec::new())
        } else {
            None
        };

        if let Some(ref mut rw_out) = rob_weights_out
            && prev_overlap_len > 0
        {
            let prev_rw = mem::take(&mut self.overlap_buffer_robustness_weights);
            for (i, (&prev_val, &curr_val)) in prev_rw
                .iter()
                .zip(result.robustness_weights.iter())
                .take(prev_overlap_len)
                .enumerate()
            {
                let merged = match self.config.merge_strategy {
                    MergeStrategy::Average => (prev_val + curr_val) / T::from(2.0).unwrap(),
                    MergeStrategy::WeightedAverage => {
                        let weight = T::from(i as f64 / prev_overlap_len as f64).unwrap();
                        prev_val * (T::one() - weight) + curr_val * weight
                    }
                    MergeStrategy::TakeFirst => prev_val,
                    MergeStrategy::TakeLast => curr_val,
                };
                rw_out.push(merged);
            }
        }

        // Add non-overlap portion
        if return_start < overlap_start {
            y_smooth_out.extend_from_slice(&smoothed[return_start..overlap_start]);
            if let Some(ref mut rw_out) = rob_weights_out {
                rw_out.extend_from_slice(&result.robustness_weights[return_start..overlap_start]);
            }
        }

        // Calculate residuals for output
        let residuals_out = if self.config.compute_residuals {
            let y_slice = &combined_y[return_start..return_start + y_smooth_out.len()];
            Some(
                y_slice
                    .iter()
                    .zip(y_smooth_out.iter())
                    .map(|(y, s)| *y - *s)
                    .collect(),
            )
        } else {
            None
        };

        // Buffer overlap for next chunk
        if overlap_start < combined_points {
            let overlap_start_x = overlap_start * dimensions;
            self.overlap_buffer_x = combined_x[overlap_start_x..].to_vec();
            self.overlap_buffer_y = combined_y[overlap_start..].to_vec();
            self.overlap_buffer_smoothed = smoothed[overlap_start..].to_vec();
            if self.config.return_robustness_weights {
                self.overlap_buffer_robustness_weights =
                    result.robustness_weights[overlap_start..].to_vec();
            }
        } else {
            self.overlap_buffer_x.clear();
            self.overlap_buffer_y.clear();
            self.overlap_buffer_smoothed.clear();
            self.overlap_buffer_robustness_weights.clear();
        }

        // Note: We return results in the order they were processed (combined chunk/overlap).
        // The KD-tree implementation does not require data to be globally sorted.
        let return_start_x = return_start * dimensions;
        let x_out_len = y_smooth_out.len() * dimensions;
        let x_out = combined_x[return_start_x..return_start_x + x_out_len].to_vec();

        // Update diagnostics cumulatively
        let diagnostics = if let Some(ref mut state) = self.diagnostics_state {
            let y_emitted = &combined_y[return_start..return_start + y_smooth_out.len()];
            state.update(y_emitted, &y_smooth_out);
            Some(state.finalize())
        } else {
            None
        };

        Ok(LoessResult {
            x: x_out,
            dimensions: self.config.dimensions,
            distance_metric: self.config.distance_metric.clone(),
            polynomial_degree: self.config.polynomial_degree,
            y: y_smooth_out,
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals: residuals_out,
            robustness_weights: rob_weights_out,
            diagnostics,
            iterations_used: result.iterations,
            fraction_used: self.config.fraction,
            cv_scores: None,
            enp: None,
            trace_hat: None,
            delta1: None,
            delta2: None,
            residual_scale: None,
            leverage: None,
        })
    }

    // Finalize processing and get any remaining buffered data.
    pub fn finalize(&mut self) -> Result<LoessResult<T>, LoessError> {
        if self.overlap_buffer_x.is_empty() {
            return Ok(LoessResult {
                x: Vec::new(),
                dimensions: self.config.dimensions,
                distance_metric: self.config.distance_metric.clone(),
                polynomial_degree: self.config.polynomial_degree,
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
                fraction_used: self.config.fraction,
                cv_scores: None,
                enp: None,
                trace_hat: None,
                delta1: None,
                delta2: None,
                residual_scale: None,
                leverage: None,
            });
        }

        // Return buffered overlap data
        let residuals = if self.config.compute_residuals {
            let mut res = Vec::with_capacity(self.overlap_buffer_x.len());
            for (i, &smoothed) in self.overlap_buffer_smoothed.iter().enumerate() {
                res.push(self.overlap_buffer_y[i] - smoothed);
            }
            Some(res)
        } else {
            None
        };

        let robustness_weights = if self.config.return_robustness_weights {
            Some(mem::take(&mut self.overlap_buffer_robustness_weights))
        } else {
            None
        };

        // Update diagnostics for the final overlap
        let diagnostics = if let Some(ref mut state) = self.diagnostics_state {
            state.update(&self.overlap_buffer_y, &self.overlap_buffer_smoothed);
            Some(state.finalize())
        } else {
            None
        };

        let result = LoessResult {
            x: self.overlap_buffer_x.clone(),
            dimensions: self.config.dimensions,
            distance_metric: self.config.distance_metric.clone(),
            polynomial_degree: self.config.polynomial_degree,
            y: self.overlap_buffer_smoothed.clone(),
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals,
            robustness_weights,
            diagnostics,
            iterations_used: None,
            fraction_used: self.config.fraction,
            cv_scores: None,
            enp: None,
            trace_hat: None,
            delta1: None,
            delta2: None,
            residual_scale: None,
            leverage: None,
        };

        // Clear buffers
        self.overlap_buffer_x.clear();
        self.overlap_buffer_y.clear();
        self.overlap_buffer_smoothed.clear();
        self.overlap_buffer_robustness_weights.clear();

        Ok(result)
    }

    // Reset the processor state.
    pub fn reset(&mut self) {
        self.overlap_buffer_x.clear();
        self.overlap_buffer_y.clear();
        self.overlap_buffer_smoothed.clear();
        self.overlap_buffer_robustness_weights.clear();
    }
}
