//! High-level API for LOESS smoothing.
//!
//! This module provides the primary user-facing entry point for LOESS. It
//! implements a fluent builder pattern for configuring regression parameters
//! and choosing an execution adapter (Batch, Streaming, or Online).

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::string::{String, ToString};
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use core::marker::PhantomData;

// Internal dependencies
use crate::adapters::batch::BatchLoessBuilder;
use crate::adapters::online::OnlineLoessBuilder;
use crate::adapters::online::UpdateMode;
use crate::adapters::streaming::MergeStrategy;
use crate::adapters::streaming::StreamingLoessBuilder;
use crate::algorithms::regression::{PolynomialDegree, SolverLinalg, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::SurfaceMode;
use crate::engine::executor::{CVPassFn, IntervalPassFn, SmoothPassFn};
use crate::evaluation::cv::CVKind;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::distance::DistanceLinalg;
use crate::math::distance::DistanceMetric;
use crate::math::kernel::WeightFunction;
use crate::math::linalg::FloatLinalg;
use crate::math::scaling::ScalingMethod;
use crate::primitives::backend::Backend;

// Publicly re-exported non-enum API types
pub use crate::engine::output::LoessResult;
pub use crate::primitives::errors::LoessError;

// Mode markers — zero-sized types that select which processor build() produces.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchMode;
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamingMode;
#[derive(Debug, Clone, Copy, Default)]
pub struct OnlineMode;

// Convenience type aliases: entry points that mirror the bindings API.
pub type Loess<T = f64> = LoessBuilder<T, BatchMode>;
pub type StreamingLoess<T = f64> = LoessBuilder<T, StreamingMode>;
pub type OnlineLoess<T = f64> = LoessBuilder<T, OnlineMode>;

// Fluent builder for configuring LOESS parameters and execution modes.
#[derive(Debug, Clone)]
pub struct LoessBuilder<
    T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync,
    Mode = BatchMode,
> {
    // Smoothing fraction (0..1].
    pub fraction: Option<T>,

    // Robustness iterations.
    pub iterations: Option<usize>,

    // Kernel weight function.
    pub weight_function: Option<WeightFunction>,

    // Outlier downweighting method.
    pub robustness_method: Option<RobustnessMethod>,

    // Residual scaling method (MAR or MAD).
    pub scaling_method: Option<ScalingMethod>,

    // interval estimation configuration.
    pub interval_type: Option<IntervalMethod<T>>,

    // Candidate bandwidths for cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    // CV strategy (K-Fold/LOOCV).
    pub(crate) cv_kind: Option<CVKind>,

    // CV seed for reproducibility.
    pub(crate) cv_seed: Option<u64>,

    // Relative convergence tolerance.
    pub auto_converge: Option<T>,

    // Enable performance/statistical diagnostics.
    pub return_diagnostics: Option<bool>,

    // Return original residuals r_i.
    pub compute_residuals: Option<bool>,

    // Return final robustness weights w_i.
    pub return_robustness_weights: Option<bool>,

    // Policy for handling data boundaries (default: Extend).
    pub boundary_policy: Option<BoundaryPolicy>,

    // Behavior when local neighborhood weights are zero (default: UseLocalMean).
    pub zero_weight_fallback: Option<ZeroWeightFallback>,

    // Merging strategy for overlapping chunks (Streaming only).
    pub merge_strategy: Option<MergeStrategy>,

    // Incremental update mode (Online only).
    pub update_mode: Option<UpdateMode>,

    // Chunk size for streaming (Streaming only).
    pub chunk_size: Option<usize>,

    // Overlap size for streaming chunks (Streaming only).
    pub overlap: Option<usize>,

    // Window capacity for sliding window (Online only).
    pub window_capacity: Option<usize>,

    // Minimum points required for a valid fit (Online only).
    pub min_points: Option<usize>,

    // Polynomial degree for local regression (0=constant, 1=linear, 2=quadratic).
    pub polynomial_degree: Option<PolynomialDegree>,

    // Number of predictor dimensions (default: 1).
    pub dimensions: Option<usize>,

    // Distance metric for nD neighborhood computation (default: Euclidean).
    pub distance_metric: Option<DistanceMetric<T>>,

    // Surface evaluation mode (default: Interpolation).
    pub surface_mode: Option<SurfaceMode>,

    // Cell size for interpolation subdivision (default: 0.2).
    pub cell: Option<T>,

    // Maximum number of vertices for interpolation.
    pub interpolation_vertices: Option<usize>,

    // Whether to reduce polynomial degree at boundary vertices during interpolation.
    // When `true` (default), vertices outside the tight data bounds use Linear fits.
    // Set to `false` to match R's loess behavior exactly.
    pub boundary_degree_fallback: Option<bool>,

    // User-defined case weights (one per observation).
    // When provided, multiplies each local kernel weight: `w_ij = custom_weights[j] * K(d_ij/h)`.
    // Must have the same length as `y`. Only used in Batch mode.
    pub custom_weights: Option<Vec<T>>,

    // CV method string for string-based cross-validation API ("kfold" or "loocv").
    pub cv_method_str: Option<String>,

    // K value for K-fold CV (default: 5).
    pub cv_k_val: usize,

    // Per-dimension scale weights for the "weighted" distance metric.
    pub weighted_metric_weights: Option<Vec<T>>,

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

    // Execution backend hint.
    #[doc(hidden)]
    pub backend: Option<Backend>,

    // Parallel execution hint.
    #[doc(hidden)]
    pub parallel: Option<bool>,

    // Tracks if any parameter was set multiple times (for validation).
    #[doc(hidden)]
    pub duplicate_param: Option<&'static str>,

    // Parse errors from string-accepting builder methods; reported together by `build()`.
    #[doc(hidden)]
    pub parse_errors: Vec<LoessError>,

    // Zero-sized mode marker — selects which processor build() produces.
    #[doc(hidden)]
    pub _mode: PhantomData<Mode>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync, Mode: Default> Default
    for LoessBuilder<T, Mode>
{
    fn default() -> Self {
        Self::new()
    }
}

#[allow(private_bounds)]
impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + 'static + SolverLinalg, Mode: Default>
    LoessBuilder<T, Mode>
{
    // Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            fraction: None,
            iterations: None,
            weight_function: None,
            robustness_method: None,
            scaling_method: None,
            interval_type: None,
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            auto_converge: None,
            return_diagnostics: None,
            compute_residuals: None,
            return_robustness_weights: None,
            boundary_policy: None,
            zero_weight_fallback: None,
            merge_strategy: None,
            update_mode: None,
            chunk_size: None,
            overlap: None,
            window_capacity: None,
            min_points: None,
            polynomial_degree: None,
            dimensions: None,
            distance_metric: None,
            surface_mode: None,
            cell: None,
            interpolation_vertices: None,
            boundary_degree_fallback: None,
            custom_weights: None,
            cv_method_str: None,
            cv_k_val: 5,
            weighted_metric_weights: None,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            backend: None,
            parallel: None,
            duplicate_param: None,
            parse_errors: Vec::new(),
            _mode: PhantomData,
        }
    }

    // Set behavior for handling zero-weight neighborhoods.
    pub fn zero_weight_fallback(mut self, policy: impl AsRef<str>) -> Self {
        if self.zero_weight_fallback.is_some() {
            self.duplicate_param = Some("zero_weight_fallback");
        }
        match policy.as_ref().parse() {
            Ok(p) => self.zero_weight_fallback = Some(p),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: impl AsRef<str>) -> Self {
        if self.boundary_policy.is_some() {
            self.duplicate_param = Some("boundary_policy");
        }
        match policy.as_ref().parse() {
            Ok(p) => self.boundary_policy = Some(p),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the merging strategy for overlapping chunks (Streaming only).
    pub fn merge_strategy(mut self, strategy: impl AsRef<str>) -> Self {
        if self.merge_strategy.is_some() {
            self.duplicate_param = Some("merge_strategy");
        }
        match strategy.as_ref().parse() {
            Ok(s) => self.merge_strategy = Some(s),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the incremental update mode (Online only).
    pub fn update_mode(mut self, mode: impl AsRef<str>) -> Self {
        if self.update_mode.is_some() {
            self.duplicate_param = Some("update_mode");
        }
        match mode.as_ref().parse() {
            Ok(m) => self.update_mode = Some(m),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the chunk size for streaming (Streaming only).
    pub fn chunk_size(mut self, size: usize) -> Self {
        if self.chunk_size.is_some() {
            self.duplicate_param = Some("chunk_size");
        }
        self.chunk_size = Some(size);
        self
    }

    // Set the overlap size for streaming chunks (Streaming only).
    pub fn overlap(mut self, overlap: usize) -> Self {
        if self.overlap.is_some() {
            self.duplicate_param = Some("overlap");
        }
        self.overlap = Some(overlap);
        self
    }

    // Set the window capacity for online processing (Online only).
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        if self.window_capacity.is_some() {
            self.duplicate_param = Some("window_capacity");
        }
        self.window_capacity = Some(capacity);
        self
    }

    // Set the minimum points required for a valid fit (Online only).
    pub fn min_points(mut self, points: usize) -> Self {
        if self.min_points.is_some() {
            self.duplicate_param = Some("min_points");
        }
        self.min_points = Some(points);
        self
    }

    // Set the smoothing fraction (bandwidth alpha).
    pub fn fraction(mut self, fraction: T) -> Self {
        if self.fraction.is_some() {
            self.duplicate_param = Some("fraction");
        }
        self.fraction = Some(fraction);
        self
    }

    // Set the number of robustness iterations (typically 0-4).
    pub fn iterations(mut self, iterations: usize) -> Self {
        if self.iterations.is_some() {
            self.duplicate_param = Some("iterations");
        }
        self.iterations = Some(iterations);
        self
    }

    // Set the kernel weight function.
    pub fn weight_function(mut self, wf: impl AsRef<str>) -> Self {
        if self.weight_function.is_some() {
            self.duplicate_param = Some("weight_function");
        }
        match wf.as_ref().parse() {
            Ok(w) => self.weight_function = Some(w),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness weighting method.
    pub fn robustness_method(mut self, rm: impl AsRef<str>) -> Self {
        if self.robustness_method.is_some() {
            self.duplicate_param = Some("robustness_method");
        }
        match rm.as_ref().parse() {
            Ok(r) => self.robustness_method = Some(r),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the residual scaling method (MAR/MAD).
    pub fn scaling_method(mut self, sm: impl AsRef<str>) -> Self {
        if self.scaling_method.is_some() {
            self.duplicate_param = Some("scaling_method");
        }
        match sm.as_ref().parse() {
            Ok(s) => self.scaling_method = Some(s),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Enable standard error computation.
    pub fn return_se(mut self) -> Self {
        if self.interval_type.is_none() {
            self.interval_type = Some(IntervalMethod::se());
        }
        self
    }

    // Enable confidence intervals at the specified level (e.g., 0.95).
    pub fn confidence_intervals(mut self, level: T) -> Self {
        if self.interval_type.as_ref().is_some_and(|it| it.confidence) {
            self.duplicate_param = Some("confidence_intervals");
        }
        self.interval_type = Some(match self.interval_type {
            Some(existing) if existing.prediction => IntervalMethod {
                level,
                confidence: true,
                prediction: true,
                se: true,
            },
            _ => IntervalMethod::confidence(level),
        });
        self
    }

    // Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        if self.interval_type.as_ref().is_some_and(|it| it.prediction) {
            self.duplicate_param = Some("prediction_intervals");
        }
        self.interval_type = Some(match self.interval_type {
            Some(existing) if existing.confidence => IntervalMethod {
                level,
                confidence: true,
                prediction: true,
                se: true,
            },
            _ => IntervalMethod::prediction(level),
        });
        self
    }

    // Enable automatic bandwidth selection via cross-validation.
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
        if self.cv_fractions.is_some() {
            self.duplicate_param = Some("cv_fractions");
        }
        self.cv_fractions = Some(fractions);
        self
    }

    // Set the random seed for reproducible K-fold fold splitting.
    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.cv_seed = Some(seed);
        self
    }

    // Set per-dimension weights for the `"weighted"` distance metric.
    pub fn weighted_metric_weights(mut self, weights: Vec<T>) -> Self {
        self.weighted_metric_weights = Some(weights);
        self
    }

    // Enable automatic convergence detection based on relative change.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        if self.auto_converge.is_some() {
            self.duplicate_param = Some("auto_converge");
        }
        self.auto_converge = Some(tolerance);
        self
    }

    // Include statistical diagnostics (Metric, R², etc.) in output.
    pub fn return_diagnostics(mut self) -> Self {
        self.return_diagnostics = Some(true);
        self
    }

    // Include residuals in output.
    pub fn return_residuals(mut self) -> Self {
        self.compute_residuals = Some(true);
        self
    }

    // Include final robustness weights in output.
    pub fn return_robustness_weights(mut self) -> Self {
        self.return_robustness_weights = Some(true);
        self
    }

    // Set the polynomial degree for local regression.
    //
    // Accepts case-insensitive strings:
    // "constant", "linear", "quadratic", "cubic", "quartic".
    pub fn degree(mut self, degree: impl AsRef<str>) -> Self {
        if self.polynomial_degree.is_some() {
            self.duplicate_param = Some("degree");
        }
        match degree.as_ref().parse() {
            Ok(d) => self.polynomial_degree = Some(d),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the number of predictor dimensions (default: 1).
    pub fn dimensions(mut self, dims: usize) -> Self {
        if self.dimensions.is_some() {
            self.duplicate_param = Some("dimensions");
        }
        self.dimensions = Some(dims);
        self
    }

    // Set the distance metric for nD neighborhood computation.
    pub fn distance_metric(mut self, metric: impl AsRef<str>) -> Self
    where
        T: core::str::FromStr,
    {
        if self.distance_metric.is_some() {
            self.duplicate_param = Some("distance_metric");
        }
        match metric.as_ref().parse() {
            Ok(m) => self.distance_metric = Some(m),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the surface evaluation mode (Interpolation or Direct).
    pub fn surface_mode(mut self, mode: impl AsRef<str>) -> Self {
        if self.surface_mode.is_some() {
            self.duplicate_param = Some("surface_mode");
        }
        match mode.as_ref().parse() {
            Ok(m) => self.surface_mode = Some(m),
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the interpolation cell size (default: 0.2).
    pub fn cell(mut self, cell: T) -> Self {
        if self.cell.is_some() {
            self.duplicate_param = Some("cell");
        }
        self.cell = Some(cell);
        self
    }

    // Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        if self.interpolation_vertices.is_some() {
            self.duplicate_param = Some("interpolation_vertices");
        }
        self.interpolation_vertices = Some(vertices);
        self
    }

    // Set whether to reduce polynomial degree at boundary vertices during interpolation.
    //
    // When `true` (default), vertices outside the tight data bounds use Linear fits
    // to avoid unstable extrapolation.
    // Set to `false` to match R's loess behavior exactly.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        if self.boundary_degree_fallback.is_some() {
            self.duplicate_param = Some("boundary_degree_fallback");
        }
        self.boundary_degree_fallback = Some(enabled);
        self
    }

    // Set User-defined case weights (one per observation).
    //
    // Weights multiply the local kernel weight at each neighborhood point:
    // `w_ij = custom_weights[j] * K(d_ij / h) * robustness_j`.
    //
    // Higher weights increase the influence of the corresponding observation
    // on nearby local fits — analogous to `weights` in R's `stats::loess`.
    //
    // Must have the same length as `y`. Only applied in Batch mode.
    pub fn custom_weights(mut self, weights: Vec<T>) -> Self {
        if self.custom_weights.is_some() {
            self.duplicate_param = Some("custom_weights");
        }
        self.custom_weights = Some(weights);
        self
    }

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++

    // Set a custom smooth pass function for execution (only for dev)
    #[doc(hidden)]
    pub fn custom_smooth_pass(mut self, pass: SmoothPassFn<T>) -> Self {
        self.custom_smooth_pass = Some(pass);
        self
    }

    // Set a custom cross-validation pass function (only for dev)
    #[doc(hidden)]
    pub fn custom_cv_pass(mut self, pass: CVPassFn<T>) -> Self {
        self.custom_cv_pass = Some(pass);
        self
    }

    // Set a custom interval estimation pass function (only for dev)
    #[doc(hidden)]
    pub fn custom_interval_pass(mut self, pass: IntervalPassFn<T>) -> Self {
        self.custom_interval_pass = Some(pass);
        self
    }

    // Set the execution backend hint (only for dev)
    #[doc(hidden)]
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    // Set parallel execution hint (only for dev)
    #[doc(hidden)]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = Some(parallel);
        self
    }
}

// BatchMode build: produces a serial in-memory BatchLoess processor.
#[allow(private_bounds)]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static>
    LoessBuilder<T, BatchMode>
{
    pub fn build(self) -> Result<crate::adapters::batch::BatchLoess<T>, LoessError> {
        Batch::convert(self).build()
    }
}

// Low-level adapter dispatch — used by bindings/internals, not the public API.
//
// Allows transitioning from any LoessBuilder<T, Mode> to a specialized execution
// builder for cases where the mode cannot be expressed statically (e.g. FFI glue
// that selects Batch / Streaming / Online at runtime based on user input).
#[allow(private_bounds)]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static, Mode: Default>
    LoessBuilder<T, Mode>
{
    #[doc(hidden)]
    pub fn adapter<A>(self, _adapter: A) -> A::Output
    where
        A: LoessAdapter<T>,
    {
        A::convert(self)
    }
}

// StreamingMode build: produces a serial StreamingLoess processor.
#[allow(private_bounds)]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static>
    LoessBuilder<T, StreamingMode>
{
    pub fn build(self) -> Result<crate::adapters::streaming::StreamingLoess<T>, LoessError> {
        Streaming::convert(self).build()
    }
}

// OnlineMode build: produces a serial OnlineLoess processor.
#[allow(private_bounds)]
impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static>
    LoessBuilder<T, OnlineMode>
{
    pub fn build(self) -> Result<crate::adapters::online::OnlineLoess<T>, LoessError> {
        Online::convert(self).build()
    }
}

// Trait for transitioning a LoessBuilder into a specialized execution builder.
pub trait LoessAdapter<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> {
    // The output execution builder.
    type Output;

    // Convert a [`LoessBuilder`] (any mode) into a specialized execution builder.
    fn convert<Mode>(builder: LoessBuilder<T, Mode>) -> Self::Output;
}

// Marker for in-memory batch processing.
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> LoessAdapter<T>
    for Batch
{
    type Output = BatchLoessBuilder<T>;

    fn convert<Mode>(builder: LoessBuilder<T, Mode>) -> Self::Output {
        let mut result = BatchLoessBuilder::default();

        if let Some(fraction) = builder.fraction {
            result.fraction = fraction;
        }
        if let Some(iterations) = builder.iterations {
            result.iterations = iterations;
        }
        if let Some(wf) = builder.weight_function {
            result.weight_function = wf;
        }
        if let Some(rm) = builder.robustness_method {
            result.robustness_method = rm;
        }
        if let Some(sm) = builder.scaling_method {
            result.scaling_method = sm;
        }
        if let Some(it) = builder.interval_type {
            result.interval_type = Some(it);
        }
        if let Some(cvf) = builder.cv_fractions {
            result.cv_fractions = Some(cvf);
        }
        if let Some(cvk) = builder.cv_kind {
            result.cv_kind = Some(cvk);
        }
        result.cv_seed = builder.cv_seed;
        // Convert string-based CV method (from cv_method()/cv_k() builder methods)
        if result.cv_kind.is_none()
            && let Some(method_str) = builder.cv_method_str
        {
            let lower = method_str.to_lowercase();
            match lower.as_str() {
                "kfold" | "k_fold" | "k-fold" => {
                    result.cv_kind = Some(CVKind::KFold(builder.cv_k_val));
                }
                "loocv" | "loo_cv" | "loo-cv" => {
                    result.cv_kind = Some(CVKind::LOOCV);
                }
                _ => {
                    result.deferred_error = Some(LoessError::InvalidOption {
                        option: "cv_method",
                        value: method_str,
                        valid: "kfold, loocv",
                    });
                }
            }
        }
        if let Some(ac) = builder.auto_converge {
            result.auto_converge = Some(ac);
        }
        if let Some(zwf) = builder.zero_weight_fallback {
            result.zero_weight_fallback = zwf;
        }
        if let Some(bp) = builder.boundary_policy {
            result.boundary_policy = bp;
        }

        if let Some(rw) = builder.return_robustness_weights {
            result.return_robustness_weights = rw;
        }
        if let Some(rd) = builder.return_diagnostics {
            result.return_diagnostics = rd;
        }
        if let Some(cr) = builder.compute_residuals {
            result.compute_residuals = cr;
        }
        if let Some(pd) = builder.polynomial_degree {
            result.polynomial_degree = pd;
        }
        if let Some(dims) = builder.dimensions {
            result.dimensions = dims;
        }
        if let Some(mut dm) = builder.distance_metric {
            if let DistanceMetric::Weighted(ref mut w) = dm
                && let Some(wmw) = builder.weighted_metric_weights
            {
                *w = wmw;
            }
            result.distance_metric = dm;
        }
        if let Some(cell) = builder.cell {
            result.cell = Some(cell.to_f64().unwrap());
        }
        if let Some(iv) = builder.interpolation_vertices {
            result.interpolation_vertices = Some(iv);
        }
        if let Some(sm) = builder.surface_mode {
            result.surface_mode = sm;
        }
        if let Some(bdf) = builder.boundary_degree_fallback {
            result.boundary_degree_fallback = bdf;
        }
        if let Some(uw) = builder.custom_weights {
            result.custom_weights = Some(uw);
        }

        // ++++++++++++++++++++++++++++++++++++++
        // +               DEV                  +
        // ++++++++++++++++++++++++++++++++++++++
        if let Some(sp) = builder.custom_smooth_pass {
            result.custom_smooth_pass = Some(sp);
        }
        if let Some(cp) = builder.custom_cv_pass {
            result.custom_cv_pass = Some(cp);
        }
        if let Some(ip) = builder.custom_interval_pass {
            result.custom_interval_pass = Some(ip);
        }
        if let Some(b) = builder.backend {
            result.backend = Some(b);
        }
        if let Some(p) = builder.parallel {
            result.parallel = Some(p);
        }

        result.duplicate_param = builder.duplicate_param;

        if !builder.parse_errors.is_empty() {
            result.deferred_error = Some(LoessError::ParseErrors(builder.parse_errors));
        }

        result
    }
}

// Marker for chunked streaming processing.
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> LoessAdapter<T>
    for Streaming
{
    type Output = StreamingLoessBuilder<T>;

    fn convert<Mode>(builder: LoessBuilder<T, Mode>) -> Self::Output {
        let mut result = StreamingLoessBuilder::default();

        // Override with user-provided values
        if let Some(chunk_size) = builder.chunk_size {
            result.chunk_size = chunk_size;
        }
        if let Some(overlap) = builder.overlap {
            result.overlap = overlap;
        }
        if let Some(fraction) = builder.fraction {
            result.fraction = fraction;
        }
        if let Some(iterations) = builder.iterations {
            result.iterations = iterations;
        }
        if let Some(wf) = builder.weight_function {
            result.weight_function = wf;
        }
        if let Some(bp) = builder.boundary_policy {
            result.boundary_policy = bp;
        }
        if let Some(rm) = builder.robustness_method {
            result.robustness_method = rm;
        }
        if let Some(sm) = builder.scaling_method {
            result.scaling_method = sm;
        }
        if let Some(zwf) = builder.zero_weight_fallback {
            result.zero_weight_fallback = zwf;
        }
        if let Some(ms) = builder.merge_strategy {
            result.merge_strategy = ms;
        }

        if let Some(rw) = builder.return_robustness_weights {
            result.return_robustness_weights = rw;
        }
        if let Some(rd) = builder.return_diagnostics {
            result.return_diagnostics = rd;
        }
        if let Some(cr) = builder.compute_residuals {
            result.compute_residuals = cr;
        }
        if let Some(ac) = builder.auto_converge {
            result.auto_converge = Some(ac);
        }
        if let Some(pd) = builder.polynomial_degree {
            result.polynomial_degree = pd;
        }
        if let Some(dims) = builder.dimensions {
            result.dimensions = dims;
        }
        if let Some(mut dm) = builder.distance_metric {
            if let DistanceMetric::Weighted(ref mut w) = dm
                && let Some(wmw) = builder.weighted_metric_weights
            {
                *w = wmw;
            }
            result.distance_metric = dm;
        }
        if let Some(cell) = builder.cell {
            result.cell = Some(cell.to_f64().unwrap());
        }
        if let Some(iv) = builder.interpolation_vertices {
            result.interpolation_vertices = Some(iv);
        }
        if let Some(sm) = builder.surface_mode {
            result.surface_mode = sm;
        }
        if let Some(bdf) = builder.boundary_degree_fallback {
            result.boundary_degree_fallback = bdf;
        }

        // ++++++++++++++++++++++++++++++++++++++
        // +               DEV                  +
        // ++++++++++++++++++++++++++++++++++++++

        if let Some(sp) = builder.custom_smooth_pass {
            result.custom_smooth_pass = Some(sp);
        }
        if let Some(cp) = builder.custom_cv_pass {
            result.custom_cv_pass = Some(cp);
        }
        if let Some(ip) = builder.custom_interval_pass {
            result.custom_interval_pass = Some(ip);
        }
        if let Some(b) = builder.backend {
            result.backend = Some(b);
        }
        if let Some(p) = builder.parallel {
            result.parallel = Some(p);
        }
        result.duplicate_param = builder.duplicate_param;

        if !builder.parse_errors.is_empty() {
            result.deferred_error = Some(LoessError::ParseErrors(builder.parse_errors));
        }

        result
    }
}

// Marker for incremental online processing.
#[derive(Debug, Clone, Copy)]
pub struct Online;

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> LoessAdapter<T>
    for Online
{
    type Output = OnlineLoessBuilder<T>;

    fn convert<Mode>(builder: LoessBuilder<T, Mode>) -> Self::Output {
        let mut result = OnlineLoessBuilder::default();

        // Override with user-provided values
        if let Some(window_capacity) = builder.window_capacity {
            result.window_capacity = window_capacity;
        }
        if let Some(min_points) = builder.min_points {
            result.min_points = min_points;
        }
        if let Some(fraction) = builder.fraction {
            result.fraction = fraction;
        }
        if let Some(iterations) = builder.iterations {
            result.iterations = iterations;
        }
        if let Some(wf) = builder.weight_function {
            result.weight_function = wf;
        }
        if let Some(um) = builder.update_mode {
            result.update_mode = um;
        }
        if let Some(rm) = builder.robustness_method {
            result.robustness_method = rm;
        }
        if let Some(sm) = builder.scaling_method {
            result.scaling_method = sm;
        }
        if let Some(bp) = builder.boundary_policy {
            result.boundary_policy = bp;
        }
        if let Some(zwf) = builder.zero_weight_fallback {
            result.zero_weight_fallback = zwf;
        }

        if let Some(cr) = builder.compute_residuals {
            result.compute_residuals = cr;
        }
        if let Some(rw) = builder.return_robustness_weights {
            result.return_robustness_weights = rw;
        }
        if let Some(ac) = builder.auto_converge {
            result.auto_converge = Some(ac);
        }
        if let Some(pd) = builder.polynomial_degree {
            result.polynomial_degree = pd;
        }
        if let Some(dims) = builder.dimensions {
            result.dimensions = dims;
        }
        if let Some(mut dm) = builder.distance_metric {
            if let DistanceMetric::Weighted(ref mut w) = dm
                && let Some(wmw) = builder.weighted_metric_weights
            {
                *w = wmw;
            }
            result.distance_metric = dm;
        }
        if let Some(cell) = builder.cell {
            result.cell = Some(cell.to_f64().unwrap());
        }
        if let Some(iv) = builder.interpolation_vertices {
            result.interpolation_vertices = Some(iv);
        }
        if let Some(sm) = builder.surface_mode {
            result.surface_mode = sm;
        }
        if let Some(bdf) = builder.boundary_degree_fallback {
            result.boundary_degree_fallback = bdf;
        }

        // ++++++++++++++++++++++++++++++++++++++
        // +               DEV                  +
        // ++++++++++++++++++++++++++++++++++++++

        if let Some(sp) = builder.custom_smooth_pass {
            result.custom_smooth_pass = Some(sp);
        }
        if let Some(cp) = builder.custom_cv_pass {
            result.custom_cv_pass = Some(cp);
        }
        if let Some(ip) = builder.custom_interval_pass {
            result.custom_interval_pass = Some(ip);
        }
        if let Some(b) = builder.backend {
            result.backend = Some(b);
        }
        if let Some(p) = builder.parallel {
            result.parallel = Some(p);
        }
        result.duplicate_param = builder.duplicate_param;

        if !builder.parse_errors.is_empty() {
            result.deferred_error = Some(LoessError::ParseErrors(builder.parse_errors));
        }

        result
    }
}

// ─── String-to-enum conversion ────────────────────────────────────────────────
//
// All `impl FromStr` blocks for option types.  These are the single source of
// truth consumed by both the `IntoEnum` trait (builder ergonomics) and the
// language-binding helpers below.

use core::str::FromStr;
use num_traits::Float;

// WeightFunction

impl FromStr for WeightFunction {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(WeightFunction::Cosine),
            "epanechnikov" => Ok(WeightFunction::Epanechnikov),
            "gaussian" => Ok(WeightFunction::Gaussian),
            "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
            "triangle" | "triangular" => Ok(WeightFunction::Triangle),
            "tricube" => Ok(WeightFunction::Tricube),
            "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
            _ => Err(LoessError::InvalidOption {
                option: "weight_function",
                value: s.to_string(),
                valid: "tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            }),
        }
    }
}

// BoundaryPolicy

impl FromStr for BoundaryPolicy {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "extend" | "pad" => Ok(BoundaryPolicy::Extend),
            "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
            "zero" => Ok(BoundaryPolicy::Zero),
            "noboundary" | "none" => Ok(BoundaryPolicy::NoBoundary),
            _ => Err(LoessError::InvalidOption {
                option: "boundary_policy",
                value: s.to_string(),
                valid: "extend, reflect, zero, noboundary",
            }),
        }
    }
}

// DistanceMetric<T>

impl<T> FromStr for DistanceMetric<T>
where
    T: Float + FromStr,
{
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        if let Some(p_str) = lower.strip_prefix("minkowski:") {
            let p: T = p_str.parse().map_err(|_| LoessError::InvalidOption {
                option: "distance_metric",
                value: s.to_string(),
                valid: "normalized, euclidean, manhattan, chebyshev, minkowski, minkowski:<p>, weighted",
            })?;
            return Ok(DistanceMetric::Minkowski(p));
        }
        match lower.as_str() {
            "normalized" | "norm" => Ok(DistanceMetric::Normalized),
            "euclidean" | "euclid" => Ok(DistanceMetric::Euclidean),
            "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
            "chebyshev" | "linf" => Ok(DistanceMetric::Chebyshev),
            "minkowski" => Ok(DistanceMetric::Minkowski(T::from(2.0).unwrap())),
            "weighted" | "weighted_euclidean" => Ok(DistanceMetric::Weighted(Vec::new())),
            _ => Err(LoessError::InvalidOption {
                option: "distance_metric",
                value: s.to_string(),
                valid: "normalized, euclidean, manhattan, chebyshev, minkowski, minkowski:<p>, weighted",
            }),
        }
    }
}

// ScalingMethod

impl FromStr for ScalingMethod {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mar" | "median_absolute_residual" => Ok(ScalingMethod::MAR),
            "mad" | "median_absolute_deviation" => Ok(ScalingMethod::MAD),
            "mean" | "mean_absolute_residual" => Ok(ScalingMethod::Mean),
            _ => Err(LoessError::InvalidOption {
                option: "scaling_method",
                value: s.to_string(),
                valid: "mad, mar, mean",
            }),
        }
    }
}

// RobustnessMethod

impl FromStr for RobustnessMethod {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
            "huber" => Ok(RobustnessMethod::Huber),
            "talwar" => Ok(RobustnessMethod::Talwar),
            _ => Err(LoessError::InvalidOption {
                option: "robustness_method",
                value: s.to_string(),
                valid: "bisquare, huber, talwar",
            }),
        }
    }
}

// SurfaceMode

impl FromStr for SurfaceMode {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "interpolation" | "interp" | "interpolate" => Ok(SurfaceMode::Interpolation),
            "direct" => Ok(SurfaceMode::Direct),
            _ => Err(LoessError::InvalidOption {
                option: "surface_mode",
                value: s.to_string(),
                valid: "interpolation, direct",
            }),
        }
    }
}

// PolynomialDegree

impl FromStr for PolynomialDegree {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "constant" | "0" => Ok(PolynomialDegree::Constant),
            "linear" | "1" => Ok(PolynomialDegree::Linear),
            "quadratic" | "2" => Ok(PolynomialDegree::Quadratic),
            "cubic" | "3" => Ok(PolynomialDegree::Cubic),
            "quartic" | "4" => Ok(PolynomialDegree::Quartic),
            _ => Err(LoessError::InvalidOption {
                option: "degree",
                value: s.to_string(),
                valid: "constant (0), linear (1), quadratic (2), cubic (3), quartic (4)",
            }),
        }
    }
}

// ZeroWeightFallback

impl FromStr for ZeroWeightFallback {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
            "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
            "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
            _ => Err(LoessError::InvalidOption {
                option: "zero_weight_fallback",
                value: s.to_string(),
                valid: "use_local_mean, return_original, return_none",
            }),
        }
    }
}

// MergeStrategy

impl FromStr for MergeStrategy {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average" | "mean" => Ok(MergeStrategy::Average),
            "weighted_average" | "weighted" | "weightedaverage" => {
                Ok(MergeStrategy::WeightedAverage)
            }
            "take_first" | "first" | "takefirst" | "left" => Ok(MergeStrategy::TakeFirst),
            "take_last" | "last" | "takelast" | "right" => Ok(MergeStrategy::TakeLast),
            _ => Err(LoessError::InvalidOption {
                option: "merge_strategy",
                value: s.to_string(),
                valid: "average, weighted_average, take_first, take_last",
            }),
        }
    }
}

// UpdateMode

impl FromStr for UpdateMode {
    type Err = LoessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" | "resmooth" => Ok(UpdateMode::Full),
            "incremental" | "single" => Ok(UpdateMode::Incremental),
            _ => Err(LoessError::InvalidOption {
                option: "update_mode",
                value: s.to_string(),
                valid: "full, incremental",
            }),
        }
    }
}

// Binding helpers (only with the `dev` feature)
//
// Parse and canonical-name wrappers used by the binding layer.  Only compiled
// when the `dev` feature is active; re-exported through
// `loess_rs::internals::alias`.

#[cfg(feature = "dev")]
pub mod helpers {
    use super::{
        BoundaryPolicy, DistanceMetric, LoessError, MergeStrategy, PolynomialDegree,
        RobustnessMethod, ScalingMethod, SurfaceMode, UpdateMode, WeightFunction,
        ZeroWeightFallback,
    };

    // Parse helpers

    pub fn parse_weight_function(s: &str) -> Result<WeightFunction, LoessError> {
        s.parse()
    }

    pub fn parse_robustness_method(s: &str) -> Result<RobustnessMethod, LoessError> {
        s.parse()
    }

    pub fn parse_zero_weight_fallback(s: &str) -> Result<ZeroWeightFallback, LoessError> {
        s.parse()
    }

    pub fn parse_boundary_policy(s: &str) -> Result<BoundaryPolicy, LoessError> {
        s.parse()
    }

    pub fn parse_scaling_method(s: &str) -> Result<ScalingMethod, LoessError> {
        s.parse()
    }

    pub fn parse_polynomial_degree(s: &str) -> Result<PolynomialDegree, LoessError> {
        s.parse()
    }

    pub fn parse_distance_metric(s: &str) -> Result<DistanceMetric<f64>, LoessError> {
        s.parse()
    }

    pub fn parse_surface_mode(s: &str) -> Result<SurfaceMode, LoessError> {
        s.parse()
    }

    pub fn parse_update_mode(s: &str) -> Result<UpdateMode, LoessError> {
        s.parse()
    }

    pub fn parse_merge_strategy(s: &str) -> Result<MergeStrategy, LoessError> {
        s.parse()
    }

    // Canonical-name helpers
    //
    // Round-trip guarantee: `X_str(v).parse::<X>().unwrap() == v` for all `v`.
    // `DistanceMetric` is excluded because `Minkowski(p)` requires a formatted
    // string; use `distance_metric_components` in the binding layer instead.

    pub fn weight_function_str(v: WeightFunction) -> &'static str {
        match v {
            WeightFunction::Tricube => "tricube",
            WeightFunction::Epanechnikov => "epanechnikov",
            WeightFunction::Gaussian => "gaussian",
            WeightFunction::Uniform => "uniform",
            WeightFunction::Biweight => "biweight",
            WeightFunction::Triangle => "triangle",
            WeightFunction::Cosine => "cosine",
        }
    }

    pub fn robustness_method_str(v: RobustnessMethod) -> &'static str {
        match v {
            RobustnessMethod::Bisquare => "bisquare",
            RobustnessMethod::Huber => "huber",
            RobustnessMethod::Talwar => "talwar",
        }
    }

    pub fn scaling_method_str(v: ScalingMethod) -> &'static str {
        match v {
            ScalingMethod::MAD => "mad",
            ScalingMethod::MAR => "mar",
            ScalingMethod::Mean => "mean",
        }
    }

    pub fn zero_weight_fallback_str(v: ZeroWeightFallback) -> &'static str {
        match v {
            ZeroWeightFallback::UseLocalMean => "use_local_mean",
            ZeroWeightFallback::ReturnOriginal => "return_original",
            ZeroWeightFallback::ReturnNone => "return_none",
        }
    }

    pub fn boundary_policy_str(v: BoundaryPolicy) -> &'static str {
        match v {
            BoundaryPolicy::Extend => "extend",
            BoundaryPolicy::Reflect => "reflect",
            BoundaryPolicy::Zero => "zero",
            BoundaryPolicy::NoBoundary => "noboundary",
        }
    }

    pub fn polynomial_degree_str(v: PolynomialDegree) -> &'static str {
        match v {
            PolynomialDegree::Constant => "constant",
            PolynomialDegree::Linear => "linear",
            PolynomialDegree::Quadratic => "quadratic",
            PolynomialDegree::Cubic => "cubic",
            PolynomialDegree::Quartic => "quartic",
        }
    }

    pub fn surface_mode_str(v: SurfaceMode) -> &'static str {
        match v {
            SurfaceMode::Interpolation => "interpolation",
            SurfaceMode::Direct => "direct",
        }
    }

    pub fn update_mode_str(v: UpdateMode) -> &'static str {
        match v {
            UpdateMode::Full => "full",
            UpdateMode::Incremental => "incremental",
        }
    }

    pub fn merge_strategy_str(v: MergeStrategy) -> &'static str {
        match v {
            MergeStrategy::Average => "average",
            MergeStrategy::WeightedAverage => "weighted_average",
            MergeStrategy::TakeFirst => "take_first",
            MergeStrategy::TakeLast => "take_last",
        }
    }
}