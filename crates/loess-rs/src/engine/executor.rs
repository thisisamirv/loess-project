//! Execution engine for LOESS smoothing operations.
//!
//! This module provides the core execution engine that orchestrates LOESS
//! smoothing operations. It handles the iteration loop, robustness weight
//! updates, convergence checking, cross-validation, and variance estimation.
//! The executor is the central component that coordinates all lower-level
//! algorithms to produce smoothed results.
// ## srrstats Compliance
//
// @srrstats {RE2.0} Core LOESS execution: boundary handling, iteration loop, convergence.
// @srrstats {G2.2} Auto-convergence tolerance for early stopping of iterations.
// Configurable robustness iterations with convergence monitoring.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::sync::Arc;
#[cfg(feature = "std")]
use std::vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering::Equal;
use core::fmt::{Debug, Display, Formatter};
use num_traits::Float;

// Internal dependencies
use crate::algorithms::defaults::*;
use crate::algorithms::interpolation::InterpolationSurface;
use crate::algorithms::regression::context::RegressionContext;
use crate::algorithms::regression::specialized::SolverLinalg;
use crate::engine::defaults::*;
use crate::evaluation::cv::CVKind;
use crate::evaluation::diagnostics::Diagnostics;
use crate::evaluation::intervals::{BootstrapConfig, BootstrapOutput, IntervalMethod};
use crate::primitives::errors::LoessError;
use crate::primitives::policies::ExtrapolationPolicy;

use crate::math::defaults::*;
use crate::math::distance::DistanceLinalg;
use crate::math::linalg::FloatLinalg;
use crate::math::neighborhood::{KDTree, Neighborhood, NodeDistance, PointDistance};
use crate::primitives::backend::Backend;
use crate::primitives::buffer::{
    CVBuffer, CachedNeighborhood, FittingBuffer, LoessBuffer, NeighborhoodSearchBuffer,
};
use crate::primitives::policies::{
    BoundaryPolicy, DistanceMetric, PolynomialDegree, RobustnessMethod, ScalingMethod,
    WeightFunction, ZeroWeightFallback,
};
use crate::primitives::window::Window;

#[derive(Debug, Clone, Copy)]
pub struct CVRunOptions<T> {
    pub kind: CVKind,
    pub seed: Option<u64>,
    pub tolerance: Option<T>,
}

#[derive(Debug, Clone)]
pub struct PredictQuery<T> {
    pub(crate) return_se: bool,
    pub(crate) confidence_intervals: Option<T>,
    pub(crate) prediction_intervals: Option<T>,
    pub(crate) bootstrap: Option<BootstrapConfig>,
    pub(crate) return_derivative: bool,
    pub(crate) extrapolation: ExtrapolationPolicy,
    pub(crate) max_extrapolation_distance: Option<T>,
    pub(crate) max_neighbor_distance: Option<T>,
}
impl<T> PredictQuery<T> {
    // Whether the local fit's gradient is included in the output (used by fastLoess's
    // parallel predict pass, which needs this outside `loess-rs` itself). No trait
    // bounds needed: this just reads a plain `bool` field.
    pub fn return_derivative(&self) -> bool {
        self.return_derivative
    }
}

pub type RawPredictValues<T> = Result<(Vec<T>, Option<Vec<T>>, Option<Vec<T>>), LoessError>;
pub type PredictPassFn<T> = fn(
    &PredictState<T>,
    &[T], // new_x (flattened, `dimensions` values per query point)
    &PredictQuery<T>,
    bool, // need_se
) -> RawPredictValues<T>;
#[derive(Debug, Clone)]
pub struct PredictState<T: Float> {
    // Boundary-padded, flattened training predictors (`x.len() == dimensions * n_total`).
    pub x: Vec<T>,

    // Number of predictor dimensions.
    pub dimensions: usize,

    // Boundary-padded training responses, aligned with `x`.
    pub y: Vec<T>,

    // Final (post-robustness-iteration) weights, aligned with `x`/`y`.
    pub robustness_weights: Vec<T>,

    // Neighbor count (span), already resolved from `fraction`.
    pub window_size: usize,

    // Kernel weight function used during fitting.
    pub weight_function: WeightFunction,

    // Zero-weight fallback policy.
    pub zero_weight_fallback: ZeroWeightFallback,

    // Degree of local polynomial used during fitting.
    pub polynomial_degree: PolynomialDegree,

    // Distance metric used for neighborhood search during fitting.
    pub distance_metric: DistanceMetric<T>,

    // Per-dimension normalization scales (used when `distance_metric` is `Normalized`),
    // computed from the unpadded training data's 10%-trimmed sample standard deviation.
    pub scales: Vec<T>,

    // Per-observation case weights, aligned with `x`/`y`, if provided.
    pub custom_weights: Option<Vec<T>>,

    // Global residual scale used to widen prediction intervals beyond the local standard
    // error: the same `sqrt(RSS / delta1)` value as `LoessResult::residual_scale` if
    // `.return_se()` was set on the original `fit()`, otherwise a MAD-based fallback
    // (matching `Diagnostics.residual_sd`).
    pub residual_sd: T,

    // Standard error for in-range queries under `SurfaceMode::Interpolation`, precomputed
    // with the exact same uniform approximate-leverage heuristic `fit()` itself falls back
    // to there (`sigma * sqrt(eff_fraction / n)`). Keeps `predict()`'s SE self-consistent
    // with whichever surface mode produced `y`, instead of pairing a fast/approximate `y`
    // with an unrelated exact-leverage SE. Unused (and meaningless) when `surface` is `None`.
    pub interpolation_se: T,

    // Per-dimension minimum/maximum of the REAL (unpadded) training predictors, used to
    // decide whether a query point is out-of-range for `ExtrapolationPolicy`. `x` above is
    // boundary-*padded* and can extend well beyond this range on each dimension.
    pub train_min: Vec<T>,
    pub train_max: Vec<T>,

    // KD-tree over `x`, built once when the model is retained rather than rebuilt on every
    // `predict()` call (it would otherwise need to be reconstructed from scratch each time).
    pub kdtree: KDTree<T>,

    // Interpolation surface built during `fit()`, retained when `surface_mode()` was
    // `Interpolation` (the default). `predict_one_full` reuses it (via `evaluate()`) for
    // in-range query points, so the returned value matches `fit()`'s own `y_smooth` at
    // training points instead of diverging via a separate exact per-point regression.
    // `None` under `SurfaceMode::Direct`.
    pub surface: Option<InterpolationSurface<T>>,

    // Custom (e.g. parallel) predict pass, injected by extension crates like fastLoess.
    pub custom_predict_pass: Option<PredictPassFn<T>>,
    pub bootstrap_predictor: Option<Arc<dyn BootstrapPredictor<T>>>,
}
pub type BootstrapPredictionFn<'a, T> =
    dyn FnMut(&PredictState<T>) -> Result<Vec<T>, LoessError> + 'a;
pub trait BootstrapPredictor<T: Float>: Debug + Send + Sync + core::panic::RefUnwindSafe {
    fn original_residuals(&self) -> Option<&[T]> {
        None
    }

    fn compute(
        &self,
        bootstrap: BootstrapConfig,
        method: &IntervalMethod<T>,
        n_output: usize,
        predict: &mut BootstrapPredictionFn<'_, T>,
    ) -> Result<BootstrapOutput<T>, LoessError>;
}
impl<T: Float + PartialEq> PartialEq for PredictState<T> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.dimensions == other.dimensions
            && self.y == other.y
            && self.robustness_weights == other.robustness_weights
            && self.window_size == other.window_size
            && self.weight_function == other.weight_function
            && self.zero_weight_fallback == other.zero_weight_fallback
            && self.polynomial_degree == other.polynomial_degree
            && self.distance_metric == other.distance_metric
            && self.scales == other.scales
            && self.custom_weights == other.custom_weights
            && self.residual_sd == other.residual_sd
            && self.interpolation_se == other.interpolation_se
            && self.train_min == other.train_min
            && self.train_max == other.train_max
    }
}

// Comprehensive LOESS output containing smoothed values and diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct LoessResult<T: Float> {
    // Input x-values (independent variable). Flattened for nD.
    pub x: Vec<T>,

    // Number of predictor dimensions.
    pub dimensions: usize,

    // Distance metric used for neighborhood search.
    pub distance_metric: DistanceMetric<T>,

    // Degree of local polynomial (0 for local mean, 1 for local linear, 2 for local quadratic).
    pub polynomial_degree: PolynomialDegree,

    // Smoothed y-values (dependent variable).
    pub y: Vec<T>,

    // Standard errors of the fit at each point.
    pub standard_errors: Option<Vec<T>>,

    // Lower bounds of the confidence intervals for the mean response.
    pub confidence_lower: Option<Vec<T>>,

    // Upper bounds of the confidence intervals for the mean response.
    pub confidence_upper: Option<Vec<T>>,

    // Lower bounds of the prediction intervals for new observations.
    pub prediction_lower: Option<Vec<T>>,

    // Upper bounds of the prediction intervals for new observations.
    pub prediction_upper: Option<Vec<T>>,

    // Residuals from the fit (y_i - y_hat_i).
    pub residuals: Option<Vec<T>>,

    // Final robustness weights from the iterative refinement process.
    pub robustness_weights: Option<Vec<T>>,

    // Comprehensive diagnostic metrics (RMSE, R^2, AIC, etc.).
    pub diagnostics: Option<Diagnostics<T>>,

    // Number of robustness iterations actually performed.
    pub iterations_used: Option<usize>,

    // Smoothing fraction used for the fit (optimal if selected by CV).
    pub fraction_used: T,

    // RMSE scores for each tested fraction during cross-validation.
    pub cv_scores: Option<Vec<T>>,

    // Equivalent Number of Parameters (trace of hat matrix).
    // This measures the effective model complexity.
    pub enp: Option<T>,

    // Trace of the hat matrix (same as ENP for LOESS).
    pub trace_hat: Option<T>,

    // Delta1 for proper SE computation: tr((I-L)(I-L)').
    // Used as the denominator for residual scale estimation.
    pub delta1: Option<T>,

    // Delta2 for SE computation: tr(((I-L)(I-L)')^2).
    // Used for confidence interval width adjustment.
    pub delta2: Option<T>,

    // Residual scale estimate: sqrt(RSS / delta1).
    // This is the proper estimate of sigma for inference.
    pub residual_scale: Option<T>,

    // Leverage (hat matrix diagonal) at each point.
    // l_ii measures how much influence point i has on its own fitted value.
    pub leverage: Option<Vec<T>>,

    // Per-point local fit gradient (flattened, `dimensions` values per point), if
    // `return_gradient` was set. Only computed in `SurfaceMode::Direct`; `None` in
    // `SurfaceMode::Interpolation`.
    pub gradient: Option<Vec<T>>,

    // Retained fitted-model state for `predict()`, if `retain_model` was set. Wrapped in
    // `Arc` so cloning a `LoessResult` (e.g. to hand to multiple worker threads) is a
    // cheap refcount bump instead of deep-copying the whole padded training set.
    pub predict_state: Option<Arc<PredictState<T>>>,
}

impl<T: Float> LoessResult<T> {
    // Check if confidence intervals were computed.
    pub fn has_confidence_intervals(&self) -> bool {
        self.confidence_lower.is_some() && self.confidence_upper.is_some()
    }

    // Check if prediction intervals were computed.
    pub fn has_prediction_intervals(&self) -> bool {
        self.prediction_lower.is_some() && self.prediction_upper.is_some()
    }

    // Check if cross-validation was performed.
    pub fn has_cv_scores(&self) -> bool {
        self.cv_scores.is_some()
    }

    // Get the best (minimum) CV score.
    pub fn best_cv_score(&self) -> Option<T> {
        self.cv_scores.as_ref().and_then(|scores| {
            scores
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))
        })
    }
}

impl<T: Float + Display + Debug> Display for LoessResult<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Summary:")?;
        let n = self.y.len();
        writeln!(f, "  Data points: {}", n)?;
        writeln!(f, "  Dimensions:  {}", self.dimensions)?;
        writeln!(f, "  Distance:    {:?}", self.distance_metric)?;
        writeln!(f, "  Degree:      {:?}", self.polynomial_degree)?;
        writeln!(f, "  Fraction:    {}", self.fraction_used)?;

        if let Some(iters) = self.iterations_used {
            writeln!(f, "  Iterations: {}", iters)?;
        }

        // Show robustness status
        if self.robustness_weights.is_some() {
            writeln!(f, "  Robustness: Applied")?;
        }

        if self.has_cv_scores()
            && let Some(best_score) = self.best_cv_score()
        {
            writeln!(f, "  Best CV score: {}", best_score)?;
        }
        writeln!(f)?;

        if let Some(diag) = &self.diagnostics {
            writeln!(f, "{}", diag)?;
        }

        writeln!(f, "Smoothed Data:")?;

        // Determine which columns to show
        let has_std_err = self.standard_errors.is_some();
        let has_conf = self.has_confidence_intervals();
        let has_pred = self.has_prediction_intervals();
        let has_resid = self.residuals.is_some();
        let has_weights = self.robustness_weights.is_some();

        // Build header
        if self.dimensions == 1 {
            write!(f, "{:>8} {:>12}", "X", "Y_smooth")?;
        } else {
            write!(f, "{:>8} {:>12}", "X (nD)", "Y_smooth")?;
        }
        if has_std_err {
            write!(f, " {:>12}", "Std_Err")?;
        }
        if has_conf {
            write!(f, " {:>12} {:>12}", "Conf_Lower", "Conf_Upper")?;
        }
        if has_pred {
            write!(f, " {:>12} {:>12}", "Pred_Lower", "Pred_Upper")?;
        }
        if has_resid {
            write!(f, " {:>12}", "Residual")?;
        }
        if has_weights {
            write!(f, " {:>10}", "Rob_Weight")?;
        }
        writeln!(f)?;

        // Separator line
        let line_width = 21
            + if has_std_err { 13 } else { 0 }
            + if has_conf { 26 } else { 0 }
            + if has_pred { 26 } else { 0 }
            + if has_resid { 13 } else { 0 }
            + if has_weights { 11 } else { 0 };
        writeln!(f, "{:-<width$}", "", width = line_width)?;

        // Data rows (show first 10 and last 10 if more than 20 points)
        let n = self.x.len();
        let show_all = n <= 20;
        let rows_to_show: Vec<usize> = if show_all {
            (0..n).collect()
        } else {
            (0..10).chain(n - 10..n).collect()
        };

        let mut prev_idx = 0;
        for (i, &idx) in rows_to_show.iter().enumerate() {
            // Add ellipsis if we skipped rows
            if i > 0 && idx != prev_idx + 1 {
                writeln!(f, "{:>8}", "...")?;
            }
            prev_idx = idx;

            if self.dimensions == 1 {
                write!(f, "{:>8.2} {:>12.6}", self.x[idx], self.y[idx])?;
            } else {
                write!(f, "{:>8.2} {:>12.6}", "[...]", self.y[idx])?;
            }

            // Standard error
            if has_std_err && let Some(se) = &self.standard_errors {
                write!(f, " {:>12.6}", se[idx])?;
            }

            // Confidence intervals
            if has_conf
                && let (Some(lower), Some(upper)) = (&self.confidence_lower, &self.confidence_upper)
            {
                write!(f, " {:>12.6} {:>12.6}", lower[idx], upper[idx])?;
            }

            // Prediction intervals
            if has_pred
                && let (Some(lower), Some(upper)) = (&self.prediction_lower, &self.prediction_upper)
            {
                write!(f, " {:>12.6} {:>12.6}", lower[idx], upper[idx])?;
            }

            // Residuals
            if has_resid && let Some(resid) = &self.residuals {
                write!(f, " {:>12.6}", resid[idx])?;
            }

            // Robustness weights
            if has_weights && let Some(weights) = &self.robustness_weights {
                write!(f, " {:>10.4}", weights[idx])?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

fn loess_normalization_scales<T: Float>(x: &[T], n: usize, dims: usize) -> Vec<T> {
    let mut scales = vec![T::one(); dims];
    if dims <= 1 || n == 0 {
        return scales;
    }

    let trim = n.div_ceil(10);
    let mut values = Vec::with_capacity(n);
    for dim in 0..dims {
        values.clear();
        values.extend((0..n).map(|row| x[row * dims + dim]));
        values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Equal));

        let start = trim.min(n);
        let end = n.saturating_sub(trim);
        let (start, end) = if end.saturating_sub(start) < 2 {
            (0, n)
        } else {
            (start, end)
        };
        let retained = &values[start..end];
        let count = T::from(retained.len()).unwrap_or(T::one());
        let mean = retained
            .iter()
            .copied()
            .fold(T::zero(), |sum, value| sum + value)
            / count;
        let squared_deviations = retained.iter().fold(T::zero(), |sum, &value| {
            let deviation = value - mean;
            sum + deviation * deviation
        });
        let variance = squared_deviations / T::from(retained.len() - 1).unwrap_or(T::one());
        let standard_deviation = variance.sqrt();
        if standard_deviation > T::zero() && standard_deviation.is_finite() {
            scales[dim] = T::one() / standard_deviation;
        }
    }

    scales
}

// Standard LOESS distance calculator.
//
// Implements `PointDistance` using either Euclidean or Normalized Euclidean metrics.
pub struct LoessDistanceCalculator<'a, T: FloatLinalg + DistanceLinalg + SolverLinalg> {
    // The distance metric to use (Euclidean or Normalized).
    pub metric: DistanceMetric<T>,
    // Normalization scales for each dimension (used if metric is Normalized).
    pub scales: &'a [T],
}

impl<'a, T: FloatLinalg + DistanceLinalg + SolverLinalg> PointDistance<T>
    for LoessDistanceCalculator<'a, T>
{
    fn split_distance(&self, dim: usize, split_val: T, query_val: T) -> T {
        let diff = (query_val - split_val).abs();
        match &self.metric {
            DistanceMetric::Normalized => diff * self.scales[dim],
            DistanceMetric::Euclidean => diff,
            DistanceMetric::Manhattan => diff,
            DistanceMetric::Chebyshev => diff,
            DistanceMetric::Minkowski(_) => diff,
            DistanceMetric::Weighted(w) => diff * w[dim].sqrt(),
        }
    }

    fn distance_squared(&self, a: &[T], b: &[T]) -> T {
        match &self.metric {
            DistanceMetric::Normalized => DistanceMetric::normalized_squared(a, b, self.scales),
            DistanceMetric::Euclidean => DistanceMetric::euclidean_squared(a, b),
            DistanceMetric::Weighted(w) => DistanceMetric::weighted_squared(a, b, w),
            DistanceMetric::Manhattan => DistanceMetric::manhattan_squared(a, b),
            DistanceMetric::Chebyshev => DistanceMetric::chebyshev_squared(a, b),
            DistanceMetric::Minkowski(p) => DistanceMetric::minkowski_squared(a, b, *p),
        }
    }

    fn split_distance_squared(&self, dim: usize, split_val: T, query_val: T) -> T {
        let diff = query_val - split_val;
        match &self.metric {
            DistanceMetric::Normalized => {
                let d = diff * self.scales[dim];
                d * d
            }
            DistanceMetric::Euclidean => diff * diff,
            DistanceMetric::Weighted(w) => diff * diff * w[dim],
            // Squaring the linear split distance |diff| is valid for comparison
            // against squared total distances due to monotonicity.
            _ => {
                let d = self.split_distance(dim, split_val, query_val);
                d * d
            }
        }
    }

    fn post_process_distance(&self, d: T) -> T {
        // Get the linear distance from the squared distance
        d.sqrt()
    }
}

// Mode for surface evaluation.
//
// Controls whether to use interpolation surface (faster, less accurate) or
// direct per-point fitting (slower, more accurate).
use crate::primitives::policies::SurfaceMode;

// Signature for custom smooth pass function
pub type SmoothPassFn<T> = fn(
    &[T],               // x (query points)
    &[T],               // y (associated values)
    &[T],               // x_search (augmented data for neighbor search)
    &[T],               // y_search (augmented data for fitting)
    usize,              // dimensions
    usize,              // window_size
    bool,               // use_robustness
    &[T],               // robustness_weights
    &mut [T],           // output (y_smooth)
    WeightFunction,     // weight_function
    ZeroWeightFallback, // zero_weight_fallback
    PolynomialDegree,   // polynomial_degree
    &DistanceMetric<T>, // distance_metric
    &[T],               // scales (normalization scales per dimension)
    Option<&[T]>,       // custom_weights (per-observation user weights)
);

// Signature for custom cross-validation pass function
pub type CVPassFn<T> = fn(
    &[T],            // x
    &[T],            // y
    &[T],            // candidate fractions
    CVKind,          // CV strategy
    &LoessConfig<T>, // Config for internal fits
) -> (T, Vec<T>); // (best_fraction, scores)

// Signature for custom interval estimation pass function
pub type IntervalPassFn<T> = fn(
    &[T],               // x (query points)
    &[T],               // y (associated values)
    &[T],               // x_search (augmented data for neighbor search)
    &[T],               // y_search (augmented data for fitting)
    &[T],               // y_smooth
    usize,              // dimensions
    usize,              // window_size
    &[T],               // robustness_weights
    WeightFunction,     // weight_function
    &IntervalMethod<T>, // interval configuration
    PolynomialDegree,   // polynomial_degree
    &DistanceMetric<T>, // distance_metric
    &[T],               // scales (normalization scales per dimension)
    Option<&[T]>,
) -> Vec<T>; // standard errors

// Signature for custom iteration batch pass function.
pub type FitPassFn<T> = fn(
    &[T],            // x
    &[T],            // y
    &LoessConfig<T>, // full configuration
) -> (
    Vec<T>,         // smoothed
    Option<Vec<T>>, // std_errors
    usize,          // iterations
    Vec<T>,         // robustness_weights
);

use crate::algorithms::interpolation::VertexPassFn;

// Signature for custom KD-tree builder function.
pub type KDTreeBuilderFn<T> = fn(points: &[T], dims: usize) -> KDTree<T>;

// Signature for custom gradient pass function (Direct mode only).
pub type GradientPassFn<T> = fn(
    &[T],               // x (query points)
    &[T],               // x_search (augmented data for neighbor search)
    &[T],               // y_search (augmented data for fitting)
    usize,              // dimensions
    usize,              // window_size
    &[T],               // robustness_weights (final, converged values)
    WeightFunction,     // weight_function
    ZeroWeightFallback, // zero_weight_fallback
    PolynomialDegree,   // polynomial_degree
    &DistanceMetric<T>, // distance_metric
    &[T],               // scales (normalization scales per dimension)
    Option<&[T]>,       // custom_weights (per-observation user weights)
) -> Vec<T>; // flattened per-point gradient (n * dimensions)

#[derive(Debug)]
pub(crate) struct RetainedBootstrapFit<T: FloatLinalg + SolverLinalg> {
    pub x: Vec<T>,
    pub smoothed: Vec<T>,
    pub residuals: Vec<T>,
    pub config: LoessConfig<T>,
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync + 'static>
    BootstrapPredictor<T> for RetainedBootstrapFit<T>
{
    fn original_residuals(&self) -> Option<&[T]> {
        Some(&self.residuals)
    }

    fn compute(
        &self,
        bootstrap: BootstrapConfig,
        method: &IntervalMethod<T>,
        n_output: usize,
        predict: &mut BootstrapPredictionFn<'_, T>,
    ) -> Result<BootstrapOutput<T>, crate::primitives::errors::LoessError> {
        let mut config = self.config.clone();
        config.cv_fractions = None;
        config.cv_kind = None;
        config.return_variance = None;
        config.return_gradient = false;
        config.retain_model = true;
        bootstrap.compute_at(method, &self.smoothed, &self.residuals, n_output, |batch| {
            batch
                .iter()
                .map(|response| {
                    let fitted = LoessExecutor::run_with_config(&self.x, response, config.clone());
                    let state = fitted
                        .predict_state
                        .ok_or(crate::primitives::errors::LoessError::PredictionUnavailable)?;
                    predict(&state)
                })
                .collect()
        })
    }
}

// Output from LOESS execution.
#[derive(Debug, Clone)]
pub struct ExecutorOutput<T: FloatLinalg> {
    // Smoothed y-values.
    pub smoothed: Vec<T>,

    // Standard errors (if SE estimation or intervals were requested).
    pub std_errors: Option<Vec<T>>,

    // Number of iterations performed (if auto-convergence was active).
    pub iterations: Option<usize>,

    // Smoothing fraction used (selected by CV or configured).
    pub used_fraction: T,

    // RMSE scores for each tested fraction (if CV was performed).
    pub cv_scores: Option<Vec<T>>,

    // Final robustness weights from iterative refinement.
    pub robustness_weights: Vec<T>,

    // Leverage values (hat matrix diagonal) for each point.
    // Only computed when intervals are requested.
    pub leverage: Option<Vec<T>>,

    // Per-point local fit gradient (flattened, `dimensions` values per point), if
    // `return_gradient` was set. Only computed in `SurfaceMode::Direct`.
    pub gradient: Option<Vec<T>>,

    // Retained fitted-model state for `Predict::call()`, if `retain_model` was set.
    pub predict_state: Option<Arc<PredictState<T>>>,
}

// Configuration for LOESS execution.
#[derive(Debug, Clone)]
pub struct LoessConfig<T: FloatLinalg + SolverLinalg> {
    // Smoothing fraction (0, 1].
    // If `None` and `cv_fractions` are provided, bandwidth selection is performed.
    pub fraction: Option<T>,

    // Number of robustness iterations (0 means initial fit only).
    pub iterations: usize,

    // Kernel weight function used for local regression.
    pub weight_function: WeightFunction,

    // Zero-weight fallback policy.
    pub zero_weight_fallback: ZeroWeightFallback,

    // Robustness weighting method for outlier downweighting.
    pub robustness_method: RobustnessMethod,

    // Residual scaling method (MAR or MAD).
    pub scaling_method: ScalingMethod,

    // Candidate fractions to evaluate during cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    // Cross-validation strategy (e.g., K-Fold or LOOCV).
    pub cv_kind: Option<CVKind>,

    // Seed for random number generation in cross-validation.
    pub cv_seed: Option<u64>,

    // Convergence tolerance for early stopping of robustness iterations.
    pub auto_converge: Option<T>,

    // Configuration for standard errors and intervals.
    pub return_variance: Option<IntervalMethod<T>>,

    // Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    // Polynomial degree for local regression (0=constant, 1=linear, 2=quadratic).
    pub polynomial_degree: PolynomialDegree,

    // Number of predictor dimensions (default: 1).
    pub dimensions: usize,

    // Distance metric for nD neighborhood computation.
    pub distance_metric: DistanceMetric<T>,

    // Surface evaluation mode (Interpolation or Direct).
    pub surface_mode: SurfaceMode,

    // Maximum number of vertices for the interpolation surface.
    pub interpolation_vertices: Option<usize>,

    // Cell size as a fraction of the smoothing span (default: 0.2).
    // Used to determine subdivision when `surface_mode` is `Interpolation`.
    pub cell: Option<f64>,

    // Whether to reduce polynomial degree to Linear at boundary vertices during interpolation.
    // When `true` (default), vertices outside the tight data bounds use Linear fits to avoid
    // unstable extrapolation. Set to `false` to match R's loess behavior exactly.
    pub boundary_degree_fallback: bool,

    // User-defined case weights (one per observation).
    // When provided, each weight multiplies the kernel weight in the local WLS:
    // `w_ij = custom_weights[j] * K(d_ij / h) * robustness_j`.
    // Must have the same length as `y`. Only supported for Batch mode.
    pub custom_weights: Option<Vec<T>>,

    // Retain the fitted model's training data/weights, enabling `Predict::call()`.
    // Off by default (no extra memory/clone cost unless requested). Only supported for Batch mode.
    pub retain_model: bool,

    // Include the per-point local fit gradient (flattened, `dimensions` values per point) in
    // the output. Only computed in `SurfaceMode::Direct`; `None` in `SurfaceMode::Interpolation`.
    // Only supported for Batch mode.
    pub return_gradient: bool,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    // Custom smooth pass function (enables parallel execution).
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    // Custom cross-validation pass function.
    pub custom_cv_pass: Option<CVPassFn<T>>,

    // Custom interval estimation pass function.
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    // Custom gradient pass function (Direct mode only).
    pub custom_gradient_pass: Option<GradientPassFn<T>>,

    // Custom iteration batch pass function.
    pub custom_fit_pass: Option<FitPassFn<T>>,

    // Custom vertex pass function (Interpolation mode).
    pub custom_vertex_pass: Option<VertexPassFn<T>>,

    // Custom KD-tree builder function.
    pub custom_kdtree_builder: Option<KDTreeBuilderFn<T>>,

    // Execution backend hint for extension crates.
    pub backend: Option<Backend>,

    // Whether to use parallel execution
    pub parallel: bool,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + SolverLinalg> Default
    for LoessConfig<T>
{
    fn default() -> Self {
        Self {
            fraction: None,
            iterations: DEFAULT_ITERATIONS,
            weight_function: DEFAULT_WEIGHT_FUNCTION_ENUM,
            zero_weight_fallback: DEFAULT_ZERO_WEIGHT_FALLBACK_ENUM,
            robustness_method: DEFAULT_ROBUSTNESS_METHOD_ENUM,
            scaling_method: DEFAULT_SCALING_METHOD_ENUM,
            cv_fractions: None,
            cv_kind: None,
            cv_seed: None,
            auto_converge: None,
            return_variance: None,
            boundary_policy: DEFAULT_BOUNDARY_POLICY_ENUM,
            polynomial_degree: DEFAULT_POLYNOMIAL_DEGREE_ENUM,
            dimensions: DEFAULT_DIMENSIONS,
            distance_metric: default_distance_metric(),
            surface_mode: DEFAULT_SURFACE_MODE_ENUM,
            interpolation_vertices: None,
            cell: None,
            boundary_degree_fallback: DEFAULT_BOUNDARY_DEGREE_FALLBACK,
            custom_weights: None,
            retain_model: false,
            return_gradient: DEFAULT_RETURN_GRADIENT,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_gradient_pass: None,
            custom_fit_pass: None,
            custom_vertex_pass: None,
            custom_kdtree_builder: None,
            parallel: false,
            backend: None,
        }
    }
}

// Unified executor for LOESS smoothing operations.
#[derive(Debug, Clone)]
pub struct LoessExecutor<T: FloatLinalg + SolverLinalg> {
    // Smoothing fraction (0, 1].
    pub fraction: T,

    // Number of robustness iterations.
    pub iterations: usize,

    // Kernel weight function.
    pub weight_function: WeightFunction,

    // Zero weight fallback flag.
    pub zero_weight_fallback: ZeroWeightFallback,

    // Robustness method for iterative refinement.
    pub robustness_method: RobustnessMethod,

    // Residual scaling method.
    pub scaling_method: ScalingMethod,

    // Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    // Polynomial degree for local regression.
    pub polynomial_degree: PolynomialDegree,

    // Number of predictor dimensions.
    pub dimensions: usize,

    // Distance metric for nD neighborhood computation.
    pub distance_metric: DistanceMetric<T>,

    // Surface evaluation mode (Interpolation or Direct).
    pub surface_mode: SurfaceMode,

    // Maximum number of vertices for interpolation.
    pub interpolation_vertices: Option<usize>,

    // Cell size for interpolation subdivision.
    pub cell: Option<f64>,

    // Whether to reduce polynomial degree to Linear at boundary vertices during interpolation.
    pub boundary_degree_fallback: bool,

    // User-defined case weights (one per observation). See `LoessConfig::custom_weights`.
    pub custom_weights: Option<Vec<T>>,

    // Retain the fitted model's training data/weights, enabling `Predict::call()`.
    pub retain_model: bool,

    // Include the per-point local fit gradient in the output (Direct mode only).
    pub return_gradient: bool,

    // ++++++++++++++++++++++++++++++++++++++
    // +               DEV                  +
    // ++++++++++++++++++++++++++++++++++++++
    // Custom smooth pass function (e.g., for parallel execution).
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    // Custom cross-validation pass function.
    pub custom_cv_pass: Option<CVPassFn<T>>,

    // Custom interval estimation pass function.
    pub custom_interval_pass: Option<IntervalPassFn<T>>,

    // Custom gradient pass function (Direct mode only).
    pub custom_gradient_pass: Option<GradientPassFn<T>>,

    // Custom iteration batch pass function.
    pub custom_fit_pass: Option<FitPassFn<T>>,

    // Custom vertex pass function (Interpolation mode).
    pub custom_vertex_pass: Option<VertexPassFn<T>>,

    // Custom KD-tree builder function.
    pub custom_kdtree_builder: Option<KDTreeBuilderFn<T>>,

    // Execution backend hint for extension crates.
    pub backend: Option<Backend>,

    // Whether to use parallel execution
    pub parallel: bool,
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + 'static + SolverLinalg> Default
    for LoessExecutor<T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatLinalg + DistanceLinalg + Debug + Send + Sync + 'static + SolverLinalg>
    LoessExecutor<T>
{
    pub(crate) fn bootstrap_fit(
        x: &[T],
        y: &[T],
        smoothed: &[T],
        mut config: LoessConfig<T>,
        bootstrap: BootstrapConfig,
        method: &IntervalMethod<T>,
    ) -> Result<BootstrapOutput<T>, crate::primitives::errors::LoessError> {
        config.cv_fractions = None;
        config.cv_kind = None;
        config.return_variance = None;
        config.retain_model = false;
        config.return_gradient = false;
        let residuals: Vec<T> = y
            .iter()
            .zip(smoothed)
            .map(|(&value, &fit)| value - fit)
            .collect();
        bootstrap.compute(method, smoothed, &residuals, |batch| {
            Ok(batch
                .iter()
                .map(|response| Self::run_with_config(x, response, config.clone()).smoothed)
                .collect())
        })
    }

    // Create a new executor with default parameters.
    pub fn new() -> Self {
        Self {
            fraction: T::from(DEFAULT_FRACTION).unwrap_or_else(|| T::from(0.5).unwrap()),
            iterations: DEFAULT_ITERATIONS,
            weight_function: DEFAULT_WEIGHT_FUNCTION_ENUM,
            zero_weight_fallback: DEFAULT_ZERO_WEIGHT_FALLBACK_ENUM,
            robustness_method: DEFAULT_ROBUSTNESS_METHOD_ENUM,
            scaling_method: DEFAULT_SCALING_METHOD_ENUM,
            boundary_policy: DEFAULT_BOUNDARY_POLICY_ENUM,
            polynomial_degree: DEFAULT_POLYNOMIAL_DEGREE_ENUM,
            dimensions: DEFAULT_DIMENSIONS,
            distance_metric: default_distance_metric(),
            surface_mode: DEFAULT_SURFACE_MODE_ENUM,
            interpolation_vertices: None,
            cell: None,
            boundary_degree_fallback: DEFAULT_BOUNDARY_DEGREE_FALLBACK,
            custom_weights: None,
            retain_model: false,
            return_gradient: DEFAULT_RETURN_GRADIENT,
            custom_smooth_pass: None,
            custom_cv_pass: None,
            custom_interval_pass: None,
            custom_gradient_pass: None,
            custom_fit_pass: None,
            custom_vertex_pass: None,
            custom_kdtree_builder: None,
            parallel: false,
            backend: None,
        }
    }

    // Create a new executor from a `LoessConfig`.
    pub fn from_config(config: &LoessConfig<T>) -> Self {
        let default_frac = T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap());
        let mut exec = Self::new()
            .fraction(config.fraction.unwrap_or(default_frac))
            .iterations(config.iterations)
            .weight_function(config.weight_function)
            .zero_weight_fallback(config.zero_weight_fallback)
            .robustness_method(config.robustness_method)
            .scaling_method(config.scaling_method)
            .boundary_policy(config.boundary_policy)
            .polynomial_degree(config.polynomial_degree)
            .dimensions(config.dimensions)
            .distance_metric(config.distance_metric.clone())
            .surface_mode(config.surface_mode)
            .interpolation_vertices(config.interpolation_vertices)
            .cell(config.cell)
            .boundary_degree_fallback(config.boundary_degree_fallback)
            .retain_model(config.retain_model)
            .return_gradient(config.return_gradient)
            // ++++++++++++++++++++++++++++++++++++++
            // +               DEV                  +
            // ++++++++++++++++++++++++++++++++++++++
            .custom_smooth_pass(config.custom_smooth_pass)
            .custom_cv_pass(config.custom_cv_pass)
            .custom_interval_pass(config.custom_interval_pass)
            .custom_gradient_pass(config.custom_gradient_pass)
            .custom_fit_pass(config.custom_fit_pass)
            .custom_vertex_pass(config.custom_vertex_pass)
            .custom_kdtree_builder(config.custom_kdtree_builder)
            .parallel(config.parallel)
            .backend(config.backend);
        if let Some(cw) = config.custom_weights.clone() {
            exec = exec.custom_weights(cw);
        }
        exec
    }

    // Set the smoothing fraction (bandwidth).
    pub fn fraction(mut self, frac: T) -> Self {
        self.fraction = frac;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, niter: usize) -> Self {
        self.iterations = niter;
        self
    }

    // Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    // Set the zero weight fallback policy flag.
    pub fn zero_weight_fallback(mut self, flag: ZeroWeightFallback) -> Self {
        self.zero_weight_fallback = flag;
        self
    }

    // Set the robustness method for iterative refinement.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    // Set the residual scaling method (MAR/MAD).
    pub fn scaling_method(mut self, method: ScalingMethod) -> Self {
        self.scaling_method = method;
        self
    }

    // Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    // Set the polynomial degree for local regression.
    pub fn polynomial_degree(mut self, degree: PolynomialDegree) -> Self {
        self.polynomial_degree = degree;
        self
    }

    // Set the number of predictor dimensions.
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    // Set the distance metric for nD neighborhood computation.
    pub fn distance_metric(mut self, metric: DistanceMetric<T>) -> Self {
        self.distance_metric = metric;
        self
    }

    // Set the surface evaluation mode (Interpolation or Direct).
    pub fn surface_mode(mut self, mode: SurfaceMode) -> Self {
        self.surface_mode = mode;
        self
    }

    // Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: Option<usize>) -> Self {
        self.interpolation_vertices = vertices;
        self
    }

    // Set the interpolation cell size.
    pub fn cell(mut self, cell: Option<f64>) -> Self {
        self.cell = cell;
        self
    }

    // Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.boundary_degree_fallback = enabled;
        self
    }

    // Set User-defined case weights (one per observation).
    pub fn custom_weights(mut self, weights: Vec<T>) -> Self {
        self.custom_weights = Some(weights);
        self
    }

    // Set whether to retain the fitted model's training data, enabling `Predict::call()`.
    pub fn retain_model(mut self, retain: bool) -> Self {
        self.retain_model = retain;
        self
    }

    // Set whether to include the per-point local fit gradient in the output.
    pub fn return_gradient(mut self, return_gradient: bool) -> Self {
        self.return_gradient = return_gradient;
        self
    }

    pub fn custom_smooth_pass(mut self, smooth_pass_fn: Option<SmoothPassFn<T>>) -> Self {
        self.custom_smooth_pass = smooth_pass_fn;
        self
    }

    // Set a custom cross-validation pass function.
    pub fn custom_cv_pass(mut self, cv_pass_fn: Option<CVPassFn<T>>) -> Self {
        self.custom_cv_pass = cv_pass_fn;
        self
    }

    // Set a custom interval estimation pass function.
    pub fn custom_interval_pass(mut self, interval_pass_fn: Option<IntervalPassFn<T>>) -> Self {
        self.custom_interval_pass = interval_pass_fn;
        self
    }

    // Set a custom vertex pass function (Interpolation mode).
    pub fn custom_vertex_pass(mut self, vertex_pass_fn: Option<VertexPassFn<T>>) -> Self {
        self.custom_vertex_pass = vertex_pass_fn;
        self
    }

    // Set a custom gradient pass function (Direct mode only).
    pub fn custom_gradient_pass(mut self, gradient_pass_fn: Option<GradientPassFn<T>>) -> Self {
        self.custom_gradient_pass = gradient_pass_fn;
        self
    }

    // Set a custom iteration batch pass function.
    pub fn custom_fit_pass(mut self, fit_pass_fn: Option<FitPassFn<T>>) -> Self {
        self.custom_fit_pass = fit_pass_fn;
        self
    }

    // Set a custom KD-tree builder function.
    pub fn custom_kdtree_builder(mut self, kdtree_builder_fn: Option<KDTreeBuilderFn<T>>) -> Self {
        self.custom_kdtree_builder = kdtree_builder_fn;
        self
    }

    // Set whether to use parallel execution.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    // Set the execution backend hint.
    pub fn backend(mut self, backend: Option<Backend>) -> Self {
        self.backend = backend;
        self
    }

    // Smooth data using a `LoessConfig` payload.
    pub fn run_with_config(x: &[T], y: &[T], config: LoessConfig<T>) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let executor = LoessExecutor::from_config(&config);
        let dims = executor.dimensions;
        let n = x.len() / dims;
        let eff_fraction = config.fraction.unwrap_or(executor.fraction);
        let window_size = Window::calculate_span(n, eff_fraction);
        let n_coeffs = executor.polynomial_degree.num_coefficients_nd(dims);

        // Create a workspace to be reused across CV and final fit
        let mut workspace =
            LoessBuffer::<T, NodeDistance<T>, Neighborhood<T>>::new(n, dims, window_size, n_coeffs);

        // Handle cross-validation if configured
        if let Some(ref cv_fracs) = config.cv_fractions {
            let cv_kind = config.cv_kind.unwrap_or(CVKind::KFold(5));

            // Run CV to find best fraction
            let (best_frac, scores) = if let Some(callback) = config.custom_cv_pass {
                callback(x, y, cv_fracs, cv_kind, &config)
            } else {
                let LoessBuffer {
                    ref mut cv_buffer, ..
                } = workspace;
                executor.cross_validate_with_options(
                    x,
                    y,
                    cv_fracs,
                    CVRunOptions {
                        kind: cv_kind,
                        seed: config.cv_seed,
                        tolerance: config.auto_converge,
                    },
                    cv_buffer,
                )
            };

            // Run final pass with best fraction
            let mut output = executor.run(
                x,
                y,
                Some(best_frac),
                Some(config.iterations),
                config.auto_converge,
                config.return_variance.as_ref(),
                Some(&mut workspace),
            );
            output.cv_scores = Some(scores);
            output.used_fraction = best_frac;
            output
        } else {
            // Direct run (no CV)
            executor.run(
                x,
                y,
                config.fraction,
                Some(config.iterations),
                config.auto_converge,
                config.return_variance.as_ref(),
                Some(&mut workspace),
            )
        }
    }

    pub fn cross_validate_with_options(
        &self,
        x: &[T],
        y: &[T],
        fractions: &[T],
        options: CVRunOptions<T>,
        buffer: &mut CVBuffer<T>,
    ) -> (T, Vec<T>) {
        let executor = self.clone().retain_model(false).return_gradient(false);
        let subset_weights = |indices: &[usize]| {
            self.custom_weights.as_ref().map(|weights| {
                indices
                    .iter()
                    .map(|&index| weights[index])
                    .collect::<Vec<_>>()
            })
        };
        let predictor = if self.dimensions > 1 {
            Some(
                |training_x: &[T],
                 training_y: &[T],
                 indices: &[usize],
                 query_x: &[T],
                 fraction: T| {
                    let mut subset_executor = executor.clone();
                    subset_executor.custom_weights = subset_weights(indices);
                    // Retain the actual fold fit so held-out predictions use its final
                    // robustness weights, boundary padding, and interpolation surface.
                    let fold_output = subset_executor.clone().retain_model(true).run(
                        training_x,
                        training_y,
                        Some(fraction),
                        None,
                        options.tolerance,
                        None,
                        None,
                    );
                    let state = fold_output
                        .predict_state
                        .expect("retained CV fold fit must produce prediction state");
                    let dimensions = state.dimensions;
                    subset_executor.custom_weights = state.custom_weights.clone();

                    let mut predictions = Vec::with_capacity(query_x.len() / dimensions);
                    for query_point in query_x.chunks_exact(dimensions) {
                        let mut eval_point = query_point.to_vec();
                        let mut out_of_range = false;
                        for dimension in 0..dimensions {
                            if query_point[dimension] < state.train_min[dimension] {
                                eval_point[dimension] = state.train_min[dimension];
                                out_of_range = true;
                            } else if query_point[dimension] > state.train_max[dimension] {
                                eval_point[dimension] = state.train_max[dimension];
                                out_of_range = true;
                            }
                        }

                        if !out_of_range && let Some(surface) = state.surface.as_ref() {
                            predictions.push(surface.evaluate(query_point));
                        } else {
                            let prediction = subset_executor.predict(
                                &state.x,
                                &state.y,
                                &state.robustness_weights,
                                &eval_point,
                                state.window_size,
                                &state.scales,
                                &state.kdtree,
                            );
                            predictions.push(prediction.first().copied().unwrap_or(T::zero()));
                        }
                    }
                    predictions
                },
            )
        } else {
            None
        };
        options.kind.run_with_indices(
            x,
            y,
            self.dimensions,
            fractions,
            options.seed,
            |training_x, training_y, indices, fraction| {
                let mut subset_executor = executor.clone();
                subset_executor.custom_weights = subset_weights(indices);
                subset_executor
                    .run(
                        training_x,
                        training_y,
                        Some(fraction),
                        None,
                        options.tolerance,
                        None,
                        None,
                    )
                    .smoothed
            },
            predictor,
            buffer,
        )
    }

    // Execute smoothing with explicit overrides for specific parameters.
    //
    // Uses interpolation surface for efficient evaluation - fits only at
    // cell vertices and interpolates for all other points.
    //
    // # Special Cases
    //
    // * **Insufficient data** (n < 2): Returns original y-values.
    // * **Global regression** (fraction >= 1.0): Performs OLS on the entire dataset.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &[T],
        y: &[T],
        fraction: Option<T>,
        max_iter: Option<usize>,
        tolerance: Option<T>,
        confidence_method: Option<&IntervalMethod<T>>,
        workspace: Option<&mut LoessBuffer<T, NodeDistance<T>, Neighborhood<T>>>,
    ) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let dims = self.dimensions;
        let n = x.len() / dims;
        let eff_fraction = fraction.unwrap_or(self.fraction);

        // Calculate window size
        let window_size = Window::calculate_span(n, eff_fraction);
        let target_iterations = max_iter.unwrap_or(self.iterations);
        let n_coeffs = self.polynomial_degree.num_coefficients_nd(dims);

        // Apply boundary policy (unified)
        let (ax, ay, mapping) = self.boundary_policy.apply(x, y, dims, window_size);
        let n_total = ay.len();
        let is_augmented = n_total > n;

        // Expand custom weights to the augmented (boundary-padded) data space.
        // Boundary points are assigned the weight of their nearest original point.
        let custom_weights_aug: Option<Vec<T>> = self
            .custom_weights
            .as_ref()
            .map(|uw| mapping.iter().map(|&orig_idx| uw[orig_idx]).collect());

        let mut new_workspace;
        let workspace = if let Some(ws) = workspace {
            ws.ensure_capacity(n_total, dims, window_size, n_coeffs);
            ws
        } else {
            new_workspace = LoessBuffer::<T, NodeDistance<T>, Neighborhood<T>>::new(
                n_total,
                dims,
                window_size,
                n_coeffs,
            );
            &mut new_workspace
        };

        // R LOESS normalizes multivariate predictors by their trimmed sample SD.
        workspace.executor_buffer.ensure_capacity(n_total, dims);
        let scales_local = loess_normalization_scales(x, n, dims);
        workspace.executor_buffer.scales.resize(dims, T::one());
        workspace
            .executor_buffer
            .scales
            .copy_from_slice(&scales_local);

        // Build KD-Tree for efficient kNN
        let kdtree = if let Some(builder) = self.custom_kdtree_builder {
            builder(&ax, dims)
        } else {
            KDTree::new(&ax, dims)
        };

        // Define distance calculator
        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales: &scales_local,
        };

        // Resolution First: no default limit unless explicitly provided
        let max_vertices = self.interpolation_vertices.unwrap_or(usize::MAX);

        let n_coeffs = self.polynomial_degree.num_coefficients_nd(dims);
        workspace.ensure_capacity(n_total, dims, window_size, n_coeffs);

        let mut y_smooth = vec![T::zero(); n];
        workspace
            .executor_buffer
            .robustness_weights
            .resize(n_total, T::one());
        workspace.executor_buffer.residuals.resize(n, T::zero());

        // For interpolation mode: build surface once before iterations
        let mut _surface_opt: Option<InterpolationSurface<T>> = None;
        if self.surface_mode == SurfaceMode::Interpolation {
            let fitter = |vertex: &[T],
                          neighborhood: &Neighborhood<T>,
                          fb: &mut FittingBuffer<T>,
                          degree: PolynomialDegree| {
                let mut context = RegressionContext::new(
                    &ax,
                    dims,
                    &ay,
                    0, // query_idx is not used when query_point is Some
                    Some(vertex),
                    neighborhood,
                    false, // use_robustness
                    &workspace.executor_buffer.robustness_weights,
                    self.weight_function,
                    self.zero_weight_fallback,
                    degree,
                    false, // compute_leverage
                    Some(fb),
                );
                if let Some(ref uw) = custom_weights_aug {
                    context = context.with_custom_weights(uw);
                }
                context.fit_with_coefficients()
            };

            let cell_fraction = T::from(self.cell.unwrap_or(DEFAULT_CELL))
                .unwrap_or_else(|| T::from(DEFAULT_CELL).unwrap());

            let surface = InterpolationSurface::build(
                &ax,
                &ay,
                dims,
                eff_fraction,
                window_size,
                &dist_calc,
                &kdtree,
                max_vertices,
                fitter,
                &mut workspace.search_buffer,
                &mut workspace.neighborhood,
                &mut workspace.fitting_buffer,
                cell_fraction,
                self.custom_vertex_pass,
                &scales_local,
                self.weight_function,
                self.zero_weight_fallback,
                self.polynomial_degree,
                &self.distance_metric,
                self.boundary_degree_fallback,
                custom_weights_aug.as_deref(),
            );
            _surface_opt = Some(surface);

            // Re-evaluate surface at all original points
            let surface = _surface_opt.as_ref().unwrap();
            for (i, val) in y_smooth.iter_mut().enumerate().take(n) {
                let query_offset = i * dims;
                // Use original x (not augmented) for query points
                let query_point = &x[query_offset..query_offset + dims];
                *val = surface.evaluate(query_point);
            }
        } else {
            // Direct mode: initial fit - populate neighborhood cache
            workspace.executor_buffer.neighborhood_cache.entries.clear();

            // Check for custom smooth pass callback
            if let Some(callback) = self.custom_smooth_pass {
                // Use custom parallel/accelerated implementation
                callback(
                    x,
                    y,
                    &ax,
                    &ay,
                    dims,
                    window_size,
                    false, // use_robustness (first pass)
                    &workspace.executor_buffer.robustness_weights,
                    &mut y_smooth,
                    self.weight_function,
                    self.zero_weight_fallback,
                    self.polynomial_degree,
                    &self.distance_metric,
                    &scales_local,
                    custom_weights_aug.as_deref(),
                );
                // Mark cache as invalid since we bypassed the internal method
                workspace.executor_buffer.neighborhood_cache.is_valid = false;
            } else {
                self.smooth_pass(
                    &ax,
                    &ay,
                    x, // x_query
                    y, // y_query
                    window_size,
                    &workspace.executor_buffer.robustness_weights,
                    false,
                    &scales_local,
                    &mut y_smooth,
                    n,
                    &kdtree,
                    &mut workspace.search_buffer,
                    &mut workspace.neighborhood,
                    &mut workspace.fitting_buffer,
                    None, // No leverage collection during initial fit
                    Some(&mut workspace.executor_buffer.neighborhood_cache.entries), // Populate cache
                    None, // Not using cache yet
                    custom_weights_aug.as_deref(),
                    None, // No gradient collection during initial fit
                );
                workspace.executor_buffer.neighborhood_cache.is_valid = true;
            }
        }

        let mut iterations_performed = 1;

        // Robustness iteration loop
        for iter in 1..target_iterations {
            iterations_performed = iter + 1;

            // Update robustness weights based on residuals from previous pass
            T::batch_abs_residuals(
                &y[..n],
                &y_smooth[..n],
                &mut workspace.executor_buffer.residuals[..n],
            );

            // 1. Compute new weights using the unified robustness method
            // This handles scaling (MAR/MAD) and weight function application
            let n_res = n; // length of residuals
            let mut new_weights = vec![T::zero(); n];

            // Re-use sorted_residuals as scratch space for median computation
            workspace
                .executor_buffer
                .sorted_residuals
                .resize(n_res, T::zero());

            let stop_robustness = self.robustness_method.apply_robustness_weights(
                &workspace.executor_buffer.residuals[..n_res],
                &mut new_weights,
                self.scaling_method,
                &mut workspace.executor_buffer.sorted_residuals,
            );
            if stop_robustness {
                break;
            }

            // 2. Sync to robustness_weights and check convergence
            let mut max_change = T::zero();

            if is_augmented {
                for (aug_idx, &orig_idx) in mapping.iter().enumerate() {
                    let old_w = workspace.executor_buffer.robustness_weights[aug_idx];
                    let new_w = new_weights[orig_idx];
                    workspace.executor_buffer.robustness_weights[aug_idx] = new_w;

                    let change = (new_w - old_w).abs();
                    if change > max_change {
                        max_change = change;
                    }
                }
            } else {
                for (i, &new_w) in new_weights.iter().enumerate().take(n) {
                    let old_w = workspace.executor_buffer.robustness_weights[i];
                    workspace.executor_buffer.robustness_weights[i] = new_w;

                    let change = (new_w - old_w).abs();
                    if change > max_change {
                        max_change = change;
                    }
                }
            }

            if let Some(tol) = tolerance
                && max_change < tol
            {
                break;
            }

            // Re-fit with new robustness weights
            match self.surface_mode {
                SurfaceMode::Interpolation => {
                    // Refit vertex values using existing surface structure
                    if let Some(ref mut surface) = _surface_opt {
                        let fitter =
                            |vertex: &[T],
                             neighborhood: &Neighborhood<T>,
                             fb: &mut FittingBuffer<T>,
                             degree: PolynomialDegree| {
                                let mut context = RegressionContext::new(
                                    &ax,
                                    dims,
                                    &ay,
                                    0, // query_idx is not used when query_point is Some
                                    Some(vertex),
                                    neighborhood,
                                    true, // use_robustness
                                    &workspace.executor_buffer.robustness_weights,
                                    self.weight_function,
                                    self.zero_weight_fallback,
                                    degree,
                                    false, // compute_leverage
                                    Some(fb),
                                );
                                if let Some(ref uw) = custom_weights_aug {
                                    context = context.with_custom_weights(uw);
                                }
                                context.fit_with_coefficients()
                            };

                        surface.refit_values(
                            &ax,
                            &ay,
                            fitter,
                            &mut workspace.neighborhood,
                            &mut workspace.fitting_buffer,
                            self.custom_vertex_pass,
                            self.weight_function,
                            self.zero_weight_fallback,
                            self.polynomial_degree,
                            &self.distance_metric,
                            &scales_local,
                            &workspace.executor_buffer.robustness_weights,
                            self.boundary_degree_fallback,
                            custom_weights_aug.as_deref(),
                        );

                        // Re-evaluate at data points using original x (not augmented)
                        for (i, val) in y_smooth.iter_mut().enumerate().take(n) {
                            let query_offset = i * dims;
                            let query_point = &x[query_offset..query_offset + dims];
                            *val = surface.evaluate(query_point);
                        }
                    }
                }
                SurfaceMode::Direct => {
                    // Check for custom smooth pass callback
                    if let Some(callback) = self.custom_smooth_pass {
                        // Use custom parallel/accelerated implementation
                        callback(
                            x,
                            y,
                            &ax,
                            &ay,
                            dims,
                            window_size,
                            true, // use_robustness
                            &workspace.executor_buffer.robustness_weights,
                            &mut y_smooth,
                            self.weight_function,
                            self.zero_weight_fallback,
                            self.polynomial_degree,
                            &self.distance_metric,
                            &scales_local,
                            custom_weights_aug.as_deref(),
                        );
                    } else {
                        // Use cached neighborhoods to skip KD-tree searches
                        let cache_ref = if workspace.executor_buffer.neighborhood_cache.is_valid {
                            Some(
                                workspace
                                    .executor_buffer
                                    .neighborhood_cache
                                    .entries
                                    .as_slice(),
                            )
                        } else {
                            None
                        };
                        self.smooth_pass(
                            &ax,
                            &ay,
                            x, // x_query
                            y, // y_query
                            window_size,
                            &workspace.executor_buffer.robustness_weights,
                            true,
                            &scales_local,
                            &mut y_smooth,
                            n,
                            &kdtree,
                            &mut workspace.search_buffer,
                            &mut workspace.neighborhood,
                            &mut workspace.fitting_buffer,
                            None,      // No leverage collection during robustness iterations
                            None,      // Not populating cache
                            cache_ref, // Use cached neighborhoods
                            custom_weights_aug.as_deref(),
                            None, // No gradient collection during robustness iterations
                        );
                    }
                }
            }
        }

        // Collect leverage and/or gradient values when requested. Both come from the same
        // per-point WLS solve `fit()` already performs (leverage via a second `solve_normal`
        // against the fit's own normal equations, gradient via `RegressionContext`'s
        // `gradient_out` side-channel) - requesting them together runs a SINGLE combined
        // pass instead of two separate re-fits. Only supported in `SurfaceMode::Direct`:
        // the interpolation surface only stores value(+gradient) at sparse vertices, not
        // enough to reconstruct an exact per-point leverage/gradient without re-deriving
        // the Hermite interpolant's own derivative, so both stay `None` under
        // `SurfaceMode::Interpolation`.
        let need_leverage = confidence_method.is_some() && self.surface_mode == SurfaceMode::Direct;
        let need_gradient = self.return_gradient && self.surface_mode == SurfaceMode::Direct;

        let (leverage_values, gradient_values) = if need_leverage || need_gradient {
            let cache_ref = if workspace.executor_buffer.neighborhood_cache.is_valid {
                Some(
                    workspace
                        .executor_buffer
                        .neighborhood_cache
                        .entries
                        .as_slice(),
                )
            } else {
                None
            };

            if need_gradient && let Some(callback) = self.custom_gradient_pass {
                // Custom gradient callback runs its own pass; leverage (if also
                // requested) still goes through the built-in combined pass below.
                let leverages = need_leverage.then(|| {
                    let mut leverages = Vec::with_capacity(n);
                    self.smooth_pass(
                        &ax,
                        &ay,
                        x, // x_query
                        y, // y_query
                        window_size,
                        &workspace.executor_buffer.robustness_weights,
                        true,
                        &scales_local,
                        &mut y_smooth,
                        n,
                        &kdtree,
                        &mut workspace.search_buffer,
                        &mut workspace.neighborhood,
                        &mut workspace.fitting_buffer,
                        Some(&mut leverages),
                        None, // Not populating cache
                        cache_ref,
                        custom_weights_aug.as_deref(),
                        None, // Not collecting gradient here (custom callback handles it)
                    );
                    leverages
                });
                let gradient = callback(
                    x,
                    &ax,
                    &ay,
                    dims,
                    window_size,
                    &workspace.executor_buffer.robustness_weights,
                    self.weight_function,
                    self.zero_weight_fallback,
                    self.polynomial_degree,
                    &self.distance_metric,
                    &scales_local,
                    custom_weights_aug.as_deref(),
                );
                (leverages, Some(gradient))
            } else {
                let mut leverages = need_leverage.then(|| Vec::with_capacity(n));
                let mut gradients = need_gradient.then(Vec::new);
                self.smooth_pass(
                    &ax,
                    &ay,
                    x, // x_query
                    y, // y_query
                    window_size,
                    &workspace.executor_buffer.robustness_weights,
                    true,
                    &scales_local,
                    &mut y_smooth,
                    n,
                    &kdtree,
                    &mut workspace.search_buffer,
                    &mut workspace.neighborhood,
                    &mut workspace.fitting_buffer,
                    leverages.as_mut(),
                    None, // Not populating cache
                    cache_ref,
                    custom_weights_aug.as_deref(),
                    gradients.as_mut(),
                );
                (leverages, gradients)
            }
        } else {
            (None, None)
        };

        // Standard errors (now using actual leverage if available)
        let se = if let Some(interval_method) = confidence_method {
            if let Some(callback) = self.custom_interval_pass {
                // Use custom parallel/accelerated implementation
                Some(callback(
                    x,
                    y,
                    &ax,
                    &ay,
                    &y_smooth,
                    dims,
                    window_size,
                    &workspace.executor_buffer.robustness_weights,
                    self.weight_function,
                    interval_method,
                    self.polynomial_degree,
                    &self.distance_metric,
                    &scales_local,
                    custom_weights_aug.as_deref(),
                ))
            } else if dims == 1
                && !is_augmented
                && self.polynomial_degree == PolynomialDegree::Linear
            {
                let mut standard_errors = Vec::with_capacity(n);
                for query_index in 0..n {
                    kdtree.find_kernel_neighborhood(
                        &x[query_index..query_index + 1],
                        window_size,
                        &dist_calc,
                        self.weight_function,
                        &mut workspace.search_buffer,
                        &mut workspace.neighborhood,
                    );
                    let neighborhood = &workspace.neighborhood;
                    let bandwidth = neighborhood.max_distance;
                    let standard_error = if bandwidth <= T::epsilon() {
                        T::zero()
                    } else {
                        IntervalMethod::compute_local_se((0..neighborhood.len()).map(|neighbor| {
                            let index = neighborhood.indices[neighbor];
                            let weight = self
                                .weight_function
                                .compute_weight(neighborhood.distances[neighbor] / bandwidth)
                                * workspace.executor_buffer.robustness_weights[index]
                                * custom_weights_aug
                                    .as_ref()
                                    .map_or(T::one(), |weights| weights[index]);
                            (
                                x[index] - x[query_index],
                                weight,
                                y[index] - y_smooth[index],
                            )
                        }))
                    };
                    standard_errors.push(standard_error);
                }
                Some(standard_errors)
            } else if let Some(ref lev) = leverage_values {
                // Use actual leverage values
                T::batch_abs_residuals(
                    &y[..n],
                    &y_smooth[..n],
                    &mut workspace.executor_buffer.residuals[..n],
                );
                let mut sorted_residuals = workspace.executor_buffer.residuals.clone();
                let median_idx = n / 2;
                if median_idx < sorted_residuals.len() {
                    sorted_residuals.select_nth_unstable_by(median_idx, |a: &T, b| {
                        a.partial_cmp(b).unwrap_or(Equal)
                    });
                }
                let median_residual = sorted_residuals[median_idx];
                let sigma = median_residual * T::from(1.4826).unwrap();

                // SE = sigma * sqrt(leverage)
                let mut se_vec = vec![T::zero(); n];
                T::batch_sqrt_scale(lev, sigma, &mut se_vec);
                Some(se_vec)
            } else {
                // Fallback to approximate leverage (for Interpolation mode).
                // NOTE: This is a coarse heuristic based on the smoothing fraction.
                // It assumes uniform leverage across all points, which is rarely true
                // but provides a stable baseline for standard errors when local
                // hat matrix diagonals are not available.
                T::batch_abs_residuals(
                    &y[..n],
                    &y_smooth[..n],
                    &mut workspace.executor_buffer.residuals[..n],
                );
                let mut sorted_residuals = workspace.executor_buffer.residuals.clone();
                let median_idx = n / 2;
                if median_idx < sorted_residuals.len() {
                    sorted_residuals.select_nth_unstable_by(median_idx, |a: &T, b| {
                        a.partial_cmp(b).unwrap_or(Equal)
                    });
                }
                let median_residual = sorted_residuals[median_idx];
                let sigma = median_residual * T::from(1.4826).unwrap();

                let approx_leverage = eff_fraction / T::from(n).unwrap();
                let se_vec: Vec<T> = (0..n).map(|_| sigma * approx_leverage.sqrt()).collect();
                Some(se_vec)
            }
        } else {
            None
        };

        // Extract robustness weights for original points only
        let final_robustness_weights = if is_augmented {
            let mut rw = vec![T::one(); n];
            for (i, &idx) in mapping.iter().enumerate().take(n_total) {
                if idx < n {
                    rw[idx] = workspace.executor_buffer.robustness_weights[i];
                }
            }
            rw
        } else {
            workspace.executor_buffer.robustness_weights[..n].to_vec()
        };

        let predict_state = self.retain_model.then(|| {
            let mut train_min = x[..dims].to_vec();
            let mut train_max = x[..dims].to_vec();
            for i in 1..n {
                for d in 0..dims {
                    let val = x[i * dims + d];
                    if val < train_min[d] {
                        train_min[d] = val;
                    }
                    if val > train_max[d] {
                        train_max[d] = val;
                    }
                }
            }
            let residuals: Vec<T> = y[..n]
                .iter()
                .zip(y_smooth[..n].iter())
                .map(|(&yi, &fi)| yi - fi)
                .collect();
            let residual_sd = IntervalMethod::calculate_residual_sd(&residuals, None);

            // Mirrors fit()'s own `SurfaceMode::Interpolation` SE fallback exactly (same
            // uncentered median-abs-residual and `eff_fraction/n` approximate leverage), so
            // `predict()` stays self-consistent with whichever mode produced `y` instead of
            // always falling back to a per-point exact-leverage value from an unrelated fit.
            let interpolation_se = {
                let mut abs_residuals: Vec<T> = residuals.iter().map(|r| r.abs()).collect();
                let median_idx = n / 2;
                if median_idx < abs_residuals.len() {
                    abs_residuals.select_nth_unstable_by(median_idx, |a: &T, b| {
                        a.partial_cmp(b).unwrap_or(Equal)
                    });
                }
                let sigma = abs_residuals[median_idx] * T::from(1.4826).unwrap();
                let approx_leverage = eff_fraction / T::from(n).unwrap();
                sigma * approx_leverage.sqrt()
            };

            Arc::new(PredictState {
                x: ax.clone(),
                dimensions: dims,
                y: ay.clone(),
                robustness_weights: workspace.executor_buffer.robustness_weights.to_vec(),
                window_size,
                weight_function: self.weight_function,
                zero_weight_fallback: self.zero_weight_fallback,
                polynomial_degree: self.polynomial_degree,
                distance_metric: self.distance_metric.clone(),
                scales: scales_local.to_vec(),
                custom_weights: custom_weights_aug.clone(),
                residual_sd,
                interpolation_se,
                train_min,
                train_max,
                kdtree: kdtree.clone(),
                // `evaluate()` only needs `cells`/`vertex_data`/`vertices`; drop the much
                // larger per-vertex neighborhood cache (only needed for refitting during
                // `fit()`'s own robustness iterations, already done by this point) so
                // `.retain_model(true)` doesn't pay for it.
                surface: _surface_opt.clone().map(|mut s| {
                    s.vertex_neighborhoods = Vec::new();
                    s
                }),
                custom_predict_pass: None,
                bootstrap_predictor: None,
            })
        });

        ExecutorOutput {
            smoothed: y_smooth,
            std_errors: se,
            // Report iterations whenever robustness is configured (target_iterations > 1),
            // regardless of whether auto-converge tolerance was used.
            iterations: (target_iterations > 1).then_some(iterations_performed),
            used_fraction: eff_fraction,
            cv_scores: None,
            robustness_weights: final_robustness_weights,
            leverage: leverage_values,
            gradient: gradient_values,
            predict_state,
        }
    }

    // Predict values at arbitrary points using the provided training data.
    //
    // This is used for out-of-sample prediction, specifically during cross-validation.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &self,
        x_train: &[T],
        y_train: &[T],
        robustness_weights: &[T],
        x_query: &[T],
        window_size: usize,
        scales: &[T],
        kdtree: &KDTree<T>,
    ) -> Vec<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let dims = self.dimensions;
        let n_query = x_query.len() / dims;
        let mut y_pred = vec![T::zero(); n_query];

        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales,
        };

        let n_points_train = x_train.len() / dims;
        let n_coeffs = self.polynomial_degree.num_coefficients_nd(dims);
        let mut workspace = LoessBuffer::<T, NodeDistance<T>, Neighborhood<T>>::new(
            n_points_train,
            dims,
            window_size,
            n_coeffs,
        );

        for (i, pred) in y_pred.iter_mut().enumerate() {
            let query_offset = i * dims;
            let query_point = &x_query[query_offset..query_offset + dims];

            // Find neighbors in training data (KD-tree is always available)
            kdtree.find_kernel_neighborhood(
                query_point,
                window_size,
                &dist_calc,
                self.weight_function,
                &mut workspace.search_buffer,
                &mut workspace.neighborhood,
            );
            let neighborhood = &workspace.neighborhood;

            // Fit local polynomial
            let mut context = RegressionContext::new(
                x_train,
                dims,
                y_train,
                0, // query_idx is not used when query_point is Some
                Some(query_point),
                neighborhood,
                true, // use_robustness
                robustness_weights,
                self.weight_function,
                ZeroWeightFallback::UseLocalMean, // CV fallback
                self.polynomial_degree,
                false, // compute_leverage
                Some(&mut workspace.fitting_buffer),
            );
            if let Some(weights) = self.custom_weights.as_deref() {
                context = context.with_custom_weights(weights);
            }
            if let Some((val, _)) = context.fit() {
                *pred = val;
            } else {
                // If fitting fails, fallback to something or zero
                // For CV, zero is better than crashing, but we could try to find a global mean
                *pred = T::zero();
            }
        }
        y_pred
    }

    // Perform a single smoothing pass over all nD points (Direct mode).
    //
    // # Caching Behavior
    //
    // * `populate_cache`: If `Some(&mut cache)`, populate the cache with neighborhoods during this pass.
    // * `cached_neighborhoods`: If `Some(&cache)`, use cached neighborhoods instead of calling `find_k_nearest`.
    //
    // Typically, the first pass populates the cache, and subsequent robustness iterations use it.
    #[allow(clippy::too_many_arguments)]
    fn smooth_pass(
        &self,
        x_context: &[T],
        y_context: &[T],
        x_query: &[T],
        y_query: &[T],
        window_size: usize,
        robustness_weights: &[T],
        use_robustness: bool,
        scales: &[T],
        y_smooth: &mut [T],
        original_n: usize,
        kdtree: &KDTree<T>,
        search_buffer: &mut NeighborhoodSearchBuffer<NodeDistance<T>>,
        neighborhood: &mut Neighborhood<T>,
        fitting_buffer: &mut FittingBuffer<T>,
        mut leverage_out: Option<&mut Vec<T>>,
        mut populate_cache: Option<&mut Vec<CachedNeighborhood<T>>>,
        cached_neighborhoods: Option<&[CachedNeighborhood<T>]>,
        custom_weights: Option<&[T]>,
        // Flattened (`dimensions` values per point), like `LoessResult::gradient`. When
        // `Some`, populated for free from the same solve `fit()` already performs via
        // `RegressionContext::with_gradient_out`, instead of a separate `gradient_pass`.
        mut gradient_out: Option<&mut Vec<T>>,
    ) where
        T: Float + Debug + Send + Sync + 'static,
    {
        let dims = self.dimensions;
        let dist_calc = LoessDistanceCalculator {
            metric: self.distance_metric.clone(),
            scales,
        };
        let compute_leverage = leverage_out.is_some();
        if let Some(ref mut grad_vec) = gradient_out {
            grad_vec.resize(original_n * dims, T::zero());
        }

        // Prepare cache for population if requested
        if let Some(ref mut cache) = populate_cache.as_ref() {
            // Pre-allocate capacity but will push during iteration
            let _ = cache; // just to check it's mutable
        }

        for i in 0..original_n {
            let query_offset = i * dims;
            let query_point = &x_query[query_offset..query_offset + dims];

            // Either use cached neighborhood or compute fresh
            if let Some(cache) = cached_neighborhoods {
                // Use cached neighborhood
                let cached = &cache[i];
                neighborhood.indices.clear();
                neighborhood.indices.extend_from_slice(&cached.indices);
                neighborhood.distances.clear();
                neighborhood.distances.extend_from_slice(&cached.distances);
                neighborhood.max_distance = cached.max_distance;
            } else {
                // Compute neighborhood via KD-tree
                kdtree.find_kernel_neighborhood(
                    query_point,
                    window_size,
                    &dist_calc,
                    self.weight_function,
                    search_buffer,
                    neighborhood,
                );

                // Populate cache if requested
                if let Some(ref mut cache) = populate_cache {
                    cache.push(CachedNeighborhood {
                        indices: neighborhood.indices.clone(),
                        distances: neighborhood.distances.clone(),
                        max_distance: neighborhood.max_distance,
                    });
                }
            }

            let neighborhood_ref = &*neighborhood;

            let mut context = RegressionContext::new(
                x_context,
                dims,
                y_context,
                i,
                Some(query_point),
                neighborhood_ref,
                use_robustness,
                robustness_weights,
                self.weight_function,
                self.zero_weight_fallback,
                self.polynomial_degree,
                compute_leverage,
                Some(fitting_buffer),
            );
            if let Some(uw) = custom_weights {
                context = context.with_custom_weights(uw);
            }
            if let Some(ref mut grad_vec) = gradient_out {
                context =
                    context.with_gradient_out(&mut grad_vec[query_offset..query_offset + dims]);
            }

            if let Some((val, lev)) = context.fit() {
                y_smooth[i] = val;
                if let Some(ref mut lev_vec) = leverage_out {
                    if lev_vec.len() <= i {
                        lev_vec.resize(i + 1, T::zero());
                    }
                    lev_vec[i] = lev;
                }
            } else {
                y_smooth[i] = y_query[i];
                if let Some(ref mut lev_vec) = leverage_out {
                    if lev_vec.len() <= i {
                        lev_vec.resize(i + 1, T::zero());
                    }
                    lev_vec[i] = T::zero();
                }
            }
        }
    }
}

#[cfg(test)]
mod normalization_tests {
    use super::loess_normalization_scales;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    use approx::assert_relative_eq;

    #[test]
    fn uses_r_loess_trimmed_sample_standard_deviation() {
        let mut x = Vec::new();
        for index in 0..10 {
            x.push(if index == 9 {
                100.0
            } else {
                index as f64 / 10.0
            });
            x.push(if index == 9 {
                1000.0
            } else {
                index as f64 * 10.0
            });
        }

        let scales = loess_normalization_scales(&x, 10, 2);

        assert_relative_eq!(scales[0], 1.0 / 0.06_f64.sqrt(), epsilon = 1e-12);
        assert_relative_eq!(scales[1], 1.0 / 600.0_f64.sqrt(), epsilon = 1e-12);
        assert_relative_eq!(scales[0] / scales[1], 100.0, epsilon = 1e-10);
    }
}
