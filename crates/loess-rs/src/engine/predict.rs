//! Out-of-sample prediction for fitted Batch LOESS models.
//!
//! This module holds the fitted-model state retained by `Predict::call()`
//! (Batch adapter only, opt-in via `.retain_model(true)`) and the logic that
//! evaluates the local polynomial fit at arbitrary query points not in the
//! training set. Supports the full nD / polynomial-degree / distance-metric
//! generality of the Batch adapter by reusing the same `RegressionContext` and
//! `KDTree` neighbor search used during fitting.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::interpolation::InterpolationSurface;
use crate::algorithms::regression::{
    PolynomialDegree, RegressionContext, SolverLinalg, ZeroWeightFallback,
};
use crate::api::IntoEnum;
use crate::engine::executor::LoessDistanceCalculator;
use crate::engine::output::LoessResult;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::distance::{DistanceLinalg, DistanceMetric};
use crate::math::kernel::WeightFunction;
use crate::math::linalg::FloatLinalg;
use crate::math::neighborhood::{KDTree, Neighborhood, NodeDistance};
use crate::primitives::buffer::{FittingBuffer, NeighborhoodSearchBuffer};
use crate::primitives::errors::LoessError;

// Policy for evaluating query points outside the retained per-dimension training range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtrapolationPolicy {
    // Clamp each out-of-range dimension to the nearest training boundary (default;
    // matches `predict()`'s original behavior).
    #[default]
    Clamp,

    // Linearly extrapolate from the boundary point's local fit and gradient (first-order
    // Taylor expansion from the clamped point).
    Linear,

    // Fail the whole `predict()` call with `LoessError::PredictOutOfRange` if any query
    // point falls outside the training range on any dimension.
    Error,
}

// Fluent, deferred-validation configuration for a `Predict::call()` invocation. Call
// `.build()` to validate and obtain the ready-to-call `Predict`.
#[derive(Debug, Clone)]
pub struct PredictBuilder<T> {
    // Include standard errors in the output.
    pub return_se: bool,

    // Confidence interval coverage level (e.g. `Some(0.95)`), or `None` to skip.
    pub confidence_intervals: Option<T>,

    // Prediction interval coverage level (e.g. `Some(0.95)`), or `None` to skip.
    pub prediction_intervals: Option<T>,

    // Include the local fit's gradient (`dimensions` values per query point, flattened)
    // in the output.
    pub return_derivative: bool,

    // Behavior for query points outside the training range.
    pub extrapolation: ExtrapolationPolicy,

    // Under `ExtrapolationPolicy::Linear`, the maximum allowed per-dimension distance
    // beyond the training boundary before `predict()` errors with
    // `LoessError::ExtrapolationTooFar`, instead of returning the first-order Taylor
    // extension's unbounded value. `None` (default) preserves the original, uncapped
    // behavior. Ignored under `Clamp`/`Error`.
    pub max_extrapolation_distance: Option<T>,

    // Maximum allowed distance to the farthest point in a query's k-nearest-neighbor
    // window before `predict()` errors with `LoessError::SparseNeighborhood`. Guards
    // against the per-dimension bounding-box range check's blind spot: a point can sit
    // inside every dimension's min/max range yet fall in an empty region far from any
    // real training data (e.g. an empty "corner" of non-rectangularly distributed data).
    // Measured as a plain (raw-coordinate) Euclidean distance, independent of
    // `distance_metric` - the same unit space as `max_extrapolation_distance`, so the two
    // caps can be reasoned about together regardless of which metric the model was fit
    // with (e.g. under the default `Normalized` metric, `neighborhood` search distances
    // are metric-space, not raw-coordinate, values). `None` (default) preserves the
    // original behavior of silently extrapolating there. Applies regardless of
    // `extrapolation`/whether the bounding-box check flagged the point as out-of-range.
    pub max_neighbor_distance: Option<T>,

    // Set by `extrapolation(...)` when given an invalid string; surfaced by `build()`.
    pub pending_error: Option<LoessError>,
}

impl<T: FloatLinalg> Default for PredictBuilder<T> {
    fn default() -> Self {
        Self {
            return_se: false,
            confidence_intervals: None,
            prediction_intervals: None,
            return_derivative: false,
            extrapolation: ExtrapolationPolicy::default(),
            max_extrapolation_distance: None,
            max_neighbor_distance: None,
            pending_error: None,
        }
    }
}

impl<T: FloatLinalg> PredictBuilder<T> {
    // Create a new `PredictBuilder` with default values, matching `Loess::new()`'s
    // constructor-style entry point.
    pub fn new() -> Self {
        Self::default()
    }

    // Include standard errors in the output.
    pub fn return_se(mut self) -> Self {
        self.return_se = true;
        self
    }

    // Request a confidence interval at the given coverage level (e.g. `0.95`).
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.confidence_intervals = Some(level);
        self
    }

    // Request a prediction interval at the given coverage level (e.g. `0.95`).
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.prediction_intervals = Some(level);
        self
    }

    // Include the local fit's gradient (`dimensions` values per query point, flattened)
    // in the output.
    pub fn return_derivative(mut self) -> Self {
        self.return_derivative = true;
        self
    }

    // Behavior for query points outside the training range: `"clamp"` (default),
    // `"linear"`, `"error"`, or an `ExtrapolationPolicy` variant directly.
    #[allow(private_bounds)]
    pub fn extrapolation(mut self, policy: impl IntoEnum<ExtrapolationPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.extrapolation = p,
            Err(e) => self.pending_error = Some(e),
        }
        self
    }

    // Under `"linear"` extrapolation, the maximum allowed per-dimension distance beyond
    // the training boundary before `call()` errors instead of returning an unbounded value.
    pub fn max_extrapolation_distance(mut self, distance: T) -> Self {
        self.max_extrapolation_distance = Some(distance);
        self
    }

    // Maximum allowed distance to the farthest point in a query's k-nearest-neighbor window
    // before `call()` errors, catching in-range-but-sparse query points.
    pub fn max_neighbor_distance(mut self, distance: T) -> Self {
        self.max_neighbor_distance = Some(distance);
        self
    }

    // Validates this configuration and produces a ready-to-call `PredictQuery`. Mandatory:
    // `PredictQuery` has no public constructor of its own, so `.build()` is the only way to
    // obtain one.
    pub fn build(self) -> Result<PredictQuery<T>, LoessError> {
        if let Some(e) = self.pending_error {
            return Err(e);
        }
        Ok(PredictQuery {
            return_se: self.return_se,
            confidence_intervals: self.confidence_intervals,
            prediction_intervals: self.prediction_intervals,
            return_derivative: self.return_derivative,
            extrapolation: self.extrapolation,
            max_extrapolation_distance: self.max_extrapolation_distance,
            max_neighbor_distance: self.max_neighbor_distance,
        })
    }
}

// `Predict::new()` is the friendly, common-case entry point for `PredictBuilder`, matching
// `Loess` being an alias for `LoessBuilder<T, BatchMode>`.
pub type Predict<T = f64> = PredictBuilder<T>;

// Validated, ready-to-call configuration for a `Predict::call()` invocation, produced by
// `PredictBuilder::build()`. Fields are private; the only way to construct one is via the
// builder, so a `.build()` call can never be skipped.
#[derive(Debug, Clone)]
pub struct PredictQuery<T> {
    return_se: bool,
    confidence_intervals: Option<T>,
    prediction_intervals: Option<T>,
    return_derivative: bool,
    extrapolation: ExtrapolationPolicy,
    max_extrapolation_distance: Option<T>,
    max_neighbor_distance: Option<T>,
}

impl<T> PredictQuery<T> {
    // Whether the local fit's gradient is included in the output (used by fastLoess's
    // parallel predict pass, which needs this outside `loess-rs` itself). No trait
    // bounds needed: this just reads a plain `bool` field.
    pub fn return_derivative(&self) -> bool {
        self.return_derivative
    }
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync> PredictQuery<T> {
    // Evaluate `result` (a fitted Batch model) at arbitrary out-of-sample query points
    // (flattened, `dimensions` values per point), per these options. Returns a
    // `PredictOutput` with one entry per query point.
    //
    // Requires `.retain_model(true)` on the builder that produced `result` (Batch adapter
    // only); returns `LoessError::PredictionUnavailable` otherwise.
    //
    // Reuses `fit()`'s own interpolation surface (when built under the default
    // `SurfaceMode::Interpolation`) for in-range query points, so it exactly reproduces
    // `fit()`'s value at any point already in the training set, regardless of
    // `surface_mode()`. A fresh local regression is only run when `return_derivative`,
    // an out-of-range extrapolation, or `max_neighbor_distance` needs the actual
    // gradient/leverage at the query point.
    pub fn call(
        &self,
        result: &LoessResult<T>,
        new_x: &[T],
    ) -> Result<PredictOutput<T>, LoessError> {
        let state = result
            .predict_state
            .as_ref()
            .ok_or(LoessError::PredictionUnavailable)?;
        predict_batch(state, new_x, self)
    }
}

// Result of a `Predict::call()` invocation.
#[derive(Debug, Clone)]
pub struct PredictOutput<T> {
    // Predicted y-values, one per query point in `new_x`.
    pub y: Vec<T>,

    // Standard errors, if `return_se`/`confidence_intervals`/`prediction_intervals` was requested.
    pub standard_errors: Option<Vec<T>>,

    // Confidence interval bounds for the mean response, if `confidence_intervals` was set.
    pub confidence_lower: Option<Vec<T>>,
    pub confidence_upper: Option<Vec<T>>,

    // Prediction interval bounds for a new observation, if `prediction_intervals` was set.
    pub prediction_lower: Option<Vec<T>>,
    pub prediction_upper: Option<Vec<T>>,

    // Local fit's gradient at each query point, if `return_derivative` was set
    // (`dimensions` values per query point, flattened like `new_x`).
    pub derivative: Option<Vec<T>>,
}

// Per-point predict results before shared confidence/prediction interval math is applied:
// `(y, optional flattened gradient, optional standard error)`.
pub type RawPredictValues<T> = Result<(Vec<T>, Option<Vec<T>>, Option<Vec<T>>), LoessError>;

// Signature for a custom (e.g. parallel) predict pass function. Computes only the
// per-point values (y, optional gradient, optional standard error); the shared
// confidence/prediction interval math is applied afterward by `predict_batch`.
#[doc(hidden)]
pub type PredictPassFn<T> = fn(
    &PredictState<T>,
    &[T], // new_x (flattened, `dimensions` values per query point)
    &PredictQuery<T>,
    bool, // need_se
) -> RawPredictValues<T>;

// Fitted-model state retained by a Batch `fit()` call when `.retain_model(true)` was set,
// enabling `Predict::call()` to evaluate the fit at out-of-sample query points.
//
// `x`/`y`/`robustness_weights`/`custom_weights` are the boundary-*padded* arrays actually
// used for local fitting (not the shorter, unpadded arrays returned in `LoessResult`), so
// that predictions near the edges of the training range are consistent with `fit()`.
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
    // computed from the padded training data's min-max range.
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
    #[doc(hidden)]
    pub custom_predict_pass: Option<PredictPassFn<T>>,
}

// Manual `PartialEq` that ignores `kdtree`/`surface` (cached derived structures, not part
// of model identity) and `custom_predict_pass` (function pointer comparisons aren't
// meaningful - addresses aren't guaranteed unique across codegen units).
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

fn zero_vec<T: FloatLinalg>(n: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(T::zero());
    }
    v
}

// Evaluate the fitted model at a single out-of-sample query point, returning
// `(y, optional flattened gradient, optional standard error)`. `search_buffer`/
// `neighborhood` are reusable working space for the KD-tree neighbor search.
//
// `pub` (hidden) so extension crates like fastLoess can reuse it for a parallel
// `PredictPassFn` implementation.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn predict_one_full<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>(
    state: &PredictState<T>,
    query_point: &[T],
    kdtree: &KDTree<T>,
    dist_calc: &LoessDistanceCalculator<T>,
    search_buffer: &mut NeighborhoodSearchBuffer<NodeDistance<T>>,
    neighborhood: &mut Neighborhood<T>,
    options: &PredictQuery<T>,
    need_se: bool,
) -> Result<(T, Option<Vec<T>>, Option<T>), LoessError> {
    let dims = state.dimensions;

    // Determine per-dimension out-of-range status and the clamped boundary point.
    let mut out_of_range = false;
    let mut clamped = query_point.to_vec();
    for d in 0..dims {
        if query_point[d] < state.train_min[d] {
            if options.extrapolation == ExtrapolationPolicy::Error {
                return Err(LoessError::PredictOutOfRange {
                    dimension: d,
                    query: query_point[d].to_f64().unwrap_or(0.0),
                    min: state.train_min[d].to_f64().unwrap_or(0.0),
                    max: state.train_max[d].to_f64().unwrap_or(0.0),
                });
            }
            clamped[d] = state.train_min[d];
            out_of_range = true;
        } else if query_point[d] > state.train_max[d] {
            if options.extrapolation == ExtrapolationPolicy::Error {
                return Err(LoessError::PredictOutOfRange {
                    dimension: d,
                    query: query_point[d].to_f64().unwrap_or(0.0),
                    min: state.train_min[d].to_f64().unwrap_or(0.0),
                    max: state.train_max[d].to_f64().unwrap_or(0.0),
                });
            }
            clamped[d] = state.train_max[d];
            out_of_range = true;
        }
    }

    // Both `Clamp` and `Linear` evaluate the local fit AT the clamped (in-range) point;
    // `Linear` additionally extends that fit using its own gradient.
    let extrapolate_linear = out_of_range && options.extrapolation == ExtrapolationPolicy::Linear;
    let eval_point = clamped;

    if extrapolate_linear && let Some(max_dist) = options.max_extrapolation_distance {
        for d in 0..dims {
            let dist = (query_point[d] - eval_point[d]).abs();
            if dist > max_dist {
                return Err(LoessError::ExtrapolationTooFar {
                    dimension: d,
                    distance: dist.to_f64().unwrap_or(0.0),
                    max_distance: max_dist.to_f64().unwrap_or(0.0),
                });
            }
        }
    }

    let need_gradient = options.return_derivative || extrapolate_linear;

    // A requested neighbor-sparsity check needs a real KD-tree search to measure against,
    // so it must bypass the surface-only fast path below even when nothing else would.
    let need_neighborhood_check = options.max_neighbor_distance.is_some();

    // Fast path: for an in-range query point with no gradient requested, an available
    // interpolation surface (`SurfaceMode::Interpolation`, the default) can answer the query
    // directly - no neighborhood search or regression solve needed, and (unlike the exact
    // per-point fit below) it matches `fit()`'s own `y_smooth`/SE at training points. SE, if
    // requested, uses the same uniform approximate-leverage heuristic `fit()` itself falls
    // back to under `SurfaceMode::Interpolation`, rather than an unrelated exact-leverage
    // value from a fit that `y` no longer even depends on.
    if !out_of_range
        && !need_gradient
        && !need_neighborhood_check
        && let Some(surface) = &state.surface
    {
        let se = need_se.then_some(state.interpolation_se);
        return Ok((surface.evaluate(query_point), None, se));
    }

    kdtree.find_k_nearest(
        &eval_point,
        state.window_size,
        dist_calc,
        None,
        search_buffer,
        neighborhood,
    );

    if let Some(max_dist) = options.max_neighbor_distance {
        // Recomputed as a plain Euclidean distance (not `neighborhood.max_distance`,
        // which is measured in whatever `distance_metric` space the KD-tree search used -
        // e.g. dimensionless under the default `Normalized` metric), so this cap lives in
        // the same raw-coordinate units as `max_extrapolation_distance` above.
        let mut farthest = T::zero();
        for &idx in &neighborhood.indices {
            let neighbor_point = &state.x[idx * dims..idx * dims + dims];
            let mut sq_dist = T::zero();
            for d in 0..dims {
                let diff = eval_point[d] - neighbor_point[d];
                sq_dist = sq_dist + diff * diff;
            }
            let dist = sq_dist.sqrt();
            if dist > farthest {
                farthest = dist;
            }
        }
        if farthest > max_dist {
            return Err(LoessError::SparseNeighborhood {
                distance: farthest.to_f64().unwrap_or(0.0),
                max_distance: max_dist.to_f64().unwrap_or(0.0),
            });
        }
    }

    let gradient = if need_gradient {
        // `fit_with_coefficients()` only has a buffered implementation (the non-buffered
        // path is a stub that always falls back to a zero-gradient degenerate case), so a
        // real buffer must be supplied here even though it's not reused across calls.
        let n_coeffs = state.polynomial_degree.num_coefficients_nd(dims);
        let mut buffer = FittingBuffer::new(state.window_size, n_coeffs);
        let mut context = RegressionContext::new(
            &state.x,
            dims,
            &state.y,
            0, // query_idx is not used when query_point is Some
            Some(eval_point.as_slice()),
            neighborhood,
            true, // use_robustness
            &state.robustness_weights,
            state.weight_function,
            state.zero_weight_fallback,
            state.polynomial_degree,
            false, // compute_leverage (fit_with_coefficients doesn't support it)
            Some(&mut buffer),
        );
        if dims == 1 {
            context = context.with_global_x_range(state.train_max[0] - state.train_min[0]);
        }
        let mut context = if let Some(cw) = state.custom_weights.as_deref() {
            context.with_custom_weights(cw)
        } else {
            context
        };
        let coeffs = context
            .fit_with_coefficients()
            .unwrap_or_else(|| zero_vec(dims + 1));
        Some(coeffs[1..].to_vec())
    } else {
        None
    };

    let mut context = RegressionContext::new(
        &state.x,
        dims,
        &state.y,
        0,
        Some(eval_point.as_slice()),
        neighborhood,
        true,
        &state.robustness_weights,
        state.weight_function,
        state.zero_weight_fallback,
        state.polynomial_degree,
        need_se,
        None,
    );
    if dims == 1 {
        context = context.with_global_x_range(state.train_max[0] - state.train_min[0]);
    }
    if let Some(cw) = state.custom_weights.as_deref() {
        context = context.with_custom_weights(cw);
    }
    let (mut y, leverage) = context.fit().unwrap_or((T::zero(), T::zero()));

    // In-range (so the fast path above was skipped only because a gradient was needed):
    // prefer the surface's value for `y` anyway, so it still matches `fit()`'s `y_smooth` at
    // training points. The gradient always keeps coming from the exact fit above regardless
    // (never from the surface - see the module-level note on `return_derivative`).
    let surface_active = !out_of_range && state.surface.is_some();
    if let Some(surface) = &state.surface
        && !out_of_range
    {
        y = surface.evaluate(query_point);
    }

    if extrapolate_linear {
        let grad = gradient.as_deref().unwrap_or(&[]);
        for d in 0..dims {
            let g = grad.get(d).copied().unwrap_or(T::zero());
            y = y + g * (query_point[d] - eval_point[d]);
        }
    }

    // Same reasoning as the fast path above: keep SE consistent with whichever surface mode
    // produced `y`, rather than always using the exact-leverage value from this fit (which,
    // when a surface is active, only ran here to obtain the gradient, not `y`).
    let se = need_se.then(|| {
        if surface_active {
            state.interpolation_se
        } else {
            state.residual_sd * leverage.max(T::zero()).sqrt()
        }
    });

    Ok((y, gradient, se))
}

// Serial fallback for `predict_batch` when no `custom_predict_pass` is set.
fn predict_batch_serial<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>(
    state: &PredictState<T>,
    new_x: &[T],
    options: &PredictQuery<T>,
    need_se: bool,
) -> RawPredictValues<T> {
    let dims = state.dimensions;
    let n_train = state.x.len() / dims.max(1);
    let n_query = new_x.len() / dims.max(1);

    let dist_calc = LoessDistanceCalculator {
        metric: state.distance_metric.clone(),
        scales: &state.scales,
    };
    let mut search_buffer = NeighborhoodSearchBuffer::new(state.window_size.min(n_train.max(1)));
    let mut neighborhood = Neighborhood::with_capacity(state.window_size.min(n_train.max(1)));

    let mut y = Vec::with_capacity(n_query);
    let mut derivative = options
        .return_derivative
        .then(|| Vec::with_capacity(n_query * dims));
    let mut se = need_se.then(|| Vec::with_capacity(n_query));

    for i in 0..n_query {
        let query_point = &new_x[i * dims..(i + 1) * dims];
        let (yi, grad, sei) = predict_one_full(
            state,
            query_point,
            &state.kdtree,
            &dist_calc,
            &mut search_buffer,
            &mut neighborhood,
            options,
            need_se,
        )?;
        y.push(yi);
        if let Some(d) = derivative.as_mut() {
            match grad {
                Some(g) => d.extend_from_slice(&g),
                None => d.extend_from_slice(&zero_vec(dims)),
            }
        }
        if let Some(s) = se.as_mut() {
            s.push(sei.unwrap_or(T::zero()));
        }
    }

    Ok((y, derivative, se))
}

// Evaluate the fitted model at a batch of out-of-sample query points (flattened, `dimensions`
// values per point), per `options`. Delegates the per-point work to `state.custom_predict_pass`
// if set (e.g. fastLoess's Rayon-parallel implementation), otherwise evaluates serially.
pub fn predict_batch<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>(
    state: &PredictState<T>,
    new_x: &[T],
    options: &PredictQuery<T>,
) -> Result<PredictOutput<T>, LoessError> {
    if state.dimensions == 0 || !new_x.len().is_multiple_of(state.dimensions) {
        return Err(LoessError::InvalidInput(format!(
            "predict(): new_x length ({}) is not a multiple of dimensions ({})",
            new_x.len(),
            state.dimensions
        )));
    }
    for (i, &val) in new_x.iter().enumerate() {
        if !val.is_finite() {
            return Err(LoessError::InvalidNumericValue(format!(
                "new_x[{}]={}",
                i,
                val.to_f64().unwrap_or(f64::NAN)
            )));
        }
    }

    let need_se = options.return_se
        || options.confidence_intervals.is_some()
        || options.prediction_intervals.is_some();

    let (y, derivative, se) = if let Some(pass) = state.custom_predict_pass {
        pass(state, new_x, options, need_se)?
    } else {
        predict_batch_serial(state, new_x, options, need_se)?
    };

    let (confidence_lower, confidence_upper) = if let Some(level) = options.confidence_intervals {
        let se_vals = se.as_deref().unwrap_or(&[]);
        let z = IntervalMethod::<T>::approximate_z_score(level)
            .map_err(|_| LoessError::InvalidIntervals(level.to_f64().unwrap_or(0.0)))?;
        let lower: Vec<T> = y.iter().zip(se_vals).map(|(&yi, &s)| yi - z * s).collect();
        let upper: Vec<T> = y.iter().zip(se_vals).map(|(&yi, &s)| yi + z * s).collect();
        (Some(lower), Some(upper))
    } else {
        (None, None)
    };

    let (prediction_lower, prediction_upper) = if let Some(level) = options.prediction_intervals {
        let se_vals = se.as_deref().unwrap_or(&[]);
        let z = IntervalMethod::<T>::approximate_z_score(level)
            .map_err(|_| LoessError::InvalidIntervals(level.to_f64().unwrap_or(0.0)))?;
        let rsd_sq = state.residual_sd * state.residual_sd;
        let lower: Vec<T> = y
            .iter()
            .zip(se_vals)
            .map(|(&yi, &s)| yi - z * (s * s + rsd_sq).sqrt())
            .collect();
        let upper: Vec<T> = y
            .iter()
            .zip(se_vals)
            .map(|(&yi, &s)| yi + z * (s * s + rsd_sq).sqrt())
            .collect();
        (Some(lower), Some(upper))
    } else {
        (None, None)
    };

    Ok(PredictOutput {
        y,
        standard_errors: se,
        confidence_lower,
        confidence_upper,
        prediction_lower,
        prediction_upper,
        derivative,
    })
}
