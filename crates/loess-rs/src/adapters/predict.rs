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
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;

// Internal dependencies

use crate::algorithms::regression::context::RegressionContext;
use crate::algorithms::regression::specialized::SolverLinalg;
use crate::engine::executor::{LoessDistanceCalculator, LoessResult};
use crate::evaluation::intervals::{BootstrapConfig, IntervalMethod, IntervalsBuilder};
use crate::math::distance::DistanceLinalg;
use crate::math::linalg::FloatLinalg;
use crate::math::neighborhood::{KDTree, Neighborhood, NodeDistance};
use crate::primitives::buffer::{FittingBuffer, NeighborhoodSearchBuffer};
use crate::primitives::errors::LoessError;

// Policy for evaluating query points outside the retained per-dimension training range.
use crate::engine::executor::{PredictQuery, PredictState, RawPredictValues};
use crate::primitives::policies::ExtrapolationPolicy;

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

    pub bootstrap_samples: Option<usize>,
    pub seed: Option<u64>,

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

    // Invalid extrapolation/output options are surfaced by `build()`.
    pub pending_error: Option<LoessError>,
}

impl<T: FloatLinalg> Default for PredictBuilder<T> {
    fn default() -> Self {
        Self {
            return_se: false,
            confidence_intervals: None,
            prediction_intervals: None,
            bootstrap_samples: None,
            seed: None,
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

    /// Select standard errors (`"se"`) and/or flattened gradients (`"derivative"` or `"gradient"`).
    /// Selections accumulate; unsupported names are rejected by `build()`.
    pub fn outputs<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for name in names {
            match name.as_ref() {
                "se" => self.return_se = true,
                "derivative" | "gradient" => self.return_derivative = true,
                other => {
                    self.pending_error = Some(LoessError::InvalidOption {
                        option: "predict_outputs",
                        value: other.to_string(),
                        valid: "se, derivative, gradient",
                    });
                }
            }
        }
        self
    }

    pub fn intervals(mut self, options: IntervalsBuilder<T>) -> Self {
        if let Some(level) = options.confidence {
            if self.confidence_intervals.is_some() {
                self.pending_error = Some(LoessError::DuplicateParameter {
                    parameter: "intervals",
                });
            }
            self.confidence_intervals = Some(level);
        }
        if let Some(level) = options.prediction {
            if self.prediction_intervals.is_some() {
                self.pending_error = Some(LoessError::DuplicateParameter {
                    parameter: "intervals",
                });
            }
            self.prediction_intervals = Some(level);
        }
        if let Some(samples) = options.bootstrap {
            if self.bootstrap_samples.is_some() {
                self.pending_error = Some(LoessError::DuplicateParameter {
                    parameter: "intervals",
                });
            }
            self.bootstrap_samples = Some(samples);
        }
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    // Include the local fit's gradient (`dimensions` values per query point, flattened)
    // in the output.
    pub fn return_derivative(mut self) -> Self {
        self.return_derivative = true;
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
        if let Some(samples) = self.bootstrap_samples {
            crate::engine::validator::Validator::validate_bootstrap_samples(samples)?;
        }
        for level in [self.confidence_intervals, self.prediction_intervals]
            .into_iter()
            .flatten()
        {
            crate::engine::validator::Validator::validate_interval_level(level)?;
        }
        for (name, distance) in [
            (
                "max_extrapolation_distance",
                self.max_extrapolation_distance,
            ),
            ("max_neighbor_distance", self.max_neighbor_distance),
        ] {
            if let Some(distance) = distance
                && (!distance.is_finite() || distance < T::zero())
            {
                return Err(LoessError::InvalidInput(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(PredictQuery {
            return_se: self.return_se,
            confidence_intervals: self.confidence_intervals,
            prediction_intervals: self.prediction_intervals,
            bootstrap: self.bootstrap_samples.map(|n_boot| BootstrapConfig {
                n_boot,
                seed: self.seed,
            }),
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

// Signature for a custom (e.g. parallel) predict pass function. Computes only the
// per-point values (y, optional gradient, optional standard error); the shared
// confidence/prediction interval math is applied afterward by `predict_batch`.

// Fitted-model state retained by a Batch `fit()` call when `.retain_model(true)` was set,
// enabling `Predict::call()` to evaluate the fit at out-of-sample query points.
//
// `x`/`y`/`robustness_weights`/`custom_weights` are the boundary-*padded* arrays actually
// used for local fitting (not the shorter, unpadded arrays returned in `LoessResult`), so
// that predictions near the edges of the training range are consistent with `fit()`.

// Manual `PartialEq` that ignores `kdtree`/`surface` (cached derived structures, not part
// of model identity) and `custom_predict_pass` (function pointer comparisons aren't
// meaningful - addresses aren't guaranteed unique across codegen units).

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

    if state.weight_function.support().is_none() {
        kdtree.find_kernel_neighborhood(
            &eval_point,
            state.window_size,
            dist_calc,
            state.weight_function,
            search_buffer,
            neighborhood,
        );
    }

    let gradient = if need_gradient {
        // `fit_with_coefficients()` only has a buffered implementation (the non-buffered
        // path is a stub that always falls back to a zero-gradient degenerate case), so a
        // real buffer must be supplied here even though it's not reused across calls.
        let n_coeffs = state.polynomial_degree.num_coefficients_nd(dims);
        let mut buffer = FittingBuffer::new(state.window_size, n_coeffs);
        let context = RegressionContext::new(
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
        } else if dims == 1
            && state.polynomial_degree.value() == 1
            && let Some(residuals) = state
                .bootstrap_predictor
                .as_ref()
                .and_then(|predictor| predictor.original_residuals())
                .filter(|residuals| residuals.len() == state.y.len())
        {
            let bandwidth = neighborhood.max_distance;
            if bandwidth <= T::epsilon() {
                T::zero()
            } else {
                IntervalMethod::compute_local_se((0..neighborhood.len()).map(|neighbor| {
                    let index = neighborhood.indices[neighbor];
                    let weight = state
                        .weight_function
                        .compute_weight(neighborhood.distances[neighbor] / bandwidth)
                        * state.robustness_weights[index]
                        * state
                            .custom_weights
                            .as_ref()
                            .map_or(T::one(), |weights| weights[index]);
                    (state.x[index] - eval_point[0], weight, residuals[index])
                }))
            }
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

    let need_se = options.bootstrap.is_none()
        && (options.return_se
            || options.confidence_intervals.is_some()
            || options.prediction_intervals.is_some());

    let (y, derivative, se) = if let Some(pass) = state.custom_predict_pass {
        pass(state, new_x, options, need_se)?
    } else {
        predict_batch_serial(state, new_x, options, need_se)?
    };

    if let Some(bootstrap) = options.bootstrap {
        let method = IntervalMethod {
            level: options
                .confidence_intervals
                .or(options.prediction_intervals)
                .unwrap_or_else(|| T::from(0.95).unwrap()),
            confidence: options.confidence_intervals.is_some(),
            prediction_level: options.prediction_intervals,
            prediction: options.prediction_intervals.is_some(),
            se: true,
        };
        let mut query = options.clone();
        query.bootstrap = None;
        query.return_se = false;
        query.confidence_intervals = None;
        query.prediction_intervals = None;
        let mut evaluate = |refit: &PredictState<T>| Ok(predict_batch(refit, new_x, &query)?.y);
        let output = state
            .bootstrap_predictor
            .as_ref()
            .ok_or(LoessError::PredictionUnavailable)?
            .compute(
                bootstrap,
                &method,
                new_x.len() / state.dimensions,
                &mut evaluate,
            )?;
        return Ok(PredictOutput {
            y,
            derivative,
            standard_errors: Some(output.std_errors),
            confidence_lower: output.confidence_lower,
            confidence_upper: output.confidence_upper,
            prediction_lower: output.prediction_lower,
            prediction_upper: output.prediction_upper,
        });
    }

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
