//! Node.js bindings for fastLoess using N-API.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::api::LoessBuilder;
use ::fastLoess::internals::binding_support as shared_parse;
use ::fastLoess::prelude::{IntervalsBuilder, LoessResult as InnerLoessResult, Predict};

fn to_napi_error(err: shared_parse::BindingError) -> Error {
    let status = match err.category {
        shared_parse::BindingErrorCategory::InvalidArg => Status::InvalidArg,
        shared_parse::BindingErrorCategory::Runtime => Status::GenericFailure,
    };
    Error::new(status, err.message)
}

fn map_invalid_arg<T, E: ToString>(result: std::result::Result<T, E>) -> Result<T> {
    shared_parse::map_invalid_arg(result).map_err(to_napi_error)
}

fn map_runtime<T, E: ToString>(result: std::result::Result<T, E>) -> Result<T> {
    shared_parse::map_runtime(result).map_err(to_napi_error)
}

/// Diagnostic statistics for the LOESS fit.
#[napi(object)]
pub struct Diagnostics {
    /// Root Mean Squared Error.
    pub rmse: f64,
    /// Mean Absolute Error.
    pub mae: f64,
    /// R-squared (coefficient of determination).
    #[napi(js_name = "r_squared")]
    pub r_squared: f64,
    /// Akaike Information Criterion (if computed).
    pub aic: Option<f64>,
    /// Corrected AIC (if computed).
    pub aicc: Option<f64>,
    /// Effective degrees of freedom (if computed).
    #[napi(js_name = "effective_df")]
    pub effective_df: Option<f64>,
    /// Batch: robust residual scale estimate (1.4826 * MAD); Streaming: cumulative sample SD of emitted residuals.
    #[napi(js_name = "residual_sd")]
    pub residual_sd: f64,
}

/// Result of a single online update step.
#[napi(object)]
pub struct OnlineOutput {
    /// Smoothed value for the latest point.
    pub y: f64,
    /// Standard error (if computed).
    #[napi(js_name = "standard_error")]
    pub standard_error: Option<f64>,
    /// Residual (raw input y minus this output's y) (if computed).
    pub residual: Option<f64>,
    /// Robustness weight for the latest point (if computed).
    #[napi(js_name = "robustness_weight")]
    pub robustness_weight: Option<f64>,
    /// Number of robustness iterations performed (if applicable).
    #[napi(js_name = "iterations_used")]
    pub iterations_used: Option<u32>,
    /// Confidence interval lower bound (`update_mode="full"` only, if requested).
    #[napi(js_name = "confidence_lower")]
    pub confidence_lower: Option<f64>,
    /// Confidence interval upper bound (`update_mode="full"` only, if requested).
    #[napi(js_name = "confidence_upper")]
    pub confidence_upper: Option<f64>,
    /// Prediction interval lower bound (`update_mode="full"` only, if requested).
    #[napi(js_name = "prediction_lower")]
    pub prediction_lower: Option<f64>,
    /// Prediction interval upper bound (`update_mode="full"` only, if requested).
    #[napi(js_name = "prediction_upper")]
    pub prediction_upper: Option<f64>,
    /// Local fit gradient (`dimensions` values) for the latest point (if requested).
    #[napi(js_name = "gradient")]
    pub gradient: Option<Vec<f64>>,
}

/// Result of a LOESS fit.
#[napi]
pub struct LoessResult {
    inner: InnerLoessResult<f64>,
}

#[napi]
impl LoessResult {
    /// Get the x values (same order as input).
    #[napi(getter)]
    pub fn get_x(&self) -> Float64Array {
        Float64Array::from(self.inner.x.as_slice())
    }

    /// Get the smoothed y values.
    #[napi(getter)]
    pub fn get_y(&self) -> Float64Array {
        Float64Array::from(self.inner.y.as_slice())
    }

    /// Get residuals (if requested).
    #[napi(getter)]
    pub fn get_residuals(&self) -> Option<Float64Array> {
        self.inner
            .residuals
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get standard errors (if requested/computed).
    #[napi(getter, js_name = "standard_errors")]
    pub fn get_standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get lower confidence bounds (if requested).
    #[napi(getter, js_name = "confidence_lower")]
    pub fn get_confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get upper confidence bounds (if requested).
    #[napi(getter, js_name = "confidence_upper")]
    pub fn get_confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get lower prediction bounds (if requested).
    #[napi(getter, js_name = "prediction_lower")]
    pub fn get_prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get upper prediction bounds (if requested).
    #[napi(getter, js_name = "prediction_upper")]
    pub fn get_prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get robustness weights (if requested).
    #[napi(getter, js_name = "robustness_weights")]
    pub fn get_robustness_weights(&self) -> Option<Float64Array> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get the per-point local fit gradient (flattened, `dimensions` values per point) (if requested).
    #[napi(getter, js_name = "gradient")]
    pub fn get_gradient(&self) -> Option<Float64Array> {
        self.inner
            .gradient
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get diagnostics (if requested).
    #[napi(getter)]
    pub fn get_diagnostics(&self) -> Option<Diagnostics> {
        self.inner.diagnostics.as_ref().map(|d| Diagnostics {
            rmse: d.rmse,
            mae: d.mae,
            r_squared: d.r_squared,
            aic: d.aic,
            aicc: d.aicc,
            effective_df: d.effective_df,
            residual_sd: d.residual_sd,
        })
    }

    /// Get cross-validation scores (if CV was performed).
    #[napi(getter, js_name = "cv_scores")]
    pub fn get_cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get the fraction used for smoothing.
    #[napi(getter, js_name = "fraction_used")]
    pub fn get_fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    /// Get the number of iterations performed.
    #[napi(getter, js_name = "iterations_used")]
    pub fn get_iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }

    /// Get equivalent number of parameters (hat-matrix stat, if "se" was requested in outputs).
    #[napi(getter)]
    pub fn get_enp(&self) -> Option<f64> {
        self.inner.enp
    }

    /// Get trace of hat matrix (if "se" was requested in outputs).
    #[napi(getter, js_name = "trace_hat")]
    pub fn get_trace_hat(&self) -> Option<f64> {
        self.inner.trace_hat
    }

    /// Get first delta statistic (if "se" was requested in outputs).
    #[napi(getter)]
    pub fn get_delta1(&self) -> Option<f64> {
        self.inner.delta1
    }

    /// Get second delta statistic (if "se" was requested in outputs).
    #[napi(getter)]
    pub fn get_delta2(&self) -> Option<f64> {
        self.inner.delta2
    }

    /// Get residual scale estimate (if "se" was requested in outputs).
    #[napi(getter, js_name = "residual_scale")]
    pub fn get_residual_scale(&self) -> Option<f64> {
        self.inner.residual_scale
    }

    /// Get per-point leverage / hat-matrix diagonal (if "se" was requested in outputs).
    #[napi(getter)]
    pub fn get_leverage(&self) -> Option<Float64Array> {
        self.inner
            .leverage
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Get number of predictor dimensions.
    #[napi(getter)]
    pub fn get_dimensions(&self) -> u32 {
        self.inner.dimensions as u32
    }

    /// Evaluate the fitted model at out-of-sample query points not in the training set.
    ///
    /// Requires `retain_model: true` to have been set on the builder before `fit()`.
    #[napi]
    pub fn predict(
        &self,
        new_x: Float64Array,
        options: Option<PredictOptions>,
    ) -> Result<PredictOutput> {
        let opts = options.unwrap_or_default();
        validate_outputs(opts.outputs.as_ref(), &["se", "gradient", "derivative"])?;
        let output = map_invalid_arg(shared_parse::run_predict(
            &self.inner,
            new_x.as_ref(),
            shared_parse::PredictOptionSet {
                return_se: has_output(opts.outputs.as_ref(), "se"),
                confidence_level: opts.intervals.as_ref().and_then(|value| value.confidence),
                prediction_level: opts.intervals.as_ref().and_then(|value| value.prediction),
                return_derivative: has_output(opts.outputs.as_ref(), "gradient")
                    || has_output(opts.outputs.as_ref(), "derivative"),
                extrapolation: opts.extrapolation.as_deref(),
                max_extrapolation_distance: opts.max_extrapolation_distance,
                max_neighbor_distance: opts.max_neighbor_distance,
            },
        ))?;
        Ok(PredictOutput { inner: output })
    }
}

/// Options for `LoessResult.predict()`.
#[napi(object)]
#[derive(Default)]
pub struct PredictOptions {
    /// Optional output components: se, gradient (or derivative).
    pub outputs: Option<Vec<String>>,
    pub intervals: Option<IntervalsOptions>,
    /// Behavior for query points outside the training range ("clamp", "linear", "error"). Default: "clamp".
    pub extrapolation: Option<String>,
    /// Under "linear" extrapolation, the maximum allowed distance beyond the training
    /// boundary before `predict()` errors instead of returning an unbounded value.
    #[napi(js_name = "max_extrapolation_distance")]
    pub max_extrapolation_distance: Option<f64>,
    /// Maximum allowed distance to the farthest point in a query's neighbor window
    /// before `predict()` errors, catching in-range-but-sparse query points.
    #[napi(js_name = "max_neighbor_distance")]
    pub max_neighbor_distance: Option<f64>,
}

/// Result of `LoessResult.predict()`.
#[napi]
pub struct PredictOutput {
    inner: shared_parse::PredictOutput<f64>,
}

#[napi]
impl PredictOutput {
    /// Predicted y values, one per query point.
    #[napi(getter)]
    pub fn get_y(&self) -> Float64Array {
        Float64Array::from(self.inner.y.as_slice())
    }

    /// Standard errors (if requested).
    #[napi(getter, js_name = "standard_errors")]
    pub fn get_standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Lower confidence interval bounds (if requested).
    #[napi(getter, js_name = "confidence_lower")]
    pub fn get_confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Upper confidence interval bounds (if requested).
    #[napi(getter, js_name = "confidence_upper")]
    pub fn get_confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Lower prediction interval bounds (if requested).
    #[napi(getter, js_name = "prediction_lower")]
    pub fn get_prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Upper prediction interval bounds (if requested).
    #[napi(getter, js_name = "prediction_upper")]
    pub fn get_prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    /// Local fit's gradient at each query point (if requested).
    #[napi(getter)]
    pub fn get_derivative(&self) -> Option<Float64Array> {
        self.inner
            .derivative
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }
}

fn has_output(outputs: Option<&Vec<String>>, name: &str) -> bool {
    outputs.is_some_and(|values| values.iter().any(|value| value == name))
}

fn validate_outputs(outputs: Option<&Vec<String>>, allowed: &[&str]) -> Result<()> {
    if let Some(output) = outputs
        .into_iter()
        .flatten()
        .find(|value| !allowed.contains(&value.as_str()))
    {
        return Err(to_napi_error(shared_parse::BindingError::invalid_arg(
            format!(
                "unknown output '{output}'. Valid outputs: {}",
                allowed.join(", ")
            ),
        )));
    }
    Ok(())
}

#[napi(object)]
pub struct CVOptions {
    pub fractions: Vec<f64>,
    pub method: Option<String>,
    pub k: Option<u32>,
}

#[napi(object)]
pub struct IntervalsOptions {
    pub confidence: Option<f64>,
    pub prediction: Option<f64>,
}

/// Configuration options for LOESS smoothing.
#[napi(object)]
pub struct SmoothOptions {
    /// Smoothing fraction (0 < fraction <= 1). Default: 0.67.
    pub fraction: Option<f64>,
    /// Number of robustness iterations. Default: 3.
    pub iterations: Option<u32>,
    /// Weight function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube".
    #[napi(js_name = "weight_function")]
    pub weight_function: Option<String>,
    /// Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare".
    #[napi(js_name = "robustness_method")]
    pub robustness_method: Option<String>,
    /// Fallback strategy when weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean".
    #[napi(js_name = "zero_weight_fallback")]
    pub zero_weight_fallback: Option<String>,
    /// Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend".
    #[napi(js_name = "boundary_policy")]
    pub boundary_policy: Option<String>,
    /// Scaling method ("mad", "mar", "mean"). Default: "mad".
    #[napi(js_name = "scaling_method")]
    pub scaling_method: Option<String>,
    /// Auto-convergence tolerance. Default: None.
    #[napi(js_name = "auto_converge")]
    pub auto_converge: Option<f64>,
    /// Optional output components: diagnostics, residuals, weights, gradient (or derivative), se, sorted.
    pub outputs: Option<Vec<String>>,
    /// Grouped cross-validation configuration for Batch smoothing.
    pub cv: Option<CVOptions>,
    pub intervals: Option<IntervalsOptions>,
    /// Enable parallel execution. Default: true.
    pub parallel: Option<bool>,
    /// Polynomial degree ("constant", "linear", "quadratic", etc.). Default: "linear".
    pub degree: Option<String>,
    /// Number of predictor dimensions. Default: 1.
    pub dimensions: Option<u32>,
    /// Distance metric ("normalized", "euclidean", "manhattan", "chebyshev", "minkowski:p", "weighted"). Default: "normalized".
    #[napi(js_name = "distance_metric")]
    pub distance_metric: Option<String>,
    /// Per-dimension weights for the "weighted" distance metric.
    #[napi(js_name = "weighted_metric_weights")]
    pub weighted_metric_weights: Option<Vec<f64>>,
    /// Surface mode ("interpolation" or "direct"). Default: "interpolation".
    #[napi(js_name = "surface_mode")]
    pub surface_mode: Option<String>,
    /// Interpolation cell size (default 0.2). Smaller = more vertices, higher accuracy.
    pub cell: Option<f64>,
    /// Maximum number of interpolation vertices.
    #[napi(js_name = "interpolation_vertices")]
    pub interpolation_vertices: Option<u32>,
    /// Reduce polynomial degree to linear at boundary vertices (default true).
    #[napi(js_name = "boundary_degree_fallback")]
    pub boundary_degree_fallback: Option<bool>,
    /// Non-negative JavaScript safe-integer seed for reproducible K-fold cross-validation splits.
    pub seed: Option<f64>,
    /// Policy for non-finite (NaN/Inf) values in input data ("error", "drop"). Default: "error".
    #[napi(js_name = "missing")]
    pub missing: Option<String>,
    /// Retain the fitted model's training data, enabling `LoessResult.predict()`. Default: false.
    #[napi(js_name = "retain_model")]
    pub retain_model: Option<bool>,
}

/// Configuration options for streaming LOESS smoothing.
///
/// A subset of [`SmoothOptions`]: cross-validation is Batch-only and has no
/// equivalent here, so it isn't a field on this type.
#[napi(object)]
pub struct StreamingSmoothOptions {
    /// Smoothing fraction (0 < fraction <= 1). Default: 0.67.
    pub fraction: Option<f64>,
    /// Number of robustness iterations. Default: 3.
    pub iterations: Option<u32>,
    /// Weight function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube".
    #[napi(js_name = "weight_function")]
    pub weight_function: Option<String>,
    /// Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare".
    #[napi(js_name = "robustness_method")]
    pub robustness_method: Option<String>,
    /// Fallback strategy when weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean".
    #[napi(js_name = "zero_weight_fallback")]
    pub zero_weight_fallback: Option<String>,
    /// Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend".
    #[napi(js_name = "boundary_policy")]
    pub boundary_policy: Option<String>,
    /// Scaling method ("mad", "mar", "mean"). Default: "mad".
    #[napi(js_name = "scaling_method")]
    pub scaling_method: Option<String>,
    /// Auto-convergence tolerance. Default: None.
    #[napi(js_name = "auto_converge")]
    pub auto_converge: Option<f64>,
    /// Optional output components: diagnostics, residuals, weights, gradient (or derivative), se.
    pub outputs: Option<Vec<String>>,
    pub intervals: Option<IntervalsOptions>,
    /// Enable parallel execution. Default: true.
    pub parallel: Option<bool>,
    /// Polynomial degree ("constant", "linear", "quadratic", etc.). Default: "linear".
    pub degree: Option<String>,
    /// Number of predictor dimensions. Default: 1.
    pub dimensions: Option<u32>,
    /// Distance metric ("normalized", "euclidean", "manhattan", "chebyshev", "minkowski:p", "weighted"). Default: "normalized".
    #[napi(js_name = "distance_metric")]
    pub distance_metric: Option<String>,
    /// Per-dimension weights for the "weighted" distance metric.
    #[napi(js_name = "weighted_metric_weights")]
    pub weighted_metric_weights: Option<Vec<f64>>,
    /// Surface mode ("interpolation" or "direct"). Default: "interpolation".
    #[napi(js_name = "surface_mode")]
    pub surface_mode: Option<String>,
    /// Interpolation cell size (default 0.2). Smaller = more vertices, higher accuracy.
    pub cell: Option<f64>,
    /// Maximum number of interpolation vertices.
    #[napi(js_name = "interpolation_vertices")]
    pub interpolation_vertices: Option<u32>,
    /// Reduce polynomial degree to linear at boundary vertices (default true).
    #[napi(js_name = "boundary_degree_fallback")]
    pub boundary_degree_fallback: Option<bool>,
    /// Policy for non-finite (NaN/Inf) values in each chunk ("error", "drop"). Default: "error".
    #[napi(js_name = "missing")]
    pub missing: Option<String>,
}

/// Configuration options for online LOESS smoothing.
///
/// A subset of [`SmoothOptions`]: diagnostics, residuals, parallel execution,
/// and cross-validation are all no-ops for online processing (it handles one
/// point at a time, always runs sequentially, and always returns a residual
/// inline), so they aren't fields on this type. `intervals` and the "se" output
/// require `update_mode: "full"`.
#[napi(object)]
pub struct OnlineSmoothOptions {
    /// Smoothing fraction (0 < fraction <= 1). Default: 0.67.
    pub fraction: Option<f64>,
    /// Number of robustness iterations. Default: 0; positive values require
    /// `update_mode: "full"`.
    pub iterations: Option<u32>,
    /// Weight function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube".
    #[napi(js_name = "weight_function")]
    pub weight_function: Option<String>,
    /// Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare".
    #[napi(js_name = "robustness_method")]
    pub robustness_method: Option<String>,
    /// Fallback strategy when weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean".
    #[napi(js_name = "zero_weight_fallback")]
    pub zero_weight_fallback: Option<String>,
    /// Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend".
    #[napi(js_name = "boundary_policy")]
    pub boundary_policy: Option<String>,
    /// Scaling method ("mad", "mar", "mean"). Default: "mad".
    #[napi(js_name = "scaling_method")]
    pub scaling_method: Option<String>,
    /// Auto-convergence tolerance. Default: None.
    #[napi(js_name = "auto_converge")]
    pub auto_converge: Option<f64>,
    /// Optional output components: weights, gradient (or derivative), se.
    pub outputs: Option<Vec<String>>,
    pub intervals: Option<IntervalsOptions>,
    /// Polynomial degree ("constant", "linear", "quadratic", etc.). Default: "linear".
    pub degree: Option<String>,
    /// Number of predictor dimensions; Online vector updates accept one coordinate per dimension.
    pub dimensions: Option<u32>,
    /// Distance metric ("normalized", "euclidean", "manhattan", "chebyshev", "minkowski:p", "weighted"). Default: "normalized".
    #[napi(js_name = "distance_metric")]
    pub distance_metric: Option<String>,
    /// Per-dimension weights for the "weighted" distance metric.
    #[napi(js_name = "weighted_metric_weights")]
    pub weighted_metric_weights: Option<Vec<f64>>,
    /// Surface mode ("interpolation" or "direct"). Default: "interpolation".
    #[napi(js_name = "surface_mode")]
    pub surface_mode: Option<String>,
    /// Interpolation cell size (default 0.2). Smaller = more vertices, higher accuracy.
    pub cell: Option<f64>,
    /// Maximum number of interpolation vertices.
    #[napi(js_name = "interpolation_vertices")]
    pub interpolation_vertices: Option<u32>,
    /// Reduce polynomial degree to linear at boundary vertices (default true).
    #[napi(js_name = "boundary_degree_fallback")]
    pub boundary_degree_fallback: Option<bool>,
    /// Policy for non-finite (NaN/Inf) `x`/`y` values passed to `addPoint` ("error", "drop"). Default: "error".
    #[napi(js_name = "missing")]
    pub missing: Option<String>,
}

/// Build a LoessBuilder from Batch options, applying every field.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

fn batch_options_to_builder(opts: Option<&SmoothOptions>) -> Result<LoessBuilder<f64>> {
    let mut builder = LoessBuilder::<f64>::new();
    if let Some(opts) = opts {
        validate_outputs(
            opts.outputs.as_ref(),
            &[
                "diagnostics",
                "residuals",
                "weights",
                "gradient",
                "derivative",
                "se",
                "sorted",
            ],
        )?;
        let grouped_cv = opts.cv.as_ref();
        let cv_seed = opts
            .seed
            .map(|seed| {
                if !seed.is_finite() || seed < 0.0 || seed.fract() != 0.0 || seed > MAX_SAFE_INTEGER
                {
                    Err(shared_parse::BindingError::invalid_arg(format!(
                        "seed must be a non-negative JavaScript safe integer, got {seed}"
                    )))
                } else {
                    Ok(seed as u64)
                }
            })
            .transpose()
            .map_err(to_napi_error)?;
        let (configured_builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations.map(|v| v as usize),
                weight_function: opts.weight_function.as_deref(),
                robustness_method: opts.robustness_method.as_deref(),
                zero_weight_fallback: opts.zero_weight_fallback.as_deref(),
                boundary_policy: opts.boundary_policy.as_deref(),
                scaling_method: opts.scaling_method.as_deref(),
                auto_converge: opts.auto_converge,
                return_residuals: has_output(opts.outputs.as_ref(), "residuals"),
                return_robustness_weights: has_output(opts.outputs.as_ref(), "weights"),
                return_diagnostics: has_output(opts.outputs.as_ref(), "diagnostics"),
                confidence_intervals: opts.intervals.as_ref().and_then(|value| value.confidence),
                prediction_intervals: opts.intervals.as_ref().and_then(|value| value.prediction),
                parallel: opts.parallel,
                degree: opts.degree.as_deref(),
                dimensions: opts.dimensions.map(|v| v as usize),
                distance_metric: opts.distance_metric.as_deref(),
                weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                surface_mode: opts.surface_mode.as_deref(),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                return_sorted: has_output(opts.outputs.as_ref(), "sorted"),
                cell: opts.cell,
                interpolation_vertices: opts.interpolation_vertices.map(|v| v as usize),
                boundary_degree_fallback: opts.boundary_degree_fallback,
                cv_fractions: grouped_cv.map(|cv| cv.fractions.as_slice()),
                cv_method: grouped_cv.and_then(|cv| cv.method.as_deref()),
                cv_k: grouped_cv.and_then(|cv| cv.k).map(|v| v as usize),
                cv_seed,
                missing: opts.missing.as_deref(),
                retain_model: opts.retain_model,
            },
        ))?;
        builder = configured_builder;
        if has_output(opts.outputs.as_ref(), "gradient")
            || has_output(opts.outputs.as_ref(), "derivative")
        {
            builder = builder.return_gradient();
        }
    }
    Ok(builder)
}

/// Build a LoessBuilder from Streaming options, applying every field.
fn streaming_options_to_builder(
    opts: Option<&StreamingSmoothOptions>,
) -> Result<LoessBuilder<f64>> {
    let mut builder = LoessBuilder::<f64>::new();
    if let Some(opts) = opts {
        validate_outputs(
            opts.outputs.as_ref(),
            &[
                "diagnostics",
                "residuals",
                "weights",
                "gradient",
                "derivative",
                "se",
            ],
        )?;
        let (configured_builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations.map(|v| v as usize),
                weight_function: opts.weight_function.as_deref(),
                robustness_method: opts.robustness_method.as_deref(),
                zero_weight_fallback: opts.zero_weight_fallback.as_deref(),
                boundary_policy: opts.boundary_policy.as_deref(),
                scaling_method: opts.scaling_method.as_deref(),
                auto_converge: opts.auto_converge,
                return_residuals: has_output(opts.outputs.as_ref(), "residuals"),
                return_robustness_weights: has_output(opts.outputs.as_ref(), "weights"),
                return_diagnostics: has_output(opts.outputs.as_ref(), "diagnostics"),
                confidence_intervals: opts.intervals.as_ref().and_then(|value| value.confidence),
                prediction_intervals: opts.intervals.as_ref().and_then(|value| value.prediction),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                parallel: opts.parallel,
                degree: opts.degree.as_deref(),
                dimensions: opts.dimensions.map(|v| v as usize),
                distance_metric: opts.distance_metric.as_deref(),
                weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                surface_mode: opts.surface_mode.as_deref(),
                cell: opts.cell,
                interpolation_vertices: opts.interpolation_vertices.map(|v| v as usize),
                boundary_degree_fallback: opts.boundary_degree_fallback,
                missing: opts.missing.as_deref(),
                ..Default::default()
            },
        ))?;
        builder = configured_builder;
        if has_output(opts.outputs.as_ref(), "gradient")
            || has_output(opts.outputs.as_ref(), "derivative")
        {
            builder = builder.return_gradient();
        }
    }
    Ok(builder)
}

/// Build a LoessBuilder from Online options, applying every field.
fn online_options_to_builder(opts: Option<&OnlineSmoothOptions>) -> Result<LoessBuilder<f64>> {
    let mut builder = LoessBuilder::<f64>::new();
    if let Some(opts) = opts {
        validate_outputs(
            opts.outputs.as_ref(),
            &["weights", "gradient", "derivative", "se"],
        )?;
        let (configured_builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations.map(|v| v as usize),
                weight_function: opts.weight_function.as_deref(),
                robustness_method: opts.robustness_method.as_deref(),
                zero_weight_fallback: opts.zero_weight_fallback.as_deref(),
                boundary_policy: opts.boundary_policy.as_deref(),
                scaling_method: opts.scaling_method.as_deref(),
                auto_converge: opts.auto_converge,
                return_robustness_weights: has_output(opts.outputs.as_ref(), "weights"),
                degree: opts.degree.as_deref(),
                dimensions: opts.dimensions.map(|v| v as usize),
                distance_metric: opts.distance_metric.as_deref(),
                weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                surface_mode: opts.surface_mode.as_deref(),
                cell: opts.cell,
                interpolation_vertices: opts.interpolation_vertices.map(|v| v as usize),
                boundary_degree_fallback: opts.boundary_degree_fallback,
                missing: opts.missing.as_deref(),
                confidence_intervals: opts.intervals.as_ref().and_then(|value| value.confidence),
                prediction_intervals: opts.intervals.as_ref().and_then(|value| value.prediction),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                ..Default::default()
            },
        ))?;
        builder = configured_builder;
        if has_output(opts.outputs.as_ref(), "gradient")
            || has_output(opts.outputs.as_ref(), "derivative")
        {
            builder = builder.return_gradient();
        }
    }
    Ok(builder)
}

/// Batch LOESS smoothing.
#[napi]
pub struct Loess {
    options: Option<SmoothOptions>,
}

#[napi]
impl Loess {
    /// Create a new batch LOESS smoother.
    #[napi(constructor)]
    pub fn new(options: Option<SmoothOptions>) -> Self {
        Self { options }
    }

    /// Fit the model.
    #[napi]
    pub fn fit(
        &self,
        x: Float64Array,
        y: Float64Array,
        custom_weights: Option<Float64Array>,
    ) -> Result<LoessResult> {
        let builder = self.create_builder()?;
        let model = map_runtime(shared_parse::build_batch(
            builder,
            custom_weights.map(|cw| cw.as_ref().to_vec()),
        ))?;
        let result = map_runtime(model.fit(x.as_ref(), y.as_ref()))?;
        Ok(LoessResult { inner: result })
    }

    /// Fit the model asynchronously.
    #[napi(js_name = "fit_async")]
    pub fn fit_async(
        &self,
        x: Float64Array,
        y: Float64Array,
        custom_weights: Option<Float64Array>,
    ) -> Result<AsyncTask<LoessTask>> {
        let mut builder = self.create_builder()?;
        if let Some(cw) = custom_weights {
            builder = builder.custom_weights(cw.as_ref().to_vec());
        }
        let x_vec = x.as_ref().to_vec();
        let y_vec = y.as_ref().to_vec();

        Ok(AsyncTask::new(LoessTask {
            builder,
            x: x_vec,
            y: y_vec,
        }))
    }

    fn create_builder(&self) -> Result<LoessBuilder<f64>> {
        batch_options_to_builder(self.options.as_ref())
    }
}

pub struct LoessTask {
    builder: LoessBuilder<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Task for LoessTask {
    type Output = InnerLoessResult<f64>;
    type JsValue = LoessResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let model = map_runtime(shared_parse::build_batch(self.builder.clone(), None))?;
        map_runtime(model.fit(&self.x, &self.y))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(LoessResult { inner: output })
    }
}

/// Configuration options for streaming processing.
#[napi(object)]
pub struct StreamingOptions {
    /// Size of each data chunk. Default: 5000.
    #[napi(js_name = "chunk_size")]
    pub chunk_size: Option<u32>,
    /// Header/footer overlap size. Default: chunk_size / 10, min. 1.
    pub overlap: Option<u32>,
    /// Strategy for merging chunk overlaps ("average", "weighted_average", "take_first", "take_last").
    #[napi(js_name = "merge_strategy")]
    pub merge_strategy: Option<String>,
}

/// Streaming LOESS smoother for large datasets.
#[napi]
pub struct StreamingLoess {
    inner: ParallelStreamingLoess<f64>,
}

#[napi]
impl StreamingLoess {
    /// Create a new streaming LOESS smoother.
    #[napi(constructor)]
    pub fn new(
        options: Option<StreamingSmoothOptions>,
        streaming_opts: Option<StreamingOptions>,
    ) -> Result<Self> {
        let builder = streaming_options_to_builder(options.as_ref())?;

        let (chunk_size, overlap, merge_strategy) = match streaming_opts {
            Some(s) => (
                s.chunk_size.map(|v| v as usize),
                s.overlap.map(|v| v as usize),
                s.merge_strategy,
            ),
            None => (None, None, None),
        };

        let model = map_runtime(shared_parse::build_streaming(
            builder,
            chunk_size,
            overlap,
            merge_strategy.as_deref(),
        ))?;

        Ok(StreamingLoess { inner: model })
    }

    /// Process a chunk of data.
    #[napi(js_name = "process_chunk")]
    pub fn process_chunk(&mut self, x: Float64Array, y: Float64Array) -> Result<LoessResult> {
        let result: InnerLoessResult<f64> =
            map_runtime(self.inner.process_chunk(x.as_ref(), y.as_ref()))?;
        Ok(LoessResult { inner: result })
    }

    /// Process a chunk with one case weight per observation.
    #[napi(js_name = "process_chunk_weighted")]
    pub fn process_chunk_weighted(
        &mut self,
        x: Float64Array,
        y: Float64Array,
        custom_weights: Float64Array,
    ) -> Result<LoessResult> {
        let result: InnerLoessResult<f64> = map_runtime(self.inner.process_chunk_weighted(
            x.as_ref(),
            y.as_ref(),
            custom_weights.as_ref(),
        ))?;
        Ok(LoessResult { inner: result })
    }

    /// Finalize the stream and return remaining data.
    #[napi]
    pub fn finalize(&mut self) -> Result<LoessResult> {
        let result: InnerLoessResult<f64> = map_runtime(self.inner.finalize())?;
        Ok(LoessResult { inner: result })
    }
}

/// Configuration options for online processing.
#[napi(object)]
pub struct OnlineOptions {
    /// Maximum number of points to keep in the window. Default: 1000.
    #[napi(js_name = "window_capacity")]
    pub window_capacity: Option<u32>,
    /// Minimum points required before smoothing starts. Default: 2.
    #[napi(js_name = "min_points")]
    pub min_points: Option<u32>,
    /// Update mode ("full", "incremental"). Default: "incremental".
    #[napi(js_name = "update_mode")]
    pub update_mode: Option<String>,
}

/// Online LOESS smoother for real-time data.
#[napi]
pub struct OnlineLoess {
    inner: ParallelOnlineLoess<f64>,
}

#[napi]
impl OnlineLoess {
    /// Create a new online LOESS smoother.
    #[napi(constructor)]
    pub fn new(
        options: Option<OnlineSmoothOptions>,
        online_opts: Option<OnlineOptions>,
    ) -> Result<Self> {
        let builder = online_options_to_builder(options.as_ref())?;

        let (window_capacity, min_points, update_mode) = match online_opts {
            Some(o) => (
                o.window_capacity.map(|v| v as usize),
                o.min_points.map(|v| v as usize),
                o.update_mode,
            ),
            None => (None, None, None),
        };

        let model = map_runtime(shared_parse::build_online(
            builder,
            window_capacity,
            min_points,
            update_mode.as_deref(),
        ))?;

        Ok(OnlineLoess { inner: model })
    }

    /// Add a single point and get the smoothed value if enough points are available.
    #[napi(js_name = "add_point")]
    pub fn add_point(
        &mut self,
        x: f64,
        y: f64,
        weight: Option<f64>,
    ) -> Result<Option<OnlineOutput>> {
        let output = self
            .inner
            .add_point_weighted(&[x], y, weight.unwrap_or(1.0))
            .map_err(|e| to_napi_error(shared_parse::BindingError::invalid_arg(e.to_string())))?;
        Ok(output.map(|o| OnlineOutput {
            y: o.y,
            standard_error: o.standard_error,
            residual: o.residual,
            robustness_weight: o.robustness_weight,
            iterations_used: o.iterations_used.map(|i| i as u32),
            confidence_lower: o.confidence_lower,
            confidence_upper: o.confidence_upper,
            prediction_lower: o.prediction_lower,
            prediction_upper: o.prediction_upper,
            gradient: o.gradient,
        }))
    }

    /// Add a point with one coordinate per configured predictor dimension.
    #[napi(js_name = "add_point_vector")]
    pub fn add_point_vector(
        &mut self,
        x: Float64Array,
        y: f64,
        weight: Option<f64>,
    ) -> Result<Option<OnlineOutput>> {
        let x = x.as_ref().to_vec();
        let output = self
            .inner
            .add_point_weighted(&x, y, weight.unwrap_or(1.0))
            .map_err(|e| to_napi_error(shared_parse::BindingError::invalid_arg(e.to_string())))?;
        Ok(output.map(|o| OnlineOutput {
            y: o.y,
            standard_error: o.standard_error,
            residual: o.residual,
            robustness_weight: o.robustness_weight,
            iterations_used: o.iterations_used.map(|i| i as u32),
            confidence_lower: o.confidence_lower,
            confidence_upper: o.confidence_upper,
            prediction_lower: o.prediction_lower,
            prediction_upper: o.prediction_upper,
            gradient: o.gradient,
        }))
    }

    /// Compute diagnostics for the current sliding window on demand.
    #[napi(js_name = "window_diagnostics")]
    pub fn window_diagnostics(&self) -> Result<Option<Diagnostics>> {
        self.inner
            .window_diagnostics()
            .map_err(|e| to_napi_error(shared_parse::BindingError::runtime(e.to_string())))
            .map(|result| {
                result.map(|d| Diagnostics {
                    rmse: d.rmse,
                    mae: d.mae,
                    r_squared: d.r_squared,
                    aic: d.aic,
                    aicc: d.aicc,
                    effective_df: d.effective_df,
                    residual_sd: d.residual_sd,
                })
            })
    }

    /// Predict query points using a fitted model of the current sliding window.
    #[napi(js_name = "predict_window")]
    pub fn predict_window(
        &self,
        new_x: Float64Array,
        options: Option<PredictOptions>,
    ) -> Result<PredictOutput> {
        let PredictOptions {
            outputs,
            intervals,
            extrapolation,
            max_extrapolation_distance,
            max_neighbor_distance,
        } = options.unwrap_or_default();
        validate_outputs(outputs.as_ref(), &["se", "gradient", "derivative"])?;

        let mut interval_builder = IntervalsBuilder::new();
        if let Some(intervals) = intervals {
            if let Some(level) = intervals.confidence {
                interval_builder = interval_builder.confidence(level);
            }
            if let Some(level) = intervals.prediction {
                interval_builder = interval_builder.prediction(level);
            }
        }
        let mut builder = Predict::new()
            .intervals(interval_builder)
            .extrapolation(extrapolation.as_deref().unwrap_or("clamp"));
        if let Some(outputs) = outputs {
            builder = builder.outputs(outputs);
        }
        if let Some(distance) = max_extrapolation_distance {
            builder = builder.max_extrapolation_distance(distance);
        }
        if let Some(distance) = max_neighbor_distance {
            builder = builder.max_neighbor_distance(distance);
        }
        let query = map_invalid_arg(builder.build())?;
        let output = map_invalid_arg(self.inner.predict_window(new_x.as_ref(), &query))?;
        Ok(PredictOutput { inner: output })
    }
}
