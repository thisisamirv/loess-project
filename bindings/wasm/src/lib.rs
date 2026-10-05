//! WebAssembly bindings for fastLoess.

use js_sys::{Float64Array, Object, Reflect};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Returns the version of this WebAssembly binding package.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

// ============================================================================
// TypeScript interface declarations injected into the generated .d.ts
// ============================================================================

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
/** Configuration options for LOESS smoothing. */
export interface IntervalsOptions { confidence?: number; prediction?: number; }
export interface SmoothOptions {
    /** Optional output components: diagnostics, residuals, weights, gradient (or derivative), se, sorted. */
    outputs?: string[];
    intervals?: IntervalsOptions;
    /** Grouped batch cross-validation configuration. */
    cv?: { fractions: number[]; method?: string; k?: number };
    /** Smoothing fraction (0 < fraction <= 1). Default: 0.67. */
    fraction?: number;
    /** Number of robustness iterations. Default: 3. */
    iterations?: number;
    /** Kernel function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube". */
    weight_function?: string;
    /** Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare". */
    robustness_method?: string;
    /** Fallback when all weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean". */
    zero_weight_fallback?: string;
    /** Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend". */
    boundary_policy?: string;
    /** Scaling method ("mad", "mar", "mean"). Default: "mad". */
    scaling_method?: string;
    /** Auto-convergence tolerance. Disabled when absent. */
    auto_converge?: number;
    /** Enable parallel execution. Default: true. */
    parallel?: boolean;
    /** Polynomial degree ("constant", "linear", "quadratic", "cubic", "quartic"). Default: "linear". */
    degree?: string;
    /** Number of predictor dimensions. Default: 1. */
    dimensions?: number;
    /** Distance metric ("normalized", "euclidean", "manhattan", "chebyshev", "minkowski:p", "weighted"). Default: "normalized". */
    distance_metric?: string;
    /** Surface computation mode ("interpolation" or "direct"). Default: "interpolation". */
    surface_mode?: string;
    /** Per-dimension weights for the weighted distance metric. */
    weighted_metric_weights?: number[];
    /** Cell parameter for interpolation (fraction of data). Default: 0.2. */
    cell?: number;
    /** Number of interpolation vertices. Default: auto. */
    interpolation_vertices?: number;
    /** Fall back to lower polynomial degree at boundaries. Default: true. */
    boundary_degree_fallback?: boolean;
    /** Non-negative safe-integer seed for cross-validation (at most Number.MAX_SAFE_INTEGER). */
    seed?: number;
    /** Policy for non-finite (NaN/Inf) values in input data ("error", "drop"). Default: "error". */
    missing?: string;
    /** Retain the fitted model's training data, enabling `LoessResult.predict()`. Default: false. */
    retain_model?: boolean;
}

/** Options for `LoessResult.predict()`. */
export interface PredictOptions {
    /** Optional prediction components: se, gradient (or derivative). */
    outputs?: string[];
    intervals?: IntervalsOptions;
    /** Behavior for query points outside the training range ("clamp", "linear", "error"). Default: "clamp". */
    extrapolation?: string;
    /** Under "linear" extrapolation, the maximum allowed distance beyond the training boundary before `predict()` errors instead of returning an unbounded value. */
    max_extrapolation_distance?: number;
    /** Maximum allowed distance to the farthest point in a query's neighbor window before `predict()` errors, catching in-range-but-sparse query points. */
    max_neighbor_distance?: number;
}

/** Out-of-sample prediction methods added to the generated result class. */
export interface LoessResult {
    predict(newX: Float64Array, options?: PredictOptions): PredictOutput;
}

/** Result of `LoessResult.predict()`. */
export interface PredictOutput {
    /** Predicted y values, one per query point. */
    readonly y: Float64Array;
    /** Standard errors (if requested). */
    readonly standard_errors: Float64Array | undefined;
    /** Lower confidence interval bounds (if requested). */
    readonly confidence_lower: Float64Array | undefined;
    /** Upper confidence interval bounds (if requested). */
    readonly confidence_upper: Float64Array | undefined;
    /** Lower prediction interval bounds (if requested). */
    readonly prediction_lower: Float64Array | undefined;
    /** Upper prediction interval bounds (if requested). */
    readonly prediction_upper: Float64Array | undefined;
    /** Local fit's gradient at each query point (if requested). */
    readonly derivative: Float64Array | undefined;
}

/** Configuration options for streaming LOESS smoothing. A subset of `SmoothOptions`: cross-validation has no equivalent here. */
export interface StreamingSmoothOptions {
    /** Optional output components: diagnostics, residuals, weights, gradient (or derivative), se. */
    outputs?: string[];
    intervals?: IntervalsOptions;
    /** Smoothing fraction (0 < fraction <= 1). Default: 0.67. */
    fraction?: number;
    /** Number of robustness iterations. Default: 3. */
    iterations?: number;
    /** Kernel function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube". */
    weight_function?: string;
    /** Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare". */
    robustness_method?: string;
    /** Fallback when all weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean". */
    zero_weight_fallback?: string;
    /** Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend". */
    boundary_policy?: string;
    /** Scaling method ("mad", "mar", "mean"). Default: "mad". */
    scaling_method?: string;
    /** Auto-convergence tolerance. Disabled when absent. */
    auto_converge?: number;
    /** Enable parallel execution. Default: true. */
    parallel?: boolean;
    /** Polynomial degree ("constant", "linear", "quadratic", "cubic", "quartic"). Default: "linear". */
    degree?: string;
    /** Number of predictor dimensions. Default: 1. */
    dimensions?: number;
    /** Distance metric ("normalized", "euclidean", "manhattan", "chebyshev", "minkowski:p", "weighted"). Default: "normalized". */
    distance_metric?: string;
    /** Surface computation mode ("interpolation" or "direct"). Default: "interpolation". */
    surface_mode?: string;
    /** Per-dimension weights for the weighted distance metric. */
    weighted_metric_weights?: number[];
    /** Cell parameter for interpolation (fraction of data). Default: 0.2. */
    cell?: number;
    /** Number of interpolation vertices. Default: auto. */
    interpolation_vertices?: number;
    /** Fall back to lower polynomial degree at boundaries. Default: true. */
    boundary_degree_fallback?: boolean;
    /** Policy for non-finite (NaN/Inf) values in each chunk ("error", "drop"). Default: "error". */
    missing?: string;
}

/** Configuration options for online LOESS smoothing. A subset of `SmoothOptions`: diagnostics, residuals, parallel execution, and cross-validation have no equivalent here. `confidence_intervals`/`prediction_intervals` and the `se` output require `update_mode: "full"`. */
export interface OnlineSmoothOptions {
    /** Optional output components: weights, gradient (or derivative), se. */
    outputs?: string[];
    intervals?: IntervalsOptions;
    /** Smoothing fraction (0 < fraction <= 1). Default: 0.67. */
    fraction?: number;
    /** Number of robustness iterations. Default: 0; positive values require
     * `update_mode: "full"`. */
    iterations?: number;
    /** Kernel function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube". */
    weight_function?: string;
    /** Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare". */
    robustness_method?: string;
    /** Fallback when all weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean". */
    zero_weight_fallback?: string;
    /** Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend". */
    boundary_policy?: string;
    /** Scaling method ("mad", "mar", "mean"). Default: "mad". */
    scaling_method?: string;
    /** Auto-convergence tolerance. Disabled when absent. */
    auto_converge?: number;
    /** Polynomial degree ("constant", "linear", "quadratic", "cubic", "quartic"). Default: "linear". */
    degree?: string;
    /** Number of predictor dimensions. Default: 1. */
    dimensions?: number;
    /** Distance metric ("normalized", "euclidean", "manhattan", "chebyshev", "minkowski:p", "weighted"). Default: "normalized". */
    distance_metric?: string;
    /** Surface computation mode ("interpolation" or "direct"). Default: "interpolation". */
    surface_mode?: string;
    /** Per-dimension weights for the weighted distance metric. */
    weighted_metric_weights?: number[];
    /** Cell parameter for interpolation (fraction of data). Default: 0.2. */
    cell?: number;
    /** Number of interpolation vertices. Default: auto. */
    interpolation_vertices?: number;
    /** Fall back to lower polynomial degree at boundaries. Default: true. */
    boundary_degree_fallback?: boolean;
    /** Policy for non-finite (NaN/Inf) `x`/`y` values passed to `add_point` ("error", "drop"). Default: "error". */
    missing?: string;
}

/** Configuration options for streaming LOESS. */
export interface StreamingOptions {
    /** Size of each processing chunk. Default: 5000. */
    chunk_size?: number;
    /** Overlap between adjacent chunks. Default: chunk_size / 10, min. 1. */
    overlap?: number;
    /** Strategy for merging chunks ("average", "weighted_average", "take_first", "take_last"). Default: "weighted_average". */
    merge_strategy?: string;
}

/** Configuration options for online LOESS. */
export interface OnlineOptions {
    /** Maximum number of points to retain in the sliding window. Default: 1000. */
    window_capacity?: number;
    /** Minimum points required before smoothing starts. Default: 2. */
    min_points?: number;
    /** Update strategy ("full" or "incremental"). Default: "incremental". */
    update_mode?: string;
}

/** Batch LOESS smoother. */
export class Loess {
    free(): void;
    constructor(options?: SmoothOptions);
    /** Fit the model to data and return smoothed values. */
    fit(x: Float64Array, y: Float64Array, customWeights?: Float64Array): LoessResult;
}

/** Streaming LOESS smoother for large datasets. */
export class StreamingLoess {
    free(): void;
    constructor(options?: StreamingSmoothOptions, streamingOpts?: StreamingOptions);
    /** Process a chunk of data. */
    process_chunk(x: Float64Array, y: Float64Array): LoessResult;
    /** Process a chunk with one case weight per observation. */
    process_chunk_weighted(x: Float64Array, y: Float64Array, weights: Float64Array): LoessResult;
    /** Finalize the stream and return remaining data. */
    finalize(): LoessResult;
}

/** Online LOESS smoother for real-time data. */
export class OnlineLoess {
    free(): void;
    constructor(options?: OnlineSmoothOptions, onlineOpts?: OnlineOptions);
    /** Add a single point and get the smoothed value, or null if not enough points yet. */
    add_point(x: number, y: number): OnlineOutput | null;
    /** Add a point with one coordinate per configured predictor dimension. */
    add_point_vector(x: Float64Array, y: number): OnlineOutput | null;
    /** Add a scalar point with a case weight. */
    add_point_weighted(x: number, y: number, weight: number): OnlineOutput | null;
    /** Add a coordinate vector with a case weight. */
    add_point_vector_weighted(x: Float64Array, y: number, weight: number): OnlineOutput | null;
    /** Compute fit diagnostics for the current window, or null before warm-up. */
    window_diagnostics(): Diagnostics | null;
    /** Predict query points using a fit of the current window. */
    predict_window(newX: Float64Array, options?: PredictOptions): PredictOutput;
}

"#;

use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::api::LoessBuilder;
use ::fastLoess::internals::binding_support as shared_parse;
use ::fastLoess::prelude::{IntervalsBuilder, LoessResult as InnerLoessResult, Predict};

fn to_js_error(err: shared_parse::BindingError) -> JsValue {
    JsValue::from_str(&err.message)
}

fn map_invalid_arg<T, E: ToString>(result: Result<T, E>) -> Result<T, JsValue> {
    shared_parse::map_invalid_arg(result).map_err(to_js_error)
}

fn map_runtime<T, E: ToString>(result: Result<T, E>) -> Result<T, JsValue> {
    shared_parse::map_runtime(result).map_err(to_js_error)
}

fn validate_option_keys(value: &JsValue, name: &str, allowed: &[&str]) -> Result<(), JsValue> {
    if value.is_undefined() || value.is_null() || !value.is_object() {
        return Ok(());
    }

    let object: &Object = value.unchecked_ref();
    for key in Object::keys(object).iter() {
        if let Some(key) = key.as_string()
            && !allowed.contains(&key.as_str())
        {
            return Err(JsValue::from_str(&format!(
                "unknown {name} option '{key}'. Valid options: {}",
                allowed.join(", ")
            )));
        }
    }
    Ok(())
}

fn validate_nested_option_keys(
    value: &JsValue,
    parent: &str,
    key: &str,
    allowed: &[&str],
) -> Result<(), JsValue> {
    if value.is_undefined() || value.is_null() || !value.is_object() {
        return Ok(());
    }
    let nested = Reflect::get(value, &JsValue::from_str(key))?;
    validate_option_keys(&nested, parent, allowed)
}

fn validate_outputs(outputs: Option<&Vec<String>>, allowed: &[&str]) -> Result<(), JsValue> {
    if let Some(output) = outputs
        .into_iter()
        .flatten()
        .find(|value| !allowed.contains(&value.as_str()))
    {
        return Err(JsValue::from_str(&format!(
            "unknown output '{output}'. Valid outputs: {}",
            allowed.join(", ")
        )));
    }
    Ok(())
}

fn to_float64_array(values: &[f64]) -> Float64Array {
    Float64Array::from(values)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothOptions {
    pub outputs: Option<Vec<String>>,
    pub cv: Option<CVOptionsJs>,
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub weight_function: Option<String>,
    pub robustness_method: Option<String>,
    pub zero_weight_fallback: Option<String>,
    pub boundary_policy: Option<String>,
    pub scaling_method: Option<String>,
    pub auto_converge: Option<f64>,
    pub intervals: Option<IntervalsOptionsJs>,
    #[serde(rename = "parallel")]
    pub parallel: Option<bool>,
    pub degree: Option<String>,
    pub dimensions: Option<usize>,
    pub distance_metric: Option<String>,
    pub surface_mode: Option<String>,
    pub weighted_metric_weights: Option<Vec<f64>>,
    pub cell: Option<f64>,
    pub interpolation_vertices: Option<usize>,
    pub boundary_degree_fallback: Option<bool>,
    pub seed: Option<f64>,
    pub missing: Option<String>,
    pub retain_model: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CVOptionsJs {
    pub fractions: Vec<f64>,
    pub method: Option<String>,
    pub k: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntervalsOptionsJs {
    pub confidence: Option<f64>,
    pub prediction: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictOptionsJs {
    pub outputs: Option<Vec<String>>,
    pub intervals: Option<IntervalsOptionsJs>,
    pub extrapolation: Option<String>,
    pub max_extrapolation_distance: Option<f64>,
    pub max_neighbor_distance: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingOptions {
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub merge_strategy: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnlineOptions {
    pub window_capacity: Option<usize>,
    pub min_points: Option<usize>,
    pub update_mode: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingSmoothOptions {
    pub outputs: Option<Vec<String>>,
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub weight_function: Option<String>,
    pub robustness_method: Option<String>,
    pub zero_weight_fallback: Option<String>,
    pub boundary_policy: Option<String>,
    pub scaling_method: Option<String>,
    pub auto_converge: Option<f64>,
    pub parallel: Option<bool>,
    pub degree: Option<String>,
    pub dimensions: Option<usize>,
    pub distance_metric: Option<String>,
    pub surface_mode: Option<String>,
    pub weighted_metric_weights: Option<Vec<f64>>,
    pub cell: Option<f64>,
    pub interpolation_vertices: Option<usize>,
    pub boundary_degree_fallback: Option<bool>,
    pub missing: Option<String>,
    pub intervals: Option<IntervalsOptionsJs>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnlineSmoothOptions {
    pub outputs: Option<Vec<String>>,
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub weight_function: Option<String>,
    pub robustness_method: Option<String>,
    pub zero_weight_fallback: Option<String>,
    pub boundary_policy: Option<String>,
    pub scaling_method: Option<String>,
    pub auto_converge: Option<f64>,
    pub degree: Option<String>,
    pub dimensions: Option<usize>,
    pub distance_metric: Option<String>,
    pub surface_mode: Option<String>,
    pub weighted_metric_weights: Option<Vec<f64>>,
    pub cell: Option<f64>,
    pub interpolation_vertices: Option<usize>,
    pub boundary_degree_fallback: Option<bool>,
    pub missing: Option<String>,
    pub intervals: Option<IntervalsOptionsJs>,
}

#[wasm_bindgen]
pub struct Diagnostics {
    pub rmse: f64,
    pub mae: f64,
    #[wasm_bindgen(js_name = r_squared)]
    pub r_squared: f64,
    pub aic: Option<f64>,
    pub aicc: Option<f64>,
    #[wasm_bindgen(js_name = effective_df)]
    pub effective_df: Option<f64>,
    /// Batch: robust residual scale estimate (1.4826 * MAD); Streaming: cumulative sample SD of emitted residuals.
    #[wasm_bindgen(js_name = residual_sd)]
    pub residual_sd: f64,
}

// Result of a single online update step.
#[wasm_bindgen]
pub struct OnlineOutput {
    y: f64,
    standard_error: Option<f64>,
    residual: Option<f64>,
    robustness_weight: Option<f64>,
    iterations_used: Option<usize>,
    confidence_lower: Option<f64>,
    confidence_upper: Option<f64>,
    prediction_lower: Option<f64>,
    prediction_upper: Option<f64>,
    gradient: Option<Vec<f64>>,
}

#[wasm_bindgen]
impl OnlineOutput {
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f64 {
        self.y
    }

    #[wasm_bindgen(getter, js_name = "standard_error")]
    pub fn standard_error(&self) -> Option<f64> {
        self.standard_error
    }

    #[wasm_bindgen(getter)]
    pub fn residual(&self) -> Option<f64> {
        self.residual
    }

    #[wasm_bindgen(getter, js_name = "robustness_weight")]
    pub fn robustness_weight(&self) -> Option<f64> {
        self.robustness_weight
    }

    #[wasm_bindgen(getter, js_name = "iterations_used")]
    pub fn iterations_used(&self) -> Option<u32> {
        self.iterations_used.map(|i| i as u32)
    }

    #[wasm_bindgen(getter, js_name = "confidence_lower")]
    pub fn confidence_lower(&self) -> Option<f64> {
        self.confidence_lower
    }

    #[wasm_bindgen(getter, js_name = "confidence_upper")]
    pub fn confidence_upper(&self) -> Option<f64> {
        self.confidence_upper
    }

    #[wasm_bindgen(getter, js_name = "prediction_lower")]
    pub fn prediction_lower(&self) -> Option<f64> {
        self.prediction_lower
    }

    #[wasm_bindgen(getter, js_name = "prediction_upper")]
    pub fn prediction_upper(&self) -> Option<f64> {
        self.prediction_upper
    }

    #[wasm_bindgen(getter)]
    pub fn gradient(&self) -> Option<Float64Array> {
        self.gradient.as_deref().map(to_float64_array)
    }
}

#[wasm_bindgen]
pub struct LoessResult {
    inner: InnerLoessResult<f64>,
}

#[wasm_bindgen]
impl LoessResult {
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> Float64Array {
        to_float64_array(&self.inner.x)
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> Float64Array {
        to_float64_array(&self.inner.y)
    }

    #[wasm_bindgen(getter)]
    pub fn residuals(&self) -> Option<Float64Array> {
        self.inner.residuals.as_ref().map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = standard_errors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = confidence_lower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = confidence_upper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = prediction_lower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = prediction_upper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = robustness_weights)]
    pub fn robustness_weights(&self) -> Option<Float64Array> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter)]
    pub fn diagnostics(&self) -> Option<Diagnostics> {
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

    #[wasm_bindgen(getter, js_name = cv_scores)]
    pub fn cv_scores(&self) -> Option<Float64Array> {
        self.inner.cv_scores.as_ref().map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = fraction_used)]
    pub fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    #[wasm_bindgen(getter, js_name = iterations_used)]
    pub fn iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }

    #[wasm_bindgen(getter)]
    pub fn enp(&self) -> Option<f64> {
        self.inner.enp
    }

    #[wasm_bindgen(getter, js_name = "trace_hat")]
    pub fn trace_hat(&self) -> Option<f64> {
        self.inner.trace_hat
    }

    #[wasm_bindgen(getter)]
    pub fn delta1(&self) -> Option<f64> {
        self.inner.delta1
    }

    #[wasm_bindgen(getter)]
    pub fn delta2(&self) -> Option<f64> {
        self.inner.delta2
    }

    #[wasm_bindgen(getter, js_name = "residual_scale")]
    pub fn residual_scale(&self) -> Option<f64> {
        self.inner.residual_scale
    }

    #[wasm_bindgen(getter)]
    pub fn leverage(&self) -> Option<Float64Array> {
        self.inner.leverage.as_ref().map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter)]
    pub fn gradient(&self) -> Option<Float64Array> {
        self.inner.gradient.as_ref().map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions as u32
    }

    /// Evaluate the fitted model at out-of-sample query points not in the training set.
    ///
    /// Requires `retain_model: true` to have been set on the builder before `fit()`.
    #[wasm_bindgen(skip_typescript)]
    pub fn predict(
        &self,
        new_x: &Float64Array,
        options: JsValue,
    ) -> Result<PredictOutput, JsValue> {
        validate_option_keys(
            &options,
            "prediction",
            &[
                "outputs",
                "intervals",
                "extrapolation",
                "max_extrapolation_distance",
                "max_neighbor_distance",
            ],
        )?;
        validate_nested_option_keys(
            &options,
            "intervals",
            "intervals",
            &["confidence", "prediction"],
        )?;
        let opts: PredictOptionsJs = if options.is_undefined() || options.is_null() {
            PredictOptionsJs {
                outputs: None,
                intervals: None,
                extrapolation: None,
                max_extrapolation_distance: None,
                max_neighbor_distance: None,
            }
        } else {
            serde_wasm_bindgen::from_value(options)?
        };
        validate_outputs(opts.outputs.as_ref(), &["se", "gradient", "derivative"])?;
        let new_x_vec = new_x.to_vec();
        let output = map_invalid_arg(shared_parse::run_predict(
            &self.inner,
            &new_x_vec,
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

/// Result of `LoessResult.predict()`.
#[wasm_bindgen(skip_typescript)]
pub struct PredictOutput {
    inner: shared_parse::PredictOutput<f64>,
}

#[wasm_bindgen]
impl PredictOutput {
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> Float64Array {
        to_float64_array(&self.inner.y)
    }

    #[wasm_bindgen(getter, js_name = standard_errors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = confidence_lower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = confidence_upper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = prediction_lower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter, js_name = prediction_upper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| to_float64_array(v))
    }

    #[wasm_bindgen(getter)]
    pub fn derivative(&self) -> Option<Float64Array> {
        self.inner.derivative.as_ref().map(|v| to_float64_array(v))
    }
}

// LOESS smoother.
#[wasm_bindgen(skip_typescript)]
pub struct Loess {
    options: JsValue,
}

#[wasm_bindgen]
impl Loess {
    /// Create a new `Loess` model with the given options.
    #[wasm_bindgen(constructor, skip_typescript)]
    pub fn new(options: JsValue) -> Loess {
        Loess { options }
    }

    /// Fit the model to data and return smoothed values.
    #[wasm_bindgen(skip_typescript)]
    #[allow(non_snake_case)]
    pub fn fit(
        &self,
        x: &Float64Array,
        y: &Float64Array,
        customWeights: Option<Box<[f64]>>,
    ) -> Result<LoessResult, JsValue> {
        smooth(
            x,
            y,
            self.options.clone(),
            customWeights.map(|b| b.to_vec()),
        )
    }
}

// Build a LoessBuilder from Batch options, applying every field.
fn has_output(outputs: Option<&Vec<String>>, name: &str) -> bool {
    outputs.is_some_and(|values| values.iter().any(|value| value == name))
}

fn batch_options_to_builder(opts: Option<SmoothOptions>) -> Result<LoessBuilder<f64>, JsValue> {
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
        let cv = opts.cv.as_ref();
        let cv_fractions = cv.map(|value| value.fractions.as_slice());
        let cv_method = cv.and_then(|value| value.method.as_deref());
        let cv_k = cv.and_then(|value| value.k).map(|value| value as usize);
        let cv_seed = match opts.seed {
            Some(seed)
                if seed.is_finite()
                    && seed >= 0.0
                    && seed.fract() == 0.0
                    && seed <= 9_007_199_254_740_991.0 =>
            {
                Some(seed as u64)
            }
            Some(_) => {
                return Err(JsValue::from_str(
                    "seed must be a non-negative safe integer no greater than Number.MAX_SAFE_INTEGER",
                ));
            }
            None => None,
        };
        builder = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations,
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
                dimensions: opts.dimensions,
                distance_metric: opts.distance_metric.as_deref(),
                weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                surface_mode: opts.surface_mode.as_deref(),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                return_sorted: has_output(opts.outputs.as_ref(), "sorted"),
                cell: opts.cell,
                interpolation_vertices: opts.interpolation_vertices,
                boundary_degree_fallback: opts.boundary_degree_fallback,
                cv_fractions,
                cv_method,
                cv_k,
                cv_seed,
                missing: opts.missing.as_deref(),
                retain_model: opts.retain_model,
            },
        ))?
        .0;
        if has_output(opts.outputs.as_ref(), "gradient")
            || has_output(opts.outputs.as_ref(), "derivative")
        {
            builder = builder.return_gradient();
        }
    }
    Ok(builder)
}

// Build a LoessBuilder from Streaming options, applying every field.
fn streaming_options_to_builder(
    opts: Option<StreamingSmoothOptions>,
) -> Result<LoessBuilder<f64>, JsValue> {
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
        builder = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations,
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
                dimensions: opts.dimensions,
                distance_metric: opts.distance_metric.as_deref(),
                weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                surface_mode: opts.surface_mode.as_deref(),
                cell: opts.cell,
                interpolation_vertices: opts.interpolation_vertices,
                boundary_degree_fallback: opts.boundary_degree_fallback,
                missing: opts.missing.as_deref(),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                ..Default::default()
            },
        ))?
        .0;
        if has_output(opts.outputs.as_ref(), "gradient")
            || has_output(opts.outputs.as_ref(), "derivative")
        {
            builder = builder.return_gradient();
        }
    }
    Ok(builder)
}

// Build a LoessBuilder from Online options, applying every field.
fn online_options_to_builder(
    opts: Option<OnlineSmoothOptions>,
) -> Result<LoessBuilder<f64>, JsValue> {
    let mut builder = LoessBuilder::<f64>::new();
    if let Some(opts) = opts {
        validate_outputs(
            opts.outputs.as_ref(),
            &["weights", "gradient", "derivative", "se"],
        )?;
        builder = map_invalid_arg(shared_parse::apply_builder_options(
            builder,
            shared_parse::BuilderOptionSet {
                fraction: opts.fraction,
                iterations: opts.iterations,
                weight_function: opts.weight_function.as_deref(),
                robustness_method: opts.robustness_method.as_deref(),
                zero_weight_fallback: opts.zero_weight_fallback.as_deref(),
                boundary_policy: opts.boundary_policy.as_deref(),
                scaling_method: opts.scaling_method.as_deref(),
                auto_converge: opts.auto_converge,
                return_robustness_weights: has_output(opts.outputs.as_ref(), "weights"),
                degree: opts.degree.as_deref(),
                dimensions: opts.dimensions,
                distance_metric: opts.distance_metric.as_deref(),
                weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                surface_mode: opts.surface_mode.as_deref(),
                cell: opts.cell,
                interpolation_vertices: opts.interpolation_vertices,
                boundary_degree_fallback: opts.boundary_degree_fallback,
                missing: opts.missing.as_deref(),
                confidence_intervals: opts.intervals.as_ref().and_then(|value| value.confidence),
                prediction_intervals: opts.intervals.as_ref().and_then(|value| value.prediction),
                return_se: has_output(opts.outputs.as_ref(), "se"),
                ..Default::default()
            },
        ))?
        .0;
        if has_output(opts.outputs.as_ref(), "gradient")
            || has_output(opts.outputs.as_ref(), "derivative")
        {
            builder = builder.return_gradient();
        }
    }
    Ok(builder)
}

fn smooth(
    x: &Float64Array,
    y: &Float64Array,
    options: JsValue,
    custom_weights: Option<Vec<f64>>,
) -> Result<LoessResult, JsValue> {
    validate_option_keys(
        &options,
        "batch",
        &[
            "outputs",
            "cv",
            "fraction",
            "iterations",
            "weight_function",
            "robustness_method",
            "zero_weight_fallback",
            "boundary_policy",
            "scaling_method",
            "auto_converge",
            "intervals",
            "parallel",
            "degree",
            "dimensions",
            "distance_metric",
            "surface_mode",
            "weighted_metric_weights",
            "cell",
            "interpolation_vertices",
            "boundary_degree_fallback",
            "seed",
            "missing",
            "retain_model",
        ],
    )?;
    validate_nested_option_keys(
        &options,
        "intervals",
        "intervals",
        &["confidence", "prediction"],
    )?;
    validate_nested_option_keys(&options, "cv", "cv", &["fractions", "method", "k"])?;
    let opts = if !options.is_undefined() && !options.is_null() {
        Some(serde_wasm_bindgen::from_value::<SmoothOptions>(options)?)
    } else {
        None
    };
    let builder = batch_options_to_builder(opts)?;

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let model = map_runtime(shared_parse::build_batch(builder, custom_weights))?;
    let result = map_runtime(model.fit(&x_vec, &y_vec))?;

    Ok(LoessResult { inner: result })
}

// LOESS smoother.
#[wasm_bindgen(skip_typescript)]
pub struct StreamingLoess {
    inner: ParallelStreamingLoess<f64>,
}

#[wasm_bindgen]
impl StreamingLoess {
    // Create a new smoother.
    #[wasm_bindgen(constructor, skip_typescript)]
    #[allow(non_snake_case)]
    pub fn new(options: JsValue, streamingOpts: JsValue) -> Result<StreamingLoess, JsValue> {
        validate_option_keys(
            &options,
            "streaming",
            &[
                "outputs",
                "fraction",
                "iterations",
                "weight_function",
                "robustness_method",
                "zero_weight_fallback",
                "boundary_policy",
                "scaling_method",
                "auto_converge",
                "intervals",
                "parallel",
                "degree",
                "dimensions",
                "distance_metric",
                "surface_mode",
                "weighted_metric_weights",
                "cell",
                "interpolation_vertices",
                "boundary_degree_fallback",
                "missing",
            ],
        )?;
        validate_nested_option_keys(
            &options,
            "intervals",
            "intervals",
            &["confidence", "prediction"],
        )?;
        validate_option_keys(
            &streamingOpts,
            "streamingOpts",
            &["chunk_size", "overlap", "merge_strategy"],
        )?;
        let opts = if !options.is_undefined() && !options.is_null() {
            Some(serde_wasm_bindgen::from_value::<StreamingSmoothOptions>(
                options,
            )?)
        } else {
            None
        };
        let builder = streaming_options_to_builder(opts)?;

        let (chunk_size, overlap, merge_strategy) =
            if !streamingOpts.is_undefined() && !streamingOpts.is_null() {
                let sopts: StreamingOptions = serde_wasm_bindgen::from_value(streamingOpts)?;
                (sopts.chunk_size, sopts.overlap, sopts.merge_strategy)
            } else {
                (None, None, None)
            };

        let model = map_runtime(shared_parse::build_streaming(
            builder,
            chunk_size,
            overlap,
            merge_strategy.as_deref(),
        ))?;

        Ok(StreamingLoess { inner: model })
    }

    #[wasm_bindgen(js_name = process_chunk, skip_typescript)]
    pub fn process_chunk(
        &mut self,
        x: &Float64Array,
        y: &Float64Array,
    ) -> Result<LoessResult, JsValue> {
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let result: InnerLoessResult<f64> = map_runtime(self.inner.process_chunk(&x_vec, &y_vec))?;
        Ok(LoessResult { inner: result })
    }

    #[wasm_bindgen(js_name = process_chunk_weighted, skip_typescript)]
    pub fn process_chunk_weighted(
        &mut self,
        x: &Float64Array,
        y: &Float64Array,
        weights: &Float64Array,
    ) -> Result<LoessResult, JsValue> {
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let weights_vec = weights.to_vec();
        let result: InnerLoessResult<f64> = map_runtime(self.inner.process_chunk_weighted(
            &x_vec,
            &y_vec,
            &weights_vec,
        ))?;
        Ok(LoessResult { inner: result })
    }

    #[wasm_bindgen(skip_typescript)]
    pub fn finalize(&mut self) -> Result<LoessResult, JsValue> {
        let result: InnerLoessResult<f64> = map_runtime(self.inner.finalize())?;
        Ok(LoessResult { inner: result })
    }
}

// LOESS smoother.
#[wasm_bindgen(skip_typescript)]
pub struct OnlineLoess {
    inner: ParallelOnlineLoess<f64>,
}

#[wasm_bindgen]
impl OnlineLoess {
    // Create a new smoother.
    #[wasm_bindgen(constructor, skip_typescript)]
    #[allow(non_snake_case)]
    pub fn new(options: JsValue, onlineOpts: JsValue) -> Result<OnlineLoess, JsValue> {
        validate_option_keys(
            &options,
            "online",
            &[
                "outputs",
                "fraction",
                "iterations",
                "weight_function",
                "robustness_method",
                "zero_weight_fallback",
                "boundary_policy",
                "scaling_method",
                "auto_converge",
                "intervals",
                "degree",
                "dimensions",
                "distance_metric",
                "surface_mode",
                "weighted_metric_weights",
                "cell",
                "interpolation_vertices",
                "boundary_degree_fallback",
                "missing",
            ],
        )?;
        validate_nested_option_keys(
            &options,
            "intervals",
            "intervals",
            &["confidence", "prediction"],
        )?;
        validate_option_keys(
            &onlineOpts,
            "onlineOpts",
            &["window_capacity", "min_points", "update_mode"],
        )?;
        let opts = if !options.is_undefined() && !options.is_null() {
            Some(serde_wasm_bindgen::from_value::<OnlineSmoothOptions>(
                options,
            )?)
        } else {
            None
        };
        let builder = online_options_to_builder(opts)?;

        let (window_capacity, min_points, update_mode) =
            if !onlineOpts.is_undefined() && !onlineOpts.is_null() {
                let oopts: OnlineOptions = serde_wasm_bindgen::from_value(onlineOpts)?;
                (oopts.window_capacity, oopts.min_points, oopts.update_mode)
            } else {
                (None, None, None)
            };

        let model = map_runtime(shared_parse::build_online(
            builder,
            window_capacity,
            min_points,
            update_mode.as_deref(),
        ))?;

        Ok(OnlineLoess { inner: model })
    }

    #[wasm_bindgen(js_name = "add_point", skip_typescript)]
    pub fn add_point(&mut self, x: f64, y: f64) -> Result<JsValue, JsValue> {
        let output = map_invalid_arg(self.inner.add_point(&[x], y))?;
        Ok(match output {
            Some(o) => JsValue::from(OnlineOutput {
                y: o.y,
                standard_error: o.standard_error,
                residual: o.residual,
                robustness_weight: o.robustness_weight,
                iterations_used: o.iterations_used,
                confidence_lower: o.confidence_lower,
                confidence_upper: o.confidence_upper,
                prediction_lower: o.prediction_lower,
                prediction_upper: o.prediction_upper,
                gradient: o.gradient,
            }),
            None => JsValue::null(),
        })
    }

    #[wasm_bindgen(js_name = "add_point_vector", skip_typescript)]
    pub fn add_point_vector(&mut self, x: &Float64Array, y: f64) -> Result<JsValue, JsValue> {
        let x = x.to_vec();
        let output = map_invalid_arg(self.inner.add_point(&x, y))?;
        Ok(match output {
            Some(o) => JsValue::from(OnlineOutput {
                y: o.y,
                standard_error: o.standard_error,
                residual: o.residual,
                robustness_weight: o.robustness_weight,
                iterations_used: o.iterations_used,
                confidence_lower: o.confidence_lower,
                confidence_upper: o.confidence_upper,
                prediction_lower: o.prediction_lower,
                prediction_upper: o.prediction_upper,
                gradient: o.gradient,
            }),
            None => JsValue::null(),
        })
    }

    #[wasm_bindgen(js_name = add_point_weighted, skip_typescript)]
    pub fn add_point_weighted(&mut self, x: f64, y: f64, weight: f64) -> Result<JsValue, JsValue> {
        self.add_point_vector_weighted(&Float64Array::from(&[x][..]), y, weight)
    }

    #[wasm_bindgen(js_name = add_point_vector_weighted, skip_typescript)]
    pub fn add_point_vector_weighted(
        &mut self,
        x: &Float64Array,
        y: f64,
        weight: f64,
    ) -> Result<JsValue, JsValue> {
        let x = x.to_vec();
        let output = map_invalid_arg(self.inner.add_point_weighted(&x, y, weight))?;
        Ok(match output {
            Some(o) => JsValue::from(OnlineOutput {
                y: o.y,
                standard_error: o.standard_error,
                residual: o.residual,
                robustness_weight: o.robustness_weight,
                iterations_used: o.iterations_used,
                confidence_lower: o.confidence_lower,
                confidence_upper: o.confidence_upper,
                prediction_lower: o.prediction_lower,
                prediction_upper: o.prediction_upper,
                gradient: o.gradient,
            }),
            None => JsValue::null(),
        })
    }

    #[wasm_bindgen(js_name = window_diagnostics, skip_typescript)]
    pub fn window_diagnostics(&self) -> Result<JsValue, JsValue> {
        let diagnostics = map_runtime(self.inner.window_diagnostics())?;
        Ok(match diagnostics {
            Some(d) => JsValue::from(Diagnostics {
                rmse: d.rmse,
                mae: d.mae,
                r_squared: d.r_squared,
                aic: d.aic,
                aicc: d.aicc,
                effective_df: d.effective_df,
                residual_sd: d.residual_sd,
            }),
            None => JsValue::null(),
        })
    }

    #[wasm_bindgen(js_name = predict_window, skip_typescript)]
    pub fn predict_window(
        &self,
        new_x: &Float64Array,
        options: JsValue,
    ) -> Result<PredictOutput, JsValue> {
        validate_option_keys(
            &options,
            "prediction",
            &[
                "outputs",
                "intervals",
                "extrapolation",
                "max_extrapolation_distance",
                "max_neighbor_distance",
            ],
        )?;
        validate_nested_option_keys(
            &options,
            "intervals",
            "intervals",
            &["confidence", "prediction"],
        )?;
        let opts: PredictOptionsJs = if options.is_undefined() || options.is_null() {
            PredictOptionsJs {
                outputs: None,
                intervals: None,
                extrapolation: None,
                max_extrapolation_distance: None,
                max_neighbor_distance: None,
            }
        } else {
            serde_wasm_bindgen::from_value(options)?
        };
        validate_outputs(opts.outputs.as_ref(), &["se", "gradient", "derivative"])?;
        let mut intervals = IntervalsBuilder::new();
        if let Some(level) = opts.intervals.as_ref().and_then(|value| value.confidence) {
            intervals = intervals.confidence(level);
        }
        if let Some(level) = opts.intervals.as_ref().and_then(|value| value.prediction) {
            intervals = intervals.prediction(level);
        }
        let mut builder = Predict::new()
            .intervals(intervals)
            .extrapolation(opts.extrapolation.as_deref().unwrap_or("clamp"));
        if has_output(opts.outputs.as_ref(), "se") {
            builder = builder.return_se();
        }
        if has_output(opts.outputs.as_ref(), "gradient")
            || has_output(opts.outputs.as_ref(), "derivative")
        {
            builder = builder.return_derivative();
        }
        if let Some(distance) = opts.max_extrapolation_distance {
            builder = builder.max_extrapolation_distance(distance);
        }
        if let Some(distance) = opts.max_neighbor_distance {
            builder = builder.max_neighbor_distance(distance);
        }
        let query = map_invalid_arg(builder.build())?;
        let new_x = new_x.to_vec();
        let output = map_invalid_arg(self.inner.predict_window(&new_x, &query))?;
        Ok(PredictOutput { inner: output })
    }
}
