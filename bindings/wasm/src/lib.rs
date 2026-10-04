//! WebAssembly bindings for fastLoess.

use js_sys::Float64Array;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// ============================================================================
// TypeScript interface declarations injected into the generated .d.ts
// ============================================================================

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &'static str = r#"
/** Configuration options for LOESS smoothing. */
export interface SmoothOptions {
    /** Optional output components: diagnostics, residuals, weights, gradient (or derivative), se, sorted. */
    outputs?: string[];
    /** Grouped batch cross-validation configuration. */
    cv?: { fractions: number[]; method?: string; k?: number; seed?: number };
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
    /** Confidence interval level (e.g. 0.95). Disabled when absent. */
    confidence_intervals?: number;
    /** Prediction interval level (e.g. 0.95). Disabled when absent. */
    prediction_intervals?: number;
    /** Enable parallel execution. Default: true. */
    parallel?: boolean;
    /** Fractions to test for cross-validation. CV disabled when absent. */
    cv_fractions?: number[];
    /** CV method ("kfold" or "loocv"). Default: "kfold". */
    cv_method?: string;
    /** Number of folds for k-fold CV. Default: 5. */
    cv_k?: number;
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
    /** Random seed for cross-validation. */
    cv_seed?: number;
    /** Policy for non-finite (NaN/Inf) values in input data ("error", "drop"). Default: "error". */
    missing?: string;
    /** Retain the fitted model's training data, enabling `LoessResult.predict()`. Default: false. */
    retain_model?: boolean;
}

/** Options for `LoessResult.predict()`. */
export interface PredictOptions {
    /** Optional prediction components: se, gradient (or derivative). */
    outputs?: string[];
    /** Confidence interval coverage level (e.g. 0.95). Disabled when absent. */
    confidence_level?: number;
    /** Prediction interval coverage level (e.g. 0.95). Disabled when absent. */
    prediction_level?: number;
    /** Behavior for query points outside the training range ("clamp", "linear", "error"). Default: "clamp". */
    extrapolation?: string;
    /** Under "linear" extrapolation, the maximum allowed distance beyond the training boundary before `predict()` errors instead of returning an unbounded value. */
    max_extrapolation_distance?: number;
    /** Maximum allowed distance to the farthest point in a query's neighbor window before `predict()` errors, catching in-range-but-sparse query points. */
    max_neighbor_distance?: number;
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
    /** Confidence interval level (e.g. 0.95), computed per chunk. Disabled when absent. */
    confidence_intervals?: number;
    /** Prediction interval level (e.g. 0.95), computed per chunk. Disabled when absent. */
    prediction_intervals?: number;
}

/** Configuration options for online LOESS smoothing. A subset of `SmoothOptions`: diagnostics, residuals, parallel execution, and cross-validation have no equivalent here. `confidence_intervals`/`prediction_intervals` and the `se` output require `update_mode: "full"`. */
export interface OnlineSmoothOptions {
    /** Optional output components: weights, gradient (or derivative), se. */
    outputs?: string[];
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
    /** Confidence interval level (e.g. 0.95). Only computed under `update_mode: "full"`. Disabled when absent. */
    confidence_intervals?: number;
    /** Prediction interval level (e.g. 0.95). Only computed under `update_mode: "full"`. Disabled when absent. */
    prediction_intervals?: number;
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
    /** Finalize the stream and return remaining data. */
    finalize(): LoessResult;
}

/** Online LOESS smoother for real-time data. */
export class OnlineLoess {
    free(): void;
    constructor(options?: OnlineSmoothOptions, onlineOpts?: OnlineOptions);
    /** Add a single point and get the smoothed value, or null if not enough points yet. */
    add_point(x: number, y: number): OnlineOutput | null;
}

/** Result from a single online update step. */
export class OnlineOutput {
    free(): void;
    get y(): number;
    get standard_error(): number | undefined;
    get residual(): number | undefined;
    get robustness_weight(): number | undefined;
    get iterations_used(): number | undefined;
    get confidence_lower(): number | undefined;
    get confidence_upper(): number | undefined;
    get prediction_lower(): number | undefined;
    get prediction_upper(): number | undefined;
    get gradient(): Float64Array | undefined;
}
"#;

use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::api::LoessBuilder;
use ::fastLoess::internals::binding_support as shared_parse;
use ::fastLoess::prelude::LoessResult as InnerLoessResult;

fn to_js_error(err: shared_parse::BindingError) -> JsValue {
    JsValue::from_str(&err.message)
}

fn map_invalid_arg<T, E: ToString>(result: Result<T, E>) -> Result<T, JsValue> {
    shared_parse::map_invalid_arg(result).map_err(to_js_error)
}

fn map_runtime<T, E: ToString>(result: Result<T, E>) -> Result<T, JsValue> {
    shared_parse::map_runtime(result).map_err(to_js_error)
}

#[derive(Deserialize)]
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
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
    #[serde(rename = "parallel")]
    pub parallel: Option<bool>,
    pub cv_fractions: Option<Vec<f64>>,
    pub cv_method: Option<String>,
    pub cv_k: Option<u32>,
    pub degree: Option<String>,
    pub dimensions: Option<usize>,
    pub distance_metric: Option<String>,
    pub surface_mode: Option<String>,
    pub weighted_metric_weights: Option<Vec<f64>>,
    pub cell: Option<f64>,
    pub interpolation_vertices: Option<usize>,
    pub boundary_degree_fallback: Option<bool>,
    pub cv_seed: Option<u64>,
    pub missing: Option<String>,
    pub retain_model: Option<bool>,
}

#[derive(Deserialize)]
pub struct CVOptionsJs {
    pub fractions: Vec<f64>,
    pub method: Option<String>,
    pub k: Option<u32>,
    pub seed: Option<u64>,
}

#[derive(Deserialize)]
pub struct PredictOptionsJs {
    pub outputs: Option<Vec<String>>,
    pub confidence_level: Option<f64>,
    pub prediction_level: Option<f64>,
    pub extrapolation: Option<String>,
    pub max_extrapolation_distance: Option<f64>,
    pub max_neighbor_distance: Option<f64>,
}

#[derive(Deserialize)]
pub struct StreamingOptions {
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub merge_strategy: Option<String>,
}

#[derive(Deserialize)]
pub struct OnlineOptions {
    pub window_capacity: Option<usize>,
    pub min_points: Option<usize>,
    pub update_mode: Option<String>,
}

#[derive(Deserialize)]
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
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
}

#[derive(Deserialize)]
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
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
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
        self.gradient
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
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
        unsafe { Float64Array::view(&self.inner.x) }
    }

    #[wasm_bindgen(getter)]
    pub fn y(&self) -> Float64Array {
        unsafe { Float64Array::view(&self.inner.y) }
    }

    #[wasm_bindgen(getter)]
    pub fn residuals(&self) -> Option<Float64Array> {
        self.inner
            .residuals
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = standard_errors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidence_lower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidence_upper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = prediction_lower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = prediction_upper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = robustness_weights)]
    pub fn robustness_weights(&self) -> Option<Float64Array> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
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
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
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
        self.inner
            .leverage
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter)]
    pub fn gradient(&self) -> Option<Float64Array> {
        self.inner
            .gradient
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
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
        let opts: PredictOptionsJs = if options.is_undefined() || options.is_null() {
            PredictOptionsJs {
                outputs: None,
                confidence_level: None,
                prediction_level: None,
                extrapolation: None,
                max_extrapolation_distance: None,
                max_neighbor_distance: None,
            }
        } else {
            serde_wasm_bindgen::from_value(options)?
        };
        let new_x_vec = new_x.to_vec();
        let output = map_invalid_arg(shared_parse::run_predict(
            &self.inner,
            &new_x_vec,
            shared_parse::PredictOptionSet {
                return_se: has_output(opts.outputs.as_ref(), "se"),
                confidence_level: opts.confidence_level,
                prediction_level: opts.prediction_level,
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
        unsafe { Float64Array::view(&self.inner.y) }
    }

    #[wasm_bindgen(getter, js_name = standard_errors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidence_lower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidence_upper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = prediction_lower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = prediction_upper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter)]
    pub fn derivative(&self) -> Option<Float64Array> {
        self.inner
            .derivative
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
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
        let cv = opts.cv.as_ref();
        let cv_fractions = cv
            .map(|value| value.fractions.as_slice())
            .or(opts.cv_fractions.as_deref());
        let cv_method = cv
            .and_then(|value| value.method.as_deref())
            .or(opts.cv_method.as_deref());
        let cv_k = cv
            .and_then(|value| value.k)
            .map(|value| value as usize)
            .or(opts.cv_k.map(|value| value as usize));
        let cv_seed = cv.and_then(|value| value.seed).or(opts.cv_seed);
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
                confidence_intervals: opts.confidence_intervals,
                prediction_intervals: opts.prediction_intervals,
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
                confidence_intervals: opts.confidence_intervals,
                prediction_intervals: opts.prediction_intervals,
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
                confidence_intervals: opts.confidence_intervals,
                prediction_intervals: opts.prediction_intervals,
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
}
