//! WebAssembly bindings for fastLoess.

use js_sys::Float64Array;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "initPanicHook")]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

use ::fastLoess::api::{Batch, LoessBuilder, Online, Streaming};
use ::fastLoess::binding_support as shared_parse;
use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::api::{MergeStrategy, UpdateMode};
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
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub weight_function: Option<String>,
    pub robustness_method: Option<String>,
    pub zero_weight_fallback: Option<String>,
    pub boundary_policy: Option<String>,
    pub scaling_method: Option<String>,
    pub auto_converge: Option<f64>,
    pub return_residuals: Option<bool>,
    pub return_robustness_weights: Option<bool>,
    pub return_diagnostics: Option<bool>,
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
    pub parallel: Option<bool>,
    pub cv_fractions: Option<Vec<f64>>,
    pub cv_method: Option<String>,
    pub cv_k: Option<u32>,
    pub degree: Option<String>,
    pub dimensions: Option<usize>,
    pub distance_metric: Option<String>,
    pub surface_mode: Option<String>,
    pub return_se: Option<bool>,
    pub weighted_metric_weights: Option<Vec<f64>>,
    pub cell: Option<f64>,
    pub interpolation_vertices: Option<usize>,
    pub boundary_degree_fallback: Option<bool>,
    pub cv_seed: Option<u64>,
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

#[wasm_bindgen]
pub struct Diagnostics {
    pub rmse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub aic: Option<f64>,
    pub aicc: Option<f64>,
    pub effective_df: Option<f64>,
    pub residual_sd: f64,
}

// Result of a single online update step.
#[wasm_bindgen]
pub struct OnlineOutput {
    smoothed: f64,
    std_error: Option<f64>,
    residual: Option<f64>,
    robustness_weight: Option<f64>,
    iterations_used: Option<usize>,
}

#[wasm_bindgen]
impl OnlineOutput {
    #[wasm_bindgen(getter)]
    pub fn smoothed(&self) -> f64 {
        self.smoothed
    }

    #[wasm_bindgen(getter, js_name = "std_error")]
    pub fn std_error(&self) -> Option<f64> {
        self.std_error
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

    #[wasm_bindgen(getter, js_name = "standard_errors")]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = "confidence_lower")]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = "confidence_upper")]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = "prediction_lower")]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = "prediction_upper")]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = "robustness_weights")]
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

    #[wasm_bindgen(getter, js_name = "cv_scores")]
    pub fn cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = "fraction_used")]
    pub fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    #[wasm_bindgen(getter, js_name = "iterations_used")]
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
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions as u32
    }
}

// LOESS smoother.
#[wasm_bindgen]
pub struct Loess {
    options: JsValue,
}

#[wasm_bindgen]
impl Loess {
    /// Create a new `Loess` model with the given options.
    #[wasm_bindgen(constructor)]
    pub fn new(options: JsValue) -> Loess {
        Loess { options }
    }

    /// Fit the model to data and return smoothed values.
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

fn smooth(
    x: &Float64Array,
    y: &Float64Array,
    options: JsValue,
    custom_weights: Option<Vec<f64>>,
) -> Result<LoessResult, JsValue> {
    let mut builder = LoessBuilder::<f64>::new();
    let y_len = y.length() as usize;

    if !options.is_undefined() && !options.is_null() {
        let opts: SmoothOptions = serde_wasm_bindgen::from_value(options)?;

        let (configured_builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
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
                return_residuals: opts.return_residuals.unwrap_or(false),
                return_robustness_weights: opts.return_robustness_weights.unwrap_or(false),
                return_diagnostics: opts.return_diagnostics.unwrap_or(false),
                confidence_intervals: opts.confidence_intervals,
                prediction_intervals: opts.prediction_intervals,
                parallel: opts.parallel,
                degree: opts.degree.as_deref(),
                dimensions: opts.dimensions,
                distance_metric: opts.distance_metric.as_deref(),
                weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                surface_mode: opts.surface_mode.as_deref(),
                return_se: opts.return_se.unwrap_or(false),
                cell: opts.cell,
                interpolation_vertices: opts.interpolation_vertices,
                boundary_degree_fallback: opts.boundary_degree_fallback,
                cv_fractions: opts.cv_fractions.as_deref(),
                cv_method: opts.cv_method.as_deref(),
                cv_k: opts.cv_k.map(|v| v as usize),
                cv_seed: opts.cv_seed,
            },
        ))?;
        builder = configured_builder;
    }

    if let Some(cw) = custom_weights {
        if cw.len() != y_len {
            return Err(to_js_error(shared_parse::BindingError::invalid_arg(
                shared_parse::custom_weights_length_mismatch_message(cw.len(), y_len),
            )));
        }
        if cw.iter().any(|&w| w < 0.0) {
            return Err(to_js_error(shared_parse::BindingError::invalid_arg(
                shared_parse::CUSTOM_WEIGHTS_MUST_BE_NON_NEGATIVE,
            )));
        }
        builder = builder.custom_weights(cw);
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let model = map_runtime(builder.adapter(Batch).build())?;

    let result = map_runtime(model.fit(&x_vec, &y_vec))?;

    Ok(LoessResult { inner: result })
}

// LOESS smoother.
#[wasm_bindgen]
pub struct StreamingLoess {
    inner: ParallelStreamingLoess<f64>,
}

#[wasm_bindgen]
impl StreamingLoess {
    // Create a new smoother.
    #[wasm_bindgen(constructor)]
    #[allow(non_snake_case)]
    pub fn new(options: JsValue, streamingOpts: JsValue) -> Result<StreamingLoess, JsValue> {
        let mut builder = LoessBuilder::<f64>::new();

        if !options.is_undefined() && !options.is_null() {
            let opts: SmoothOptions = serde_wasm_bindgen::from_value(options)?;

            let (configured_builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
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
                    return_residuals: opts.return_residuals.unwrap_or(false),
                    return_robustness_weights: opts.return_robustness_weights.unwrap_or(false),
                    return_diagnostics: opts.return_diagnostics.unwrap_or(false),
                    confidence_intervals: opts.confidence_intervals,
                    prediction_intervals: opts.prediction_intervals,
                    parallel: opts.parallel,
                    degree: opts.degree.as_deref(),
                    dimensions: opts.dimensions,
                    distance_metric: opts.distance_metric.as_deref(),
                    weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                    surface_mode: opts.surface_mode.as_deref(),
                    return_se: opts.return_se.unwrap_or(false),
                    cell: opts.cell,
                    interpolation_vertices: opts.interpolation_vertices,
                    boundary_degree_fallback: opts.boundary_degree_fallback,
                    cv_fractions: opts.cv_fractions.as_deref(),
                    cv_method: opts.cv_method.as_deref(),
                    cv_k: opts.cv_k.map(|v| v as usize),
                    cv_seed: opts.cv_seed,
                },
            ))?;
            builder = configured_builder;
        }

        let mut chunk_size = 5000;
        let mut overlap = 500;
        let mut merge_strategy = MergeStrategy::WeightedAverage;

        if !streamingOpts.is_undefined() && !streamingOpts.is_null() {
            let sopts: StreamingOptions = serde_wasm_bindgen::from_value(streamingOpts)?;
            if let Some(cs) = sopts.chunk_size {
                chunk_size = cs;
            }
            if let Some(ov) = sopts.overlap {
                overlap = ov;
            }
            if let Some(ms) = sopts.merge_strategy {
                merge_strategy = map_invalid_arg(shared_parse::parse_merge_strategy(&ms))?;
            }
        }

        let model = map_runtime(
            builder
                .adapter(Streaming)
                .chunk_size(chunk_size)
                .overlap(overlap)
                .merge_strategy(merge_strategy)
                .build(),
        )?;

        Ok(StreamingLoess { inner: model })
    }

    #[wasm_bindgen(js_name = processChunk)]
    pub fn process_chunk(
        &mut self,
        x: &Float64Array,
        y: &Float64Array,
    ) -> Result<LoessResult, JsValue> {
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let result: ::fastLoess::prelude::LoessResult<f64> =
            map_runtime(self.inner.process_chunk(&x_vec, &y_vec))?;
        Ok(LoessResult { inner: result })
    }

    pub fn finalize(&mut self) -> Result<LoessResult, JsValue> {
        let result: ::fastLoess::prelude::LoessResult<f64> = map_runtime(self.inner.finalize())?;
        Ok(LoessResult { inner: result })
    }
}

// LOESS smoother.
#[wasm_bindgen]
pub struct OnlineLoess {
    inner: ParallelOnlineLoess<f64>,
}

#[wasm_bindgen]
impl OnlineLoess {
    // Create a new smoother.
    #[wasm_bindgen(constructor)]
    #[allow(non_snake_case)]
    pub fn new(options: JsValue, onlineOpts: JsValue) -> Result<OnlineLoess, JsValue> {
        let mut builder = LoessBuilder::<f64>::new();

        if !options.is_undefined() && !options.is_null() {
            let opts: SmoothOptions = serde_wasm_bindgen::from_value(options)?;
            let (configured_builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
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
                    return_residuals: opts.return_residuals.unwrap_or(false),
                    return_robustness_weights: opts.return_robustness_weights.unwrap_or(false),
                    return_diagnostics: opts.return_diagnostics.unwrap_or(false),
                    confidence_intervals: opts.confidence_intervals,
                    prediction_intervals: opts.prediction_intervals,
                    parallel: opts.parallel,
                    degree: opts.degree.as_deref(),
                    dimensions: opts.dimensions,
                    distance_metric: opts.distance_metric.as_deref(),
                    weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                    surface_mode: opts.surface_mode.as_deref(),
                    return_se: opts.return_se.unwrap_or(false),
                    cell: opts.cell,
                    interpolation_vertices: opts.interpolation_vertices,
                    boundary_degree_fallback: opts.boundary_degree_fallback,
                    cv_fractions: None,
                    cv_method: None,
                    cv_k: None,
                    cv_seed: None,
                },
            ))?;
            builder = configured_builder;
        }

        let mut window_capacity = 1000;
        let mut min_points = 3;
        let mut update_mode = UpdateMode::Full;

        if !onlineOpts.is_undefined() && !onlineOpts.is_null() {
            let oopts: OnlineOptions = serde_wasm_bindgen::from_value(onlineOpts)?;
            if let Some(wc) = oopts.window_capacity {
                window_capacity = wc;
            }
            if let Some(mp) = oopts.min_points {
                min_points = mp;
            }
            if let Some(um) = oopts.update_mode {
                update_mode = map_invalid_arg(shared_parse::parse_update_mode(&um))?;
            }
        }

        let model = map_runtime(
            builder
                .adapter(Online)
                .window_capacity(window_capacity)
                .min_points(min_points)
                .update_mode(update_mode)
                .build(),
        )?;

        Ok(OnlineLoess { inner: model })
    }

    // Add a single point and return its smoothed value, or undefined if the
    // window is not yet full enough to produce a result.
    #[wasm_bindgen(js_name = "add_point")]
    pub fn add_point(&mut self, x: f64, y: f64) -> Result<Option<OnlineOutput>, JsValue> {
        let output = map_invalid_arg(self.inner.add_point(&[x], y))?;
        Ok(output.map(|o| OnlineOutput {
            smoothed: o.smoothed,
            std_error: o.std_error,
            residual: o.residual,
            robustness_weight: o.robustness_weight,
            iterations_used: o.iterations_used,
        }))
    }
}
