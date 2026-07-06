//! WebAssembly bindings for fastLoess.

use js_sys::Float64Array;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "initPanicHook")]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

use ::fastLoess::api::{Batch, LoessBuilder, Online, Streaming};
use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::api::{
    BoundaryPolicy, DistanceMetric, MergeStrategy, PolynomialDegree, RobustnessMethod,
    ScalingMethod::{self, MAD, MAR, Mean},
    SurfaceMode, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use ::fastLoess::prelude::LoessResult as InnerLoessResult;

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

fn parse_weight_function(name: &str) -> Result<WeightFunction, JsValue> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(JsValue::from_str(&format!(
            "Unknown weight function: {}",
            name
        ))),
    }
}

fn parse_robustness_method(name: &str) -> Result<RobustnessMethod, JsValue> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(JsValue::from_str(&format!(
            "Unknown robustness method: {}",
            name
        ))),
    }
}

fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback, JsValue> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(JsValue::from_str(&format!(
            "Unknown zero weight fallback: {}",
            name
        ))),
    }
}

fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy, JsValue> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(JsValue::from_str(&format!(
            "Unknown boundary policy: {}",
            name
        ))),
    }
}

fn parse_scaling_method(name: &str) -> Result<ScalingMethod, JsValue> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        "mean" => Ok(Mean),
        _ => Err(JsValue::from_str(&format!(
            "Unknown scaling method: {}. Valid options: mad, mar, mean",
            name
        ))),
    }
}

fn parse_polynomial_degree(name: &str) -> Result<PolynomialDegree, JsValue> {
    match name.to_lowercase().as_str() {
        "constant" | "0" => Ok(PolynomialDegree::Constant),
        "linear" | "1" => Ok(PolynomialDegree::Linear),
        "quadratic" | "2" => Ok(PolynomialDegree::Quadratic),
        "cubic" | "3" => Ok(PolynomialDegree::Cubic),
        "quartic" | "4" => Ok(PolynomialDegree::Quartic),
        _ => Err(JsValue::from_str(&format!(
            "Unknown polynomial degree: {}. Valid options: constant, linear, quadratic, cubic, quartic",
            name
        ))),
    }
}

fn parse_distance_metric(name: &str) -> Result<DistanceMetric<f64>, JsValue> {
    // Handle "minkowski:p" inline format
    if let Some(p_str) = name.to_lowercase().strip_prefix("minkowski:") {
        let p: f64 = p_str
            .parse()
            .map_err(|_| JsValue::from_str(&format!("Invalid Minkowski p value: {}", p_str)))?;
        return Ok(DistanceMetric::Minkowski(p));
    }
    match name.to_lowercase().as_str() {
        "normalized" => Ok(DistanceMetric::Normalized),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        "chebyshev" | "linf" => Ok(DistanceMetric::Chebyshev),
        "minkowski" => Ok(DistanceMetric::Minkowski(2.0)),
        "weighted" => Ok(DistanceMetric::Weighted(Vec::new())),
        _ => Err(JsValue::from_str(&format!(
            "Unknown distance metric: {}. Valid options: normalized, euclidean, manhattan, chebyshev, minkowski, weighted",
            name
        ))),
    }
}

fn parse_merge_strategy(name: &str) -> Result<MergeStrategy, JsValue> {
    match name.to_lowercase().as_str() {
        "average" | "mean" => Ok(MergeStrategy::Average),
        "weighted_average" | "weighted" => Ok(MergeStrategy::WeightedAverage),
        "take_first" | "first" => Ok(MergeStrategy::TakeFirst),
        "take_last" | "last" => Ok(MergeStrategy::TakeLast),
        _ => Err(JsValue::from_str(&format!(
            "Unknown merge strategy: {}. Valid options: average, weighted_average, take_first, take_last",
            name
        ))),
    }
}

fn parse_surface_mode(name: &str) -> Result<SurfaceMode, JsValue> {
    match name.to_lowercase().as_str() {
        "interpolation" | "interpolate" => Ok(SurfaceMode::Interpolation),
        "direct" => Ok(SurfaceMode::Direct),
        _ => Err(JsValue::from_str(&format!(
            "Unknown surface mode: {}. Valid options: interpolation, direct",
            name
        ))),
    }
}

fn parse_update_mode(name: &str) -> Result<UpdateMode, JsValue> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(JsValue::from_str(&format!("Unknown update mode: {}", name))),
    }
}

#[wasm_bindgen]
pub struct Diagnostics {
    pub rmse: f64,
    pub mae: f64,
    #[wasm_bindgen(js_name = rSquared)]
    pub r_squared: f64,
    pub aic: Option<f64>,
    pub aicc: Option<f64>,
    #[wasm_bindgen(js_name = effectiveDf)]
    pub effective_df: Option<f64>,
    #[wasm_bindgen(js_name = residualSd)]
    pub residual_sd: f64,
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

    #[wasm_bindgen(getter, js_name = standardErrors)]
    pub fn standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidenceLower)]
    pub fn confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = confidenceUpper)]
    pub fn confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = predictionLower)]
    pub fn prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = predictionUpper)]
    pub fn prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = robustnessWeights)]
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

    #[wasm_bindgen(getter, js_name = cvScores)]
    pub fn cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| unsafe { Float64Array::view(v) })
    }

    #[wasm_bindgen(getter, js_name = fractionUsed)]
    pub fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    #[wasm_bindgen(getter, js_name = iterationsUsed)]
    pub fn iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }

    #[wasm_bindgen(getter)]
    pub fn enp(&self) -> Option<f64> {
        self.inner.enp
    }

    #[wasm_bindgen(getter, js_name = traceHat)]
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

    #[wasm_bindgen(getter, js_name = residualScale)]
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

    if !options.is_undefined() && !options.is_null() {
        let opts: SmoothOptions = serde_wasm_bindgen::from_value(options)?;

        if let Some(f) = opts.fraction {
            builder = builder.fraction(f);
        }
        if let Some(iter) = opts.iterations {
            builder = builder.iterations(iter);
        }
        if let Some(wf) = opts.weight_function {
            parse_weight_function(&wf)?;
            builder = builder.weight_function(&wf);
        }
        if let Some(rm) = opts.robustness_method {
            parse_robustness_method(&rm)?;
            builder = builder.robustness_method(&rm);
        }
        if let Some(zw) = opts.zero_weight_fallback {
            parse_zero_weight_fallback(&zw)?;
            builder = builder.zero_weight_fallback(&zw);
        }
        if let Some(bp) = opts.boundary_policy {
            parse_boundary_policy(&bp)?;
            builder = builder.boundary_policy(&bp);
        }
        if let Some(sm) = opts.scaling_method {
            parse_scaling_method(&sm)?;
            builder = builder.scaling_method(&sm);
        }
        if let Some(ac) = opts.auto_converge {
            builder = builder.auto_converge(ac);
        }
        if opts.return_residuals.unwrap_or(false) {
            builder = builder.return_residuals();
        }
        if opts.return_robustness_weights.unwrap_or(false) {
            builder = builder.return_robustness_weights();
        }
        if opts.return_diagnostics.unwrap_or(false) {
            builder = builder.return_diagnostics();
        }
        if let Some(ci) = opts.confidence_intervals {
            builder = builder.confidence_intervals(ci);
        }
        if let Some(pi) = opts.prediction_intervals {
            builder = builder.prediction_intervals(pi);
        }
        if let Some(par) = opts.parallel {
            builder = builder.parallel(par);
        }
        if let Some(deg) = opts.degree {
            parse_polynomial_degree(&deg)?;
            builder = builder.degree(&deg);
        }
        if let Some(dim) = opts.dimensions {
            builder = builder.dimensions(dim);
        }
        {
            let weighted_weights = opts.weighted_metric_weights.clone().unwrap_or_default();
            if let Some(dm) = opts.distance_metric {
                if dm.eq_ignore_ascii_case("weighted") {
                    builder = builder.distance_metric("weighted");
                    builder = builder.weighted_metric_weights(weighted_weights);
                } else {
                    parse_distance_metric(&dm)?;
                    builder = builder.distance_metric(&dm);
                }
            }
        }
        if let Some(sm_val) = opts.surface_mode {
            parse_surface_mode(&sm_val)?;
            builder = builder.surface_mode(&sm_val);
        }
        if opts.return_se.unwrap_or(false) {
            builder = builder.return_se();
        }
        if let Some(cw) = custom_weights {
            if cw.len() != y.length() as usize {
                return Err(JsValue::from_str(&format!(
                    "custom_weights length ({}) must match y length ({})",
                    cw.len(),
                    y.length()
                )));
            }
            if cw.iter().any(|&w| w < 0.0) {
                return Err(JsValue::from_str("custom_weights must be non-negative"));
            }
            builder = builder.custom_weights(cw);
        }
        if let Some(c) = opts.cell {
            builder = builder.cell(c);
        }
        if let Some(v) = opts.interpolation_vertices {
            builder = builder.interpolation_vertices(v);
        }
        if let Some(bdf) = opts.boundary_degree_fallback {
            builder = builder.boundary_degree_fallback(bdf);
        }

        // Cross-validation
        if let Some(fractions) = opts.cv_fractions {
            let method = opts.cv_method.as_deref().unwrap_or("kfold");
            let k = opts.cv_k.unwrap_or(5) as usize;
            let seed = opts.cv_seed;

            match method.to_lowercase().as_str() {
                "simple" | "loo" | "loocv" | "leave_one_out" => {
                    builder = builder.cv_method("loocv");
                    builder = builder.cv_fractions(fractions);
                    if let Some(s) = seed {
                        builder = builder.cv_seed(s);
                    }
                }
                "kfold" | "k_fold" | "k-fold" => {
                    builder = builder.cv_method("kfold");
                    builder = builder.cv_k(k);
                    builder = builder.cv_fractions(fractions);
                    if let Some(s) = seed {
                        builder = builder.cv_seed(s);
                    }
                }
                _ => {
                    return Err(JsValue::from_str(&format!(
                        "Unknown CV method: {}. Valid options: loocv, kfold",
                        method
                    )));
                }
            };
        }
    }

    let x_vec = x.to_vec();
    let y_vec = y.to_vec();

    let model = builder
        .adapter(Batch)
        .build()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = model
        .fit(&x_vec, &y_vec)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

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

            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter);
            }
            if let Some(wf) = opts.weight_function {
                parse_weight_function(&wf)?;
                builder = builder.weight_function(&wf);
            }
            if let Some(rm) = opts.robustness_method {
                parse_robustness_method(&rm)?;
                builder = builder.robustness_method(&rm);
            }
            if let Some(zw) = opts.zero_weight_fallback {
                parse_zero_weight_fallback(&zw)?;
                builder = builder.zero_weight_fallback(&zw);
            }
            if let Some(bp) = opts.boundary_policy {
                parse_boundary_policy(&bp)?;
                builder = builder.boundary_policy(&bp);
            }
            if let Some(sm) = opts.scaling_method {
                parse_scaling_method(&sm)?;
                builder = builder.scaling_method(&sm);
            }
            if let Some(ac) = opts.auto_converge {
                builder = builder.auto_converge(ac);
            }
            if opts.return_residuals.unwrap_or(false) {
                builder = builder.return_residuals();
            }
            if opts.return_robustness_weights.unwrap_or(false) {
                builder = builder.return_robustness_weights();
            }
            if opts.return_diagnostics.unwrap_or(false) {
                builder = builder.return_diagnostics();
            }
            if let Some(ci) = opts.confidence_intervals {
                builder = builder.confidence_intervals(ci);
            }
            if let Some(pi) = opts.prediction_intervals {
                builder = builder.prediction_intervals(pi);
            }
            if let Some(par) = opts.parallel {
                builder = builder.parallel(par);
            }
            if let Some(deg) = opts.degree {
                parse_polynomial_degree(&deg)?;
                builder = builder.degree(&deg);
            }
            if let Some(dim) = opts.dimensions {
                builder = builder.dimensions(dim);
            }
            {
                let weighted_weights = opts.weighted_metric_weights.clone().unwrap_or_default();
                if let Some(dm) = opts.distance_metric {
                    if dm.eq_ignore_ascii_case("weighted") {
                        builder = builder.distance_metric("weighted");
                        builder = builder.weighted_metric_weights(weighted_weights);
                    } else {
                        parse_distance_metric(&dm)?;
                        builder = builder.distance_metric(&dm);
                    }
                }
            }
            if let Some(sm_val) = opts.surface_mode {
                parse_surface_mode(&sm_val)?;
                builder = builder.surface_mode(&sm_val);
            }
            if opts.return_se.unwrap_or(false) {
                builder = builder.return_se();
            }
            if let Some(c) = opts.cell {
                builder = builder.cell(c);
            }
            if let Some(v) = opts.interpolation_vertices {
                builder = builder.interpolation_vertices(v);
            }
            if let Some(bdf) = opts.boundary_degree_fallback {
                builder = builder.boundary_degree_fallback(bdf);
            }
            // Cross-validation
            if let Some(fractions) = opts.cv_fractions {
                let method = opts.cv_method.as_deref().unwrap_or("kfold");
                let k = opts.cv_k.unwrap_or(5) as usize;
                let seed = opts.cv_seed;
                match method.to_lowercase().as_str() {
                    "simple" | "loo" | "loocv" | "leave_one_out" => {
                        builder = builder.cv_method("loocv");
                        builder = builder.cv_fractions(fractions);
                        if let Some(s) = seed {
                            builder = builder.cv_seed(s);
                        }
                    }
                    "kfold" | "k_fold" | "k-fold" => {
                        builder = builder.cv_method("kfold");
                        builder = builder.cv_k(k);
                        builder = builder.cv_fractions(fractions);
                        if let Some(s) = seed {
                            builder = builder.cv_seed(s);
                        }
                    }
                    _ => {
                        return Err(JsValue::from_str(&format!(
                            "Unknown CV method: {}. Valid options: loocv, kfold",
                            method
                        )));
                    }
                };
            }
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
                merge_strategy = parse_merge_strategy(&ms)?;
            }
        }

        let model = builder
            .adapter(Streaming)
            .chunk_size(chunk_size)
            .overlap(overlap)
            .merge_strategy(merge_strategy)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

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
        let result: ::fastLoess::prelude::LoessResult<f64> = self
            .inner
            .process_chunk(&x_vec, &y_vec)
            .map_err(|e: ::fastLoess::prelude::LoessError| JsValue::from_str(&e.to_string()))?;
        Ok(LoessResult { inner: result })
    }

    pub fn finalize(&mut self) -> Result<LoessResult, JsValue> {
        let result: ::fastLoess::prelude::LoessResult<f64> = self
            .inner
            .finalize()
            .map_err(|e: ::fastLoess::prelude::LoessError| JsValue::from_str(&e.to_string()))?;
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

            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter);
            }
            if let Some(wf) = opts.weight_function {
                parse_weight_function(&wf)?;
                builder = builder.weight_function(&wf);
            }
            if let Some(rm) = opts.robustness_method {
                parse_robustness_method(&rm)?;
                builder = builder.robustness_method(&rm);
            }
            if let Some(zw) = opts.zero_weight_fallback {
                parse_zero_weight_fallback(&zw)?;
                builder = builder.zero_weight_fallback(&zw);
            }
            if let Some(bp) = opts.boundary_policy {
                parse_boundary_policy(&bp)?;
                builder = builder.boundary_policy(&bp);
            }
            if let Some(sm) = opts.scaling_method {
                parse_scaling_method(&sm)?;
                builder = builder.scaling_method(&sm);
            }
            if let Some(ac) = opts.auto_converge {
                builder = builder.auto_converge(ac);
            }
            if opts.return_residuals.unwrap_or(false) {
                builder = builder.return_residuals();
            }
            if opts.return_robustness_weights.unwrap_or(false) {
                builder = builder.return_robustness_weights();
            }
            if opts.return_diagnostics.unwrap_or(false) {
                builder = builder.return_diagnostics();
            }
            if let Some(ci) = opts.confidence_intervals {
                builder = builder.confidence_intervals(ci);
            }
            if let Some(pi) = opts.prediction_intervals {
                builder = builder.prediction_intervals(pi);
            }
            if let Some(par) = opts.parallel {
                builder = builder.parallel(par);
            }
            if let Some(deg) = opts.degree {
                parse_polynomial_degree(&deg)?;
                builder = builder.degree(&deg);
            }
            if let Some(dim) = opts.dimensions {
                builder = builder.dimensions(dim);
            }
            if let Some(dim) = opts.dimensions {
                builder = builder.dimensions(dim);
            }
            {
                let weighted_weights = opts.weighted_metric_weights.clone().unwrap_or_default();
                if let Some(dm) = opts.distance_metric {
                    if dm.eq_ignore_ascii_case("weighted") {
                        builder = builder.distance_metric("weighted");
                        builder = builder.weighted_metric_weights(weighted_weights);
                    } else {
                        parse_distance_metric(&dm)?;
                        builder = builder.distance_metric(&dm);
                    }
                }
            }
            if let Some(sm_val) = opts.surface_mode {
                parse_surface_mode(&sm_val)?;
                builder = builder.surface_mode(&sm_val);
            }
            if opts.return_se.unwrap_or(false) {
                builder = builder.return_se();
            }
            if let Some(c) = opts.cell {
                builder = builder.cell(c);
            }
            if let Some(v) = opts.interpolation_vertices {
                builder = builder.interpolation_vertices(v);
            }
            if let Some(bdf) = opts.boundary_degree_fallback {
                builder = builder.boundary_degree_fallback(bdf);
            }
        }

        let mut window_capacity = 100;
        let mut min_points = 2;
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
                update_mode = parse_update_mode(&um)?;
            }
        }

        let model = builder
            .adapter(Online)
            .window_capacity(window_capacity)
            .min_points(min_points)
            .update_mode(update_mode)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(OnlineLoess { inner: model })
    }

    #[wasm_bindgen(js_name = "addPoints")]
    pub fn add_points(
        &mut self,
        xs: &Float64Array,
        ys: &Float64Array,
    ) -> Result<Vec<f64>, JsValue> {
        let xs_vec: Vec<f64> = xs.to_vec();
        let ys_vec: Vec<f64> = ys.to_vec();
        let mut results = Vec::with_capacity(ys_vec.len());
        for (&xi, &yi) in xs_vec.iter().zip(ys_vec.iter()) {
            let result = self
                .inner
                .add_point(std::slice::from_ref(&xi), yi)
                .map_err(|e: ::fastLoess::prelude::LoessError| JsValue::from_str(&e.to_string()))?;
            results.push(result.as_ref().map_or(yi, |o| o.smoothed));
        }
        Ok(results)
    }
}
