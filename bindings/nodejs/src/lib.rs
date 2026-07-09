//! Node.js bindings for fastLoess using N-API.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::api::{Batch, LoessBuilder, Online, Streaming};
use ::fastLoess::internals::binding_support::{MergeStrategy, UpdateMode};
use ::fastLoess::internals::binding_support as shared_parse;
use ::fastLoess::prelude::LoessResult as InnerLoessResult;

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

// Diagnostic statistics for the LOESS fit.
#[napi(object)]
pub struct Diagnostics {
    // Root Mean Squared Error.
    pub rmse: f64,
    // Mean Absolute Error.
    pub mae: f64,
    // R-squared (coefficient of determination).
    #[napi(js_name = "r_squared")]
    pub r_squared: f64,
    // Akaike Information Criterion (if computed).
    pub aic: Option<f64>,
    // Corrected AIC (if computed).
    pub aicc: Option<f64>,
    // Effective degrees of freedom (if computed).
    #[napi(js_name = "effective_df")]
    pub effective_df: Option<f64>,
    // Residual standard deviation.
    #[napi(js_name = "residual_sd")]
    pub residual_sd: f64,
}

// Result of a single online update step.
#[napi(object)]
pub struct OnlineOutput {
    // Smoothed value for the latest point.
    pub smoothed: f64,
    // Standard error (if computed).
    #[napi(js_name = "std_error")]
    pub std_error: Option<f64>,
    // Residual y − smoothed (if computed).
    pub residual: Option<f64>,
    // Robustness weight for the latest point (if computed).
    #[napi(js_name = "robustness_weight")]
    pub robustness_weight: Option<f64>,
    // Number of robustness iterations performed (if applicable).
    #[napi(js_name = "iterations_used")]
    pub iterations_used: Option<u32>,
}

// Result of a LOESS fit.
#[napi]
pub struct LoessResult {
    inner: InnerLoessResult<f64>,
}

#[napi]
impl LoessResult {
    // Get the sorted x values.
    #[napi(getter)]
    pub fn get_x(&self) -> Float64Array {
        Float64Array::from(self.inner.x.as_slice())
    }

    // Get the smoothed y values.
    #[napi(getter)]
    pub fn get_y(&self) -> Float64Array {
        Float64Array::from(self.inner.y.as_slice())
    }

    // Get residuals (if requested).
    #[napi(getter)]
    pub fn get_residuals(&self) -> Option<Float64Array> {
        self.inner
            .residuals
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get standard errors (if requested/computed).
    #[napi(getter, js_name = "standard_errors")]
    pub fn get_standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get lower confidence bounds (if requested).
    #[napi(getter, js_name = "confidence_lower")]
    pub fn get_confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get upper confidence bounds (if requested).
    #[napi(getter, js_name = "confidence_upper")]
    pub fn get_confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get lower prediction bounds (if requested).
    #[napi(getter, js_name = "prediction_lower")]
    pub fn get_prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get upper prediction bounds (if requested).
    #[napi(getter, js_name = "prediction_upper")]
    pub fn get_prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get robustness weights (if requested).
    #[napi(getter, js_name = "robustness_weights")]
    pub fn get_robustness_weights(&self) -> Option<Float64Array> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get diagnostics (if requested).
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

    // Get cross-validation scores (if CV was performed).
    #[napi(getter, js_name = "cv_scores")]
    pub fn get_cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get the fraction used for smoothing.
    #[napi(getter, js_name = "fraction_used")]
    pub fn get_fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    // Get the number of iterations performed.
    #[napi(getter, js_name = "iterations_used")]
    pub fn get_iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }

    // Get equivalent number of parameters (hat-matrix stat, if return_se was set).
    #[napi(getter)]
    pub fn get_enp(&self) -> Option<f64> {
        self.inner.enp
    }

    // Get trace of hat matrix (if return_se was set).
    #[napi(getter, js_name = "trace_hat")]
    pub fn get_trace_hat(&self) -> Option<f64> {
        self.inner.trace_hat
    }

    // Get first delta statistic (if return_se was set).
    #[napi(getter)]
    pub fn get_delta1(&self) -> Option<f64> {
        self.inner.delta1
    }

    // Get second delta statistic (if return_se was set).
    #[napi(getter)]
    pub fn get_delta2(&self) -> Option<f64> {
        self.inner.delta2
    }

    // Get residual scale estimate (if return_se was set).
    #[napi(getter, js_name = "residual_scale")]
    pub fn get_residual_scale(&self) -> Option<f64> {
        self.inner.residual_scale
    }

    // Get per-point leverage / hat-matrix diagonal (if return_se was set).
    #[napi(getter)]
    pub fn get_leverage(&self) -> Option<Float64Array> {
        self.inner
            .leverage
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get number of predictor dimensions.
    #[napi(getter)]
    pub fn get_dimensions(&self) -> u32 {
        self.inner.dimensions as u32
    }
}

// Configuration options for LOESS smoothing.
#[napi(object)]
pub struct SmoothOptions {
    // Smoothing fraction (0 < fraction <= 1). Default: 0.67.
    pub fraction: Option<f64>,
    // Number of robustness iterations. Default: 3.
    pub iterations: Option<u32>,
    // Weight function ("tricube", "gaussian", etc.). Default: "tricube".
    #[napi(js_name = "weight_function")]
    pub weight_function: Option<String>,
    // Robustness method ("bisquare", "huber"). Default: "bisquare".
    #[napi(js_name = "robustness_method")]
    pub robustness_method: Option<String>,
    // Fallback strategy when weights are zero ("use_local_mean").
    #[napi(js_name = "zero_weight_fallback")]
    pub zero_weight_fallback: Option<String>,
    // Boundary handling ("extend", "reflect"). Default: "extend".
    #[napi(js_name = "boundary_policy")]
    pub boundary_policy: Option<String>,
    // Scaling method ("mad", "mar", "mean"). Default: "mad".
    #[napi(js_name = "scaling_method")]
    pub scaling_method: Option<String>,
    // Auto-convergence tolerance. Default: None.
    #[napi(js_name = "auto_converge")]
    pub auto_converge: Option<f64>,
    // Return residuals in result. Default: false.
    #[napi(js_name = "return_residuals")]
    pub return_residuals: Option<bool>,
    // Return robustness weights in result. Default: false.
    #[napi(js_name = "return_robustness_weights")]
    pub return_robustness_weights: Option<bool>,
    // Return diagnostics (RMSE, etc.). Default: false.
    #[napi(js_name = "return_diagnostics")]
    pub return_diagnostics: Option<bool>,
    // Calculate confidence intervals (e.g., 0.95). Default: None.
    #[napi(js_name = "confidence_intervals")]
    pub confidence_intervals: Option<f64>,
    // Calculate prediction intervals. Default: None.
    #[napi(js_name = "prediction_intervals")]
    pub prediction_intervals: Option<f64>,
    // Fractions to use for cross-validation.
    #[napi(js_name = "cv_fractions")]
    pub cv_fractions: Option<Vec<f64>>,
    // CV method ("loocv", "kfold"). Default: "kfold".
    #[napi(js_name = "cv_method")]
    pub cv_method: Option<String>,
    // Number of folds for K-Fold CV. Default: 5.
    #[napi(js_name = "cv_k")]
    pub cv_k: Option<u32>,
    // Enable parallel execution. Default: true.
    pub parallel: Option<bool>,
    // Polynomial degree ("constant", "linear", "quadratic", etc.). Default: "linear".
    pub degree: Option<String>,
    // Number of predictor dimensions. Default: 1.
    pub dimensions: Option<u32>,
    // Distance metric ("normalized", "euclidean", "weighted", etc.). Default: "normalized".
    #[napi(js_name = "distance_metric")]
    pub distance_metric: Option<String>,
    // Per-dimension weights for the "weighted" distance metric.
    #[napi(js_name = "weighted_metric_weights")]
    pub weighted_metric_weights: Option<Vec<f64>>,
    // Surface mode ("interpolation" or "direct"). Default: "interpolation".
    #[napi(js_name = "surface_mode")]
    pub surface_mode: Option<String>,
    // Compute hat-matrix statistics (enp, trace_hat, etc.). Default: false.
    #[napi(js_name = "return_se")]
    pub return_se: Option<bool>,
    // Interpolation cell size (default 0.2). Smaller = more vertices, higher accuracy.
    pub cell: Option<f64>,
    // Maximum number of interpolation vertices.
    #[napi(js_name = "interpolation_vertices")]
    pub interpolation_vertices: Option<u32>,
    // Reduce polynomial degree to linear at boundary vertices (default true).
    #[napi(js_name = "boundary_degree_fallback")]
    pub boundary_degree_fallback: Option<bool>,
    // Random seed for reproducible K-fold cross-validation splits.
    #[napi(js_name = "cv_seed")]
    pub cv_seed: Option<u32>,
}

// LOESS smoothing.
#[napi]
pub struct Loess {
    options: Option<SmoothOptions>,
}

#[napi]
impl Loess {
    // Create a new LOESS smoother.
    #[napi(constructor)]
    pub fn new(options: Option<SmoothOptions>) -> Self {
        Self { options }
    }

    // Fit the model.
    #[napi]
    pub fn fit(
        &self,
        x: Float64Array,
        y: Float64Array,
        custom_weights: Option<Vec<f64>>,
    ) -> Result<LoessResult> {
        let mut builder = self.create_builder()?;
        if let Some(cw) = custom_weights {
            if cw.len() != y.as_ref().len() {
                return Err(to_napi_error(shared_parse::BindingError::invalid_arg(
                    shared_parse::custom_weights_length_mismatch_message_for(
                        "customWeights",
                        cw.len(),
                        y.as_ref().len(),
                    ),
                )));
            }
            if cw.iter().any(|&w| w < 0.0) {
                return Err(to_napi_error(shared_parse::BindingError::invalid_arg(
                    shared_parse::custom_weights_must_be_non_negative_message_for("customWeights"),
                )));
            }
            builder = builder.custom_weights(cw);
        }
        let model = map_runtime(builder.adapter(Batch).build())?;

        let result = map_runtime(model.fit(x.as_ref(), y.as_ref()))?;

        Ok(LoessResult { inner: result })
    }

    // Fit the model asynchronously.
    #[napi(js_name = "fit_async")]
    pub fn fit_async(
        &self,
        x: Float64Array,
        y: Float64Array,
        custom_weights: Option<Vec<f64>>,
    ) -> Result<AsyncTask<LoessTask>> {
        let mut builder = self.create_builder()?;
        if let Some(cw) = custom_weights {
            if cw.len() != y.as_ref().len() {
                return Err(to_napi_error(shared_parse::BindingError::invalid_arg(
                    shared_parse::custom_weights_length_mismatch_message_for(
                        "customWeights",
                        cw.len(),
                        y.as_ref().len(),
                    ),
                )));
            }
            if cw.iter().any(|&w| w < 0.0) {
                return Err(to_napi_error(shared_parse::BindingError::invalid_arg(
                    shared_parse::custom_weights_must_be_non_negative_message_for("customWeights"),
                )));
            }
            builder = builder.custom_weights(cw);
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
        let mut builder = LoessBuilder::<f64>::new();
        let options = &self.options;

        if let Some(opts) = options {
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
                    return_residuals: opts.return_residuals.unwrap_or(false),
                    return_robustness_weights: opts.return_robustness_weights.unwrap_or(false),
                    return_diagnostics: opts.return_diagnostics.unwrap_or(false),
                    confidence_intervals: opts.confidence_intervals,
                    prediction_intervals: opts.prediction_intervals,
                    parallel: opts.parallel,
                    degree: opts.degree.as_deref(),
                    dimensions: opts.dimensions.map(|v| v as usize),
                    distance_metric: opts.distance_metric.as_deref(),
                    weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                    surface_mode: opts.surface_mode.as_deref(),
                    return_se: opts.return_se.unwrap_or(false),
                    cell: opts.cell,
                    interpolation_vertices: opts.interpolation_vertices.map(|v| v as usize),
                    boundary_degree_fallback: opts.boundary_degree_fallback,
                    cv_fractions: opts.cv_fractions.as_deref(),
                    cv_method: opts.cv_method.as_deref(),
                    cv_k: opts.cv_k.map(|v| v as usize),
                    cv_seed: opts.cv_seed.map(|s| s as u64),
                },
            ))?;
            builder = configured_builder;
        }
        Ok(builder)
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
        let model = map_runtime(self.builder.clone().adapter(Batch).build())?;

        map_runtime(model.fit(&self.x, &self.y))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(LoessResult { inner: output })
    }
}

// Configuration options for processing.
#[napi(object)]
pub struct StreamingOptions {
    // Size of each data chunk. Default: 5000.
    #[napi(js_name = "chunk_size")]
    pub chunk_size: Option<u32>,
    // Header/footer overlap size. Default: 500.
    pub overlap: Option<u32>,
    // Strategy for merging chunk overlaps ("average", "weighted_average", "take_first", "take_last").
    #[napi(js_name = "merge_strategy")]
    pub merge_strategy: Option<String>,
}

// LOESS smoother for large datasets.
#[napi]
pub struct StreamingLoess {
    inner: ParallelStreamingLoess<f64>,
}

#[napi]
impl StreamingLoess {
    // Create a new LOESS smoother.
    #[napi(constructor)]
    pub fn new(
        options: Option<SmoothOptions>,
        streaming_opts: Option<StreamingOptions>,
    ) -> Result<Self> {
        let mut builder = LoessBuilder::<f64>::new();

        if let Some(opts) = options {
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
                    return_residuals: opts.return_residuals.unwrap_or(false),
                    return_robustness_weights: opts.return_robustness_weights.unwrap_or(false),
                    return_diagnostics: opts.return_diagnostics.unwrap_or(false),
                    confidence_intervals: opts.confidence_intervals,
                    prediction_intervals: opts.prediction_intervals,
                    parallel: opts.parallel,
                    degree: opts.degree.as_deref(),
                    dimensions: opts.dimensions.map(|v| v as usize),
                    distance_metric: opts.distance_metric.as_deref(),
                    weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                    surface_mode: opts.surface_mode.as_deref(),
                    return_se: opts.return_se.unwrap_or(false),
                    cell: opts.cell,
                    interpolation_vertices: opts.interpolation_vertices.map(|v| v as usize),
                    boundary_degree_fallback: opts.boundary_degree_fallback,
                    cv_fractions: opts.cv_fractions.as_deref(),
                    cv_method: opts.cv_method.as_deref(),
                    cv_k: opts.cv_k.map(|v| v as usize),
                    cv_seed: opts.cv_seed.map(|s| s as u64),
                },
            ))?;
            builder = configured_builder;
        }

        let mut chunk_size = 5000;
        let mut overlap = 500;
        let mut merge_strategy = MergeStrategy::WeightedAverage;

        if let Some(sopts) = streaming_opts {
            if let Some(cs) = sopts.chunk_size {
                chunk_size = cs as usize;
            }
            if let Some(ov) = sopts.overlap {
                overlap = ov as usize;
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

    // Process a chunk of data.
    #[napi(js_name = "process_chunk")]
    pub fn process_chunk(&mut self, x: Float64Array, y: Float64Array) -> Result<LoessResult> {
        let result: InnerLoessResult<f64> =
            map_runtime(self.inner.process_chunk(x.as_ref(), y.as_ref()))?;
        Ok(LoessResult { inner: result })
    }

    // Finalize the stream and return remaining data.
    #[napi]
    pub fn finalize(&mut self) -> Result<LoessResult> {
        let result: InnerLoessResult<f64> = map_runtime(self.inner.finalize())?;
        Ok(LoessResult { inner: result })
    }
}

// Configuration options for processing.
#[napi(object)]
pub struct OnlineOptions {
    // Maximum number of points to keep in the window. Default: 100.
    #[napi(js_name = "window_capacity")]
    pub window_capacity: Option<u32>,
    // Minimum points required before smoothing starts. Default: 2.
    #[napi(js_name = "min_points")]
    pub min_points: Option<u32>,
    // Update mode ("full", "incremental"). Default: "full".
    #[napi(js_name = "update_mode")]
    pub update_mode: Option<String>,
}

// LOESS smoother for real-time data.
#[napi]
pub struct OnlineLoess {
    inner: ParallelOnlineLoess<f64>,
}

#[napi]
impl OnlineLoess {
    // Create a new LOESS smoother.
    #[napi(constructor)]
    pub fn new(options: Option<SmoothOptions>, online_opts: Option<OnlineOptions>) -> Result<Self> {
        let mut builder = LoessBuilder::<f64>::new();

        if let Some(opts) = options {
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
                    return_residuals: opts.return_residuals.unwrap_or(false),
                    return_robustness_weights: opts.return_robustness_weights.unwrap_or(false),
                    return_diagnostics: opts.return_diagnostics.unwrap_or(false),
                    confidence_intervals: opts.confidence_intervals,
                    prediction_intervals: opts.prediction_intervals,
                    parallel: opts.parallel,
                    degree: opts.degree.as_deref(),
                    dimensions: opts.dimensions.map(|v| v as usize),
                    distance_metric: opts.distance_metric.as_deref(),
                    weighted_metric_weights: opts.weighted_metric_weights.as_deref(),
                    surface_mode: opts.surface_mode.as_deref(),
                    return_se: opts.return_se.unwrap_or(false),
                    cell: opts.cell,
                    interpolation_vertices: opts.interpolation_vertices.map(|v| v as usize),
                    boundary_degree_fallback: opts.boundary_degree_fallback,
                    cv_fractions: opts.cv_fractions.as_deref(),
                    cv_method: opts.cv_method.as_deref(),
                    cv_k: opts.cv_k.map(|v| v as usize),
                    cv_seed: opts.cv_seed.map(|s| s as u64),
                },
            ))?;
            builder = configured_builder;
        }

        let mut window_capacity = 1000;
        let mut min_points = 3;
        let mut update_mode = UpdateMode::Full;

        if let Some(oopts) = online_opts {
            if let Some(wc) = oopts.window_capacity {
                window_capacity = wc as usize;
            }
            if let Some(mp) = oopts.min_points {
                min_points = mp as usize;
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

    // Add a single point and return its smoothed value, or null if the window
    // is not yet full enough to produce a result.
    #[napi(js_name = "add_point")]
    pub fn add_point(&mut self, x: f64, y: f64) -> Result<Option<OnlineOutput>> {
        let output = self
            .inner
            .add_point(&[x], y)
            .map_err(|e| to_napi_error(shared_parse::BindingError::invalid_arg(e.to_string())))?;
        Ok(output.map(|o| OnlineOutput {
            smoothed: o.smoothed,
            std_error: o.std_error,
            residual: o.residual,
            robustness_weight: o.robustness_weight,
            iterations_used: o.iterations_used.map(|i| i as u32),
        }))
    }
}
