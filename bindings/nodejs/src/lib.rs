//! Node.js bindings for fastLoess using N-API.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::api::{
    BoundaryPolicy, DistanceMetric, MergeStrategy, PolynomialDegree, RobustnessMethod,
    ScalingMethod, SurfaceMode, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use ::fastLoess::prelude::{
    Batch, Loess as LoessBuilder, LoessError, LoessResult as InnerLoessResult, MAD,
    MAR, Online, Streaming,
};

// Parse weight function from string
fn parse_weight_function(name: &str) -> Result<WeightFunction> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown weight function: {}", name),
        )),
    }
}

// Parse robustness method from string
fn parse_robustness_method(name: &str) -> Result<RobustnessMethod> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown robustness method: {}", name),
        )),
    }
}

// Parse zero weight fallback from string
fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown zero weight fallback: {}", name),
        )),
    }
}

// Parse boundary policy from string
fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown boundary policy: {}", name),
        )),
    }
}

// Parse scaling method from string
fn parse_scaling_method(name: &str) -> Result<ScalingMethod> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        "mean" => Ok(ScalingMethod::Mean),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!(
                "Unknown scaling method: {}. Valid options: mad, mar, mean",
                name
            ),
        )),
    }
}

// Parse polynomial degree from string
fn parse_polynomial_degree(name: &str) -> Result<PolynomialDegree> {
    match name.to_lowercase().as_str() {
        "constant" | "0" => Ok(PolynomialDegree::Constant),
        "linear" | "1" => Ok(PolynomialDegree::Linear),
        "quadratic" | "2" => Ok(PolynomialDegree::Quadratic),
        "cubic" | "3" => Ok(PolynomialDegree::Cubic),
        "quartic" | "4" => Ok(PolynomialDegree::Quartic),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown polynomial degree: {}", name),
        )),
    }
}

// Parse distance metric from string
fn parse_distance_metric(name: &str) -> Result<DistanceMetric<f64>> {
    // Handle "minkowski:p" inline format
    if let Some(p_str) = name.to_lowercase().strip_prefix("minkowski:") {
        let p: f64 = p_str.parse().map_err(|_| {
            Error::new(
                Status::InvalidArg,
                format!("Invalid Minkowski p value: {}", p_str),
            )
        })?;
        return Ok(DistanceMetric::Minkowski(p));
    }
    match name.to_lowercase().as_str() {
        "normalized" | "norm" => Ok(DistanceMetric::Normalized),
        "euclidean" | "euclid" => Ok(DistanceMetric::Euclidean),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        "chebyshev" | "linf" => Ok(DistanceMetric::Chebyshev),
        "minkowski" => Ok(DistanceMetric::Minkowski(2.0)),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!(
                "Unknown distance metric: {}. Valid options: normalized, euclidean, manhattan, chebyshev, minkowski, weighted",
                name
            ),
        )),
    }
}

// Parse merge strategy from string
fn parse_merge_strategy(name: &str) -> Result<MergeStrategy> {
    match name.to_lowercase().as_str() {
        "average" | "mean" => Ok(MergeStrategy::Average),
        "weighted_average" | "weighted" => Ok(MergeStrategy::WeightedAverage),
        "take_first" | "first" => Ok(MergeStrategy::TakeFirst),
        "take_last" | "last" => Ok(MergeStrategy::TakeLast),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!(
                "Unknown merge strategy: {}. Valid options: average, weighted_average, take_first, take_last",
                name
            ),
        )),
    }
}

// Parse surface mode from string
fn parse_surface_mode(name: &str) -> Result<SurfaceMode> {
    match name.to_lowercase().as_str() {
        "interpolation" | "interp" => Ok(SurfaceMode::Interpolation),
        "direct" => Ok(SurfaceMode::Direct),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown surface mode: {}", name),
        )),
    }
}

// Parse update mode from string
fn parse_update_mode(name: &str) -> Result<UpdateMode> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown update mode: {}", name),
        )),
    }
}

// Diagnostic statistics for the LOESS fit.
#[napi(object)]
pub struct Diagnostics {
    // Root Mean Squared Error.
    pub rmse: f64,
    // Mean Absolute Error.
    pub mae: f64,
    // R-squared (coefficient of determination).
    pub r_squared: f64,
    // Akaike Information Criterion (if computed).
    pub aic: Option<f64>,
    // Corrected AIC (if computed).
    pub aicc: Option<f64>,
    // Effective degrees of freedom (if computed).
    pub effective_df: Option<f64>,
    // Residual standard deviation.
    pub residual_sd: f64,
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
    #[napi(getter)]
    pub fn get_standard_errors(&self) -> Option<Float64Array> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get lower confidence bounds (if requested).
    #[napi(getter)]
    pub fn get_confidence_lower(&self) -> Option<Float64Array> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get upper confidence bounds (if requested).
    #[napi(getter)]
    pub fn get_confidence_upper(&self) -> Option<Float64Array> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get lower prediction bounds (if requested).
    #[napi(getter)]
    pub fn get_prediction_lower(&self) -> Option<Float64Array> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get upper prediction bounds (if requested).
    #[napi(getter)]
    pub fn get_prediction_upper(&self) -> Option<Float64Array> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get robustness weights (if requested).
    #[napi(getter)]
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
    #[napi(getter)]
    pub fn get_cv_scores(&self) -> Option<Float64Array> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| Float64Array::from(v.as_slice()))
    }

    // Get the fraction used for smoothing.
    #[napi(getter)]
    pub fn get_fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    // Get the number of iterations performed.
    #[napi(getter)]
    pub fn get_iterations_used(&self) -> Option<u32> {
        self.inner.iterations_used.map(|i| i as u32)
    }

    // Get equivalent number of parameters (hat-matrix stat, if return_se was set).
    #[napi(getter)]
    pub fn get_enp(&self) -> Option<f64> {
        self.inner.enp
    }

    // Get trace of hat matrix (if return_se was set).
    #[napi(getter)]
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
    #[napi(getter)]
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
    pub weight_function: Option<String>,
    // Robustness method ("bisquare", "huber"). Default: "bisquare".
    pub robustness_method: Option<String>,
    // Fallback strategy when weights are zero ("use_local_mean").
    pub zero_weight_fallback: Option<String>,
    // Boundary handling ("extend", "reflect"). Default: "extend".
    pub boundary_policy: Option<String>,
    // Scaling method ("mad", "mar", "mean"). Default: "mad".
    pub scaling_method: Option<String>,
    // Auto-convergence tolerance. Default: None.
    pub auto_converge: Option<f64>,
    // Return residuals in result. Default: false.
    pub return_residuals: Option<bool>,
    // Return robustness weights in result. Default: false.
    pub return_robustness_weights: Option<bool>,
    // Return diagnostics (RMSE, etc.). Default: false.
    pub return_diagnostics: Option<bool>,
    // Calculate confidence intervals (e.g., 0.95). Default: None.
    pub confidence_intervals: Option<f64>,
    // Calculate prediction intervals. Default: None.
    pub prediction_intervals: Option<f64>,
    // Fractions to use for cross-validation.
    pub cv_fractions: Option<Vec<f64>>,
    // CV method ("loocv", "kfold"). Default: "kfold".
    pub cv_method: Option<String>,
    // Number of folds for K-Fold CV. Default: 5.
    pub cv_k: Option<u32>,
    // Enable parallel execution. Default: true.
    pub parallel: Option<bool>,
    // Polynomial degree ("constant", "linear", "quadratic", etc.). Default: "linear".
    pub degree: Option<String>,
    // Number of predictor dimensions. Default: 1.
    pub dimensions: Option<u32>,
    // Distance metric ("normalized", "euclidean", "weighted", etc.). Default: "normalized".
    pub distance_metric: Option<String>,
    // Per-dimension weights for the "weighted" distance metric.
    pub weighted_metric_weights: Option<Vec<f64>>,
    // Surface mode ("interpolation" or "direct"). Default: "interpolation".
    pub surface_mode: Option<String>,
    // Compute hat-matrix statistics (enp, trace_hat, etc.). Default: false.
    pub return_se: Option<bool>,
    // Interpolation cell size (default 0.2). Smaller = more vertices, higher accuracy.
    pub cell: Option<f64>,
    // Maximum number of interpolation vertices.
    pub interpolation_vertices: Option<u32>,
    // Reduce polynomial degree to linear at boundary vertices (default true).
    pub boundary_degree_fallback: Option<bool>,
    // Random seed for reproducible K-fold cross-validation splits.
    pub cv_seed: Option<u32>,
}

// Batch LOESS smoothing.
#[napi]
pub struct Loess {
    options: Option<SmoothOptions>,
}

#[napi]
impl Loess {
    // Create a new batch LOESS smoother.
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
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "customWeights length ({}) must match y length ({})",
                        cw.len(),
                        y.as_ref().len()
                    ),
                ));
            }
            if cw.iter().any(|&w| w < 0.0) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "customWeights must be non-negative".to_string(),
                ));
            }
            builder = builder.custom_weights(cw);
        }
        let model = builder
            .adapter(Batch)
            .build()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        let result = model
            .fit(x.as_ref(), y.as_ref())
            .map_err(|e: LoessError| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(LoessResult { inner: result })
    }

    // Fit the model asynchronously.
    #[napi(js_name = "fitAsync")]
    pub fn fit_async(
        &self,
        x: Float64Array,
        y: Float64Array,
        custom_weights: Option<Vec<f64>>,
    ) -> Result<AsyncTask<LoessTask>> {
        let mut builder = self.create_builder()?;
        if let Some(cw) = custom_weights {
            if cw.len() != y.as_ref().len() {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "customWeights length ({}) must match y length ({})",
                        cw.len(),
                        y.as_ref().len()
                    ),
                ));
            }
            if cw.iter().any(|&w| w < 0.0) {
                return Err(Error::new(
                    Status::InvalidArg,
                    "customWeights must be non-negative".to_string(),
                ));
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
        let mut builder = LoessBuilder::new();
        let options = &self.options;

        if let Some(opts) = options {
            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter as usize);
            }
            if let Some(wf) = &opts.weight_function {
                builder = builder.weight_function(parse_weight_function(wf)?);
            }
            if let Some(rm) = &opts.robustness_method {
                builder = builder.robustness_method(parse_robustness_method(rm)?);
            }
            if let Some(zw) = &opts.zero_weight_fallback {
                builder = builder.zero_weight_fallback(parse_zero_weight_fallback(zw)?);
            }
            if let Some(bp) = &opts.boundary_policy {
                builder = builder.boundary_policy(parse_boundary_policy(bp)?);
            }
            if let Some(sm) = &opts.scaling_method {
                builder = builder.scaling_method(parse_scaling_method(sm)?);
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
            if let Some(deg) = &opts.degree {
                builder = builder.degree(parse_polynomial_degree(deg)?);
            }
            if let Some(dims) = opts.dimensions {
                builder = builder.dimensions(dims as usize);
            }
            if let Some(dm) = &opts.distance_metric {
                let metric = if dm.to_lowercase() == "weighted" {
                    DistanceMetric::Weighted(opts.weighted_metric_weights.clone().unwrap_or_default())
                } else {
                    parse_distance_metric(dm)?
                };
                builder = builder.distance_metric(metric);
            }
            if let Some(sm) = &opts.surface_mode {
                builder = builder.surface_mode(parse_surface_mode(sm)?);
            }
            if opts.return_se.unwrap_or(false) {
                builder = builder.return_se();
            }
            if let Some(c) = opts.cell {
                builder = builder.cell(c);
            }
            if let Some(v) = opts.interpolation_vertices {
                builder = builder.interpolation_vertices(v as usize);
            }
            if let Some(bdf) = opts.boundary_degree_fallback {
                builder = builder.boundary_degree_fallback(bdf);
            }

            // Cross-validation
            if let Some(fractions) = &opts.cv_fractions {
                let method = opts.cv_method.as_deref().unwrap_or("kfold");
                let k = opts.cv_k.unwrap_or(5) as usize;
                let seed = opts.cv_seed.map(|s| s as u64);

                match method.to_lowercase().as_str() {
                    "simple" | "loo" | "loocv" | "leave_one_out" => {
                        builder = builder.cv_method("loocv");
                        builder = builder.cv_fractions(fractions.clone());
                        if let Some(s) = seed { builder = builder.cv_seed(s); }
                    }
                    "kfold" | "k_fold" | "k-fold" => {
                        builder = builder.cv_method("kfold");
                        builder = builder.cv_k(k);
                        builder = builder.cv_fractions(fractions.clone());
                        if let Some(s) = seed { builder = builder.cv_seed(s); }
                    }
                    _ => {
                        return Err(Error::new(
                            Status::InvalidArg,
                            format!("Unknown CV method: {}. Valid options: loocv, kfold", method),
                        ));
                    }
                };
            }
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
        let model = self
            .builder
            .clone()
            .adapter(Batch)
            .build()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        model
            .fit(&self.x, &self.y)
            .map_err(|e: LoessError| Error::new(Status::GenericFailure, e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(LoessResult { inner: output })
    }
}

// Configuration options for streaming processing.
#[napi(object)]
pub struct StreamingOptions {
    // Size of each data chunk. Default: 5000.
    pub chunk_size: Option<u32>,
    // Header/footer overlap size. Default: 500.
    pub overlap: Option<u32>,
    // Strategy for merging chunk overlaps ("average", "weighted_average", "take_first", "take_last").
    pub merge_strategy: Option<String>,
}

// Streaming LOESS smoother for large datasets.
#[napi]
pub struct StreamingLoess {
    inner: ParallelStreamingLoess<f64>,
}

#[napi]
impl StreamingLoess {
    // Create a new streaming LOESS smoother.
    #[napi(constructor)]
    pub fn new(
        options: Option<SmoothOptions>,
        streaming_opts: Option<StreamingOptions>,
    ) -> Result<Self> {
        let mut builder = LoessBuilder::new();

        if let Some(opts) = options {
            if let Some(f) = opts.fraction {
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter as usize);
            }
            if let Some(wf) = opts.weight_function {
                builder = builder.weight_function(parse_weight_function(&wf)?);
            }
            if let Some(rm) = opts.robustness_method {
                builder = builder.robustness_method(parse_robustness_method(&rm)?);
            }
            if let Some(zw) = opts.zero_weight_fallback {
                builder = builder.zero_weight_fallback(parse_zero_weight_fallback(&zw)?);
            }
            if let Some(bp) = opts.boundary_policy {
                builder = builder.boundary_policy(parse_boundary_policy(&bp)?);
            }
            if let Some(sm) = opts.scaling_method {
                builder = builder.scaling_method(parse_scaling_method(&sm)?);
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
            if let Some(par) = opts.parallel {
                builder = builder.parallel(par);
            }
            if let Some(deg) = opts.degree {
                builder = builder.degree(parse_polynomial_degree(&deg)?);
            }
            if let Some(dims) = opts.dimensions {
                builder = builder.dimensions(dims as usize);
            }
            if let Some(dm) = opts.distance_metric {
                let metric = if dm.to_lowercase() == "weighted" {
                    DistanceMetric::Weighted(opts.weighted_metric_weights.clone().unwrap_or_default())
                } else {
                    parse_distance_metric(&dm)?
                };
                builder = builder.distance_metric(metric);
            }
            if let Some(sm) = opts.surface_mode {
                builder = builder.surface_mode(parse_surface_mode(&sm)?);
            }
            if opts.return_se.unwrap_or(false) {
                builder = builder.return_se();
            }
            if let Some(c) = opts.cell {
                builder = builder.cell(c);
            }
            if let Some(v) = opts.interpolation_vertices {
                builder = builder.interpolation_vertices(v as usize);
            }
            if let Some(bdf) = opts.boundary_degree_fallback {
                builder = builder.boundary_degree_fallback(bdf);
            }
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
                merge_strategy = parse_merge_strategy(&ms)?;
            }
        }

        let model = builder
            .adapter(Streaming)
            .chunk_size(chunk_size)
            .overlap(overlap)
            .merge_strategy(merge_strategy)
            .build()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(StreamingLoess { inner: model })
    }

    // Process a chunk of data.
    #[napi]
    pub fn process_chunk(&mut self, x: Float64Array, y: Float64Array) -> Result<LoessResult> {
        let result: InnerLoessResult<f64> = self
            .inner
            .process_chunk(x.as_ref(), y.as_ref())
            .map_err(|e: LoessError| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(LoessResult { inner: result })
    }

    // Finalize the stream and return remaining data.
    #[napi]
    pub fn finalize(&mut self) -> Result<LoessResult> {
        let result: InnerLoessResult<f64> = self
            .inner
            .finalize()
            .map_err(|e: LoessError| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(LoessResult { inner: result })
    }
}

// Configuration options for online processing.
#[napi(object)]
pub struct OnlineOptions {
    // Maximum number of points to keep in the window. Default: 100.
    pub window_capacity: Option<u32>,
    // Minimum points required before smoothing starts. Default: 2.
    pub min_points: Option<u32>,
    // Update mode ("full", "incremental"). Default: "full".
    pub update_mode: Option<String>,
}

// Online LOESS smoother for real-time data.
#[napi]
pub struct OnlineLoess {
    inner: ParallelOnlineLoess<f64>,
    dimensions: usize,
    degree: PolynomialDegree,
    distance_metric: DistanceMetric<f64>,
    fraction_used: f64,
}

#[napi]
impl OnlineLoess {
    // Create a new online LOESS smoother.
    #[napi(constructor)]
    pub fn new(options: Option<SmoothOptions>, online_opts: Option<OnlineOptions>) -> Result<Self> {
        let mut builder = LoessBuilder::new();
        let mut dimensions = 1usize;
        let mut degree = PolynomialDegree::Linear;
        let mut distance_metric = DistanceMetric::Normalized;
        let mut fraction_used = 0.2f64;

        if let Some(opts) = options {
            if let Some(f) = opts.fraction {
                fraction_used = f;
                builder = builder.fraction(f);
            }
            if let Some(iter) = opts.iterations {
                builder = builder.iterations(iter as usize);
            }
            if let Some(wf) = &opts.weight_function {
                builder = builder.weight_function(parse_weight_function(wf)?);
            }
            if let Some(rm) = &opts.robustness_method {
                builder = builder.robustness_method(parse_robustness_method(rm)?);
            }
            if let Some(zw) = &opts.zero_weight_fallback {
                builder = builder.zero_weight_fallback(parse_zero_weight_fallback(zw)?);
            }
            if let Some(bp) = &opts.boundary_policy {
                builder = builder.boundary_policy(parse_boundary_policy(bp)?);
            }
            if let Some(sm) = &opts.scaling_method {
                builder = builder.scaling_method(parse_scaling_method(sm)?);
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
            if let Some(deg_str) = &opts.degree {
                degree = parse_polynomial_degree(deg_str)?;
                builder = builder.degree(degree);
            }
            if let Some(dims) = opts.dimensions {
                dimensions = dims as usize;
                builder = builder.dimensions(dimensions);
            }
            if let Some(dm_str) = &opts.distance_metric {
                distance_metric = if dm_str.to_lowercase() == "weighted" {
                    DistanceMetric::Weighted(opts.weighted_metric_weights.clone().unwrap_or_default())
                } else {
                    parse_distance_metric(dm_str)?
                };
                builder = builder.distance_metric(distance_metric.clone());
            }
            if let Some(sm) = &opts.surface_mode {
                builder = builder.surface_mode(parse_surface_mode(sm)?);
            }
            if opts.return_se.unwrap_or(false) {
                builder = builder.return_se();
            }
            if let Some(c) = opts.cell {
                builder = builder.cell(c);
            }
            if let Some(v) = opts.interpolation_vertices {
                builder = builder.interpolation_vertices(v as usize);
            }
            if let Some(bdf) = opts.boundary_degree_fallback {
                builder = builder.boundary_degree_fallback(bdf);
            }
            // Cross-validation
            if let Some(fractions) = &opts.cv_fractions {
                let method = opts.cv_method.as_deref().unwrap_or("kfold");
                let k = opts.cv_k.unwrap_or(5) as usize;
                let seed = opts.cv_seed.map(|s| s as u64);
                match method.to_lowercase().as_str() {
                    "simple" | "loo" | "loocv" | "leave_one_out" => {
                        builder = builder.cv_method("loocv");
                        builder = builder.cv_fractions(fractions.clone());
                        if let Some(s) = seed { builder = builder.cv_seed(s); }
                    }
                    "kfold" | "k_fold" | "k-fold" => {
                        builder = builder.cv_method("kfold");
                        builder = builder.cv_k(k);
                        builder = builder.cv_fractions(fractions.clone());
                        if let Some(s) = seed { builder = builder.cv_seed(s); }
                    }
                    _ => {
                        return Err(Error::new(
                            Status::InvalidArg,
                            format!("Unknown CV method: {}", method),
                        ));
                    }
                };
            }
        }

        let mut window_capacity = 100;
        let mut min_points = 2;
        let mut update_mode = UpdateMode::Full;

        if let Some(oopts) = online_opts {
            if let Some(wc) = oopts.window_capacity {
                window_capacity = wc as usize;
            }
            if let Some(mp) = oopts.min_points {
                min_points = mp as usize;
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
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(OnlineLoess {
            inner: model,
            dimensions,
            degree,
            distance_metric,
            fraction_used,
        })
    }

    // Add new points to the window and get smoothed values.
    #[napi]
    pub fn add_points(&mut self, x: Float64Array, y: Float64Array) -> Result<LoessResult> {
        let x_slice = x.as_ref();
        let y_slice = y.as_ref();
        let x_vec = x_slice.to_vec();

        let mut smoothed = Vec::with_capacity(y_slice.len());
        for (&xi, &yi) in x_slice.iter().zip(y_slice.iter()) {
            let output = self
                .inner
                .add_point(std::slice::from_ref(&xi), yi)
                .map_err(|e: LoessError| Error::new(Status::GenericFailure, e.to_string()))?;
            smoothed.push(output.as_ref().map_or(yi, |o| o.smoothed));
        }

        let inner_result = InnerLoessResult {
            x: x_vec,
            y: smoothed,
            dimensions: self.dimensions,
            distance_metric: self.distance_metric.clone(),
            polynomial_degree: self.degree,
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals: None,
            robustness_weights: None,
            diagnostics: None,
            iterations_used: None,
            fraction_used: self.fraction_used,
            cv_scores: None,
            enp: None,
            trace_hat: None,
            delta1: None,
            delta2: None,
            residual_scale: None,
            leverage: None,
        };

        Ok(LoessResult {
            inner: inner_result,
        })
    }
}
