//! Shared helpers for language bindings.
//!
//! This module centralizes string option parsing used by C/C++, Julia,
//! Node.js, Python, R, and WASM bindings so option aliases and validation
//! behavior stay consistent across all binding frontends.

use crate::adapters::batch::ParallelBatchLoessBuilder;
use crate::adapters::online::ParallelOnlineLoessBuilder;
use crate::adapters::streaming::ParallelStreamingLoessBuilder;
use crate::api::LoessBuilder;
use crate::parse::IntoEnum;
use crate::prelude::{LoessError, LoessResult};
pub use loess_rs::internals::adapters::online::UpdateMode;
pub use loess_rs::internals::adapters::streaming::MergeStrategy;
use loess_rs::internals::algorithms::regression::SolverLinalg;
pub use loess_rs::internals::algorithms::regression::{PolynomialDegree, ZeroWeightFallback};
pub use loess_rs::internals::algorithms::robustness::RobustnessMethod;
pub use loess_rs::internals::engine::executor::SurfaceMode;
use loess_rs::internals::evaluation::intervals::IntervalMethod;
pub use loess_rs::internals::math::boundary::BoundaryPolicy;
use loess_rs::internals::math::distance::DistanceLinalg;
pub use loess_rs::internals::math::distance::DistanceMetric;
pub use loess_rs::internals::math::kernel::WeightFunction;
use loess_rs::internals::math::linalg::FloatLinalg;
pub use loess_rs::internals::math::scaling::ScalingMethod;
pub use loess_rs::internals::primitives::backend::Backend;
use std::ffi::CString;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingErrorCategory {
    InvalidArg,
    Runtime,
}

#[derive(Debug, Clone)]
pub struct BindingError {
    pub category: BindingErrorCategory,
    pub message: String,
}

impl BindingError {
    pub fn invalid_arg(msg: impl Into<String>) -> Self {
        Self {
            category: BindingErrorCategory::InvalidArg,
            message: msg.into(),
        }
    }

    pub fn runtime(msg: impl Into<String>) -> Self {
        Self {
            category: BindingErrorCategory::Runtime,
            message: msg.into(),
        }
    }
}

pub fn map_invalid_arg<T, E: ToString>(result: Result<T, E>) -> Result<T, BindingError> {
    result.map_err(|e| BindingError::invalid_arg(e.to_string()))
}

pub fn map_runtime<T, E: ToString>(result: Result<T, E>) -> Result<T, BindingError> {
    result.map_err(|e| BindingError::runtime(e.to_string()))
}

pub const PANIC_FALLBACK_MESSAGE: &str = "Panic in Rust library";
pub const CONFIG_POINTER_IS_NULL: &str = "Config pointer is null";
pub const MODEL_POINTER_IS_NULL: &str = "Model pointer is null";
pub const PROCESSOR_POINTER_IS_NULL: &str = "Processor pointer is null";
pub const INVALID_DATA_INPUTS: &str = "Invalid data inputs";
pub const XY_ARRAYS_MUST_NOT_BE_NULL: &str = "x and y arrays must not be null";
pub const ARRAY_LENGTH_MUST_BE_GREATER_THAN_ZERO: &str = "Array length must be greater than 0";
pub const CUSTOM_WEIGHTS_MUST_BE_NON_NEGATIVE: &str = "custom_weights must be non-negative";

pub fn sanitize_error_message(msg: &str) -> String {
    msg.replace('\0', " ")
}

pub fn to_cstring_lossy(msg: &str) -> CString {
    CString::new(sanitize_error_message(msg)).unwrap_or_default()
}

pub fn panic_fallback_message() -> &'static str {
    PANIC_FALLBACK_MESSAGE
}

pub fn dims_mismatch_message(x_len: usize, y_len: usize, dimensions: usize) -> String {
    format!(
        "x length ({}) must equal y length ({}) * dimensions ({})",
        x_len, y_len, dimensions
    )
}

pub fn custom_weights_length_mismatch_message(weights_len: usize, y_len: usize) -> String {
    format!(
        "custom_weights length ({}) must match y length ({})",
        weights_len, y_len
    )
}

pub fn custom_weights_length_mismatch_message_for(
    label: &str,
    weights_len: usize,
    y_len: usize,
) -> String {
    format!(
        "{} length ({}) must match y length ({})",
        label, weights_len, y_len
    )
}

pub fn custom_weights_must_be_non_negative_message_for(label: &str) -> String {
    format!("{} must be non-negative", label)
}

pub fn required_option_message(option_name: &str) -> String {
    format!("{} must be provided", option_name)
}

pub fn mutex_poisoned_message(details: &str) -> String {
    format!("Mutex poisoned: {}", details)
}

pub struct OnlineResultMetadata {
    pub dimensions: usize,
    pub degree: PolynomialDegree,
    pub distance_metric: DistanceMetric<f64>,
    pub fraction_used: f64,
    pub iterations_used: Option<usize>,
}

pub fn online_add_points_to_result<F>(
    x: &[f64],
    y: &[f64],
    metadata: &OnlineResultMetadata,
    mut process_point: F,
) -> Result<LoessResult<f64>, String>
where
    F: FnMut(&[f64], f64) -> Result<Option<f64>, LoessError>,
{
    let d = metadata.dimensions.max(1);
    if x.len() != y.len() * d {
        return Err(dims_mismatch_message(x.len(), y.len(), d));
    }

    let mut smoothed = Vec::with_capacity(y.len());
    for (xi_chunk, &yi) in x.chunks(d).zip(y.iter()) {
        let out = process_point(xi_chunk, yi).map_err(|e| e.to_string())?;
        smoothed.push(out.unwrap_or(yi));
    }

    Ok(LoessResult {
        x: x.to_vec(),
        y: smoothed,
        dimensions: metadata.dimensions,
        distance_metric: metadata.distance_metric.clone(),
        polynomial_degree: metadata.degree,
        standard_errors: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: metadata.iterations_used,
        fraction_used: metadata.fraction_used,
        cv_scores: None,
        enp: None,
        trace_hat: None,
        delta1: None,
        delta2: None,
        residual_scale: None,
        leverage: None,
    })
}

pub struct BuilderOptionSet<'a> {
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub weight_function: Option<&'a str>,
    pub robustness_method: Option<&'a str>,
    pub zero_weight_fallback: Option<&'a str>,
    pub boundary_policy: Option<&'a str>,
    pub scaling_method: Option<&'a str>,
    pub auto_converge: Option<f64>,
    pub return_residuals: bool,
    pub return_robustness_weights: bool,
    pub return_diagnostics: bool,
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
    pub parallel: Option<bool>,
    pub degree: Option<&'a str>,
    pub dimensions: Option<usize>,
    pub distance_metric: Option<&'a str>,
    pub weighted_metric_weights: Option<&'a [f64]>,
    pub surface_mode: Option<&'a str>,
    pub return_se: bool,
    pub cell: Option<f64>,
    pub interpolation_vertices: Option<usize>,
    pub boundary_degree_fallback: Option<bool>,
    pub cv_fractions: Option<&'a [f64]>,
    pub cv_method: Option<&'a str>,
    pub cv_k: Option<usize>,
    pub cv_seed: Option<u64>,
}

pub struct AppliedBuilderOptions {
    pub degree: Option<PolynomialDegree>,
    pub distance_metric: Option<DistanceMetric<f64>>,
}

pub struct TypedBuilderOptionSet {
    pub fraction: Option<f64>,
    pub iterations: Option<usize>,
    pub weight_function: Option<WeightFunction>,
    pub robustness_method: Option<RobustnessMethod>,
    pub zero_weight_fallback: Option<ZeroWeightFallback>,
    pub boundary_policy: Option<BoundaryPolicy>,
    pub scaling_method: Option<ScalingMethod>,
    pub auto_converge: Option<f64>,
    pub return_residuals: bool,
    pub return_robustness_weights: bool,
    pub return_diagnostics: bool,
    pub confidence_intervals: Option<f64>,
    pub prediction_intervals: Option<f64>,
    pub parallel: Option<bool>,
    pub degree: Option<PolynomialDegree>,
    pub dimensions: Option<usize>,
    pub distance_metric: Option<DistanceMetric<f64>>,
    pub surface_mode: Option<SurfaceMode>,
    pub return_se: bool,
    pub cell: Option<f64>,
    pub interpolation_vertices: Option<usize>,
    pub boundary_degree_fallback: Option<bool>,
    pub cv_fractions: Option<Vec<f64>>,
    pub cv_method: Option<String>,
    pub cv_k: Option<usize>,
    pub cv_seed: Option<u64>,
}

pub fn parse_weight_function(name: &str) -> Result<WeightFunction, String> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(format!(
            "Unknown weight function: {}. Valid options: tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            name
        )),
    }
}

pub fn parse_robustness_method(name: &str) -> Result<RobustnessMethod, String> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(format!(
            "Unknown robustness method: {}. Valid options: bisquare, huber, talwar",
            name
        )),
    }
}

pub fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback, String> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(format!(
            "Unknown zero weight fallback: {}. Valid options: use_local_mean, return_original, return_none",
            name
        )),
    }
}

pub fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy, String> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(format!(
            "Unknown boundary policy: {}. Valid options: extend, reflect, zero, noboundary",
            name
        )),
    }
}

pub fn parse_scaling_method(name: &str) -> Result<ScalingMethod, String> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(ScalingMethod::MAD),
        "mar" => Ok(ScalingMethod::MAR),
        "mean" => Ok(ScalingMethod::Mean),
        _ => Err(format!(
            "Unknown scaling method: {}. Valid options: mad, mar, mean",
            name
        )),
    }
}

pub fn parse_polynomial_degree(name: &str) -> Result<PolynomialDegree, String> {
    match name.to_lowercase().as_str() {
        "constant" | "0" => Ok(PolynomialDegree::Constant),
        "linear" | "1" => Ok(PolynomialDegree::Linear),
        "quadratic" | "2" => Ok(PolynomialDegree::Quadratic),
        "cubic" | "3" => Ok(PolynomialDegree::Cubic),
        "quartic" | "4" => Ok(PolynomialDegree::Quartic),
        _ => Err(format!(
            "Unknown polynomial degree: {}. Valid options: constant, linear, quadratic, cubic, quartic",
            name
        )),
    }
}

pub fn parse_distance_metric(name: &str) -> Result<DistanceMetric<f64>, String> {
    let lower = name.to_lowercase();
    if let Some(p_str) = lower.strip_prefix("minkowski:") {
        let p: f64 = p_str
            .parse()
            .map_err(|_| format!("Invalid Minkowski p value: {}", p_str))?;
        return Ok(DistanceMetric::Minkowski(p));
    }

    match lower.as_str() {
        "normalized" | "norm" => Ok(DistanceMetric::Normalized),
        "euclidean" | "euclid" => Ok(DistanceMetric::Euclidean),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        "chebyshev" | "linf" => Ok(DistanceMetric::Chebyshev),
        "minkowski" => Ok(DistanceMetric::Minkowski(2.0)),
        "weighted" => Ok(DistanceMetric::Weighted(Vec::new())),
        _ => Err(format!(
            "Unknown distance metric: {}. Valid options: normalized, euclidean, manhattan, chebyshev, minkowski, weighted",
            name
        )),
    }
}

pub fn weight_function_str(value: WeightFunction) -> &'static str {
    match value {
        WeightFunction::Tricube => "tricube",
        WeightFunction::Epanechnikov => "epanechnikov",
        WeightFunction::Gaussian => "gaussian",
        WeightFunction::Uniform => "uniform",
        WeightFunction::Biweight => "biweight",
        WeightFunction::Triangle => "triangle",
        WeightFunction::Cosine => "cosine",
    }
}

pub fn robustness_method_str(value: RobustnessMethod) -> &'static str {
    match value {
        RobustnessMethod::Bisquare => "bisquare",
        RobustnessMethod::Huber => "huber",
        RobustnessMethod::Talwar => "talwar",
    }
}

pub fn scaling_method_str(value: ScalingMethod) -> &'static str {
    match value {
        ScalingMethod::MAD => "mad",
        ScalingMethod::MAR => "mar",
        ScalingMethod::Mean => "mean",
    }
}

pub fn zero_weight_fallback_str(value: ZeroWeightFallback) -> &'static str {
    match value {
        ZeroWeightFallback::UseLocalMean => "use_local_mean",
        ZeroWeightFallback::ReturnOriginal => "return_original",
        ZeroWeightFallback::ReturnNone => "return_none",
    }
}

pub fn boundary_policy_str(value: BoundaryPolicy) -> &'static str {
    match value {
        BoundaryPolicy::Extend => "extend",
        BoundaryPolicy::Reflect => "reflect",
        BoundaryPolicy::Zero => "zero",
        BoundaryPolicy::NoBoundary => "noboundary",
    }
}

pub fn polynomial_degree_str(value: PolynomialDegree) -> &'static str {
    match value {
        PolynomialDegree::Constant => "constant",
        PolynomialDegree::Linear => "linear",
        PolynomialDegree::Quadratic => "quadratic",
        PolynomialDegree::Cubic => "cubic",
        PolynomialDegree::Quartic => "quartic",
    }
}

pub fn surface_mode_str(value: SurfaceMode) -> &'static str {
    match value {
        SurfaceMode::Interpolation => "interpolation",
        SurfaceMode::Direct => "direct",
    }
}

pub fn distance_metric_components(value: &DistanceMetric<f64>) -> (String, Option<&[f64]>) {
    match value {
        DistanceMetric::Normalized => ("normalized".to_string(), None),
        DistanceMetric::Euclidean => ("euclidean".to_string(), None),
        DistanceMetric::Manhattan => ("manhattan".to_string(), None),
        DistanceMetric::Chebyshev => ("chebyshev".to_string(), None),
        DistanceMetric::Minkowski(p) => (format!("minkowski:{}", p), None),
        DistanceMetric::Weighted(w) => ("weighted".to_string(), Some(w.as_slice())),
    }
}

pub fn build_distance_metric(
    name: &str,
    weighted_metric_weights: Option<&[f64]>,
) -> Result<DistanceMetric<f64>, String> {
    if name.eq_ignore_ascii_case("weighted") {
        return match weighted_metric_weights {
            Some(weights) => Ok(DistanceMetric::Weighted(weights.to_vec())),
            None => Err(
                "weighted_metric_weights must be provided when distance_metric is 'weighted'"
                    .to_string(),
            ),
        };
    }

    match parse_distance_metric(name)? {
        DistanceMetric::Weighted(_) => {
            let weights = weighted_metric_weights.ok_or_else(|| {
                "weighted_metric_weights must be provided when distance_metric is 'weighted'"
                    .to_string()
            })?;
            Ok(DistanceMetric::Weighted(weights.to_vec()))
        }
        other => Ok(other),
    }
}

pub fn apply_distance_metric_value(
    mut builder: LoessBuilder<f64>,
    distance_metric: &DistanceMetric<f64>,
) -> LoessBuilder<f64> {
    match distance_metric {
        DistanceMetric::Normalized => builder.distance_metric("normalized"),
        DistanceMetric::Euclidean => builder.distance_metric("euclidean"),
        DistanceMetric::Manhattan => builder.distance_metric("manhattan"),
        DistanceMetric::Chebyshev => builder.distance_metric("chebyshev"),
        DistanceMetric::Minkowski(p) => {
            let metric = format!("minkowski:{}", p);
            builder.distance_metric(&metric)
        }
        DistanceMetric::Weighted(weights) => {
            builder = builder.distance_metric("weighted");
            builder.weighted_metric_weights(weights.clone())
        }
    }
}

pub fn apply_distance_metric(
    builder: LoessBuilder<f64>,
    name: &str,
    weighted_metric_weights: Option<&[f64]>,
) -> Result<(LoessBuilder<f64>, DistanceMetric<f64>), String> {
    let parsed_metric = build_distance_metric(name, weighted_metric_weights)?;
    let builder = apply_distance_metric_value(builder, &parsed_metric);
    Ok((builder, parsed_metric))
}

pub fn apply_cross_validation(
    mut builder: LoessBuilder<f64>,
    fractions: Option<&[f64]>,
    method: Option<&str>,
    k: Option<usize>,
    seed: Option<u64>,
) -> Result<LoessBuilder<f64>, String> {
    let Some(fractions) = fractions else {
        return Ok(builder);
    };

    let method = method.unwrap_or("kfold");
    let k = k.unwrap_or(5);

    match method.to_lowercase().as_str() {
        "simple" | "loo" | "loocv" | "leave_one_out" => {
            builder = builder.cv_method("loocv");
            builder = builder.cv_fractions(fractions.to_vec());
            if let Some(s) = seed {
                builder = builder.cv_seed(s);
            }
            Ok(builder)
        }
        "kfold" | "k_fold" | "k-fold" => {
            builder = builder.cv_method("kfold");
            builder = builder.cv_k(k);
            builder = builder.cv_fractions(fractions.to_vec());
            if let Some(s) = seed {
                builder = builder.cv_seed(s);
            }
            Ok(builder)
        }
        _ => Err(format!(
            "Unknown CV method: {}. Valid options: loocv, kfold",
            method
        )),
    }
}

pub fn apply_builder_options(
    builder: LoessBuilder<f64>,
    options: BuilderOptionSet<'_>,
) -> Result<(LoessBuilder<f64>, AppliedBuilderOptions), String> {
    let typed = TypedBuilderOptionSet {
        fraction: options.fraction,
        iterations: options.iterations,
        weight_function: options
            .weight_function
            .map(parse_weight_function)
            .transpose()?,
        robustness_method: options
            .robustness_method
            .map(parse_robustness_method)
            .transpose()?,
        zero_weight_fallback: options
            .zero_weight_fallback
            .map(parse_zero_weight_fallback)
            .transpose()?,
        boundary_policy: options
            .boundary_policy
            .map(parse_boundary_policy)
            .transpose()?,
        scaling_method: options
            .scaling_method
            .map(parse_scaling_method)
            .transpose()?,
        auto_converge: options.auto_converge,
        return_residuals: options.return_residuals,
        return_robustness_weights: options.return_robustness_weights,
        return_diagnostics: options.return_diagnostics,
        confidence_intervals: options.confidence_intervals,
        prediction_intervals: options.prediction_intervals,
        parallel: options.parallel,
        degree: options.degree.map(parse_polynomial_degree).transpose()?,
        dimensions: options.dimensions,
        distance_metric: options
            .distance_metric
            .map(|dm| build_distance_metric(dm, options.weighted_metric_weights))
            .transpose()?,
        surface_mode: options.surface_mode.map(parse_surface_mode).transpose()?,
        return_se: options.return_se,
        cell: options.cell,
        interpolation_vertices: options.interpolation_vertices,
        boundary_degree_fallback: options.boundary_degree_fallback,
        cv_fractions: options.cv_fractions.map(|v| v.to_vec()),
        cv_method: options.cv_method.map(str::to_string),
        cv_k: options.cv_k,
        cv_seed: options.cv_seed,
    };

    apply_typed_builder_options(builder, typed)
}

pub fn apply_typed_builder_options(
    mut builder: LoessBuilder<f64>,
    options: TypedBuilderOptionSet,
) -> Result<(LoessBuilder<f64>, AppliedBuilderOptions), String> {
    let mut applied_degree = None;
    let mut applied_distance_metric = None;

    if let Some(f) = options.fraction {
        builder = builder.fraction(f);
    }
    if let Some(iter) = options.iterations {
        builder = builder.iterations(iter);
    }
    if let Some(wf) = options.weight_function {
        builder = builder.weight_function(weight_function_str(wf));
    }
    if let Some(rm) = options.robustness_method {
        builder = builder.robustness_method(robustness_method_str(rm));
    }
    if let Some(zw) = options.zero_weight_fallback {
        builder = builder.zero_weight_fallback(zero_weight_fallback_str(zw));
    }
    if let Some(bp) = options.boundary_policy {
        builder = builder.boundary_policy(boundary_policy_str(bp));
    }
    if let Some(sm) = options.scaling_method {
        builder = builder.scaling_method(scaling_method_str(sm));
    }
    if let Some(ac) = options.auto_converge {
        builder = builder.auto_converge(ac);
    }
    if options.return_residuals {
        builder = builder.return_residuals();
    }
    if options.return_robustness_weights {
        builder = builder.return_robustness_weights();
    }
    if options.return_diagnostics {
        builder = builder.return_diagnostics();
    }
    if let Some(ci) = options.confidence_intervals {
        builder = builder.confidence_intervals(ci);
    }
    if let Some(pi) = options.prediction_intervals {
        builder = builder.prediction_intervals(pi);
    }
    if let Some(par) = options.parallel {
        builder = builder.parallel(par);
    }
    if let Some(deg) = options.degree {
        applied_degree = Some(deg);
        builder = builder.degree(polynomial_degree_str(deg));
    }
    if let Some(dims) = options.dimensions {
        builder = builder.dimensions(dims);
    }
    if let Some(dm) = options.distance_metric {
        applied_distance_metric = Some(dm.clone());
        builder = apply_distance_metric_value(builder, &dm);
    }
    if let Some(sm) = options.surface_mode {
        builder = builder.surface_mode(surface_mode_str(sm));
    }
    if options.return_se {
        builder = builder.return_se();
    }
    if let Some(c) = options.cell {
        builder = builder.cell(c);
    }
    if let Some(v) = options.interpolation_vertices {
        builder = builder.interpolation_vertices(v);
    }
    if let Some(bdf) = options.boundary_degree_fallback {
        builder = builder.boundary_degree_fallback(bdf);
    }

    builder = apply_cross_validation(
        builder,
        options.cv_fractions.as_deref(),
        options.cv_method.as_deref(),
        options.cv_k,
        options.cv_seed,
    )?;

    Ok((
        builder,
        AppliedBuilderOptions {
            degree: applied_degree,
            distance_metric: applied_distance_metric,
        },
    ))
}

pub fn parse_surface_mode(name: &str) -> Result<SurfaceMode, String> {
    match name.to_lowercase().as_str() {
        "interpolation" | "interp" | "interpolate" => Ok(SurfaceMode::Interpolation),
        "direct" => Ok(SurfaceMode::Direct),
        _ => Err(format!(
            "Unknown surface mode: {}. Valid options: interpolation, direct",
            name
        )),
    }
}

pub fn parse_update_mode(name: &str) -> Result<UpdateMode, String> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(format!(
            "Unknown update mode: {}. Valid options: full, incremental",
            name
        )),
    }
}

pub fn parse_merge_strategy(name: &str) -> Result<MergeStrategy, String> {
    match name.to_lowercase().as_str() {
        "average" | "mean" => Ok(MergeStrategy::Average),
        "weighted" | "weighted_average" | "weightedaverage" => Ok(MergeStrategy::WeightedAverage),
        "first" | "take_first" | "takefirst" | "left" => Ok(MergeStrategy::TakeFirst),
        "last" | "take_last" | "takelast" | "right" => Ok(MergeStrategy::TakeLast),
        _ => Err(format!(
            "Unknown merge strategy: {}. Valid options: average, weighted_average, take_first, take_last",
            name
        )),
    }
}

// ─── Builder setter methods ───────────────────────────────────────────────────
//
// These impl blocks are in binding_support.rs (compiled only with the `dev`
// feature) so the adapter files (batch.rs, online.rs, streaming.rs) stay free
// of any `#[cfg]` attributes.

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelBatchLoessBuilder<T>
{
    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the kernel weight function.
    #[allow(private_bounds)]
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness method for outlier handling.
    #[allow(private_bounds)]
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the residual scaling method (MAR/MAD).
    #[allow(private_bounds)]
    pub fn scaling_method(mut self, method: impl IntoEnum<ScalingMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.scaling_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the zero-weight fallback policy.
    #[allow(private_bounds)]
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    #[allow(private_bounds)]
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the polynomial degree.
    #[allow(private_bounds)]
    pub fn polynomial_degree(mut self, degree: impl IntoEnum<PolynomialDegree>) -> Self {
        match degree.into_enum() {
            Ok(d) => self.base.polynomial_degree = d,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the number of dimensions explicitly.
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.base.dimensions = dims;
        self
    }

    // Set the distance metric.
    #[allow(private_bounds)]
    pub fn distance_metric(mut self, metric: impl IntoEnum<DistanceMetric<T>>) -> Self {
        match metric.into_enum() {
            Ok(m) => self.base.distance_metric = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the surface evaluation mode (Direct or Interpolation).
    #[allow(private_bounds)]
    pub fn surface_mode(mut self, mode: impl IntoEnum<SurfaceMode>) -> Self {
        match mode.into_enum() {
            Ok(m) => self.base.surface_mode = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the cell size for interpolation mode.
    pub fn cell(mut self, cell: f64) -> Self {
        self.base.cell = Some(cell);
        self
    }

    // Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.base.boundary_degree_fallback = enabled;
        self
    }

    // Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.base.interpolation_vertices = Some(vertices);
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }

    // Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.base.interval_type = Some(match self.base.interval_type {
            Some(existing) if existing.prediction => IntervalMethod {
                level,
                confidence: true,
                prediction: true,
                se: true,
            },
            _ => IntervalMethod::confidence(level),
        });
        self
    }

    // Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.base.interval_type = Some(match self.base.interval_type {
            Some(existing) if existing.confidence => IntervalMethod {
                level,
                confidence: true,
                prediction: true,
                se: true,
            },
            _ => IntervalMethod::prediction(level),
        });
        self
    }

    // Set the cross-validation method: `"kfold"` or `"loocv"`.
    pub fn cv_method(mut self, method: &str) -> Self {
        self.cv_method_str = Some(method.to_string());
        self
    }

    // Set the number of folds for K-fold cross-validation (default: 5).
    pub fn cv_k(mut self, k: usize) -> Self {
        self.cv_k_val = k;
        self
    }

    // Set the candidate fractions to evaluate during cross-validation.
    pub fn cv_fractions(mut self, fractions: Vec<T>) -> Self {
        self.base.cv_fractions = Some(fractions);
        self
    }

    // Set the random seed for reproducible cross-validation fold splitting.
    pub fn cv_seed(mut self, seed: u64) -> Self {
        self.base.cv_seed = Some(seed);
        self
    }

    // Set per-dimension weights for the `"weighted"` distance metric.
    //
    // Calling this method selects the weighted Euclidean metric and supplies
    // the weight vector (one entry per predictor dimension). Must be combined
    // with `.distance_metric("weighted")` or called on its own.
    pub fn weighted_metric_weights(mut self, weights: Vec<T>) -> Self {
        self.weighted_metric_weights = Some(weights);
        self
    }

    // Enable returning standard errors in the result.
    pub fn return_se(mut self, enabled: bool) -> Self {
        if enabled && self.base.interval_type.is_none() {
            self.base.interval_type = Some(IntervalMethod::se());
        }
        self
    }

    // Set user-defined case weights (one per observation).
    //
    // Weights multiply the local kernel weight at each neighborhood point:
    // `w_ij = custom_weights[j] * K(d_ij / h) * robustness_j`.
    //
    // Analogous to `weights` in R's `stats::loess`. Must have the same length as `y`.
    pub fn custom_weights(mut self, weights: Vec<T>) -> Self {
        self.base.custom_weights = Some(weights);
        self
    }
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelOnlineLoessBuilder<T>
{
    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the kernel weight function.
    #[allow(private_bounds)]
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness method for outlier handling.
    #[allow(private_bounds)]
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the residual scaling method (MAR/MAD).
    #[allow(private_bounds)]
    pub fn scaling_method(mut self, method: impl IntoEnum<ScalingMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.scaling_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the zero-weight fallback policy.
    #[allow(private_bounds)]
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    #[allow(private_bounds)]
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the polynomial degree.
    #[allow(private_bounds)]
    pub fn polynomial_degree(mut self, degree: impl IntoEnum<PolynomialDegree>) -> Self {
        match degree.into_enum() {
            Ok(d) => self.base.polynomial_degree = d,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the number of dimensions explicitly (though usually inferred from input).
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.base.dimensions = dims;
        self
    }

    // Set the distance metric.
    #[allow(private_bounds)]
    pub fn distance_metric(mut self, metric: impl IntoEnum<DistanceMetric<T>>) -> Self {
        match metric.into_enum() {
            Ok(m) => self.base.distance_metric = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the surface evaluation mode (Direct or Interpolation).
    #[allow(private_bounds)]
    pub fn surface_mode(mut self, mode: impl IntoEnum<SurfaceMode>) -> Self {
        match mode.into_enum() {
            Ok(m) => self.base.surface_mode = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the cell size for interpolation mode.
    pub fn cell(mut self, cell: f64) -> Self {
        self.base.cell = Some(cell);
        self
    }

    // Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.base.interpolation_vertices = Some(vertices);
        self
    }

    // Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.base.boundary_degree_fallback = enabled;
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Set whether to compute residuals.
    pub fn compute_residuals(mut self, compute: bool) -> Self {
        self.base.compute_residuals = compute;
        self
    }

    // Set whether to return robustness weights.
    pub fn return_robustness_weights(mut self, ret: bool) -> Self {
        self.base.return_robustness_weights = ret;
        self
    }

    // Set the window capacity.
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.base.window_capacity = capacity;
        self
    }

    // Set the minimum points required before smoothing.
    pub fn min_points(mut self, min: usize) -> Self {
        self.base.min_points = min;
        self
    }

    // Set the update mode (Incremental/Full).
    #[allow(private_bounds)]
    pub fn update_mode(mut self, mode: impl IntoEnum<UpdateMode>) -> Self {
        match mode.into_enum() {
            Ok(m) => self.base.update_mode = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set per-dimension weights for the `"weighted"` distance metric.
    pub fn weighted_metric_weights(mut self, weights: Vec<T>) -> Self {
        self.weighted_metric_weights = Some(weights);
        self
    }
}

impl<T: FloatLinalg + DistanceLinalg + SolverLinalg + Debug + Send + Sync>
    ParallelStreamingLoessBuilder<T>
{
    // Set parallel execution mode.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.base.parallel = Some(parallel);
        self
    }

    // Set the execution backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.base.backend = Some(backend);
        self
    }

    // Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.base.fraction = fraction;
        self
    }

    // Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.base.iterations = iterations;
        self
    }

    // Set the kernel weight function.
    #[allow(private_bounds)]
    pub fn weight_function(mut self, wf: impl IntoEnum<WeightFunction>) -> Self {
        match wf.into_enum() {
            Ok(w) => self.base.weight_function = w,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the robustness method for outlier handling.
    #[allow(private_bounds)]
    pub fn robustness_method(mut self, method: impl IntoEnum<RobustnessMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.robustness_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the residual scaling method (MAR/MAD).
    #[allow(private_bounds)]
    pub fn scaling_method(mut self, method: impl IntoEnum<ScalingMethod>) -> Self {
        match method.into_enum() {
            Ok(m) => self.base.scaling_method = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the zero-weight fallback policy.
    #[allow(private_bounds)]
    pub fn zero_weight_fallback(mut self, fallback: impl IntoEnum<ZeroWeightFallback>) -> Self {
        match fallback.into_enum() {
            Ok(f) => self.base.zero_weight_fallback = f,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the boundary handling policy.
    #[allow(private_bounds)]
    pub fn boundary_policy(mut self, policy: impl IntoEnum<BoundaryPolicy>) -> Self {
        match policy.into_enum() {
            Ok(p) => self.base.boundary_policy = p,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the polynomial degree.
    #[allow(private_bounds)]
    pub fn polynomial_degree(mut self, degree: impl IntoEnum<PolynomialDegree>) -> Self {
        match degree.into_enum() {
            Ok(d) => self.base.polynomial_degree = d,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the number of dimensions explicitly.
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.base.dimensions = dims;
        self
    }

    // Set the distance metric.
    #[allow(private_bounds)]
    pub fn distance_metric(mut self, metric: impl IntoEnum<DistanceMetric<T>>) -> Self {
        match metric.into_enum() {
            Ok(m) => self.base.distance_metric = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the surface evaluation mode (Direct or Interpolation).
    #[allow(private_bounds)]
    pub fn surface_mode(mut self, mode: impl IntoEnum<SurfaceMode>) -> Self {
        match mode.into_enum() {
            Ok(m) => self.base.surface_mode = m,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set the cell size for interpolation mode.
    pub fn cell(mut self, cell: f64) -> Self {
        self.base.cell = Some(cell);
        self
    }

    // Set the maximum number of vertices for interpolation.
    pub fn interpolation_vertices(mut self, vertices: usize) -> Self {
        self.base.interpolation_vertices = Some(vertices);
        self
    }

    // Set whether to reduce polynomial degree at boundary vertices.
    pub fn boundary_degree_fallback(mut self, enabled: bool) -> Self {
        self.base.boundary_degree_fallback = enabled;
        self
    }

    // Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.base.auto_converge = Some(tolerance);
        self
    }

    // Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.base.compute_residuals = enabled;
        self
    }

    // Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.base.return_robustness_weights = enabled;
        self
    }

    // Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.base.return_diagnostics = enabled;
        self
    }

    // Set chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.base.chunk_size = size;
        self
    }

    // Set overlap between chunks.
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.base.overlap = overlap;
        self
    }

    // Set the merge strategy for overlapping chunks.
    #[allow(private_bounds)]
    pub fn merge_strategy(mut self, strategy: impl IntoEnum<MergeStrategy>) -> Self {
        match strategy.into_enum() {
            Ok(s) => self.base.merge_strategy = s,
            Err(e) => self.parse_errors.push(e),
        }
        self
    }

    // Set per-dimension weights for the `"weighted"` distance metric.
    pub fn weighted_metric_weights(mut self, weights: Vec<T>) -> Self {
        self.weighted_metric_weights = Some(weights);
        self
    }
}
