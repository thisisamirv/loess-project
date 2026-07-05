//! C/C++ bindings for fastLoess.
//!
//! Provides C access to the fastLoess Rust library via C FFI.
//! A C++ wrapper header (fastloess.hpp) provides idiomatic C++ usage.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::ptr;

use fastLoess::internals::adapters::online::ParallelOnlineLoess;
use fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use fastLoess::internals::api::{
    BoundaryPolicy, DistanceMetric, MergeStrategy, PolynomialDegree, RobustnessMethod,
    ScalingMethod, SurfaceMode, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use fastLoess::prelude::{
    Batch, KFold, LOOCV, Loess as LoessBuilder, LoessResult, MAD, MAR, Online, Streaming,
};

/// Result struct that can be passed across FFI boundary.
/// All arrays are allocated by Rust and must be freed by Rust.
#[repr(C)]
pub struct CppLoessResult {
    /// Sorted x values (length = n)
    pub x: *mut c_double,
    /// Smoothed y values (length = n)
    pub y: *mut c_double,
    /// Number of data points
    pub n: c_ulong,

    /// Standard errors (NULL if not computed)
    pub standard_errors: *mut c_double,
    /// Lower confidence bounds (NULL if not computed)
    pub confidence_lower: *mut c_double,
    /// Upper confidence bounds (NULL if not computed)
    pub confidence_upper: *mut c_double,
    /// Lower prediction bounds (NULL if not computed)
    pub prediction_lower: *mut c_double,
    /// Upper prediction bounds (NULL if not computed)
    pub prediction_upper: *mut c_double,
    /// Residuals (NULL if not computed)
    pub residuals: *mut c_double,
    /// Robustness weights (NULL if not computed)
    pub robustness_weights: *mut c_double,

    /// Fraction used for smoothing
    pub fraction_used: c_double,
    /// Number of iterations performed (-1 if not available)
    pub iterations_used: c_int,

    /// Diagnostics (NaN if not computed)
    pub rmse: c_double,
    pub mae: c_double,
    pub r_squared: c_double,
    pub aic: c_double,
    pub aicc: c_double,
    pub effective_df: c_double,
    pub residual_sd: c_double,

    /// Hat-matrix statistics (NaN / NULL if not computed; set return_se = 1 to enable)
    pub enp: c_double,
    pub trace_hat: c_double,
    pub delta1: c_double,
    pub delta2: c_double,
    pub residual_scale: c_double,
    /// Per-point leverage / hat-matrix diagonal (NULL if not computed, length = n)
    pub leverage: *mut c_double,
    /// Number of predictor dimensions used
    pub dimensions: c_int,

    /// Error message (NULL if no error)
    pub error: *mut c_char,
}

impl Default for CppLoessResult {
    fn default() -> Self {
        CppLoessResult {
            x: ptr::null_mut(),
            y: ptr::null_mut(),
            n: 0,
            standard_errors: ptr::null_mut(),
            confidence_lower: ptr::null_mut(),
            confidence_upper: ptr::null_mut(),
            prediction_lower: ptr::null_mut(),
            prediction_upper: ptr::null_mut(),
            residuals: ptr::null_mut(),
            robustness_weights: ptr::null_mut(),
            fraction_used: 0.0,
            iterations_used: -1,
            rmse: f64::NAN,
            mae: f64::NAN,
            r_squared: f64::NAN,
            aic: f64::NAN,
            aicc: f64::NAN,
            effective_df: f64::NAN,
            residual_sd: f64::NAN,
            enp: f64::NAN,
            trace_hat: f64::NAN,
            delta1: f64::NAN,
            delta2: f64::NAN,
            residual_scale: f64::NAN,
            leverage: ptr::null_mut(),
            dimensions: 1,
            error: ptr::null_mut(),
        }
    }
}

/// Convert a Vec<f64> to a raw pointer.
fn vec_to_ptr(v: Vec<f64>) -> *mut c_double {
    let mut boxed = v.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

/// Convert an optional Vec<f64> to a raw pointer.
fn opt_vec_to_ptr(v: Option<Vec<f64>>) -> *mut c_double {
    match v {
        Some(vec) => vec_to_ptr(vec),
        None => ptr::null_mut(),
    }
}

/// Create an error result with the given message.
fn error_result(msg: &str) -> CppLoessResult {
    let mut result = CppLoessResult::default();
    let c_string = std::ffi::CString::new(msg).unwrap_or_default();
    result.error = c_string.into_raw();
    result
}

/// Parse a C string safely.
unsafe fn parse_c_str(s: *const c_char, default: &str) -> &str {
    if s.is_null() {
        default
    } else {
        CStr::from_ptr(s).to_str().unwrap_or(default)
    }
}

/// Parse weight function from string.
fn parse_weight_function(name: &str) -> Result<WeightFunction, String> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(format!(
            "Unknown weight function: {}. Valid: tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            name
        )),
    }
}

/// Parse robustness method from string.
fn parse_robustness_method(name: &str) -> Result<RobustnessMethod, String> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(format!(
            "Unknown robustness method: {}. Valid: bisquare, huber, talwar",
            name
        )),
    }
}

/// Parse zero weight fallback from string.
fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback, String> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(format!(
            "Unknown zero weight fallback: {}. Valid: use_local_mean, return_original, return_none",
            name
        )),
    }
}

/// Parse boundary policy from string.
fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy, String> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(format!(
            "Unknown boundary policy: {}. Valid: extend, reflect, zero, noboundary",
            name
        )),
    }
}

/// Parse scaling method from string.
fn parse_scaling_method(name: &str) -> Result<ScalingMethod, String> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        "mean" => Ok(ScalingMethod::Mean),
        _ => Err(format!(
            "Unknown scaling method: {}. Valid: mad, mar, mean",
            name
        )),
    }
}

/// Parse update mode from string.
fn parse_update_mode(name: &str) -> Result<UpdateMode, String> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(format!(
            "Unknown update mode: {}. Valid: full, incremental",
            name
        )),
    }
}

/// Parse merge strategy from string.
fn parse_merge_strategy(name: &str) -> Result<MergeStrategy, String> {
    match name.to_lowercase().as_str() {
        "average" | "mean" => Ok(MergeStrategy::Average),
        "weighted" | "weighted_average" | "weightedaverage" => Ok(MergeStrategy::WeightedAverage),
        "first" | "take_first" | "takefirst" | "left" => Ok(MergeStrategy::TakeFirst),
        "last" | "take_last" | "takelast" | "right" => Ok(MergeStrategy::TakeLast),
        _ => Err(format!(
            "Unknown merge strategy: {}. Valid: average, weighted, first, last",
            name
        )),
    }
}

/// Parse polynomial degree from string.
fn parse_polynomial_degree(name: &str) -> Result<PolynomialDegree, String> {
    match name.to_lowercase().as_str() {
        "constant" | "0" => Ok(PolynomialDegree::Constant),
        "linear" | "1" => Ok(PolynomialDegree::Linear),
        "quadratic" | "2" => Ok(PolynomialDegree::Quadratic),
        "cubic" | "3" => Ok(PolynomialDegree::Cubic),
        "quartic" | "4" => Ok(PolynomialDegree::Quartic),
        _ => Err(format!(
            "Unknown degree: {}. Valid: constant, linear, quadratic, cubic, quartic",
            name
        )),
    }
}

/// Parse surface mode from string.
fn parse_surface_mode(name: &str) -> Result<SurfaceMode, String> {
    match name.to_lowercase().as_str() {
        "direct" => Ok(SurfaceMode::Direct),
        "interpolation" | "interp" => Ok(SurfaceMode::Interpolation),
        _ => Err(format!(
            "Unknown surface mode: {}. Valid: direct, interpolation",
            name
        )),
    }
}

/// Parse distance metric from string.
fn parse_distance_metric(name: &str) -> Result<DistanceMetric<f64>, String> {
    match name.to_lowercase().as_str() {
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "normalized" | "norm" => Ok(DistanceMetric::Normalized),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        "chebyshev" | "linf" => Ok(DistanceMetric::Chebyshev),
        _ => Err(format!(
            "Unknown distance metric: {}. Valid: euclidean, normalized, manhattan, chebyshev",
            name
        )),
    }
}

impl From<LoessResult<f64>> for CppLoessResult {
    fn from(result: LoessResult<f64>) -> Self {
        let n = result.y.len();

        let (rmse, mae, r_squared, aic, aicc, effective_df, residual_sd) =
            if let Some(ref d) = result.diagnostics {
                (
                    d.rmse,
                    d.mae,
                    d.r_squared,
                    d.aic.unwrap_or(f64::NAN),
                    d.aicc.unwrap_or(f64::NAN),
                    d.effective_df.unwrap_or(f64::NAN),
                    d.residual_sd,
                )
            } else {
                (
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                )
            };

        CppLoessResult {
            x: vec_to_ptr(result.x),
            y: vec_to_ptr(result.y),
            n: n as c_ulong,
            standard_errors: opt_vec_to_ptr(result.standard_errors),
            confidence_lower: opt_vec_to_ptr(result.confidence_lower),
            confidence_upper: opt_vec_to_ptr(result.confidence_upper),
            prediction_lower: opt_vec_to_ptr(result.prediction_lower),
            prediction_upper: opt_vec_to_ptr(result.prediction_upper),
            residuals: opt_vec_to_ptr(result.residuals),
            robustness_weights: opt_vec_to_ptr(result.robustness_weights),
            fraction_used: result.fraction_used,
            iterations_used: result.iterations_used.map(|i| i as c_int).unwrap_or(-1),
            rmse,
            mae,
            r_squared,
            aic,
            aicc,
            effective_df,
            residual_sd,
            enp: result.enp.unwrap_or(f64::NAN),
            trace_hat: result.trace_hat.unwrap_or(f64::NAN),
            delta1: result.delta1.unwrap_or(f64::NAN),
            delta2: result.delta2.unwrap_or(f64::NAN),
            residual_scale: result.residual_scale.unwrap_or(f64::NAN),
            leverage: opt_vec_to_ptr(result.leverage),
            dimensions: result.dimensions as c_int,
            error: ptr::null_mut(),
        }
    }
}

/// Opaque handle to a Loess batch model.
pub struct CppLoess {
    builder: Option<LoessBuilder<f64>>,
    // Store CV options to apply lazily because of lifetime constraints
    cv_fractions: Option<Vec<f64>>,
    cv_method: Option<String>,
    cv_k: usize,
    // User-defined case weights
    custom_weights: Option<Vec<f64>>,
}

/// Opaque handle to a Loess streaming model.
pub struct CppStreamingLoess {
    builder: LoessBuilder<f64>,
    streaming_opts: Option<(usize, usize, MergeStrategy)>,
    model: Option<ParallelStreamingLoess<f64>>,
}

/// Opaque handle to a Loess online model.
pub struct CppOnlineLoess {
    builder: LoessBuilder<f64>,
    online_opts: Option<(usize, usize, UpdateMode)>,
    model: Option<ParallelOnlineLoess<f64>>,
    dimensions: usize,
}

/// C++ wrapper constructor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null. Arrays must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_new(
    fraction: c_double,
    iterations: c_int,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    confidence_intervals: c_double,
    prediction_intervals: c_double,
    return_diagnostics: c_int,
    return_residuals: c_int,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    cv_fractions: *const c_double,
    cv_fractions_len: c_ulong,
    cv_method: *const c_char,
    cv_k: c_int,
    parallel: c_int,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    return_se: c_int,
) -> *mut CppLoess {
    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");

    let wf = match parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let rm = match parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let sm = match parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let bp = match parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let zwf = match parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };

    let mut builder = LoessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.parallel(parallel != 0);

    if !confidence_intervals.is_nan() {
        builder = builder.confidence_intervals(confidence_intervals);
    }
    if !prediction_intervals.is_nan() {
        builder = builder.prediction_intervals(prediction_intervals);
    }
    if return_diagnostics != 0 {
        builder = builder.return_diagnostics();
    }
    if return_residuals != 0 {
        builder = builder.return_residuals();
    }
    if return_robustness_weights != 0 {
        builder = builder.return_robustness_weights();
    }
    if !auto_converge.is_nan() {
        builder = builder.auto_converge(auto_converge);
    }

    // Store CV options
    let cv_fractions_vec = if !cv_fractions.is_null() && cv_fractions_len > 0 {
        let fractions = std::slice::from_raw_parts(cv_fractions, cv_fractions_len as usize);
        Some(fractions.to_vec())
    } else {
        None
    };

    let cv_method_str = parse_c_str(cv_method, "kfold").to_string();

    // Apply LOESS-specific options
    if !degree.is_null() {
        let deg_str = parse_c_str(degree, "linear");
        match parse_polynomial_degree(deg_str) {
            Ok(d) => builder = builder.degree(d),
            Err(_) => return ptr::null_mut(),
        }
    }
    if dimensions > 0 {
        builder = builder.dimensions(dimensions as usize);
    }
    if !distance_metric.is_null() {
        let dm_str = parse_c_str(distance_metric, "normalized");
        match parse_distance_metric(dm_str) {
            Ok(m) => builder = builder.distance_metric(m),
            Err(_) => return ptr::null_mut(),
        }
    }
    if !surface_mode.is_null() {
        let sm_str = parse_c_str(surface_mode, "interpolation");
        match parse_surface_mode(sm_str) {
            Ok(s) => builder = builder.surface_mode(s),
            Err(_) => return ptr::null_mut(),
        }
    }
    if return_se != 0 {
        builder = builder.return_se();
    }

    Box::into_raw(Box::new(CppLoess {
        builder: Some(builder),
        cv_fractions: cv_fractions_vec,
        cv_method: Some(cv_method_str),
        cv_k: cv_k as usize,
        custom_weights: None,
    }))
}

/// Set user-defined case weights for the next fit call.
///
/// Weights multiply the local kernel weight: `w_ij = custom_weights[j] * K(d_ij/h) * rob_j`.
/// Must have the same length as the `y` array passed to `cpp_loess_fit`.
///
/// # Safety
/// ptr must be a valid mutable pointer returned by cpp_loess_new.
/// weights must be a valid array of length n.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_set_custom_weights(
    ptr: *mut CppLoess,
    weights: *const c_double,
    n: c_ulong,
) {
    if ptr.is_null() || weights.is_null() || n == 0 {
        return;
    }
    let loess = unsafe { &mut *ptr };
    let slice = unsafe { std::slice::from_raw_parts(weights, n as usize) };
    loess.custom_weights = Some(slice.to_vec());
}

/// Fit the batch model.
///
/// # Safety
/// `ptr` must be a valid CppLoess pointer. `x` and `y` must be valid arrays of length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_fit(
    ptr: *mut CppLoess,
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
) -> CppLoessResult {
    if ptr.is_null() {
        return error_result("Model pointer is null");
    }
    if x.is_null() || y.is_null() || n == 0 {
        return error_result("Invalid data inputs");
    }

    let loess = &mut *ptr;
    let x_slice = std::slice::from_raw_parts(x, n as usize);
    let y_slice = std::slice::from_raw_parts(y, n as usize);

    if let Some(mut builder) = loess.builder.clone() {
        // Apply CV options if present
        if let Some(fractions) = &loess.cv_fractions
            && let Some(method) = &loess.cv_method
        {
            match method.to_lowercase().as_str() {
                "simple" | "loo" | "loocv" | "leave_one_out" => {
                    builder = builder.cross_validate(LOOCV(fractions));
                }
                "kfold" | "k_fold" | "k-fold" => {
                    builder = builder.cross_validate(KFold(loess.cv_k, fractions));
                }
                _ => return error_result("Unknown CV method"),
            }
        }
        // Apply custom weights if provided
        if let Some(ref uw) = loess.custom_weights {
            builder = builder.custom_weights(uw.clone());
        }

        match builder.adapter(Batch).build() {
            Ok(m) => match m.fit(x_slice, y_slice) {
                Ok(r) => r.into(),
                Err(e) => error_result(&e.to_string()),
            },
            Err(e) => error_result(&e.to_string()),
        }
    } else {
        error_result("Model initialization failed")
    }
}

/// Free batch model.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `cpp_loess_new` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_free(ptr: *mut CppLoess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

/// Create a new Streaming Loess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_new(
    fraction: c_double,
    iterations: c_int,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    return_diagnostics: c_int,
    return_residuals: c_int,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    parallel: c_int,
    // Streaming opts
    chunk_size: c_int,
    overlap: c_int,
    merge_strategy: *const c_char,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    return_se: c_int,
) -> *mut CppStreamingLoess {
    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
    let ms_str = parse_c_str(merge_strategy, "weighted");

    let wf = match parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let rm = match parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let sm = match parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let bp = match parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let zwf = match parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let ms = match parse_merge_strategy(ms_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };

    let mut builder = LoessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.parallel(parallel != 0);

    if return_diagnostics != 0 {
        builder = builder.return_diagnostics();
    }
    if return_residuals != 0 {
        builder = builder.return_residuals();
    }
    if return_robustness_weights != 0 {
        builder = builder.return_robustness_weights();
    }
    if !auto_converge.is_nan() {
        builder = builder.auto_converge(auto_converge);
    }

    let chunk_size = chunk_size as usize;
    let overlap_size = if overlap < 0 {
        let default = chunk_size / 10;
        default.min(chunk_size.saturating_sub(10)).max(1)
    } else {
        overlap as usize
    };

    // Apply LOESS-specific options
    if !degree.is_null() {
        let deg_str = parse_c_str(degree, "linear");
        match parse_polynomial_degree(deg_str) {
            Ok(d) => builder = builder.degree(d),
            Err(_) => return ptr::null_mut(),
        }
    }
    if dimensions > 0 {
        builder = builder.dimensions(dimensions as usize);
    }
    if !distance_metric.is_null() {
        let dm_str = parse_c_str(distance_metric, "normalized");
        match parse_distance_metric(dm_str) {
            Ok(m) => builder = builder.distance_metric(m),
            Err(_) => return ptr::null_mut(),
        }
    }
    if !surface_mode.is_null() {
        let sm_str = parse_c_str(surface_mode, "interpolation");
        match parse_surface_mode(sm_str) {
            Ok(s) => builder = builder.surface_mode(s),
            Err(_) => return ptr::null_mut(),
        }
    }
    if return_se != 0 {
        builder = builder.return_se();
    }

    Box::into_raw(Box::new(CppStreamingLoess {
        builder,
        streaming_opts: Some((chunk_size, overlap_size, ms)),
        model: None,
    }))
}

#[unsafe(no_mangle)]
/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be valid. `x` and `y` must be valid arrays of length `n`.
pub unsafe extern "C" fn cpp_streaming_process(
    ptr: *mut CppStreamingLoess,
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
) -> CppLoessResult {
    if ptr.is_null() {
        return error_result("Model pointer is null");
    }
    let loess = &mut *ptr;
    if x.is_null() || y.is_null() || n == 0 {
        return error_result("Invalid data inputs");
    }
    let x_slice = std::slice::from_raw_parts(x, n as usize);
    let y_slice = std::slice::from_raw_parts(y, n as usize);

    if loess.model.is_none()
        && let Some((cs, ov, ms)) = loess.streaming_opts
    {
        match loess
            .builder
            .clone()
            .adapter(Streaming)
            .chunk_size(cs)
            .overlap(ov)
            .merge_strategy(ms)
            .build()
        {
            Ok(m) => loess.model = Some(m),
            Err(e) => return error_result(&e.to_string()),
        }
    }

    if let Some(model) = &mut loess.model {
        match model.process_chunk(x_slice, y_slice) {
            Ok(r) => r.into(),
            Err(e) => error_result(&e.to_string()),
        }
    } else {
        error_result("Streaming model initialization failed")
    }
}

#[unsafe(no_mangle)]
/// Finalize the streaming process.
///
/// # Safety
/// `ptr` must be valid.
pub unsafe extern "C" fn cpp_streaming_finalize(ptr: *mut CppStreamingLoess) -> CppLoessResult {
    if ptr.is_null() {
        return error_result("Model pointer is null");
    }
    let loess = &mut *ptr;
    if let Some(model) = &mut loess.model {
        match model.finalize() {
            Ok(r) => r.into(),
            Err(e) => error_result(&e.to_string()),
        }
    } else {
        error_result("Streaming model not initialized")
    }
}

#[unsafe(no_mangle)]
/// Free streaming model.
///
/// # Safety
/// `ptr` must be valid or null.
pub unsafe extern "C" fn cpp_streaming_free(ptr: *mut CppStreamingLoess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

/// Create a new Online Loess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_new(
    fraction: c_double,
    iterations: c_int,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    parallel: c_int,
    // Online opts
    window_capacity: c_int,
    min_points: c_int,
    update_mode: *const c_char,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    return_se: c_int,
) -> *mut CppOnlineLoess {
    let wf_str = parse_c_str(weight_function, "tricube");
    let rm_str = parse_c_str(robustness_method, "bisquare");
    let sm_str = parse_c_str(scaling_method, "mad");
    let bp_str = parse_c_str(boundary_policy, "extend");
    let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
    let um_str = parse_c_str(update_mode, "full");

    let wf = match parse_weight_function(wf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let rm = match parse_robustness_method(rm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let sm = match parse_scaling_method(sm_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let bp = match parse_boundary_policy(bp_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let zwf = match parse_zero_weight_fallback(zwf_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };
    let um = match parse_update_mode(um_str) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };

    let mut builder = LoessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations as usize);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.parallel(parallel != 0);

    if return_robustness_weights != 0 {
        builder = builder.return_robustness_weights();
    }
    if !auto_converge.is_nan() {
        builder = builder.auto_converge(auto_converge);
    }

    // Apply LOESS-specific options
    let configured_dimensions = if dimensions > 0 {
        dimensions as usize
    } else {
        1
    };
    if dimensions > 0 {
        builder = builder.dimensions(configured_dimensions);
    }
    if !degree.is_null() {
        let deg_str = parse_c_str(degree, "linear");
        match parse_polynomial_degree(deg_str) {
            Ok(d) => builder = builder.degree(d),
            Err(_) => return ptr::null_mut(),
        }
    }
    if !distance_metric.is_null() {
        let dm_str = parse_c_str(distance_metric, "normalized");
        match parse_distance_metric(dm_str) {
            Ok(m) => builder = builder.distance_metric(m),
            Err(_) => return ptr::null_mut(),
        }
    }
    if !surface_mode.is_null() {
        let sm_str = parse_c_str(surface_mode, "interpolation");
        match parse_surface_mode(sm_str) {
            Ok(s) => builder = builder.surface_mode(s),
            Err(_) => return ptr::null_mut(),
        }
    }
    if return_se != 0 {
        builder = builder.return_se();
    }

    Box::into_raw(Box::new(CppOnlineLoess {
        builder,
        online_opts: Some((window_capacity as usize, min_points as usize, um)),
        model: None,
        dimensions: configured_dimensions,
    }))
}

#[unsafe(no_mangle)]
/// Add points to online model.
///
/// # Safety
/// `ptr` must be valid. `x` and `y` must be valid arrays of length `n`.
pub unsafe extern "C" fn cpp_online_add_points(
    ptr: *mut CppOnlineLoess,
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
) -> CppLoessResult {
    if ptr.is_null() {
        return error_result("Model pointer is null");
    }
    let loess = &mut *ptr;
    if x.is_null() || y.is_null() || n == 0 {
        return error_result("Invalid data inputs");
    }
    let x_slice = std::slice::from_raw_parts(x, n as usize);
    let y_slice = std::slice::from_raw_parts(y, n as usize);

    if loess.model.is_none()
        && let Some((wc, mp, um)) = loess.online_opts
    {
        match loess
            .builder
            .clone()
            .adapter(Online)
            .window_capacity(wc)
            .min_points(mp)
            .update_mode(um)
            .build()
        {
            Ok(m) => loess.model = Some(m),
            Err(e) => return error_result(&e.to_string()),
        }
    }

    if let Some(model) = &mut loess.model {
        // The new LOESS online API processes one point at a time via add_point.
        let mut smoothed = Vec::with_capacity(y_slice.len());
        for (&xi, &yi) in x_slice.iter().zip(y_slice.iter()) {
            match model.add_point(std::slice::from_ref(&xi), yi) {
                Ok(output) => smoothed.push(output.as_ref().map_or(yi, |o| o.smoothed)),
                Err(e) => return error_result(&e.to_string()),
            }
        }

        let result = LoessResult {
            x: x_slice.to_vec(),
            y: smoothed,
            dimensions: loess.dimensions,
            distance_metric: DistanceMetric::Normalized,
            polynomial_degree: PolynomialDegree::Linear,
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals: None,
            robustness_weights: None,
            diagnostics: None,
            iterations_used: None,
            fraction_used: 0.0,
            cv_scores: None,
            enp: None,
            trace_hat: None,
            delta1: None,
            delta2: None,
            residual_scale: None,
            leverage: None,
        };
        result.into()
    } else {
        error_result("Online model initialization failed")
    }
}

#[unsafe(no_mangle)]
/// Free online model.
///
/// # Safety
/// `ptr` must be valid or null.
pub unsafe extern "C" fn cpp_online_free(ptr: *mut CppOnlineLoess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

/// Free a CppLoessResult.
///
/// # Safety
/// `result` must be a valid pointer to a CppLoessResult struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_free_result(result: *mut CppLoessResult) {
    if result.is_null() {
        return;
    }

    let r = &mut *result;
    let n = r.n as usize;

    // Free arrays
    if !r.x.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.x, n));
    }
    if !r.y.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.y, n));
    }
    if !r.standard_errors.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.standard_errors, n));
    }
    if !r.confidence_lower.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.confidence_lower, n));
    }
    if !r.confidence_upper.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.confidence_upper, n));
    }
    if !r.prediction_lower.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.prediction_lower, n));
    }
    if !r.prediction_upper.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.prediction_upper, n));
    }
    if !r.residuals.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.residuals, n));
    }
    if !r.robustness_weights.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.robustness_weights, n));
    }
    if !r.leverage.is_null() {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.leverage, n));
    }

    // Free error string
    if !r.error.is_null() {
        let _ = std::ffi::CString::from_raw(r.error);
    }
}
