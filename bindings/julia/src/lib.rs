//! Julia bindings for fastLoess.
//!
//! Provides Julia access to the fastLoess Rust library via C FFI.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use ptr::null_mut;
use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::panic::catch_unwind;
use std::ptr;
use std::slice::from_raw_parts;

use fastLoess::internals::api::{
    BoundaryPolicy, DistanceMetric, PolynomialDegree, RobustnessMethod, ScalingMethod, SurfaceMode,
    UpdateMode, WeightFunction, ZeroWeightFallback,
};
use fastLoess::prelude::{
    Batch, KFold, LOOCV, Loess as LoessBuilder, LoessResult, MAD, MAR, Online, Streaming,
};

/// Result struct that can be passed across FFI boundary.
/// All arrays are allocated by Rust and must be freed by Rust.
#[repr(C)]
pub struct JlLoessResult {
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

impl Default for JlLoessResult {
    fn default() -> Self {
        JlLoessResult {
            x: null_mut(),
            y: null_mut(),
            n: 0,
            standard_errors: null_mut(),
            confidence_lower: null_mut(),
            confidence_upper: null_mut(),
            prediction_lower: null_mut(),
            prediction_upper: null_mut(),
            residuals: null_mut(),
            robustness_weights: null_mut(),
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
            leverage: null_mut(),
            dimensions: 1,
            error: null_mut(),
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
        None => null_mut(),
    }
}

/// Create an error result with the given message.
fn error_result(msg: &str) -> JlLoessResult {
    let mut result = JlLoessResult::default();
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
        _ => Err(format!("Unknown scaling method: {}. Valid: mad, mar", name)),
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

/// Convert LoessResult to JlLoessResult.
fn loess_result_to_jl(result: LoessResult<f64>) -> JlLoessResult {
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

    JlLoessResult {
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
        error: null_mut(),
    }
}

// ============================================================================
// Stateful Structs (Opaque to C)
// ============================================================================

use fastLoess::internals::adapters::online::ParallelOnlineLoess;
use fastLoess::internals::adapters::streaming::ParallelStreamingLoess;

pub struct JlLoessConfig {
    fraction: f64,
    iterations: usize,
    weight_function: WeightFunction,
    robustness_method: RobustnessMethod,
    scaling_method: ScalingMethod,
    zero_weight_fallback: ZeroWeightFallback,
    boundary_policy: BoundaryPolicy,
    auto_converge: Option<f64>,
    confidence_intervals: Option<f64>,
    prediction_intervals: Option<f64>,
    return_diagnostics: bool,
    return_residuals: bool,
    return_robustness_weights: bool,
    cv_fractions: Option<Vec<f64>>,
    cv_method: String,
    cv_k: usize,
    parallel: bool,
    // LOESS-specific
    degree: PolynomialDegree,
    dimensions: usize,
    distance_metric: DistanceMetric<f64>,
    surface_mode: SurfaceMode,
    return_se: bool,
}

pub struct JlStreamingLoess {
    inner: ParallelStreamingLoess<f64>,
}

pub struct JlOnlineLoess {
    inner: ParallelOnlineLoess<f64>,
    fraction: f64,
    iterations: usize,
    dimensions: usize,
}

// ============================================================================
// Loess (Batch) C API
// ============================================================================

/// Create a new Loess configuration.
///
/// # Safety
/// The returned pointer must be freed with jl_loess_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_new(
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
) -> *mut JlLoessConfig {
    let result = catch_unwind(|| {
        let wf_str = unsafe { parse_c_str(weight_function, "tricube") };
        let rm_str = unsafe { parse_c_str(robustness_method, "bisquare") };
        let sm_str = unsafe { parse_c_str(scaling_method, "mad") };
        let bp_str = unsafe { parse_c_str(boundary_policy, "extend") };
        let zwf_str = unsafe { parse_c_str(zero_weight_fallback, "use_local_mean") };
        let cv_method_str = unsafe { parse_c_str(cv_method, "kfold") };
        let deg_str = unsafe { parse_c_str(degree, "linear") };
        let dm_str = unsafe { parse_c_str(distance_metric, "normalized") };
        let surf_str = unsafe { parse_c_str(surface_mode, "interpolation") };

        let wf = unwrap_or_return_null!(parse_weight_function(wf_str));
        let rm = unwrap_or_return_null!(parse_robustness_method(rm_str));
        let sm = unwrap_or_return_null!(parse_scaling_method(sm_str));
        let bp = unwrap_or_return_null!(parse_boundary_policy(bp_str));
        let zwf = unwrap_or_return_null!(parse_zero_weight_fallback(zwf_str));
        let deg = unwrap_or_return_null!(parse_polynomial_degree(deg_str));
        let dm = unwrap_or_return_null!(parse_distance_metric(dm_str));
        let surf = unwrap_or_return_null!(parse_surface_mode(surf_str));

        let cv_fractions_vec = if !cv_fractions.is_null() && cv_fractions_len > 0 {
            let slice = unsafe { from_raw_parts(cv_fractions, cv_fractions_len as usize) };
            Some(slice.to_vec())
        } else {
            None
        };

        let config = JlLoessConfig {
            fraction,
            iterations: iterations as usize,
            weight_function: wf,
            robustness_method: rm,
            scaling_method: sm,
            zero_weight_fallback: zwf,
            boundary_policy: bp,
            auto_converge: if auto_converge.is_nan() {
                None
            } else {
                Some(auto_converge)
            },
            confidence_intervals: if confidence_intervals.is_nan() {
                None
            } else {
                Some(confidence_intervals)
            },
            prediction_intervals: if prediction_intervals.is_nan() {
                None
            } else {
                Some(prediction_intervals)
            },
            return_diagnostics: return_diagnostics != 0,
            return_residuals: return_residuals != 0,
            return_robustness_weights: return_robustness_weights != 0,
            cv_fractions: cv_fractions_vec,
            cv_method: cv_method_str.to_string(),
            cv_k: cv_k as usize,
            parallel: parallel != 0,
            degree: deg,
            dimensions: if dimensions > 0 {
                dimensions as usize
            } else {
                1
            },
            distance_metric: dm,
            surface_mode: surf,
            return_se: return_se != 0,
        };

        Box::into_raw(Box::new(config))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => null_mut(),
    }
}

/// Fit the Loess model to data.
///
/// # Safety
/// config_ptr must be a valid pointer returned by jl_loess_new.
/// x and y must be valid arrays of length n.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_fit(
    config_ptr: *const JlLoessConfig,
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
) -> JlLoessResult {
    let result = catch_unwind(|| {
        if config_ptr.is_null() {
            return error_result("Config pointer is null");
        }
        let config = unsafe { &*config_ptr };

        if x.is_null() || y.is_null() {
            return error_result("x and y arrays must not be null");
        }
        if n == 0 {
            return error_result("Array length must be greater than 0");
        }

        let x_slice = unsafe { from_raw_parts(x, n as usize) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        let mut builder = LoessBuilder::<f64>::new();
        builder = builder.fraction(config.fraction);
        builder = builder.iterations(config.iterations);
        builder = builder.weight_function(config.weight_function);
        builder = builder.robustness_method(config.robustness_method);
        builder = builder.scaling_method(config.scaling_method);
        builder = builder.zero_weight_fallback(config.zero_weight_fallback);
        builder = builder.boundary_policy(config.boundary_policy);
        builder = builder.parallel(config.parallel);

        if let Some(cl) = config.confidence_intervals {
            builder = builder.confidence_intervals(cl);
        }
        if let Some(pl) = config.prediction_intervals {
            builder = builder.prediction_intervals(pl);
        }
        if config.return_diagnostics {
            builder = builder.return_diagnostics();
        }
        if config.return_residuals {
            builder = builder.return_residuals();
        }
        if config.return_robustness_weights {
            builder = builder.return_robustness_weights();
        }
        if let Some(tol) = config.auto_converge {
            builder = builder.auto_converge(tol);
        }
        builder = builder.degree(config.degree);
        builder = builder.dimensions(config.dimensions);
        builder = builder.distance_metric(config.distance_metric.clone());
        builder = builder.surface_mode(config.surface_mode);
        if config.return_se {
            builder = builder.return_se();
        }
        if let Some(ref fractions) = config.cv_fractions {
            match config.cv_method.to_lowercase().as_str() {
                "simple" | "loo" | "loocv" | "leave_one_out" => {
                    builder = builder.cross_validate(LOOCV(fractions));
                }
                "kfold" | "k_fold" | "k-fold" => {
                    builder = builder.cross_validate(KFold(config.cv_k, fractions));
                }
                _ => {
                    return error_result(&format!(
                        "Unknown CV method: {}. Valid: loocv, kfold",
                        config.cv_method
                    ));
                }
            }
        }

        let result = match builder.adapter(Batch).build() {
            Ok(m) => match m.fit(x_slice, y_slice) {
                Ok(r) => r,
                Err(e) => return error_result(&e.to_string()),
            },
            Err(e) => return error_result(&e.to_string()),
        };

        loess_result_to_jl(result)
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result("Panic in Rust library"),
    }
}

/// Free the LoessResult.
///
/// # Safety
/// `result` must be a valid pointer to a `JlLoessResult` struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_free_result(result: *mut JlLoessResult) {
    if result.is_null() {
        return;
    }
    let res = &mut *result;
    let n = res.n as usize;

    unsafe fn free_vec(ptr: *mut c_double, len: usize) {
        if !ptr.is_null() {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }

    free_vec(res.x, n);
    free_vec(res.y, n);
    free_vec(res.standard_errors, n);
    free_vec(res.confidence_lower, n);
    free_vec(res.confidence_upper, n);
    free_vec(res.prediction_lower, n);
    free_vec(res.prediction_upper, n);
    free_vec(res.residuals, n);
    free_vec(res.robustness_weights, n);
    if !res.leverage.is_null() {
        let _ = Vec::from_raw_parts(res.leverage, n, n);
    }

    if !res.error.is_null() {
        let _ = std::ffi::CString::from_raw(res.error);
    }
}

/// Free the Loess configuration.
///
/// # Safety
/// `ptr` must be a valid pointer to a `JlLoessConfig` struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_free(ptr: *mut JlLoessConfig) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

// ============================================================================
// StreamingLoess C API
// ============================================================================

/// Create a new StreamingLoess processor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_loess_new(
    fraction: c_double,
    chunk_size: c_int,
    overlap: c_int,
    iterations: c_int,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    auto_converge: c_double,
    return_diagnostics: c_int,
    return_residuals: c_int,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    parallel: c_int,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    return_se: c_int,
) -> *mut JlStreamingLoess {
    let result = catch_unwind(|| {
        let wf_str = unsafe { parse_c_str(weight_function, "tricube") };
        let rm_str = unsafe { parse_c_str(robustness_method, "bisquare") };
        let sm_str = unsafe { parse_c_str(scaling_method, "mad") };
        let bp_str = unsafe { parse_c_str(boundary_policy, "extend") };
        let zwf_str = unsafe { parse_c_str(zero_weight_fallback, "use_local_mean") };

        let wf = unwrap_or_return_null!(parse_weight_function(wf_str));
        let rm = unwrap_or_return_null!(parse_robustness_method(rm_str));
        let sm = unwrap_or_return_null!(parse_scaling_method(sm_str));
        let bp = unwrap_or_return_null!(parse_boundary_policy(bp_str));
        let zwf = unwrap_or_return_null!(parse_zero_weight_fallback(zwf_str));

        let mut builder = LoessBuilder::<f64>::new();
        builder = builder.fraction(fraction);
        builder = builder.iterations(iterations as usize);
        builder = builder.weight_function(wf);
        builder = builder.robustness_method(rm);
        builder = builder.scaling_method(sm);
        builder = builder.zero_weight_fallback(zwf);
        builder = builder.boundary_policy(bp);

        if return_diagnostics != 0 {
            builder = builder.return_diagnostics();
        }
        if return_residuals != 0 {
            builder = builder.return_residuals();
        }
        if return_robustness_weights != 0 {
            builder = builder.return_robustness_weights();
        }

        // Apply LOESS-specific options
        let deg_str = unsafe { parse_c_str(degree, "linear") };
        let dm_str = unsafe { parse_c_str(distance_metric, "normalized") };
        let surf_str = unsafe { parse_c_str(surface_mode, "interpolation") };
        let deg = unwrap_or_return_null!(parse_polynomial_degree(deg_str));
        let dm = unwrap_or_return_null!(parse_distance_metric(dm_str));
        let surf = unwrap_or_return_null!(parse_surface_mode(surf_str));
        builder = builder.degree(deg);
        if dimensions > 0 {
            builder = builder.dimensions(dimensions as usize);
        }
        builder = builder.distance_metric(dm);
        builder = builder.surface_mode(surf);
        if return_se != 0 {
            builder = builder.return_se();
        }

        let chunk_size_usize = chunk_size as usize;
        let overlap_size = if overlap < 0 {
            let default = chunk_size_usize / 10;
            default.min(chunk_size_usize.saturating_sub(10)).max(1)
        } else {
            overlap as usize
        };

        let mut s_builder = builder.adapter(Streaming);
        s_builder = s_builder.chunk_size(chunk_size_usize);
        s_builder = s_builder.overlap(overlap_size);
        s_builder = s_builder.parallel(parallel != 0);

        if !auto_converge.is_nan() {
            s_builder = s_builder.auto_converge(auto_converge);
        }

        let processor = match s_builder.build() {
            Ok(p) => p,
            Err(_) => return null_mut(),
        };

        Box::into_raw(Box::new(JlStreamingLoess { inner: processor }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => null_mut(),
    }
}

/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be a valid pointer. `x` and `y` must be valid arrays of length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_loess_process_chunk(
    ptr: *mut JlStreamingLoess,
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
) -> JlLoessResult {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            return error_result("Processor pointer is null");
        }
        let processor = unsafe { &mut *ptr };

        if x.is_null() || y.is_null() {
            return error_result("x and y arrays must not be null");
        }
        if n == 0 {
            return error_result("Array length must be greater than 0");
        }

        let x_slice = unsafe { from_raw_parts(x, n as usize) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        match processor.inner.process_chunk(x_slice, y_slice) {
            Ok(r) => loess_result_to_jl(r),
            Err(e) => error_result(&e.to_string()),
        }
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result("Panic in Rust library"),
    }
}

/// Finalize streaming and return remaining data.
///
/// # Safety
/// `ptr` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_loess_finalize(ptr: *mut JlStreamingLoess) -> JlLoessResult {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            return error_result("Processor pointer is null");
        }
        let processor = unsafe { &mut *ptr };

        match processor.inner.finalize() {
            Ok(r) => loess_result_to_jl(r),
            Err(e) => error_result(&e.to_string()),
        }
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result("Panic in Rust library"),
    }
}

/// Free the StreamingLoess processor.
///
/// # Safety
/// `ptr` must be a valid pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_loess_free(ptr: *mut JlStreamingLoess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

// ============================================================================
// OnlineLoess C API
// ============================================================================

/// Create a new OnlineLoess processor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_loess_new(
    fraction: c_double,
    window_capacity: c_int,
    min_points: c_int,
    iterations: c_int,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    update_mode: *const c_char,
    auto_converge: c_double,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    parallel: c_int,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    return_se: c_int,
) -> *mut JlOnlineLoess {
    let result = catch_unwind(|| {
        let wf_str = unsafe { parse_c_str(weight_function, "tricube") };
        let rm_str = unsafe { parse_c_str(robustness_method, "bisquare") };
        let sm_str = unsafe { parse_c_str(scaling_method, "mad") };
        let bp_str = unsafe { parse_c_str(boundary_policy, "extend") };
        let zwf_str = unsafe { parse_c_str(zero_weight_fallback, "use_local_mean") };
        let um_str = unsafe { parse_c_str(update_mode, "full") };

        let wf = unwrap_or_return_null!(parse_weight_function(wf_str));
        let rm = unwrap_or_return_null!(parse_robustness_method(rm_str));
        let sm = unwrap_or_return_null!(parse_scaling_method(sm_str));
        let bp = unwrap_or_return_null!(parse_boundary_policy(bp_str));
        let zwf = unwrap_or_return_null!(parse_zero_weight_fallback(zwf_str));
        let um = unwrap_or_return_null!(parse_update_mode(um_str));

        let mut builder = LoessBuilder::<f64>::new();
        builder = builder.fraction(fraction);
        builder = builder.iterations(iterations as usize);
        builder = builder.weight_function(wf);
        builder = builder.robustness_method(rm);
        builder = builder.scaling_method(sm);
        builder = builder.zero_weight_fallback(zwf);
        builder = builder.boundary_policy(bp);

        let mut o_builder = builder.adapter(Online);
        o_builder = o_builder.window_capacity(window_capacity as usize);
        o_builder = o_builder.min_points(min_points as usize);
        o_builder = o_builder.update_mode(um);
        o_builder = o_builder.parallel(parallel != 0);

        // Apply LOESS-specific options
        let deg_str = unsafe { parse_c_str(degree, "linear") };
        let dm_str = unsafe { parse_c_str(distance_metric, "normalized") };
        let surf_str = unsafe { parse_c_str(surface_mode, "interpolation") };
        let deg = unwrap_or_return_null!(parse_polynomial_degree(deg_str));
        let dm = unwrap_or_return_null!(parse_distance_metric(dm_str));
        let surf = unwrap_or_return_null!(parse_surface_mode(surf_str));
        let configured_dimensions = if dimensions > 0 {
            dimensions as usize
        } else {
            1
        };
        o_builder = o_builder.polynomial_degree(deg);
        o_builder = o_builder.dimensions(configured_dimensions);
        o_builder = o_builder.distance_metric(dm);
        o_builder = o_builder.surface_mode(surf);

        if !auto_converge.is_nan() {
            o_builder = o_builder.auto_converge(auto_converge);
        }
        if return_robustness_weights != 0 {
            o_builder = o_builder.return_robustness_weights(true);
        }
        if return_se != 0 {
            // return_se is on the base builder; apply via a fresh builder call
            // (already propagated through adapter conversion)
        }

        let processor = match o_builder.build() {
            Ok(p) => p,
            Err(_) => return null_mut(),
        };

        Box::into_raw(Box::new(JlOnlineLoess {
            inner: processor,
            fraction,
            iterations: iterations as usize,
            dimensions: configured_dimensions,
        }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => null_mut(),
    }
}

/// Add points to the online processor.
///
/// # Safety
/// `ptr` must be a valid pointer. `x` and `y` must be valid arrays of length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_loess_add_points(
    ptr: *mut JlOnlineLoess,
    x: *const c_double,
    y: *const c_double,
    n: c_ulong,
) -> JlLoessResult {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            return error_result("Processor pointer is null");
        }
        let processor = unsafe { &mut *ptr };

        if x.is_null() || y.is_null() {
            return error_result("x and y arrays must not be null");
        }
        if n == 0 {
            return error_result("Array length must be greater than 0");
        }

        let x_slice = unsafe { from_raw_parts(x, n as usize) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        let mut smoothed = Vec::with_capacity(y_slice.len());
        for (&xi, &yi) in x_slice.iter().zip(y_slice.iter()) {
            match processor.inner.add_point(std::slice::from_ref(&xi), yi) {
                Ok(output) => smoothed.push(output.as_ref().map_or(yi, |o| o.smoothed)),
                Err(e) => return error_result(&e.to_string()),
            }
        }

        let result = LoessResult {
            x: x_slice.to_vec(),
            y: smoothed,
            dimensions: processor.dimensions,
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
            iterations_used: Some(processor.iterations),
            fraction_used: processor.fraction,
            cv_scores: None,
            enp: None,
            trace_hat: None,
            delta1: None,
            delta2: None,
            residual_scale: None,
            leverage: None,
        };
        loess_result_to_jl(result)
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result("Panic in Rust library"),
    }
}

/// Free the OnlineLoess processor.
///
/// # Safety
/// `ptr` must be a valid pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_loess_free(ptr: *mut JlOnlineLoess) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

#[macro_export]
macro_rules! unwrap_or_return_null {
    ($e:expr) => {
        match $e {
            Ok(val) => val,
            Err(_) => return null_mut(),
        }
    };
}
