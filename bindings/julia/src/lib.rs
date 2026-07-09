//! Julia bindings for fastLoess.
//!
//! Provides Julia access to the fastLoess Rust library via C FFI.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use ptr::null_mut;
use std::cell::RefCell;
use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::panic::catch_unwind;
use std::ptr;
use std::slice::from_raw_parts;

use fastLoess::internals::api::LoessBuilder;
use fastLoess::internals::binding_support as shared_parse;
use fastLoess::internals::binding_support::DistanceMetric;
use fastLoess::prelude::LoessResult;

thread_local! {
    static JL_LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error_message(msg: &str) {
    let cmsg = shared_parse::to_cstring_lossy(msg);
    JL_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(cmsg);
    });
}

fn clear_last_error_message() {
    JL_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn null_with_last_error<T>(msg: &str) -> *mut T {
    set_last_error_message(msg);
    null_mut()
}

fn setter_unsupported_constructor_only(name: &str) {
    set_last_error_message(&shared_parse::setter_unsupported_constructor_only_message(
        name,
    ));
}

#[unsafe(no_mangle)]
pub extern "C" fn jl_last_error_message() -> *const c_char {
    JL_LAST_ERROR.with(|slot| {
        if let Some(msg) = slot.borrow().as_ref() {
            msg.as_ptr()
        } else {
            ptr::null()
        }
    })
}

// Per-point result from an online update, passed across the FFI boundary.
// has_value = 1 means the window is ready and smoothed is valid; 0 means
// the window is still filling (all other fields are undefined in that case).
#[repr(C)]
pub struct JlOnlineOutput {
    pub has_value: c_int,
    pub smoothed: c_double,
    pub std_error: c_double,         // f64::NAN when not computed
    pub residual: c_double,          // f64::NAN when not computed
    pub robustness_weight: c_double, // f64::NAN when not computed
    pub iterations_used: c_int,      // -1 when not computed
}

impl Default for JlOnlineOutput {
    fn default() -> Self {
        JlOnlineOutput {
            has_value: 0,
            smoothed: f64::NAN,
            std_error: f64::NAN,
            residual: f64::NAN,
            robustness_weight: f64::NAN,
            iterations_used: -1,
        }
    }
}

// Result struct that can be passed across FFI boundary.
// All arrays are allocated by Rust and must be freed by Rust.
#[repr(C)]
pub struct JlLoessResult {
    // Sorted x values (length = n)
    pub x: *mut c_double,
    // Smoothed y values (length = n)
    pub y: *mut c_double,
    // Number of data points
    pub n: c_ulong,

    // Standard errors (NULL if not computed)
    pub standard_errors: *mut c_double,
    // Lower confidence bounds (NULL if not computed)
    pub confidence_lower: *mut c_double,
    // Upper confidence bounds (NULL if not computed)
    pub confidence_upper: *mut c_double,
    // Lower prediction bounds (NULL if not computed)
    pub prediction_lower: *mut c_double,
    // Upper prediction bounds (NULL if not computed)
    pub prediction_upper: *mut c_double,
    // Residuals (NULL if not computed)
    pub residuals: *mut c_double,
    // Robustness weights (NULL if not computed)
    pub robustness_weights: *mut c_double,

    // Fraction used for smoothing
    pub fraction_used: c_double,
    // Number of iterations performed (-1 if not available)
    pub iterations_used: c_int,

    // Diagnostics (NaN if not computed)
    pub rmse: c_double,
    pub mae: c_double,
    pub r_squared: c_double,
    pub aic: c_double,
    pub aicc: c_double,
    pub effective_df: c_double,
    pub residual_sd: c_double,

    // Hat-matrix statistics (NaN / NULL if not computed; set return_se = 1 to enable)
    pub enp: c_double,
    pub trace_hat: c_double,
    pub delta1: c_double,
    pub delta2: c_double,
    pub residual_scale: c_double,
    // Per-point leverage / hat-matrix diagonal (NULL if not computed, length = n)
    pub leverage: *mut c_double,
    // Number of predictor dimensions used
    pub dimensions: c_int,
    // Cross-validation scores (NULL if not computed, length = cv_scores_len)
    pub cv_scores: *mut c_double,
    pub cv_scores_len: c_ulong,

    // Error message (NULL if no error)
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
            cv_scores: null_mut(),
            cv_scores_len: 0,
            error: null_mut(),
        }
    }
}

// Create an error result with the given message.
fn error_result(msg: &str) -> JlLoessResult {
    let mut result = JlLoessResult::default();
    let c_string = shared_parse::to_cstring_lossy(msg);
    result.error = c_string.into_raw();
    result
}

fn error_result_from(err: shared_parse::BindingError) -> JlLoessResult {
    error_result(&err.message)
}

fn map_runtime_result<T, E: ToString>(
    result: std::result::Result<T, E>,
) -> std::result::Result<T, Box<JlLoessResult>> {
    shared_parse::map_runtime(result).map_err(|e| Box::new(error_result_from(e)))
}

// Parse a C string safely.
unsafe fn parse_c_str(s: *const c_char, default: &str) -> &str {
    if s.is_null() {
        default
    } else {
        CStr::from_ptr(s).to_str().unwrap_or(default)
    }
}

// Convert LoessResult to JlLoessResult.
fn loess_result_to_jl(result: LoessResult<f64>) -> JlLoessResult {
    let n = result.y.len();
    let (rmse, mae, r_squared, aic, aicc, effective_df, residual_sd) =
        shared_parse::extract_diagnostics(&result);

    JlLoessResult {
        x: shared_parse::vec_to_raw_ptr(result.x),
        y: shared_parse::vec_to_raw_ptr(result.y),
        n: n as c_ulong,
        standard_errors: shared_parse::opt_vec_to_raw_ptr(result.standard_errors),
        confidence_lower: shared_parse::opt_vec_to_raw_ptr(result.confidence_lower),
        confidence_upper: shared_parse::opt_vec_to_raw_ptr(result.confidence_upper),
        prediction_lower: shared_parse::opt_vec_to_raw_ptr(result.prediction_lower),
        prediction_upper: shared_parse::opt_vec_to_raw_ptr(result.prediction_upper),
        residuals: shared_parse::opt_vec_to_raw_ptr(result.residuals),
        robustness_weights: shared_parse::opt_vec_to_raw_ptr(result.robustness_weights),
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
        leverage: shared_parse::opt_vec_to_raw_ptr(result.leverage),
        dimensions: result.dimensions as c_int,
        cv_scores: shared_parse::opt_vec_to_raw_ptr(result.cv_scores.clone()),
        cv_scores_len: result
            .cv_scores
            .as_ref()
            .map(|v| v.len() as c_ulong)
            .unwrap_or(0),
        error: null_mut(),
    }
}

// Stateful Structs (Opaque to C)

use fastLoess::internals::adapters::online::ParallelOnlineLoess;
use fastLoess::internals::adapters::streaming::ParallelStreamingLoess;

pub struct JlLoessConfig {
    base_builder: LoessBuilder<f64>,
    // Saved for weighted metric override resolution at fit time
    distance_metric: DistanceMetric<f64>,
    // Mutable via jl_loess_set_weighted_metric
    weighted_metric_weights: Option<Vec<f64>>,
    // Mutable via jl_loess_set_custom_weights; applied to build_batch
    custom_weights: Option<Vec<f64>>,
    // Needed to compute x_n = n * dimensions in jl_loess_fit
    dimensions: usize,
}

pub struct JlStreamingLoess {
    inner: ParallelStreamingLoess<f64>,
    dimensions: usize,
}

pub struct JlOnlineLoess {
    inner: ParallelOnlineLoess<f64>,
}

// Loess () C API

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
    clear_last_error_message();
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

        let wf = unwrap_or_return_null!(shared_parse::parse_weight_function(wf_str));
        let rm = unwrap_or_return_null!(shared_parse::parse_robustness_method(rm_str));
        let sm = unwrap_or_return_null!(shared_parse::parse_scaling_method(sm_str));
        let bp = unwrap_or_return_null!(shared_parse::parse_boundary_policy(bp_str));
        let zwf = unwrap_or_return_null!(shared_parse::parse_zero_weight_fallback(zwf_str));
        let deg = unwrap_or_return_null!(shared_parse::parse_polynomial_degree(deg_str));
        let dm = unwrap_or_return_null!(shared_parse::parse_distance_metric(dm_str));
        let surf = unwrap_or_return_null!(shared_parse::parse_surface_mode(surf_str));

        let cv_fractions_vec =
            shared_parse::option_vec_from_ptr(cv_fractions, cv_fractions_len as usize);

        let dims = if dimensions > 0 {
            dimensions as usize
        } else {
            1
        };

        let (base_builder, _) = unwrap_or_return_null!(shared_parse::apply_typed_builder_options(
            LoessBuilder::<f64>::new(),
            shared_parse::TypedBuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations as usize),
                weight_function: Some(wf),
                robustness_method: Some(rm),
                zero_weight_fallback: Some(zwf),
                boundary_policy: Some(bp),
                scaling_method: Some(sm),
                auto_converge: if auto_converge.is_nan() {
                    None
                } else {
                    Some(auto_converge)
                },
                return_residuals: return_residuals != 0,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: return_diagnostics != 0,
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
                parallel: Some(parallel != 0),
                degree: Some(deg),
                dimensions: Some(dims),
                distance_metric: Some(dm.clone()),
                surface_mode: Some(surf),
                return_se: return_se != 0,
                cell: None,
                interpolation_vertices: None,
                boundary_degree_fallback: None,
                cv_fractions: cv_fractions_vec,
                cv_method: Some(cv_method_str.to_string()),
                cv_k: Some(cv_k as usize),
                cv_seed: None,
            },
        ));

        Box::into_raw(Box::new(JlLoessConfig {
            base_builder,
            distance_metric: dm,
            weighted_metric_weights: None,
            custom_weights: None,
            dimensions: dims,
        }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error_message(shared_parse::panic_fallback_message());
            null_mut()
        }
    }
}

/// Set user-defined case weights for the next fit call.
///
/// Weights multiply the local kernel weight: `w_ij = custom_weights[j] * K(d_ij/h) * rob_j`.
/// Must have the same length as the `y` array passed to `jl_loess_fit`.
///
/// # Safety
/// config_ptr must be a valid mutable pointer returned by jl_loess_new.
/// weights must be a valid array of length n.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_set_custom_weights(
    config_ptr: *mut JlLoessConfig,
    weights: *const c_double,
    n: c_ulong,
) {
    if config_ptr.is_null() || weights.is_null() || n == 0 {
        return;
    }
    let config = unsafe { &mut *config_ptr };
    let slice = unsafe { from_raw_parts(weights, n as usize) };
    config.custom_weights = Some(slice.to_vec());
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
            return error_result(shared_parse::CONFIG_POINTER_IS_NULL);
        }
        let config = unsafe { &*config_ptr };

        if x.is_null() || y.is_null() {
            return error_result(shared_parse::XY_ARRAYS_MUST_NOT_BE_NULL);
        }
        if n == 0 {
            return error_result(shared_parse::ARRAY_LENGTH_MUST_BE_GREATER_THAN_ZERO);
        }

        // For multi-dimensional LOESS, x has n * dimensions elements (row-major flat).
        let x_n = n as usize * config.dimensions.max(1);
        let x_slice = unsafe { from_raw_parts(x, x_n) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        // Clone the pre-built builder; re-apply weighted metric if the setter was called
        let mut builder = config.base_builder.clone();
        if let Some(ref weights) = config.weighted_metric_weights {
            let dm = shared_parse::resolve_typed_distance_metric(
                config.distance_metric.clone(),
                Some(weights),
            );
            builder = shared_parse::apply_distance_metric_value(builder, &dm);
        }

        let model = match shared_parse::build_batch(builder, config.custom_weights.clone()) {
            Ok(m) => m,
            Err(e) => return error_result(&e.message),
        };

        let result = match map_runtime_result(model.fit(x_slice, y_slice)) {
            Ok(r) => r,
            Err(e) => return *e,
        };

        loess_result_to_jl(result)
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result(shared_parse::panic_fallback_message()),
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
    if !res.cv_scores.is_null() {
        let _ = Vec::from_raw_parts(
            res.cv_scores,
            res.cv_scores_len as usize,
            res.cv_scores_len as usize,
        );
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

/// Legacy setter retained for ABI compatibility.
///
/// Configure this in `jl_loess_new` instead.
///
/// # Safety
/// config_ptr must be a valid mutable pointer returned by jl_loess_new.
/// weights must be a valid array of length n (number of predictor dimensions).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_set_weighted_metric(
    config_ptr: *mut JlLoessConfig,
    weights: *const c_double,
    n: c_ulong,
) {
    if config_ptr.is_null() {
        set_last_error_message(shared_parse::CONFIG_POINTER_IS_NULL);
        return;
    }
    if weights.is_null() || n == 0 {
        set_last_error_message(shared_parse::INVALID_DATA_INPUTS);
        return;
    }
    // Store the weights for use during fit
    let slice = unsafe { from_raw_parts(weights, n as usize) };
    unsafe { (*config_ptr).weighted_metric_weights = Some(slice.to_vec()) };
}

/// Legacy setter retained for ABI compatibility.
///
/// Configure this in `jl_loess_new` instead.
///
/// # Safety
/// config_ptr must be a valid mutable pointer returned by jl_loess_new.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_set_cell(config_ptr: *mut JlLoessConfig, cell: c_double) {
    let _ = cell;
    if config_ptr.is_null() {
        set_last_error_message(shared_parse::CONFIG_POINTER_IS_NULL);
        return;
    }
    setter_unsupported_constructor_only("jl_loess_set_cell");
}

/// Legacy setter retained for ABI compatibility.
///
/// Configure this in `jl_loess_new` instead.
///
/// # Safety
/// config_ptr must be a valid mutable pointer returned by jl_loess_new.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_set_interpolation_vertices(
    config_ptr: *mut JlLoessConfig,
    vertices: c_ulong,
) {
    let _ = vertices;
    if config_ptr.is_null() {
        set_last_error_message(shared_parse::CONFIG_POINTER_IS_NULL);
        return;
    }
    setter_unsupported_constructor_only("jl_loess_set_interpolation_vertices");
}

/// Legacy setter retained for ABI compatibility.
///
/// Configure this in `jl_loess_new` instead.
///
/// # Safety
/// config_ptr must be a valid mutable pointer returned by jl_loess_new.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_loess_set_boundary_degree_fallback(
    config_ptr: *mut JlLoessConfig,
    enabled: c_int,
) {
    let _ = enabled;
    if config_ptr.is_null() {
        set_last_error_message(shared_parse::CONFIG_POINTER_IS_NULL);
        return;
    }
    setter_unsupported_constructor_only("jl_loess_set_boundary_degree_fallback");
}

/// Legacy setter retained for ABI compatibility.
///
/// Configure this in `jl_loess_new` instead.
///
/// # Safety
/// config_ptr must be a valid mutable pointer returned by jl_loess_new.
#[unsafe(no_mangle)]
#[allow(clippy::useless_conversion)] // c_ulong is u32 on Windows, u64 on Linux/macOS
pub unsafe extern "C" fn jl_loess_set_cv_seed(config_ptr: *mut JlLoessConfig, seed: c_ulong) {
    let _ = seed;
    if config_ptr.is_null() {
        set_last_error_message(shared_parse::CONFIG_POINTER_IS_NULL);
        return;
    }
    setter_unsupported_constructor_only("jl_loess_set_cv_seed");
}

// StreamingLoess C API

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
    merge_strategy: *const c_char,
    parallel: c_int,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    return_se: c_int,
    // Advanced options
    confidence_intervals: c_double,
    prediction_intervals: c_double,
    cell: c_double,
    interpolation_vertices: c_int,
    boundary_degree_fallback: c_int,
    weighted_metric_weights: *const c_double,
    weighted_metric_weights_len: c_ulong,
) -> *mut JlStreamingLoess {
    clear_last_error_message();
    let result = catch_unwind(|| {
        let wf_str = unsafe { parse_c_str(weight_function, "tricube") };
        let rm_str = unsafe { parse_c_str(robustness_method, "bisquare") };
        let sm_str = unsafe { parse_c_str(scaling_method, "mad") };
        let bp_str = unsafe { parse_c_str(boundary_policy, "extend") };
        let zwf_str = unsafe { parse_c_str(zero_weight_fallback, "use_local_mean") };
        let ms_str = unsafe { parse_c_str(merge_strategy, "weighted_average") };
        let deg_str = unsafe { parse_c_str(degree, "linear") };
        let surf_str = unsafe { parse_c_str(surface_mode, "interpolation") };
        let dm_str = unsafe { parse_c_str(distance_metric, "normalized") };

        let chunk_size_usize = unwrap_or_return_null!(shared_parse::require_positive_usize(
            "chunk_size",
            chunk_size
        ));
        let iterations_usize = unwrap_or_return_null!(shared_parse::require_non_negative_usize(
            "iterations",
            iterations
        ));

        let weighted_metric = shared_parse::option_slice_from_ptr(
            weighted_metric_weights,
            weighted_metric_weights_len as usize,
        );
        let effective_metric =
            shared_parse::resolve_distance_metric_for_builder(Some(dm_str), weighted_metric);
        let configured_dimensions = (dimensions as usize).max(1);

        let (builder, _) = match shared_parse::map_invalid_arg(shared_parse::apply_builder_options(
            LoessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations_usize),
                weight_function: Some(wf_str),
                robustness_method: Some(rm_str),
                zero_weight_fallback: Some(zwf_str),
                boundary_policy: Some(bp_str),
                scaling_method: Some(sm_str),
                auto_converge: (!auto_converge.is_nan()).then_some(auto_converge),
                return_residuals: return_residuals != 0,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: return_diagnostics != 0,
                confidence_intervals: (!confidence_intervals.is_nan())
                    .then_some(confidence_intervals),
                prediction_intervals: (!prediction_intervals.is_nan())
                    .then_some(prediction_intervals),
                parallel: Some(parallel != 0),
                degree: Some(deg_str),
                dimensions: Some(configured_dimensions),
                distance_metric: effective_metric,
                weighted_metric_weights: weighted_metric,
                surface_mode: Some(surf_str),
                return_se: return_se != 0,
                cell: (!cell.is_nan()).then_some(cell),
                interpolation_vertices: (interpolation_vertices > 0)
                    .then_some(interpolation_vertices as usize),
                boundary_degree_fallback: (boundary_degree_fallback >= 0)
                    .then_some(boundary_degree_fallback != 0),
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        )) {
            Ok(v) => v,
            Err(e) => return null_with_last_error(&e.message),
        };

        let overlap_size = if overlap < 0 {
            shared_parse::default_overlap(chunk_size_usize)
        } else {
            overlap as usize
        };

        let processor = match shared_parse::build_streaming(
            builder,
            Some(chunk_size_usize),
            Some(overlap_size),
            Some(ms_str),
        ) {
            Ok(p) => p,
            Err(e) => return null_with_last_error(&e.message),
        };

        Box::into_raw(Box::new(JlStreamingLoess {
            inner: processor,
            dimensions: configured_dimensions,
        }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error_message(shared_parse::panic_fallback_message());
            null_mut()
        }
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
            return error_result(shared_parse::PROCESSOR_POINTER_IS_NULL);
        }
        let processor = unsafe { &mut *ptr };

        if x.is_null() || y.is_null() {
            return error_result(shared_parse::XY_ARRAYS_MUST_NOT_BE_NULL);
        }
        if n == 0 {
            return error_result(shared_parse::ARRAY_LENGTH_MUST_BE_GREATER_THAN_ZERO);
        }

        // For multi-dimensional chunks, x has n * dimensions elements (row-major flat).
        let x_n = n as usize * processor.dimensions.max(1);
        let x_slice = unsafe { from_raw_parts(x, x_n) };
        let y_slice = unsafe { from_raw_parts(y, n as usize) };

        match map_runtime_result(processor.inner.process_chunk(x_slice, y_slice)) {
            Ok(r) => loess_result_to_jl(r),
            Err(e) => *e,
        }
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result(shared_parse::panic_fallback_message()),
    }
}

/// Finalize and return remaining data.
///
/// # Safety
/// `ptr` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_streaming_loess_finalize(ptr: *mut JlStreamingLoess) -> JlLoessResult {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            return error_result(shared_parse::PROCESSOR_POINTER_IS_NULL);
        }
        let processor = unsafe { &mut *ptr };

        match map_runtime_result(processor.inner.finalize()) {
            Ok(r) => loess_result_to_jl(r),
            Err(e) => *e,
        }
    });

    match result {
        Ok(res) => res,
        Err(_) => error_result(shared_parse::panic_fallback_message()),
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

// OnlineLoess C API

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
    return_diagnostics: c_int,
    return_residuals: c_int,
    zero_weight_fallback: *const c_char,
    parallel: c_int,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    return_se: c_int,
    // Advanced options
    confidence_intervals: c_double,
    prediction_intervals: c_double,
    cell: c_double,
    interpolation_vertices: c_int,
    boundary_degree_fallback: c_int,
    weighted_metric_weights: *const c_double,
    weighted_metric_weights_len: c_ulong,
) -> *mut JlOnlineLoess {
    clear_last_error_message();
    let result = catch_unwind(|| {
        let wf_str = unsafe { parse_c_str(weight_function, "tricube") };
        let rm_str = unsafe { parse_c_str(robustness_method, "bisquare") };
        let sm_str = unsafe { parse_c_str(scaling_method, "mad") };
        let bp_str = unsafe { parse_c_str(boundary_policy, "extend") };
        let zwf_str = unsafe { parse_c_str(zero_weight_fallback, "use_local_mean") };
        let um_str = unsafe { parse_c_str(update_mode, "full") };

        let deg_str = unsafe { parse_c_str(degree, "linear") };
        let surf_str = unsafe { parse_c_str(surface_mode, "interpolation") };
        let dm_str = unsafe { parse_c_str(distance_metric, "normalized") };

        let iterations_usize = unwrap_or_return_null!(shared_parse::require_non_negative_usize(
            "iterations",
            iterations
        ));
        let window_capacity_usize = unwrap_or_return_null!(shared_parse::require_positive_usize(
            "window_capacity",
            window_capacity
        ));
        let min_points_usize = unwrap_or_return_null!(shared_parse::require_positive_usize(
            "min_points",
            min_points
        ));

        let weighted_metric = shared_parse::option_slice_from_ptr(
            weighted_metric_weights,
            weighted_metric_weights_len as usize,
        );
        let effective_metric =
            shared_parse::resolve_distance_metric_for_builder(Some(dm_str), weighted_metric);
        let configured_dimensions = if dimensions > 0 {
            dimensions as usize
        } else {
            1
        };

        let (builder, _) = match shared_parse::map_invalid_arg(shared_parse::apply_builder_options(
            LoessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations_usize),
                weight_function: Some(wf_str),
                robustness_method: Some(rm_str),
                zero_weight_fallback: Some(zwf_str),
                boundary_policy: Some(bp_str),
                scaling_method: Some(sm_str),
                auto_converge: (!auto_converge.is_nan()).then_some(auto_converge),
                return_residuals: return_residuals != 0,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: return_diagnostics != 0,
                confidence_intervals: (!confidence_intervals.is_nan())
                    .then_some(confidence_intervals),
                prediction_intervals: (!prediction_intervals.is_nan())
                    .then_some(prediction_intervals),
                parallel: Some(parallel != 0),
                degree: Some(deg_str),
                dimensions: Some(configured_dimensions),
                distance_metric: effective_metric,
                weighted_metric_weights: weighted_metric,
                surface_mode: Some(surf_str),
                return_se: return_se != 0,
                cell: (!cell.is_nan()).then_some(cell),
                interpolation_vertices: (interpolation_vertices > 0)
                    .then_some(interpolation_vertices as usize),
                boundary_degree_fallback: (boundary_degree_fallback >= 0)
                    .then_some(boundary_degree_fallback != 0),
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        )) {
            Ok(v) => v,
            Err(e) => return null_with_last_error(&e.message),
        };

        let processor = match shared_parse::build_online(
            builder,
            Some(window_capacity_usize),
            Some(min_points_usize),
            Some(um_str),
        ) {
            Ok(p) => p,
            Err(e) => return null_with_last_error(&e.message),
        };

        Box::into_raw(Box::new(JlOnlineLoess { inner: processor }))
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error_message(shared_parse::panic_fallback_message());
            null_mut()
        }
    }
}

/// Add a single point to the processor and return the smoothed value for that
/// point, or a result with `has_value = 0` if the window is still filling.
///
/// # Safety
/// `ptr` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_loess_add_point(
    ptr: *mut JlOnlineLoess,
    x: c_double,
    y: c_double,
) -> JlOnlineOutput {
    let result = catch_unwind(|| {
        if ptr.is_null() {
            set_last_error_message(shared_parse::PROCESSOR_POINTER_IS_NULL);
            return JlOnlineOutput::default();
        }
        let processor = unsafe { &mut *ptr };

        match processor.inner.add_point(&[x], y) {
            Err(e) => {
                set_last_error_message(&e.to_string());
                JlOnlineOutput::default()
            }
            Ok(None) => JlOnlineOutput::default(),
            Ok(Some(o)) => JlOnlineOutput {
                has_value: 1,
                smoothed: o.smoothed,
                std_error: o.std_error.unwrap_or(f64::NAN),
                residual: o.residual.unwrap_or(f64::NAN),
                robustness_weight: o.robustness_weight.unwrap_or(f64::NAN),
                iterations_used: o.iterations_used.map(|i| i as c_int).unwrap_or(-1),
            },
        }
    });

    result.unwrap_or_default()
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
            Err(err) => {
                set_last_error_message(&err.to_string());
                return null_mut();
            }
        }
    };
}
