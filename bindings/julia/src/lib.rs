//! Julia bindings for fastLoess.
//!
//! Provides Julia access to the fastLoess Rust library via C FFI.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use ptr::null_mut;
use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::panic::catch_unwind;
use std::ptr;
use std::slice::from_raw_parts;

use fastLoess::internals::api::LoessBuilder;
use fastLoess::internals::binding_support as shared_parse;
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
    pub error: *mut c_char,          // NULL if no error
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
            error: ptr::null_mut(),
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
    JlLoessResult {
        error: shared_parse::into_raw_error_c_string(msg),
        ..Default::default()
    }
}

fn error_result_from(err: shared_parse::BindingError) -> JlLoessResult {
    error_result(&err.message)
}

fn map_runtime_result<T, E: ToString>(
    result: std::result::Result<T, E>,
) -> std::result::Result<T, Box<JlLoessResult>> {
    shared_parse::map_runtime(result).map_err(|e| Box::new(error_result_from(e)))
}

// Convert LoessResult to JlLoessResult.
fn loess_result_to_jl(result: LoessResult<f64>) -> JlLoessResult {
    let p = shared_parse::extract_ffi_loess_result(result);
    JlLoessResult {
        x: p.x,
        y: p.y,
        n: p.n as c_ulong,
        standard_errors: p.standard_errors,
        confidence_lower: p.confidence_lower,
        confidence_upper: p.confidence_upper,
        prediction_lower: p.prediction_lower,
        prediction_upper: p.prediction_upper,
        residuals: p.residuals,
        robustness_weights: p.robustness_weights,
        fraction_used: p.fraction_used,
        iterations_used: p.iterations_used,
        rmse: p.rmse,
        mae: p.mae,
        r_squared: p.r_squared,
        aic: p.aic,
        aicc: p.aicc,
        effective_df: p.effective_df,
        residual_sd: p.residual_sd,
        enp: p.enp,
        trace_hat: p.trace_hat,
        delta1: p.delta1,
        delta2: p.delta2,
        residual_scale: p.residual_scale,
        leverage: p.leverage,
        dimensions: p.dimensions,
        cv_scores: p.cv_scores,
        cv_scores_len: p.cv_scores_len as c_ulong,
        error: null_mut(),
    }
}

// Stateful Structs (Opaque to C)

use fastLoess::internals::adapters::online::ParallelOnlineLoess;
use fastLoess::internals::adapters::streaming::ParallelStreamingLoess;

pub struct JlLoessConfig {
    base_builder: LoessBuilder<f64>,
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
    weighted_metric_weights: *const c_double,
    weighted_metric_weights_len: c_ulong,
) -> *mut JlLoessConfig {
    clear_last_error_message();
    let result = catch_unwind(|| {
        let wf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                weight_function,
                shared_parse::DEFAULT_WEIGHT_FUNCTION,
            )
        };
        let rm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                robustness_method,
                shared_parse::DEFAULT_ROBUSTNESS_METHOD,
            )
        };
        let sm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                scaling_method,
                shared_parse::DEFAULT_SCALING_METHOD,
            )
        };
        let bp_str = unsafe {
            shared_parse::parse_c_str_or_default(
                boundary_policy,
                shared_parse::DEFAULT_BOUNDARY_POLICY,
            )
        };
        let zwf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                zero_weight_fallback,
                shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
            )
        };
        let cv_method_str = unsafe {
            shared_parse::parse_c_str_or_default(cv_method, shared_parse::DEFAULT_CV_METHOD)
        };
        let deg_str =
            unsafe { shared_parse::parse_c_str_or_default(degree, shared_parse::DEFAULT_DEGREE) };
        let dm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                distance_metric,
                shared_parse::DEFAULT_DISTANCE_METRIC,
            )
        };
        let surf_str = unsafe {
            shared_parse::parse_c_str_or_default(surface_mode, shared_parse::DEFAULT_SURFACE_MODE)
        };

        let wmw_slice = shared_parse::option_slice_from_ptr(
            weighted_metric_weights,
            weighted_metric_weights_len as usize,
        );
        let effective_metric =
            shared_parse::resolve_distance_metric_for_builder(Some(dm_str), wmw_slice);

        let cv_fractions_slice =
            shared_parse::option_slice_from_ptr(cv_fractions, cv_fractions_len as usize);

        let configured_dimensions = dimensions.max(1) as usize;

        let (base_builder, _) =
            match shared_parse::map_invalid_arg(shared_parse::apply_builder_options(
                LoessBuilder::<f64>::new(),
                shared_parse::BuilderOptionSet {
                    fraction: Some(fraction),
                    iterations: Some(iterations as usize),
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
                    weighted_metric_weights: wmw_slice,
                    surface_mode: Some(surf_str),
                    return_se: return_se != 0,
                    cell: None,
                    interpolation_vertices: None,
                    boundary_degree_fallback: None,
                    cv_fractions: cv_fractions_slice,
                    cv_method: Some(cv_method_str),
                    cv_k: Some(cv_k as usize),
                    cv_seed: None,
                },
            )) {
                Ok(v) => v,
                Err(e) => return null_with_last_error(&e.message),
            };

        Box::into_raw(Box::new(JlLoessConfig {
            base_builder,
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
    custom_weights: *const c_double,
    custom_weights_n: c_ulong,
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

        let cw = shared_parse::option_vec_from_ptr(custom_weights, custom_weights_n as usize);
        let builder = config.base_builder.clone();

        let model = match shared_parse::build_batch(builder, cw) {
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
    let cv_n = res.cv_scores_len as usize;

    shared_parse::free_raw_f64_buffer(res.x, n);
    shared_parse::free_raw_f64_buffer(res.y, n);
    shared_parse::free_raw_f64_buffer(res.standard_errors, n);
    shared_parse::free_raw_f64_buffer(res.confidence_lower, n);
    shared_parse::free_raw_f64_buffer(res.confidence_upper, n);
    shared_parse::free_raw_f64_buffer(res.prediction_lower, n);
    shared_parse::free_raw_f64_buffer(res.prediction_upper, n);
    shared_parse::free_raw_f64_buffer(res.residuals, n);
    shared_parse::free_raw_f64_buffer(res.robustness_weights, n);
    shared_parse::free_raw_f64_buffer(res.leverage, n);
    shared_parse::free_raw_f64_buffer(res.cv_scores, cv_n);
    shared_parse::free_raw_c_string(res.error);
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
        let wf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                weight_function,
                shared_parse::DEFAULT_WEIGHT_FUNCTION,
            )
        };
        let rm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                robustness_method,
                shared_parse::DEFAULT_ROBUSTNESS_METHOD,
            )
        };
        let sm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                scaling_method,
                shared_parse::DEFAULT_SCALING_METHOD,
            )
        };
        let bp_str = unsafe {
            shared_parse::parse_c_str_or_default(
                boundary_policy,
                shared_parse::DEFAULT_BOUNDARY_POLICY,
            )
        };
        let zwf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                zero_weight_fallback,
                shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
            )
        };
        let ms_str = unsafe {
            shared_parse::parse_c_str_or_default(
                merge_strategy,
                shared_parse::DEFAULT_MERGE_STRATEGY,
            )
        };
        let deg_str =
            unsafe { shared_parse::parse_c_str_or_default(degree, shared_parse::DEFAULT_DEGREE) };
        let surf_str = unsafe {
            shared_parse::parse_c_str_or_default(surface_mode, shared_parse::DEFAULT_SURFACE_MODE)
        };
        let dm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                distance_metric,
                shared_parse::DEFAULT_DISTANCE_METRIC,
            )
        };

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
                ..Default::default()
            },
        )) {
            Ok(v) => v,
            Err(e) => return null_with_last_error(&e.message),
        };

        let processor = match shared_parse::build_streaming(
            builder,
            Some(chunk_size_usize),
            (overlap >= 0).then_some(overlap as usize),
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
        let wf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                weight_function,
                shared_parse::DEFAULT_WEIGHT_FUNCTION,
            )
        };
        let rm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                robustness_method,
                shared_parse::DEFAULT_ROBUSTNESS_METHOD,
            )
        };
        let sm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                scaling_method,
                shared_parse::DEFAULT_SCALING_METHOD,
            )
        };
        let bp_str = unsafe {
            shared_parse::parse_c_str_or_default(
                boundary_policy,
                shared_parse::DEFAULT_BOUNDARY_POLICY,
            )
        };
        let zwf_str = unsafe {
            shared_parse::parse_c_str_or_default(
                zero_weight_fallback,
                shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
            )
        };
        let um_str = unsafe {
            shared_parse::parse_c_str_or_default(update_mode, shared_parse::DEFAULT_UPDATE_MODE)
        };

        let deg_str =
            unsafe { shared_parse::parse_c_str_or_default(degree, shared_parse::DEFAULT_DEGREE) };
        let surf_str = unsafe {
            shared_parse::parse_c_str_or_default(surface_mode, shared_parse::DEFAULT_SURFACE_MODE)
        };
        let dm_str = unsafe {
            shared_parse::parse_c_str_or_default(
                distance_metric,
                shared_parse::DEFAULT_DISTANCE_METRIC,
            )
        };

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
                ..Default::default()
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
            return JlOnlineOutput {
                error: shared_parse::into_raw_error_c_string(
                    shared_parse::PROCESSOR_POINTER_IS_NULL,
                ),
                ..JlOnlineOutput::default()
            };
        }
        let processor = unsafe { &mut *ptr };

        match processor.inner.add_point(&[x], y) {
            Err(e) => JlOnlineOutput {
                error: shared_parse::into_raw_error_c_string(&e.to_string()),
                ..JlOnlineOutput::default()
            },
            Ok(None) => JlOnlineOutput::default(),
            Ok(Some(o)) => {
                let (std_error, residual, robustness_weight, iterations_used) =
                    shared_parse::extract_online_output(&o);
                JlOnlineOutput {
                    has_value: 1,
                    smoothed: o.smoothed,
                    std_error,
                    residual,
                    robustness_weight,
                    iterations_used,
                    error: ptr::null_mut(),
                }
            }
        }
    });

    result.unwrap_or_else(|_| JlOnlineOutput {
        error: shared_parse::into_raw_error_c_string(shared_parse::panic_fallback_message()),
        ..JlOnlineOutput::default()
    })
}

/// Free the error field of a JlOnlineOutput returned by jl_online_loess_add_point.
/// No-op when output is null or the error field is already null.
///
/// # Safety
/// `output` must be a valid pointer to a JlOnlineOutput or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jl_online_free_output(output: *mut JlOnlineOutput) {
    if !output.is_null() {
        shared_parse::free_raw_c_string((*output).error);
        (*output).error = ptr::null_mut();
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
            Err(err) => {
                set_last_error_message(&err.to_string());
                return null_mut();
            }
        }
    };
}
