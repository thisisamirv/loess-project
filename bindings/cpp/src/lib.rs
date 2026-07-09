//! C/C++ bindings for fastLoess.
//!
//! Provides C access to the fastLoess Rust library via C FFI.
//! A C++ wrapper header (fastloess.hpp) provides idiomatic C++ usage.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_ulong};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use fastLoess::internals::adapters::online::ParallelOnlineLoess;
use fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use fastLoess::internals::api::{Batch, LoessBuilder, Online, Streaming};
use fastLoess::internals::binding_support as shared_parse;
use fastLoess::internals::binding_support::{
    BoundaryPolicy, MergeStrategy, RobustnessMethod, ScalingMethod, UpdateMode, WeightFunction,
    ZeroWeightFallback,
};
use fastLoess::prelude::LoessResult;

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static CPP_LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    let cmsg = shared_parse::to_cstring_lossy(msg);
    CPP_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(cmsg);
    });
}

fn clear_last_error() {
    CPP_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn null_with_error<T>(msg: &str) -> *mut T {
    set_last_error(msg);
    ptr::null_mut()
}

fn error_result_from(err: shared_parse::BindingError) -> CppLoessResult {
    error_result(&err.message)
}

#[allow(clippy::result_large_err)]
fn map_invalid_arg_result<T, E: ToString>(
    result: std::result::Result<T, E>,
) -> std::result::Result<T, CppLoessResult> {
    shared_parse::map_invalid_arg(result).map_err(error_result_from)
}

#[allow(clippy::result_large_err)]
fn map_runtime_result<T, E: ToString>(
    result: std::result::Result<T, E>,
) -> std::result::Result<T, CppLoessResult> {
    shared_parse::map_runtime(result).map_err(error_result_from)
}

fn with_panic_result<F>(f: F) -> CppLoessResult
where
    F: FnOnce() -> CppLoessResult,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => error_result(shared_parse::panic_fallback_message()),
    }
}

fn with_panic_ptr<T, F>(f: F) -> *mut T
where
    F: FnOnce() -> *mut T,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => null_with_error(shared_parse::panic_fallback_message()),
    }
}

fn with_panic_void<F>(f: F)
where
    F: FnOnce(),
{
    if catch_unwind(AssertUnwindSafe(f)).is_err() {
        set_last_error(shared_parse::panic_fallback_message());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cpp_last_error_message() -> *const c_char {
    CPP_LAST_ERROR.with(|slot| {
        if let Some(msg) = slot.borrow().as_ref() {
            msg.as_ptr()
        } else {
            ptr::null()
        }
    })
}

// Per-point result from an online update, passed across the FFI boundary.
// has_value = 1 means the window is ready and smoothed is valid; 0 means the
// window is still filling (caller should treat it as no output yet).
// Non-computed optional fields use f64::NAN (for floats) or -1 (for int).
// error = NULL if no error, otherwise points to a null-terminated error string.
#[repr(C)]
pub struct CppOnlineOutput {
    pub has_value: c_int,
    pub smoothed: c_double,
    pub std_error: c_double,
    pub residual: c_double,
    pub robustness_weight: c_double,
    pub iterations_used: c_int,
    pub error: *mut c_char, // NULL if no error
}

// Result struct that can be passed across FFI boundary.
// All arrays are allocated by Rust and must be freed by Rust.
#[repr(C)]
pub struct CppLoessResult {
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
            cv_scores: ptr::null_mut(),
            cv_scores_len: 0,
            error: ptr::null_mut(),
        }
    }
}

// Convert a Vec<f64> to a raw pointer.
fn vec_to_ptr(v: Vec<f64>) -> *mut c_double {
    let mut boxed = v.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    ptr
}

// Convert an optional Vec<f64> to a raw pointer.
fn opt_vec_to_ptr(v: Option<Vec<f64>>) -> *mut c_double {
    match v {
        Some(vec) => vec_to_ptr(vec),
        None => ptr::null_mut(),
    }
}

// Create an error result with the given message.
fn error_result(msg: &str) -> CppLoessResult {
    let mut result = CppLoessResult::default();
    let c_string = shared_parse::to_cstring_lossy(msg);
    result.error = c_string.into_raw();
    result
}

// Parse a C string safely.
unsafe fn parse_c_str(s: *const c_char, default: &str) -> &str {
    if s.is_null() {
        default
    } else {
        CStr::from_ptr(s).to_str().unwrap_or(default)
    }
}

// Parse weight function from string.
fn parse_weight_function(name: &str) -> Result<WeightFunction, String> {
    shared_parse::parse_weight_function(name)
}

// Parse robustness method from string.
fn parse_robustness_method(name: &str) -> Result<RobustnessMethod, String> {
    shared_parse::parse_robustness_method(name)
}

// Parse zero weight fallback from string.
fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback, String> {
    shared_parse::parse_zero_weight_fallback(name)
}

// Parse boundary policy from string.
fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy, String> {
    shared_parse::parse_boundary_policy(name)
}

// Parse scaling method from string.
fn parse_scaling_method(name: &str) -> Result<ScalingMethod, String> {
    shared_parse::parse_scaling_method(name)
}

// Parse update mode from string.
fn parse_update_mode(name: &str) -> Result<UpdateMode, String> {
    shared_parse::parse_update_mode(name)
}

// Parse merge strategy from string.
fn parse_merge_strategy(name: &str) -> Result<MergeStrategy, String> {
    shared_parse::parse_merge_strategy(name)
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
            cv_scores: opt_vec_to_ptr(result.cv_scores.clone()),
            cv_scores_len: result
                .cv_scores
                .as_ref()
                .map(|v| v.len() as c_ulong)
                .unwrap_or(0),
            error: ptr::null_mut(),
        }
    }
}

// Opaque handle to a Loess model.
pub struct CppLoess {
    builder: Option<LoessBuilder<f64>>,
    // Store CV options to apply lazily because of lifetime constraints
    cv_fractions: Option<Vec<f64>>,
    cv_method: Option<String>,
    cv_k: usize,
    // User-defined case weights
    custom_weights: Option<Vec<f64>>,
    // Advanced interpolation/CV options
    cell: Option<f64>,
    interpolation_vertices: Option<usize>,
    boundary_degree_fallback: Option<bool>,
    cv_seed: Option<u64>,
}

// Opaque handle to a Loess model.
pub struct CppStreamingLoess {
    model: Option<ParallelStreamingLoess<f64>>,
}

// Opaque handle to a Loess model.
pub struct CppOnlineLoess {
    model: Option<ParallelOnlineLoess<f64>>,
}

fn setter_unsupported_eager_lifecycle(name: &str) {
    set_last_error(&format!(
        "{name} is not supported: streaming/online models are eagerly initialized at construction"
    ));
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
    // Advanced options
    cell: c_double,
    interpolation_vertices: c_int,
    boundary_degree_fallback: c_int,
    weighted_metric_weights: *const c_double,
    weighted_metric_weights_len: c_ulong,
) -> *mut CppLoess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = parse_c_str(weight_function, "tricube");
        let rm_str = parse_c_str(robustness_method, "bisquare");
        let sm_str = parse_c_str(scaling_method, "mad");
        let bp_str = parse_c_str(boundary_policy, "extend");
        let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");

        match parse_weight_function(wf_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_robustness_method(rm_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_scaling_method(sm_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_boundary_policy(bp_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_zero_weight_fallback(zwf_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };

        let cv_fractions_vec = if !cv_fractions.is_null() && cv_fractions_len > 0 {
            let fractions = std::slice::from_raw_parts(cv_fractions, cv_fractions_len as usize);
            Some(fractions.to_vec())
        } else {
            None
        };

        let cv_method_str = parse_c_str(cv_method, "kfold").to_string();
        let cv_k_usize = cv_k.max(2) as usize;
        let weighted_metric_weights_slice =
            if !weighted_metric_weights.is_null() && weighted_metric_weights_len > 0 {
                Some(std::slice::from_raw_parts(
                    weighted_metric_weights,
                    weighted_metric_weights_len as usize,
                ))
            } else {
                None
            };
        let distance_metric_str =
            (!distance_metric.is_null()).then_some(parse_c_str(distance_metric, "normalized"));
        let weighted_without_weights = distance_metric_str
            .map(|v| v.eq_ignore_ascii_case("weighted"))
            .unwrap_or(false)
            && weighted_metric_weights_slice.is_none();

        let (mut builder, _) = match shared_parse::apply_builder_options(
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
                degree: (!degree.is_null()).then_some(parse_c_str(degree, "linear")),
                dimensions: (dimensions > 0).then_some(dimensions as usize),
                distance_metric: distance_metric_str
                    .filter(|v| !v.eq_ignore_ascii_case("weighted")),
                weighted_metric_weights: weighted_metric_weights_slice,
                surface_mode: (!surface_mode.is_null())
                    .then_some(parse_c_str(surface_mode, "interpolation")),
                return_se: return_se != 0,
                cell: None,
                interpolation_vertices: None,
                boundary_degree_fallback: None,
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };
        if weighted_without_weights {
            builder = builder.distance_metric("weighted");
        }

        Box::into_raw(Box::new(CppLoess {
            builder: Some(builder),
            cv_fractions: cv_fractions_vec,
            cv_method: Some(cv_method_str),
            cv_k: cv_k_usize,
            cv_seed: None,
            custom_weights: None,
            cell: (!cell.is_nan()).then_some(cell),
            interpolation_vertices: (interpolation_vertices > 0)
                .then_some(interpolation_vertices as usize),
            boundary_degree_fallback: (boundary_degree_fallback >= 0)
                .then_some(boundary_degree_fallback != 0),
        }))
    })
}

/// Set CV seed for reproducible K-fold splits.
///
/// # Safety
/// ptr must be valid.
#[unsafe(no_mangle)]
#[allow(clippy::useless_conversion)] // c_ulong is u32 on Windows, u64 on Linux/macOS
pub unsafe extern "C" fn cpp_loess_set_cv_seed(ptr: *mut CppLoess, seed: c_ulong) {
    with_panic_void(|| {
        if !ptr.is_null() {
            unsafe { (*ptr).cv_seed = Some(u64::from(seed)) };
        }
    });
}

/// Set per-observation custom weights for the next fit call.
///
/// # Safety
/// `ptr` must be a valid `CppLoess` pointer. `weights` must be a valid array of length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_set_custom_weights(
    ptr: *mut CppLoess,
    weights: *const c_double,
    n: c_ulong,
) {
    with_panic_void(|| {
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        if weights.is_null() || n == 0 {
            unsafe { (*ptr).custom_weights = None };
            return;
        }
        let slice = unsafe { std::slice::from_raw_parts(weights, n as usize) };
        unsafe { (*ptr).custom_weights = Some(slice.to_vec()) };
    });
}

// Legacy model setters retained for ABI compatibility.
// Streaming/online models are eagerly initialized at construction, so these setters
// are unsupported and now report this through the last-error channel.

/// Set weighted-metric weights for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer. `weights` must be a
/// valid array of length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_set_weighted_metric(
    ptr: *mut CppStreamingLoess,
    weights: *const c_double,
    n: c_ulong,
) {
    with_panic_void(|| {
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        if weights.is_null() || n == 0 {
            set_last_error(shared_parse::INVALID_DATA_INPUTS);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_streaming_set_weighted_metric");
    });
}

/// Set cell tuning parameter for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_set_cell(ptr: *mut CppStreamingLoess, cell: c_double) {
    with_panic_void(|| {
        let _ = cell;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_streaming_set_cell");
    });
}

/// Set number of interpolation vertices for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_set_interpolation_vertices(
    ptr: *mut CppStreamingLoess,
    vertices: c_ulong,
) {
    with_panic_void(|| {
        let _ = vertices;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_streaming_set_interpolation_vertices");
    });
}

/// Enable or disable boundary degree fallback for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_set_boundary_degree_fallback(
    ptr: *mut CppStreamingLoess,
    enabled: c_int,
) {
    with_panic_void(|| {
        let _ = enabled;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_streaming_set_boundary_degree_fallback");
    });
}

/// Set confidence interval level for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_set_confidence_intervals(
    ptr: *mut CppStreamingLoess,
    level: c_double,
) {
    with_panic_void(|| {
        let _ = level;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_streaming_set_confidence_intervals");
    });
}

/// Set prediction interval level for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_set_prediction_intervals(
    ptr: *mut CppStreamingLoess,
    level: c_double,
) {
    with_panic_void(|| {
        let _ = level;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_streaming_set_prediction_intervals");
    });
}

// Legacy model setters retained for ABI compatibility.
// Streaming/online models are eagerly initialized at construction, so these setters
// are unsupported and now report this through the last-error channel.

/// Set weighted-metric weights for an model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer. `weights` must be a
/// valid array of length `n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_set_weighted_metric(
    ptr: *mut CppOnlineLoess,
    weights: *const c_double,
    n: c_ulong,
) {
    with_panic_void(|| {
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        if weights.is_null() || n == 0 {
            set_last_error(shared_parse::INVALID_DATA_INPUTS);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_online_set_weighted_metric");
    });
}

/// Set cell tuning parameter for an model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_set_cell(ptr: *mut CppOnlineLoess, cell: c_double) {
    with_panic_void(|| {
        let _ = cell;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_online_set_cell");
    });
}

/// Set number of interpolation vertices for an model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_set_interpolation_vertices(
    ptr: *mut CppOnlineLoess,
    vertices: c_ulong,
) {
    with_panic_void(|| {
        let _ = vertices;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_online_set_interpolation_vertices");
    });
}

/// Enable or disable boundary degree fallback for an model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_set_boundary_degree_fallback(
    ptr: *mut CppOnlineLoess,
    enabled: c_int,
) {
    with_panic_void(|| {
        let _ = enabled;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_online_set_boundary_degree_fallback");
    });
}

/// Set confidence interval level for an model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_set_confidence_intervals(
    ptr: *mut CppOnlineLoess,
    level: c_double,
) {
    with_panic_void(|| {
        let _ = level;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_online_set_confidence_intervals");
    });
}

/// Set prediction interval level for an model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_set_prediction_intervals(
    ptr: *mut CppOnlineLoess,
    level: c_double,
) {
    with_panic_void(|| {
        let _ = level;
        if ptr.is_null() {
            set_last_error(shared_parse::MODEL_POINTER_IS_NULL);
            return;
        }
        setter_unsupported_eager_lifecycle("cpp_online_set_prediction_intervals");
    });
}

/// Fit the model.
///
/// # Safety
/// `ptr` must be a valid CppLoess pointer. `x` must be a valid array of length `x_n`
/// (= n_observations * dimensions), `y` must be a valid array of length `y_n` (= n_observations).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_fit(
    ptr: *mut CppLoess,
    x: *const c_double,
    x_n: c_ulong,
    y: *const c_double,
    y_n: c_ulong,
) -> CppLoessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        if x.is_null() || y.is_null() || x_n == 0 || y_n == 0 {
            return error_result(shared_parse::INVALID_DATA_INPUTS);
        }

        let loess = &mut *ptr;
        let x_slice = std::slice::from_raw_parts(x, x_n as usize);
        let y_slice = std::slice::from_raw_parts(y, y_n as usize);

        if let Some(mut builder) = loess.builder.clone() {
            builder = match map_invalid_arg_result(shared_parse::apply_cross_validation(
                builder,
                loess.cv_fractions.as_deref(),
                loess.cv_method.as_deref(),
                Some(loess.cv_k),
                loess.cv_seed,
            )) {
                Ok(b) => b,
                Err(e) => return e,
            };
            // Apply custom weights if provided
            if let Some(ref uw) = loess.custom_weights {
                builder = builder.custom_weights(uw.clone());
            }
            if let Some(c) = loess.cell {
                builder = builder.cell(c);
            }
            if let Some(v) = loess.interpolation_vertices {
                builder = builder.interpolation_vertices(v);
            }
            if let Some(bdf) = loess.boundary_degree_fallback {
                builder = builder.boundary_degree_fallback(bdf);
            }

            let model = match map_runtime_result(builder.adapter(Batch).build()) {
                Ok(m) => m,
                Err(e) => return e,
            };
            match map_runtime_result(model.fit(x_slice, y_slice)) {
                Ok(r) => r.into(),
                Err(e) => e,
            }
        } else {
            error_result("Model initialization failed")
        }
    })
}

/// Free model.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `cpp_loess_new` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_free(ptr: *mut CppLoess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Create a new Loess model.
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
    // opts
    chunk_size: c_int,
    overlap: c_int,
    merge_strategy: *const c_char,
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
) -> *mut CppStreamingLoess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = parse_c_str(weight_function, "tricube");
        let rm_str = parse_c_str(robustness_method, "bisquare");
        let sm_str = parse_c_str(scaling_method, "mad");
        let bp_str = parse_c_str(boundary_policy, "extend");
        let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
        let ms_str = parse_c_str(merge_strategy, "weighted_average");

        match parse_weight_function(wf_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_robustness_method(rm_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_scaling_method(sm_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_boundary_policy(bp_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_zero_weight_fallback(zwf_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        let ms = match parse_merge_strategy(ms_str) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let chunk_size = chunk_size as usize;
        let overlap_size = if overlap < 0 {
            let default = chunk_size / 10;
            default.min(chunk_size.saturating_sub(10)).max(1)
        } else {
            overlap as usize
        };
        let degree_str = (!degree.is_null()).then_some(parse_c_str(degree, "linear"));
        let surface_mode_str =
            (!surface_mode.is_null()).then_some(parse_c_str(surface_mode, "interpolation"));
        let distance_metric_str =
            (!distance_metric.is_null()).then_some(parse_c_str(distance_metric, "normalized"));
        let weighted_metric_weights_slice =
            if !weighted_metric_weights.is_null() && weighted_metric_weights_len > 0 {
                Some(std::slice::from_raw_parts(
                    weighted_metric_weights,
                    weighted_metric_weights_len as usize,
                ))
            } else {
                None
            };
        let weighted_without_weights = distance_metric_str
            .map(|v| v.eq_ignore_ascii_case("weighted"))
            .unwrap_or(false)
            && weighted_metric_weights_slice.is_none();

        let (mut builder, _) = match shared_parse::apply_builder_options(
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
                confidence_intervals: None,
                prediction_intervals: None,
                parallel: Some(parallel != 0),
                degree: degree_str,
                dimensions: (dimensions > 0).then_some(dimensions as usize),
                distance_metric: distance_metric_str
                    .filter(|v| !v.eq_ignore_ascii_case("weighted")),
                weighted_metric_weights: weighted_metric_weights_slice,
                surface_mode: surface_mode_str,
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
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };
        if weighted_without_weights {
            builder = builder.distance_metric("weighted");
        }
        if !confidence_intervals.is_nan() {
            builder = builder.confidence_intervals(confidence_intervals);
        }
        if !prediction_intervals.is_nan() {
            builder = builder.prediction_intervals(prediction_intervals);
        }

        let model = match shared_parse::map_runtime(
            builder
                .clone()
                .adapter(Streaming)
                .chunk_size(chunk_size)
                .overlap(overlap_size)
                .merge_strategy(ms)
                .build(),
        ) {
            Ok(m) => m,
            Err(e) => return null_with_error(&e.message),
        };

        Box::into_raw(Box::new(CppStreamingLoess { model: Some(model) }))
    })
}

/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be valid. `x` must be a valid array of length `x_n` (= n_observations * dimensions),
/// `y` must be a valid array of length `y_n` (= n_observations).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_process(
    ptr: *mut CppStreamingLoess,
    x: *const c_double,
    x_n: c_ulong,
    y: *const c_double,
    y_n: c_ulong,
) -> CppLoessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        let loess = &mut *ptr;
        if x.is_null() || y.is_null() || x_n == 0 || y_n == 0 {
            return error_result(shared_parse::INVALID_DATA_INPUTS);
        }
        let x_slice = std::slice::from_raw_parts(x, x_n as usize);
        let y_slice = std::slice::from_raw_parts(y, y_n as usize);

        if let Some(model) = &mut loess.model {
            match map_runtime_result(model.process_chunk(x_slice, y_slice)) {
                Ok(r) => r.into(),
                Err(e) => e,
            }
        } else {
            error_result("Model initialization failed")
        }
    })
}

/// Finalize the process.
///
/// # Safety
/// `ptr` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_finalize(ptr: *mut CppStreamingLoess) -> CppLoessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        let loess = &mut *ptr;
        if let Some(model) = &mut loess.model {
            match map_runtime_result(model.finalize()) {
                Ok(r) => r.into(),
                Err(e) => e,
            }
        } else {
            error_result(" model not initialized")
        }
    })
}

/// Free model.
///
/// # Safety
/// `ptr` must be valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_free(ptr: *mut CppStreamingLoess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Create a new Loess model.
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
    return_diagnostics: c_int,
    return_residuals: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    parallel: c_int,
    // opts
    window_capacity: c_int,
    min_points: c_int,
    update_mode: *const c_char,
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
) -> *mut CppOnlineLoess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = parse_c_str(weight_function, "tricube");
        let rm_str = parse_c_str(robustness_method, "bisquare");
        let sm_str = parse_c_str(scaling_method, "mad");
        let bp_str = parse_c_str(boundary_policy, "extend");
        let zwf_str = parse_c_str(zero_weight_fallback, "use_local_mean");
        let um_str = parse_c_str(update_mode, "full");

        match parse_weight_function(wf_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_robustness_method(rm_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_scaling_method(sm_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_boundary_policy(bp_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        match parse_zero_weight_fallback(zwf_str) {
            Ok(_) => (),
            Err(e) => return null_with_error(&e),
        };
        let um = match parse_update_mode(um_str) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let configured_dimensions = if dimensions > 0 {
            dimensions as usize
        } else {
            1
        };
        let degree_str = (!degree.is_null()).then_some(parse_c_str(degree, "linear"));
        let surface_mode_str =
            (!surface_mode.is_null()).then_some(parse_c_str(surface_mode, "interpolation"));
        let distance_metric_str =
            (!distance_metric.is_null()).then_some(parse_c_str(distance_metric, "normalized"));
        let weighted_metric_weights_slice =
            if !weighted_metric_weights.is_null() && weighted_metric_weights_len > 0 {
                Some(std::slice::from_raw_parts(
                    weighted_metric_weights,
                    weighted_metric_weights_len as usize,
                ))
            } else {
                None
            };
        let weighted_without_weights = distance_metric_str
            .map(|v| v.eq_ignore_ascii_case("weighted"))
            .unwrap_or(false)
            && weighted_metric_weights_slice.is_none();

        let (mut builder, _) = match shared_parse::apply_builder_options(
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
                confidence_intervals: None,
                prediction_intervals: None,
                parallel: Some(parallel != 0),
                degree: degree_str,
                dimensions: (dimensions > 0).then_some(configured_dimensions),
                distance_metric: distance_metric_str
                    .filter(|v| !v.eq_ignore_ascii_case("weighted")),
                weighted_metric_weights: weighted_metric_weights_slice,
                surface_mode: surface_mode_str,
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
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };
        if weighted_without_weights {
            builder = builder.distance_metric("weighted");
        }
        if !confidence_intervals.is_nan() {
            builder = builder.confidence_intervals(confidence_intervals);
        }
        if !prediction_intervals.is_nan() {
            builder = builder.prediction_intervals(prediction_intervals);
        }

        let model = match shared_parse::map_runtime(
            builder
                .clone()
                .adapter(Online)
                .window_capacity(window_capacity as usize)
                .min_points(min_points as usize)
                .update_mode(um)
                .build(),
        ) {
            Ok(m) => m,
            Err(e) => return null_with_error(&e.message),
        };

        Box::into_raw(Box::new(CppOnlineLoess { model: Some(model) }))
    })
}

/// Add a single point to the model and return its smoothed value.
/// `has_value = 0` in the result means the window is still filling.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_add_point(
    ptr: *mut CppOnlineLoess,
    x: c_double,
    y: c_double,
) -> CppOnlineOutput {
    let make_error = |msg: &str| -> CppOnlineOutput {
        let c_string = shared_parse::to_cstring_lossy(msg);
        CppOnlineOutput {
            has_value: 0,
            smoothed: f64::NAN,
            std_error: f64::NAN,
            residual: f64::NAN,
            robustness_weight: f64::NAN,
            iterations_used: -1,
            error: c_string.into_raw(),
        }
    };

    match catch_unwind(AssertUnwindSafe(|| {
        if ptr.is_null() {
            return make_error(shared_parse::MODEL_POINTER_IS_NULL);
        }
        let loess = unsafe { &mut *ptr };

        if let Some(model) = &mut loess.model {
            match model.add_point(&[x], y) {
                Err(e) => make_error(&e.to_string()),
                Ok(None) => CppOnlineOutput {
                    has_value: 0,
                    smoothed: f64::NAN,
                    std_error: f64::NAN,
                    residual: f64::NAN,
                    robustness_weight: f64::NAN,
                    iterations_used: -1,
                    error: ptr::null_mut(),
                },
                Ok(Some(o)) => CppOnlineOutput {
                    has_value: 1,
                    smoothed: o.smoothed,
                    std_error: o.std_error.unwrap_or(f64::NAN),
                    residual: o.residual.unwrap_or(f64::NAN),
                    robustness_weight: o.robustness_weight.unwrap_or(f64::NAN),
                    iterations_used: o.iterations_used.map(|i| i as c_int).unwrap_or(-1),
                    error: ptr::null_mut(),
                },
            }
        } else {
            make_error("Model initialization failed")
        }
    })) {
        Ok(v) => v,
        Err(_) => make_error(shared_parse::panic_fallback_message()),
    }
}

/// Free the error string in a CppOnlineOutput (call only when error != NULL).
///
/// # Safety
/// `output` must be a valid pointer and `output->error` must have been allocated by Rust.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_free_output(output: *mut CppOnlineOutput) {
    with_panic_void(|| {
        if !output.is_null() {
            let out = unsafe { &mut *output };
            if !out.error.is_null() {
                let _ = CString::from_raw(out.error);
                out.error = ptr::null_mut();
            }
        }
    });
}

/// Free model.
///
/// # Safety
/// `ptr` must be valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_free(ptr: *mut CppOnlineLoess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Free a CppLoessResult.
///
/// # Safety
/// `result` must be a valid pointer to a CppLoessResult struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_free_result(result: *mut CppLoessResult) {
    with_panic_void(|| {
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
        if !r.cv_scores.is_null() {
            let cv_n = r.cv_scores_len as usize;
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(r.cv_scores, cv_n));
        }

        // Free error string
        if !r.error.is_null() {
            let _ = std::ffi::CString::from_raw(r.error);
        }
    });
}
