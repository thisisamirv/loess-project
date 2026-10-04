//! Go bindings for fastLoess (via cgo).
//!
//! Provides C access to the fastLoess Rust library via C FFI, consumed by
//! the `bindings/go` Go package through cgo. The generated C header
//! (fastloess_go.h) is the same shape as the C++ binding's, just with a
//! `go_`/`Go` naming scheme instead of `cpp_`/`Cpp`.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_ulonglong};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice::from_raw_parts;
use std::sync::Arc;

use fastLoess::internals::adapters::online::ParallelOnlineLoess;
use fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use fastLoess::internals::api::LoessBuilder;
use fastLoess::internals::binding_support as shared_parse;
use fastLoess::prelude::LoessResult;

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static GO_LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    let cmsg = shared_parse::to_cstring_lossy(msg);
    GO_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(cmsg);
    });
}

fn clear_last_error() {
    GO_LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn null_with_error<T>(msg: &str) -> *mut T {
    set_last_error(msg);
    ptr::null_mut()
}

fn error_result_from(err: shared_parse::BindingError) -> GoLoessResult {
    error_result(&err.message)
}

#[allow(clippy::result_large_err)]
fn map_invalid_arg_result<T, E: ToString>(result: Result<T, E>) -> Result<T, GoLoessResult> {
    shared_parse::map_invalid_arg(result).map_err(error_result_from)
}

#[allow(clippy::result_large_err)]
fn map_runtime_result<T, E: ToString>(result: Result<T, E>) -> Result<T, GoLoessResult> {
    shared_parse::map_runtime(result).map_err(error_result_from)
}

fn with_panic_result<F>(f: F) -> GoLoessResult
where
    F: FnOnce() -> GoLoessResult,
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
pub extern "C" fn go_last_error_message() -> *const c_char {
    GO_LAST_ERROR.with(|slot| {
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
pub struct GoOnlineOutput {
    pub has_value: c_int,
    pub y: c_double,
    pub standard_error: c_double,
    pub residual: c_double,
    pub robustness_weight: c_double,
    pub iterations_used: c_int,
    /// Confidence interval lower bound (`update_mode="full"` only, if requested)
    pub confidence_lower: c_double,
    /// Confidence interval upper bound (`update_mode="full"` only, if requested)
    pub confidence_upper: c_double,
    /// Prediction interval lower bound (`update_mode="full"` only, if requested)
    pub prediction_lower: c_double,
    /// Prediction interval upper bound (`update_mode="full"` only, if requested)
    pub prediction_upper: c_double,
    /// Latest point's local fit gradient, `dimensions` values, NULL if not requested
    pub gradient: *mut c_double,
    /// Number of predictor dimensions (needed to know `gradient`'s true length)
    pub dimensions: c_int,
    pub error: *mut c_char, // NULL if no error
}

// Result struct that can be passed across FFI boundary.
// All arrays are allocated by Rust and must be freed by Rust.
#[repr(C)]
pub struct GoLoessResult {
    /// x values, in the same order as the input (length = n)
    pub x: *mut c_double,
    /// Smoothed y values (length = n)
    pub y: *mut c_double,
    /// Number of data points
    pub n: usize,

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
    /// Local fit's gradient at each point, `dimensions` values per point, flattened
    /// (NULL if not requested; only takes effect when surface_mode = "direct")
    pub gradient: *mut c_double,

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
    /// Cross-validation scores (NULL if not computed, length = cv_scores_len)
    pub cv_scores: *mut c_double,
    pub cv_scores_len: usize,

    /// Opaque handle for `go_predict()`, non-NULL only if `retain_model` was set to 1.
    /// Must eventually be freed via `go_predict_handle_free`.
    pub predict_handle: *mut GoPredictHandle,

    /// Error message (NULL if no error)
    pub error: *mut c_char,
}

impl Default for GoLoessResult {
    fn default() -> Self {
        GoLoessResult {
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
            gradient: ptr::null_mut(),
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
            predict_handle: ptr::null_mut(),
            error: ptr::null_mut(),
        }
    }
}

// Create an error result with the given message.
fn error_result(msg: &str) -> GoLoessResult {
    GoLoessResult {
        error: shared_parse::into_raw_error_c_string(msg),
        ..Default::default()
    }
}

impl Default for GoOnlineOutput {
    fn default() -> Self {
        GoOnlineOutput {
            has_value: 0,
            y: f64::NAN,
            standard_error: f64::NAN,
            residual: f64::NAN,
            robustness_weight: f64::NAN,
            iterations_used: -1,
            confidence_lower: f64::NAN,
            confidence_upper: f64::NAN,
            prediction_lower: f64::NAN,
            prediction_upper: f64::NAN,
            gradient: ptr::null_mut(),
            dimensions: 1,
            error: ptr::null_mut(),
        }
    }
}

impl From<LoessResult<f64>> for GoLoessResult {
    fn from(mut result: LoessResult<f64>) -> Self {
        let gradient = result.gradient.take();
        let p = shared_parse::extract_ffi_loess_result(result);
        GoLoessResult {
            x: p.x,
            y: p.y,
            n: p.n,
            standard_errors: p.standard_errors,
            confidence_lower: p.confidence_lower,
            confidence_upper: p.confidence_upper,
            prediction_lower: p.prediction_lower,
            prediction_upper: p.prediction_upper,
            residuals: p.residuals,
            robustness_weights: p.robustness_weights,
            gradient: shared_parse::opt_vec_to_raw_ptr(gradient),
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
            cv_scores_len: p.cv_scores_len,
            predict_handle: p
                .predict_state
                .map(|state| Box::into_raw(Box::new(GoPredictHandle { state })))
                .unwrap_or(ptr::null_mut()),
            error: ptr::null_mut(),
        }
    }
}

// Opaque handle to a batch Loess model.
pub struct GoLoess {
    builder: Option<LoessBuilder<f64>>,
    // Store CV options to apply lazily because of lifetime constraints
    cv_fractions: Option<Vec<f64>>,
    cv_method: Option<String>,
    cv_k: usize,
    // Advanced interpolation/CV options
    cell: Option<f64>,
    interpolation_vertices: Option<usize>,
    boundary_degree_fallback: Option<bool>,
    cv_seed: Option<u64>,
}

// Opaque handle to a streaming Loess model.
pub struct GoStreamingLoess {
    model: Option<ParallelStreamingLoess<f64>>,
}

// Opaque handle to an online Loess model.
pub struct GoOnlineLoess {
    model: Option<ParallelOnlineLoess<f64>>,
    dimensions: usize,
}

// Opaque handle retained by `go_loess_fit` (in `GoLoessResult::predict_handle`, if
// `retain_model` was set to 1), enabling `go_predict()`. Wraps just the lightweight
// `Arc<PredictState<f64>>` extracted from the fitted model, not the whole result.
pub struct GoPredictHandle {
    state: Arc<shared_parse::PredictState<f64>>,
}

/// Go wrapper constructor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null. Arrays must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_loess_new(
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
    cv_fractions_len: usize,
    cv_method: *const c_char,
    cv_k: c_int,
    parallel: c_int,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    return_se: c_int,
    return_sorted: c_int,
    // Advanced options
    cell: c_double,
    interpolation_vertices: c_int,
    boundary_degree_fallback: c_int,
    weighted_metric_weights: *const c_double,
    weighted_metric_weights_len: usize,
    missing: *const c_char,
    retain_model: c_int,
    return_gradient: c_int,
) -> *mut GoLoess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = shared_parse::parse_c_str_or_default(
            weight_function,
            shared_parse::DEFAULT_WEIGHT_FUNCTION,
        );
        let rm_str = shared_parse::parse_c_str_or_default(
            robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm_str = shared_parse::parse_c_str_or_default(
            scaling_method,
            shared_parse::DEFAULT_SCALING_METHOD,
        );
        let bp_str = shared_parse::parse_c_str_or_default(
            boundary_policy,
            shared_parse::DEFAULT_BOUNDARY_POLICY,
        );
        let zwf_str = shared_parse::parse_c_str_or_default(
            zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );
        let missing_str =
            shared_parse::parse_c_str_or_default(missing, shared_parse::DEFAULT_MISSING_POLICY);

        let iterations = match shared_parse::require_non_negative_usize("iterations", iterations) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let cv_fractions_vec = shared_parse::option_vec_from_ptr(cv_fractions, cv_fractions_len);

        let cv_method_str =
            shared_parse::parse_c_str_or_default(cv_method, shared_parse::DEFAULT_CV_METHOD)
                .to_string();
        let cv_k_usize = match shared_parse::require_non_negative_usize("cv_k", cv_k) {
            Ok(value) => value,
            Err(error) => return null_with_error(&error),
        };
        let weighted_metric_weights_slice = shared_parse::option_slice_from_ptr(
            weighted_metric_weights,
            weighted_metric_weights_len,
        );
        let distance_metric_str =
            (!distance_metric.is_null()).then_some(shared_parse::parse_c_str_or_default(
                distance_metric,
                shared_parse::DEFAULT_DISTANCE_METRIC,
            ));

        let (builder, _) = match shared_parse::apply_builder_options(
            LoessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
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
                degree: (!degree.is_null()).then_some(shared_parse::parse_c_str_or_default(
                    degree,
                    shared_parse::DEFAULT_DEGREE,
                )),
                dimensions: (dimensions > 0).then_some(dimensions as usize),
                distance_metric: distance_metric_str,
                weighted_metric_weights: weighted_metric_weights_slice,
                surface_mode: (!surface_mode.is_null()).then_some(
                    shared_parse::parse_c_str_or_default(
                        surface_mode,
                        shared_parse::DEFAULT_SURFACE_MODE,
                    ),
                ),
                return_se: return_se != 0,
                return_sorted: return_sorted != 0,
                cell: None,
                interpolation_vertices: None,
                boundary_degree_fallback: None,
                missing: Some(missing_str),
                retain_model: Some(retain_model != 0),
                ..Default::default()
            },
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let builder = if return_gradient != 0 {
            builder.return_gradient()
        } else {
            builder
        };

        Box::into_raw(Box::new(GoLoess {
            builder: Some(builder),
            cv_fractions: cv_fractions_vec,
            cv_method: Some(cv_method_str),
            cv_k: cv_k_usize,
            cv_seed: None,
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
pub unsafe extern "C" fn go_loess_set_cv_seed(ptr: *mut GoLoess, seed: c_ulonglong) {
    with_panic_void(|| {
        if !ptr.is_null() {
            unsafe { (*ptr).cv_seed = Some(seed) };
        }
    });
}

/// Fit the model.
///
/// # Safety
/// `ptr` must be a valid GoLoess pointer. `x_values` must be a valid array of length `x_n`
/// (= n_observations * dimensions), `y_values` must be a valid array of length `y_n` (= n_observations).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_loess_fit(
    ptr: *mut GoLoess,
    x_values: *const c_double,
    x_n: usize,
    y_values: *const c_double,
    y_n: usize,
    custom_weights: *const c_double,
    custom_weights_n: usize,
) -> GoLoessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        if x_values.is_null() || y_values.is_null() || x_n == 0 || y_n == 0 {
            return error_result(shared_parse::INVALID_DATA_INPUTS);
        }

        let loess = &mut *ptr;
        let x_slice = from_raw_parts(x_values, x_n);
        let y_slice = from_raw_parts(y_values, y_n);

        let cw = shared_parse::option_vec_from_ptr(custom_weights, custom_weights_n);

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
            if let Some(c) = loess.cell {
                builder = builder.cell(c);
            }
            if let Some(v) = loess.interpolation_vertices {
                builder = builder.interpolation_vertices(v);
            }
            if let Some(bdf) = loess.boundary_degree_fallback {
                builder = builder.boundary_degree_fallback(bdf);
            }

            let model = match shared_parse::build_batch(builder, cw) {
                Ok(m) => m,
                Err(e) => return error_result(&e.message),
            };
            match map_runtime_result(model.fit(x_slice, y_slice)) {
                Ok(r) => r.into(),
                Err(e) => e,
            }
        } else {
            error_result(shared_parse::MODEL_NOT_INITIALIZED)
        }
    })
}

/// Free model.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `go_loess_new` or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_loess_free(ptr: *mut GoLoess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

// Result of `go_predict()`. All arrays are allocated by Rust and must be freed via
// `go_predict_free_result`.
#[repr(C)]
pub struct GoPredictResult {
    /// Predicted y values, one per query point (length = n)
    pub y: *mut c_double,
    /// Number of query points
    pub n: usize,
    /// Standard errors (NULL if not requested)
    pub standard_errors: *mut c_double,
    /// Lower confidence bounds (NULL if not requested)
    pub confidence_lower: *mut c_double,
    /// Upper confidence bounds (NULL if not requested)
    pub confidence_upper: *mut c_double,
    /// Lower prediction bounds (NULL if not requested)
    pub prediction_lower: *mut c_double,
    /// Upper prediction bounds (NULL if not requested)
    pub prediction_upper: *mut c_double,
    /// Local fit's gradient at each query point, `dimensions` values per point,
    /// flattened (NULL if not requested)
    pub derivative: *mut c_double,
    /// Number of predictor dimensions (needed to know `derivative`'s true length,
    /// `dimensions * n`, when freeing it)
    pub dimensions: c_int,
    /// Error message (NULL if no error)
    pub error: *mut c_char,
}

impl Default for GoPredictResult {
    fn default() -> Self {
        GoPredictResult {
            y: ptr::null_mut(),
            n: 0,
            standard_errors: ptr::null_mut(),
            confidence_lower: ptr::null_mut(),
            confidence_upper: ptr::null_mut(),
            prediction_lower: ptr::null_mut(),
            prediction_upper: ptr::null_mut(),
            derivative: ptr::null_mut(),
            dimensions: 1,
            error: ptr::null_mut(),
        }
    }
}

fn predict_error_result(msg: &str) -> GoPredictResult {
    GoPredictResult {
        error: shared_parse::into_raw_error_c_string(msg),
        ..Default::default()
    }
}

/// Evaluate a fitted model (retained via `retain_model = 1`) at out-of-sample query
/// points not in the training set.
///
/// # Safety
/// `handle` must be a valid pointer returned via `GoLoessResult::predict_handle`.
/// `new_x` must be a valid array of length `new_x_len`. `extrapolation` must be a
/// valid null-terminated string or null (defaults to "clamp").
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn go_predict(
    handle: *mut GoPredictHandle,
    new_x: *const c_double,
    new_x_len: usize,
    return_se: c_int,
    confidence_level: c_double,
    prediction_level: c_double,
    return_derivative: c_int,
    extrapolation: *const c_char,
    max_extrapolation_distance: c_double,
    max_neighbor_distance: c_double,
) -> GoPredictResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return predict_error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        if new_x.is_null() || new_x_len == 0 {
            return predict_error_result(shared_parse::INVALID_DATA_INPUTS);
        }
        let new_x_slice = from_raw_parts(new_x, new_x_len);
        let extrapolation_str = (!extrapolation.is_null())
            .then_some(shared_parse::parse_c_str_or_default(extrapolation, "clamp"));

        let state = &(*handle).state;
        let output = match shared_parse::run_predict_state(
            state,
            new_x_slice,
            shared_parse::PredictOptionSet {
                return_se: return_se != 0,
                confidence_level: (!confidence_level.is_nan()).then_some(confidence_level),
                prediction_level: (!prediction_level.is_nan()).then_some(prediction_level),
                return_derivative: return_derivative != 0,
                extrapolation: extrapolation_str,
                max_extrapolation_distance: (!max_extrapolation_distance.is_nan())
                    .then_some(max_extrapolation_distance),
                max_neighbor_distance: (!max_neighbor_distance.is_nan())
                    .then_some(max_neighbor_distance),
            },
        ) {
            Ok(o) => o,
            Err(e) => return predict_error_result(&e.message),
        };

        GoPredictResult {
            n: output.y.len(),
            y: shared_parse::vec_to_raw_ptr(output.y),
            standard_errors: shared_parse::opt_vec_to_raw_ptr(output.standard_errors),
            confidence_lower: shared_parse::opt_vec_to_raw_ptr(output.confidence_lower),
            confidence_upper: shared_parse::opt_vec_to_raw_ptr(output.confidence_upper),
            prediction_lower: shared_parse::opt_vec_to_raw_ptr(output.prediction_lower),
            prediction_upper: shared_parse::opt_vec_to_raw_ptr(output.prediction_upper),
            derivative: shared_parse::opt_vec_to_raw_ptr(output.derivative),
            dimensions: state.dimensions as c_int,
            error: ptr::null_mut(),
        }
    })) {
        Ok(v) => v,
        Err(_) => predict_error_result(shared_parse::panic_fallback_message()),
    }
}

/// Free a GoPredictResult's heap-allocated buffers.
///
/// # Safety
/// `result` must be a valid pointer to a GoPredictResult struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_predict_free_result(result: *mut GoPredictResult) {
    with_panic_void(|| {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        let n = r.n;
        shared_parse::free_raw_f64_buffer(r.y, n);
        shared_parse::free_raw_f64_buffer(r.standard_errors, n);
        shared_parse::free_raw_f64_buffer(r.confidence_lower, n);
        shared_parse::free_raw_f64_buffer(r.confidence_upper, n);
        shared_parse::free_raw_f64_buffer(r.prediction_lower, n);
        shared_parse::free_raw_f64_buffer(r.prediction_upper, n);
        shared_parse::free_raw_f64_buffer(r.derivative, n * r.dimensions.max(1) as usize);
        shared_parse::free_raw_c_string(r.error);
    });
}

/// Free a `GoPredictHandle` returned via `GoLoessResult::predict_handle`.
///
/// # Safety
/// `ptr` must be a valid pointer returned via `GoLoessResult::predict_handle`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_predict_handle_free(ptr: *mut GoPredictHandle) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Create a new Streaming Loess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_streaming_new(
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
    // Advanced options
    cell: c_double,
    interpolation_vertices: c_int,
    boundary_degree_fallback: c_int,
    weighted_metric_weights: *const c_double,
    weighted_metric_weights_len: usize,
    missing: *const c_char,
    return_gradient: c_int,
    confidence_intervals: c_double,
    prediction_intervals: c_double,
    return_se: c_int,
) -> *mut GoStreamingLoess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = shared_parse::parse_c_str_or_default(
            weight_function,
            shared_parse::DEFAULT_WEIGHT_FUNCTION,
        );
        let rm_str = shared_parse::parse_c_str_or_default(
            robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm_str = shared_parse::parse_c_str_or_default(
            scaling_method,
            shared_parse::DEFAULT_SCALING_METHOD,
        );
        let bp_str = shared_parse::parse_c_str_or_default(
            boundary_policy,
            shared_parse::DEFAULT_BOUNDARY_POLICY,
        );
        let zwf_str = shared_parse::parse_c_str_or_default(
            zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );
        let ms_str = shared_parse::parse_c_str_or_default(
            merge_strategy,
            shared_parse::DEFAULT_MERGE_STRATEGY,
        );
        let missing_str =
            shared_parse::parse_c_str_or_default(missing, shared_parse::DEFAULT_MISSING_POLICY);

        let chunk_size = match shared_parse::require_positive_usize("chunk_size", chunk_size) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };
        let degree_str = (!degree.is_null()).then_some(shared_parse::parse_c_str_or_default(
            degree,
            shared_parse::DEFAULT_DEGREE,
        ));
        let surface_mode_str = (!surface_mode.is_null()).then_some(
            shared_parse::parse_c_str_or_default(surface_mode, shared_parse::DEFAULT_SURFACE_MODE),
        );
        let distance_metric_str =
            (!distance_metric.is_null()).then_some(shared_parse::parse_c_str_or_default(
                distance_metric,
                shared_parse::DEFAULT_DISTANCE_METRIC,
            ));
        let weighted_metric_weights_slice = shared_parse::option_slice_from_ptr(
            weighted_metric_weights,
            weighted_metric_weights_len,
        );

        let (builder, _) = match shared_parse::apply_builder_options(
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
                degree: degree_str,
                dimensions: (dimensions > 0).then_some(dimensions as usize),
                distance_metric: distance_metric_str,
                weighted_metric_weights: weighted_metric_weights_slice,
                surface_mode: surface_mode_str,
                return_se: return_se != 0,
                cell: (!cell.is_nan()).then_some(cell),
                interpolation_vertices: (interpolation_vertices > 0)
                    .then_some(interpolation_vertices as usize),
                boundary_degree_fallback: (boundary_degree_fallback >= 0)
                    .then_some(boundary_degree_fallback != 0),
                missing: Some(missing_str),
                ..Default::default()
            },
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let builder = if return_gradient != 0 {
            builder.return_gradient()
        } else {
            builder
        };

        let model = match shared_parse::build_streaming(
            builder,
            Some(chunk_size),
            (overlap >= 0).then_some(overlap as usize),
            Some(ms_str),
        ) {
            Ok(m) => m,
            Err(e) => return null_with_error(&e.message),
        };

        Box::into_raw(Box::new(GoStreamingLoess { model: Some(model) }))
    })
}

/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be valid. `x_values` must be a valid array of length `x_n` (= n_observations * dimensions),
/// `y_values` must be a valid array of length `y_n` (= n_observations).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_streaming_process(
    ptr: *mut GoStreamingLoess,
    x_values: *const c_double,
    x_n: usize,
    y_values: *const c_double,
    y_n: usize,
) -> GoLoessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        let loess = &mut *ptr;
        if x_values.is_null() || y_values.is_null() || x_n == 0 || y_n == 0 {
            return error_result(shared_parse::INVALID_DATA_INPUTS);
        }
        let x_slice = from_raw_parts(x_values, x_n);
        let y_slice = from_raw_parts(y_values, y_n);

        if let Some(model) = &mut loess.model {
            match map_runtime_result(model.process_chunk(x_slice, y_slice)) {
                Ok(r) => r.into(),
                Err(e) => e,
            }
        } else {
            error_result(shared_parse::MODEL_NOT_INITIALIZED)
        }
    })
}

/// Finalize the streaming process.
///
/// # Safety
/// `ptr` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_streaming_finalize(ptr: *mut GoStreamingLoess) -> GoLoessResult {
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
            error_result(shared_parse::MODEL_NOT_INITIALIZED)
        }
    })
}

/// Free model.
///
/// # Safety
/// `ptr` must be valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_streaming_free(ptr: *mut GoStreamingLoess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Create a new Online Loess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_online_new(
    fraction: c_double,
    iterations: c_int,
    weight_function: *const c_char,
    robustness_method: *const c_char,
    scaling_method: *const c_char,
    boundary_policy: *const c_char,
    return_robustness_weights: c_int,
    zero_weight_fallback: *const c_char,
    auto_converge: c_double,
    // opts
    window_capacity: c_int,
    min_points: c_int,
    update_mode: *const c_char,
    // LOESS-specific options
    degree: *const c_char,
    dimensions: c_int,
    distance_metric: *const c_char,
    surface_mode: *const c_char,
    // Advanced options
    cell: c_double,
    interpolation_vertices: c_int,
    boundary_degree_fallback: c_int,
    weighted_metric_weights: *const c_double,
    weighted_metric_weights_len: usize,
    missing: *const c_char,
    return_gradient: c_int,
    confidence_intervals: c_double,
    prediction_intervals: c_double,
    return_se: c_int,
) -> *mut GoOnlineLoess {
    with_panic_ptr(|| {
        clear_last_error();
        let wf_str = shared_parse::parse_c_str_or_default(
            weight_function,
            shared_parse::DEFAULT_WEIGHT_FUNCTION,
        );
        let rm_str = shared_parse::parse_c_str_or_default(
            robustness_method,
            shared_parse::DEFAULT_ROBUSTNESS_METHOD,
        );
        let sm_str = shared_parse::parse_c_str_or_default(
            scaling_method,
            shared_parse::DEFAULT_SCALING_METHOD,
        );
        let bp_str = shared_parse::parse_c_str_or_default(
            boundary_policy,
            shared_parse::DEFAULT_BOUNDARY_POLICY,
        );
        let zwf_str = shared_parse::parse_c_str_or_default(
            zero_weight_fallback,
            shared_parse::DEFAULT_ZERO_WEIGHT_FALLBACK,
        );
        let um_str =
            shared_parse::parse_c_str_or_default(update_mode, shared_parse::DEFAULT_UPDATE_MODE);
        let missing_str =
            shared_parse::parse_c_str_or_default(missing, shared_parse::DEFAULT_MISSING_POLICY);

        let window_capacity =
            match shared_parse::require_positive_usize("window_capacity", window_capacity) {
                Ok(v) => v,
                Err(e) => return null_with_error(&e),
            };
        let min_points = match shared_parse::require_positive_usize("min_points", min_points) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let configured_dimensions = dimensions.max(1) as usize;
        let degree_str = (!degree.is_null()).then_some(shared_parse::parse_c_str_or_default(
            degree,
            shared_parse::DEFAULT_DEGREE,
        ));
        let surface_mode_str = (!surface_mode.is_null()).then_some(
            shared_parse::parse_c_str_or_default(surface_mode, shared_parse::DEFAULT_SURFACE_MODE),
        );
        let distance_metric_str =
            (!distance_metric.is_null()).then_some(shared_parse::parse_c_str_or_default(
                distance_metric,
                shared_parse::DEFAULT_DISTANCE_METRIC,
            ));
        let weighted_metric_weights_slice = shared_parse::option_slice_from_ptr(
            weighted_metric_weights,
            weighted_metric_weights_len,
        );

        let (builder, _) = match shared_parse::apply_builder_options(
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
                return_residuals: false,
                return_robustness_weights: return_robustness_weights != 0,
                return_diagnostics: false,
                confidence_intervals: (!confidence_intervals.is_nan())
                    .then_some(confidence_intervals),
                prediction_intervals: (!prediction_intervals.is_nan())
                    .then_some(prediction_intervals),
                parallel: None,
                degree: degree_str,
                dimensions: (dimensions > 0).then_some(configured_dimensions),
                distance_metric: distance_metric_str,
                weighted_metric_weights: weighted_metric_weights_slice,
                surface_mode: surface_mode_str,
                return_se: return_se != 0,
                cell: (!cell.is_nan()).then_some(cell),
                interpolation_vertices: (interpolation_vertices > 0)
                    .then_some(interpolation_vertices as usize),
                boundary_degree_fallback: (boundary_degree_fallback >= 0)
                    .then_some(boundary_degree_fallback != 0),
                missing: Some(missing_str),
                ..Default::default()
            },
        ) {
            Ok(v) => v,
            Err(e) => return null_with_error(&e),
        };

        let builder = if return_gradient != 0 {
            builder.return_gradient()
        } else {
            builder
        };

        let model = match shared_parse::build_online(
            builder,
            Some(window_capacity),
            Some(min_points),
            Some(um_str),
        ) {
            Ok(m) => m,
            Err(e) => return null_with_error(&e.message),
        };

        Box::into_raw(Box::new(GoOnlineLoess {
            model: Some(model),
            dimensions: configured_dimensions,
        }))
    })
}

/// Add a single point to the model and return its smoothed value.
/// `has_value = 0` in the result means the window is still filling.
///
/// # Safety
/// `ptr` must be a valid `GoOnlineLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_online_add_point(
    ptr: *mut GoOnlineLoess,
    x: c_double,
    y: c_double,
) -> GoOnlineOutput {
    let make_error = |msg: &str| -> GoOnlineOutput {
        GoOnlineOutput {
            error: shared_parse::to_cstring_lossy(msg).into_raw(),
            ..GoOnlineOutput::default()
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
                Ok(None) => GoOnlineOutput::default(),
                Ok(Some(o)) => {
                    let (
                        standard_error,
                        residual,
                        robustness_weight,
                        iterations_used,
                        confidence_lower,
                        confidence_upper,
                        prediction_lower,
                        prediction_upper,
                    ) = shared_parse::extract_online_output(&o);
                    GoOnlineOutput {
                        has_value: 1,
                        y: o.y,
                        standard_error,
                        residual,
                        robustness_weight,
                        iterations_used,
                        confidence_lower,
                        confidence_upper,
                        prediction_lower,
                        prediction_upper,
                        gradient: shared_parse::opt_vec_to_raw_ptr(o.gradient),
                        dimensions: loess.dimensions as c_int,
                        error: ptr::null_mut(),
                    }
                }
            }
        } else {
            make_error(shared_parse::MODEL_NOT_INITIALIZED)
        }
    })) {
        Ok(v) => v,
        Err(_) => make_error(shared_parse::panic_fallback_message()),
    }
}

/// Free the error field in a GoOnlineOutput (call only when error != NULL).
///
/// # Safety
/// `output` must be a valid pointer and `output->error` must have been allocated by Rust.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_online_free_output(output: *mut GoOnlineOutput) {
    with_panic_void(|| {
        if !output.is_null() {
            let out = unsafe { &mut *output };
            shared_parse::free_raw_c_string(out.error);
            out.error = ptr::null_mut();
            shared_parse::free_raw_f64_buffer(out.gradient, out.dimensions.max(1) as usize);
            out.gradient = ptr::null_mut();
        }
    });
}

/// Free model.
///
/// # Safety
/// `ptr` must be valid or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_online_free(ptr: *mut GoOnlineLoess) {
    with_panic_void(|| {
        if !ptr.is_null() {
            let _ = Box::from_raw(ptr);
        }
    });
}

/// Free a GoLoessResult.
///
/// # Safety
/// `result` must be a valid pointer to a GoLoessResult struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn go_loess_free_result(result: *mut GoLoessResult) {
    with_panic_void(|| {
        if result.is_null() {
            return;
        }

        let r = &mut *result;
        let n = r.n;
        let cv_n = r.cv_scores_len;

        shared_parse::free_raw_f64_buffer(r.x, n);
        shared_parse::free_raw_f64_buffer(r.y, n);
        shared_parse::free_raw_f64_buffer(r.standard_errors, n);
        shared_parse::free_raw_f64_buffer(r.confidence_lower, n);
        shared_parse::free_raw_f64_buffer(r.confidence_upper, n);
        shared_parse::free_raw_f64_buffer(r.prediction_lower, n);
        shared_parse::free_raw_f64_buffer(r.prediction_upper, n);
        shared_parse::free_raw_f64_buffer(r.residuals, n);
        shared_parse::free_raw_f64_buffer(r.robustness_weights, n);
        shared_parse::free_raw_f64_buffer(r.gradient, n * r.dimensions.max(1) as usize);
        shared_parse::free_raw_f64_buffer(r.leverage, n);
        shared_parse::free_raw_f64_buffer(r.cv_scores, cv_n);
        shared_parse::free_raw_c_string(r.error);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch_model() -> GoLoess {
        GoLoess {
            builder: Some(LoessBuilder::<f64>::new()),
            cv_fractions: Some(vec![0.67]),
            cv_method: Some("kfold".into()),
            cv_k: 1,
            cell: None,
            interpolation_vertices: None,
            boundary_degree_fallback: None,
            cv_seed: None,
        }
    }

    #[test]
    fn cv_seed_preserves_high_bits() {
        let mut model = batch_model();
        let seed = (1_u64 << 48) | 17;
        unsafe { go_loess_set_cv_seed(&mut model, seed) };
        assert_eq!(model.cv_seed, Some(seed));
    }

    #[test]
    fn fit_does_not_coerce_single_fold() {
        let mut model = batch_model();
        let predictors: Vec<f64> = (0..20).map(f64::from).collect();
        let observations: Vec<f64> = predictors.iter().map(|value| value.sin()).collect();
        unsafe {
            let mut result = go_loess_fit(
                &mut model,
                predictors.as_ptr(),
                predictors.len(),
                observations.as_ptr(),
                observations.len(),
                ptr::null(),
                0,
            );
            let message = shared_parse::parse_c_str_or_default(result.error, "").to_owned();
            go_loess_free_result(&mut result);
            assert!(message.to_lowercase().contains("fold"), "{message}");
        }
    }
}
