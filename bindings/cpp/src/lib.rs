//! C/C++ bindings for fastLoess.
//!
//! Provides C access to the fastLoess Rust library via C FFI.
//! A C++ wrapper header (fastloess.hpp) provides idiomatic C++ usage.

#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice::from_raw_parts;
use std::sync::Arc;

use fastLoess::internals::adapters::online::ParallelOnlineLoess;
use fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use fastLoess::internals::api::LoessBuilder;
use fastLoess::internals::binding_support as shared_parse;
use fastLoess::prelude::{IntervalsBuilder, LoessResult, Predict};

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
fn map_runtime_result<T, E: ToString>(result: Result<T, E>) -> Result<T, CppLoessResult> {
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

/// Returns the crate version as a static, null-terminated C string.
#[unsafe(no_mangle)]
pub extern "C" fn cpp_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

// Per-point result from an online update, passed across the FFI boundary.
// has_value = 1 means the window is ready and smoothed is valid; 0 means the
// window is still filling (caller should treat it as no output yet).
// Non-computed optional fields use f64::NAN (for floats) or -1 (for int).
// error = NULL if no error, otherwise points to a null-terminated error string.
#[repr(C)]
pub struct CppOnlineOutput {
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
    /// Local fit gradient (`dimensions` values) for the latest point (NULL if
    /// not computed)
    pub gradient: *mut c_double,
    pub gradient_len: usize,
    pub error: *mut c_char, // NULL if no error
}

#[repr(C)]
pub struct CppOnlineDiagnostics {
    pub has_value: c_int,
    pub rmse: c_double,
    pub mae: c_double,
    pub r_squared: c_double,
    pub aic: c_double,
    pub aicc: c_double,
    pub effective_df: c_double,
    pub residual_sd: c_double,
    pub error: *mut c_char,
}

impl Default for CppOnlineDiagnostics {
    fn default() -> Self {
        Self {
            has_value: 0,
            rmse: f64::NAN,
            mae: f64::NAN,
            r_squared: f64::NAN,
            aic: f64::NAN,
            aicc: f64::NAN,
            effective_df: f64::NAN,
            residual_sd: f64::NAN,
            error: ptr::null_mut(),
        }
    }
}

// Result struct that can be passed across FFI boundary.
// All arrays are allocated by Rust and must be freed by Rust.
#[repr(C)]
pub struct CppLoessResult {
    /// x values, in input order (flattened, length = n * dimensions)
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
    /// Per-point local fit gradient, flattened, `dimensions` values per point
    /// (NULL if not computed)
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

    /// Opaque handle for `cpp_predict()`, non-NULL only if `retain_model` was set to 1.
    /// Must eventually be freed via `cpp_predict_handle_free`.
    pub predict_handle: *mut CppPredictHandle,

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
fn error_result(msg: &str) -> CppLoessResult {
    CppLoessResult {
        error: shared_parse::into_raw_error_c_string(msg),
        ..Default::default()
    }
}

impl From<LoessResult<f64>> for CppLoessResult {
    fn from(mut result: LoessResult<f64>) -> Self {
        let gradient = shared_parse::opt_vec_to_raw_ptr(result.gradient.take());
        let p = shared_parse::extract_ffi_loess_result(result);
        CppLoessResult {
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
            gradient,
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
                .map(|state| Box::into_raw(Box::new(CppPredictHandle { state })))
                .unwrap_or(ptr::null_mut()),
            error: ptr::null_mut(),
        }
    }
}

impl Default for CppOnlineOutput {
    fn default() -> Self {
        CppOnlineOutput {
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
            gradient_len: 0,
            error: ptr::null_mut(),
        }
    }
}

// Opaque handle to a batch Loess model.
pub struct CppLoess {
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
pub struct CppStreamingLoess {
    model: Option<ParallelStreamingLoess<f64>>,
}

// Opaque handle to an online Loess model.
pub struct CppOnlineLoess {
    model: Option<ParallelOnlineLoess<f64>>,
    dimensions: usize,
}

// Opaque handle retained by `cpp_loess_fit` (in `CppLoessResult::predict_handle`, if
// `retain_model` was set to 1), enabling `cpp_predict()`. Wraps just the lightweight
// `Arc<PredictState<f64>>` extracted from the fitted model, not the whole result.
pub struct CppPredictHandle {
    state: Arc<shared_parse::PredictState<f64>>,
}

fn setter_unsupported_eager_lifecycle(name: &str) {
    set_last_error(&shared_parse::setter_unsupported_eager_message(name));
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
    return_gradient: c_int,
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
) -> *mut CppLoess {
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
        if dimensions <= 0 || interpolation_vertices < 0 {
            return null_with_error(
                "dimensions must be positive and interpolation_vertices must be non-negative",
            );
        }
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

        let model = CppLoess {
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
        };
        if let Err(error) = configured_batch_builder(&model)
            .and_then(|builder| shared_parse::build_batch(builder, None))
        {
            return null_with_error(&error.message);
        }
        Box::into_raw(Box::new(model))
    })
}

/// Set CV seed for reproducible K-fold splits.
///
/// # Safety
/// ptr must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_set_cv_seed(ptr: *mut CppLoess, seed: u64) {
    with_panic_void(|| {
        if !ptr.is_null() {
            unsafe { (*ptr).cv_seed = Some(seed) };
        }
    });
}

// Legacy model setters retained for ABI compatibility.
// Streaming/online models are eagerly initialized at construction, so these setters
// are unsupported and now report this through the last-error channel.

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
    vertices: usize,
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

/// Set cell tuning parameter for a model.
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

/// Set number of interpolation vertices for a model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_set_interpolation_vertices(
    ptr: *mut CppOnlineLoess,
    vertices: usize,
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

/// Enable or disable boundary degree fallback for a model.
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

/// Set confidence interval level for a model.
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

/// Set prediction interval level for a model.
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
/// `ptr` must be a valid CppLoess pointer. `x_values` must be a valid array of length `x_n`
/// (= n_observations * dimensions), `y_values` must be a valid array of length `y_n` (= n_observations).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_loess_fit(
    ptr: *mut CppLoess,
    x_values: *const c_double,
    x_n: usize,
    y_values: *const c_double,
    y_n: usize,
    custom_weights: *const c_double,
    custom_weights_n: usize,
) -> CppLoessResult {
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

        if loess.builder.is_some() {
            let builder = match configured_batch_builder(loess) {
                Ok(builder) => builder,
                Err(error) => return error_result_from(error),
            };
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

fn configured_batch_builder(
    model: &CppLoess,
) -> Result<LoessBuilder<f64>, shared_parse::BindingError> {
    let builder = model.builder.clone().ok_or_else(|| {
        shared_parse::BindingError::invalid_arg(shared_parse::MODEL_NOT_INITIALIZED)
    })?;
    let mut builder = shared_parse::map_invalid_arg(shared_parse::apply_cross_validation(
        builder,
        model.cv_fractions.as_deref(),
        model.cv_method.as_deref(),
        Some(model.cv_k),
        model.cv_seed,
    ))?;
    if let Some(cell) = model.cell {
        builder = builder.cell(cell);
    }
    if let Some(vertices) = model.interpolation_vertices {
        builder = builder.interpolation_vertices(vertices);
    }
    if let Some(fallback) = model.boundary_degree_fallback {
        builder = builder.boundary_degree_fallback(fallback);
    }
    Ok(builder)
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

// Result of `cpp_predict()`. All arrays are allocated by Rust and must be freed via
// `cpp_predict_free_result`.
#[repr(C)]
pub struct CppPredictResult {
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

impl Default for CppPredictResult {
    fn default() -> Self {
        CppPredictResult {
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

fn predict_error_result(msg: &str) -> CppPredictResult {
    CppPredictResult {
        error: shared_parse::into_raw_error_c_string(msg),
        ..Default::default()
    }
}

/// Evaluate a fitted model (retained via `retain_model = 1`) at out-of-sample query
/// points not in the training set.
///
/// # Safety
/// `handle` must be a valid pointer returned via `CppLoessResult::predict_handle`.
/// `new_x` must be a valid array of length `new_x_len`. `extrapolation` must be a
/// valid null-terminated string or null (defaults to "clamp").
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cpp_predict(
    handle: *mut CppPredictHandle,
    new_x: *const c_double,
    new_x_len: usize,
    return_se: c_int,
    confidence_level: c_double,
    prediction_level: c_double,
    return_derivative: c_int,
    extrapolation: *const c_char,
    max_extrapolation_distance: c_double,
    max_neighbor_distance: c_double,
) -> CppPredictResult {
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

        CppPredictResult {
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

/// Free a CppPredictResult's heap-allocated buffers.
///
/// # Safety
/// `result` must be a valid pointer to a CppPredictResult struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_predict_free_result(result: *mut CppPredictResult) {
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
        *r = CppPredictResult::default();
    });
}

/// Free a `CppPredictHandle` returned via `CppLoessResult::predict_handle`.
///
/// # Safety
/// `ptr` must be a valid pointer returned via `CppLoessResult::predict_handle`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_predict_handle_free(ptr: *mut CppPredictHandle) {
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
    return_gradient: c_int,
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
    confidence_intervals: c_double,
    prediction_intervals: c_double,
    return_se: c_int,
) -> *mut CppStreamingLoess {
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
        if dimensions <= 0 || interpolation_vertices < 0 {
            return null_with_error(
                "dimensions must be positive and interpolation_vertices must be non-negative",
            );
        }
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

        Box::into_raw(Box::new(CppStreamingLoess { model: Some(model) }))
    })
}

/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be valid. `x_values` must be a valid array of length `x_n` (= n_observations * dimensions),
/// `y_values` must be a valid array of length `y_n` (= n_observations).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_process(
    ptr: *mut CppStreamingLoess,
    x_values: *const c_double,
    x_n: usize,
    y_values: *const c_double,
    y_n: usize,
) -> CppLoessResult {
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_streaming_process_weighted(
    ptr: *mut CppStreamingLoess,
    x_values: *const c_double,
    x_n: usize,
    y_values: *const c_double,
    y_n: usize,
    weights: *const c_double,
    weights_n: usize,
) -> CppLoessResult {
    with_panic_result(|| {
        if ptr.is_null() {
            return error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        if x_values.is_null() || y_values.is_null() || weights.is_null() || x_n == 0 || y_n == 0 {
            return error_result(shared_parse::INVALID_DATA_INPUTS);
        }
        let model = unsafe { &mut *ptr };
        let x = unsafe { from_raw_parts(x_values, x_n) };
        let y = unsafe { from_raw_parts(y_values, y_n) };
        let weights = unsafe { from_raw_parts(weights, weights_n) };
        match &mut model.model {
            Some(model) => match map_runtime_result(model.process_chunk_weighted(x, y, weights)) {
                Ok(result) => result.into(),
                Err(error) => error,
            },
            None => error_result(shared_parse::MODEL_NOT_INITIALIZED),
        }
    })
}

/// Finalize the streaming process.
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
            error_result(shared_parse::MODEL_NOT_INITIALIZED)
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
    return_gradient: c_int,
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
    confidence_intervals: c_double,
    prediction_intervals: c_double,
    return_se: c_int,
) -> *mut CppOnlineLoess {
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

        let configured_dimensions =
            match shared_parse::require_positive_usize("dimensions", dimensions) {
                Ok(value) => value,
                Err(error) => return null_with_error(&error),
            };
        if interpolation_vertices < 0 {
            return null_with_error("interpolation_vertices must be non-negative");
        }
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

        Box::into_raw(Box::new(CppOnlineLoess {
            model: Some(model),
            dimensions: configured_dimensions,
        }))
    })
}

fn online_error_output(msg: &str) -> CppOnlineOutput {
    CppOnlineOutput {
        error: shared_parse::to_cstring_lossy(msg).into_raw(),
        ..CppOnlineOutput::default()
    }
}

unsafe fn online_add_point_impl(
    ptr: *mut CppOnlineLoess,
    x: &[c_double],
    y: c_double,
    weight: c_double,
) -> CppOnlineOutput {
    if ptr.is_null() {
        return online_error_output(shared_parse::MODEL_POINTER_IS_NULL);
    }
    let loess = unsafe { &mut *ptr };
    let Some(model) = &mut loess.model else {
        return online_error_output(shared_parse::MODEL_NOT_INITIALIZED);
    };
    match model.add_point_weighted(x, y, weight) {
        Err(error) => online_error_output(&error.to_string()),
        Ok(None) => CppOnlineOutput::default(),
        Ok(Some(output)) => {
            let (
                standard_error,
                residual,
                robustness_weight,
                iterations_used,
                confidence_lower,
                confidence_upper,
                prediction_lower,
                prediction_upper,
            ) = shared_parse::extract_online_output(&output);
            let gradient_len = output
                .gradient
                .as_ref()
                .map(|values| values.len())
                .unwrap_or(0);
            let gradient = shared_parse::opt_vec_to_raw_ptr(output.gradient);
            CppOnlineOutput {
                has_value: 1,
                y: output.y,
                standard_error,
                residual,
                robustness_weight,
                iterations_used,
                confidence_lower,
                confidence_upper,
                prediction_lower,
                prediction_upper,
                gradient,
                gradient_len,
                error: ptr::null_mut(),
            }
        }
    }
}

/// Add a single one-dimensional point to the model.
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
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        online_add_point_impl(ptr, &[x], y, 1.0)
    })) {
        Ok(output) => output,
        Err(_) => online_error_output(shared_parse::panic_fallback_message()),
    }
}

/// Add one point with one coordinate per configured predictor dimension.
/// `has_value = 0` in the result means the window is still filling.
///
/// # Safety
/// `ptr` must be valid. If `x_n` is nonzero, `x_values` must point to `x_n` valid values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_add_point_nd(
    ptr: *mut CppOnlineLoess,
    x_values: *const c_double,
    x_n: usize,
    y: c_double,
) -> CppOnlineOutput {
    match catch_unwind(AssertUnwindSafe(|| {
        if x_values.is_null() || x_n == 0 {
            return online_error_output(shared_parse::INVALID_DATA_INPUTS);
        }
        let x = unsafe { from_raw_parts(x_values, x_n) };
        unsafe { online_add_point_impl(ptr, x, y, 1.0) }
    })) {
        Ok(output) => output,
        Err(_) => online_error_output(shared_parse::panic_fallback_message()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_add_point_weighted(
    ptr: *mut CppOnlineLoess,
    x: c_double,
    y: c_double,
    weight: c_double,
) -> CppOnlineOutput {
    match catch_unwind(AssertUnwindSafe(|| unsafe {
        online_add_point_impl(ptr, &[x], y, weight)
    })) {
        Ok(output) => output,
        Err(_) => online_error_output(shared_parse::panic_fallback_message()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_add_point_nd_weighted(
    ptr: *mut CppOnlineLoess,
    x_values: *const c_double,
    x_n: usize,
    y: c_double,
    weight: c_double,
) -> CppOnlineOutput {
    match catch_unwind(AssertUnwindSafe(|| {
        if x_values.is_null() || x_n == 0 {
            return online_error_output(shared_parse::INVALID_DATA_INPUTS);
        }
        let x = unsafe { from_raw_parts(x_values, x_n) };
        unsafe { online_add_point_impl(ptr, x, y, weight) }
    })) {
        Ok(output) => output,
        Err(_) => online_error_output(shared_parse::panic_fallback_message()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_window_diagnostics(
    ptr: *mut CppOnlineLoess,
) -> CppOnlineDiagnostics {
    match catch_unwind(AssertUnwindSafe(|| {
        if ptr.is_null() {
            return CppOnlineDiagnostics {
                error: shared_parse::to_cstring_lossy(shared_parse::MODEL_POINTER_IS_NULL)
                    .into_raw(),
                ..CppOnlineDiagnostics::default()
            };
        }
        let processor = unsafe { &*ptr };
        let model = match &processor.model {
            Some(model) => model,
            None => {
                return CppOnlineDiagnostics {
                    error: shared_parse::to_cstring_lossy(shared_parse::MODEL_NOT_INITIALIZED)
                        .into_raw(),
                    ..CppOnlineDiagnostics::default()
                };
            }
        };
        match model.window_diagnostics() {
            Ok(None) => CppOnlineDiagnostics::default(),
            Ok(Some(d)) => CppOnlineDiagnostics {
                has_value: 1,
                rmse: d.rmse,
                mae: d.mae,
                r_squared: d.r_squared,
                aic: d.aic.unwrap_or(f64::NAN),
                aicc: d.aicc.unwrap_or(f64::NAN),
                effective_df: d.effective_df.unwrap_or(f64::NAN),
                residual_sd: d.residual_sd,
                error: ptr::null_mut(),
            },
            Err(error) => CppOnlineDiagnostics {
                error: shared_parse::to_cstring_lossy(&error.to_string()).into_raw(),
                ..CppOnlineDiagnostics::default()
            },
        }
    })) {
        Ok(result) => result,
        Err(_) => CppOnlineDiagnostics {
            error: shared_parse::to_cstring_lossy(shared_parse::panic_fallback_message())
                .into_raw(),
            ..CppOnlineDiagnostics::default()
        },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cpp_online_free_diagnostics(result: *mut CppOnlineDiagnostics) {
    with_panic_void(|| {
        if !result.is_null() {
            let result = unsafe { &mut *result };
            shared_parse::free_raw_c_string(result.error);
            result.error = ptr::null_mut();
        }
    });
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cpp_online_predict_window(
    ptr: *mut CppOnlineLoess,
    new_x: *const c_double,
    new_x_len: usize,
    return_se: c_int,
    confidence_level: c_double,
    prediction_level: c_double,
    return_derivative: c_int,
    extrapolation: *const c_char,
    max_extrapolation_distance: c_double,
    max_neighbor_distance: c_double,
) -> CppPredictResult {
    match catch_unwind(AssertUnwindSafe(|| {
        if ptr.is_null() {
            return predict_error_result(shared_parse::MODEL_POINTER_IS_NULL);
        }
        if new_x.is_null() || new_x_len == 0 {
            return predict_error_result(shared_parse::INVALID_DATA_INPUTS);
        }
        let model = unsafe { &*ptr };
        let processor = match &model.model {
            Some(processor) => processor,
            None => return predict_error_result(shared_parse::MODEL_NOT_INITIALIZED),
        };
        let new_x = unsafe { from_raw_parts(new_x, new_x_len) };
        let extrapolation = unsafe { shared_parse::parse_c_str_or_default(extrapolation, "clamp") };
        let mut intervals = IntervalsBuilder::new();
        if !confidence_level.is_nan() {
            intervals = intervals.confidence(confidence_level);
        }
        if !prediction_level.is_nan() {
            intervals = intervals.prediction(prediction_level);
        }
        let mut builder = Predict::new()
            .intervals(intervals)
            .extrapolation(extrapolation);
        if return_se != 0 {
            builder = builder.return_se();
        }
        if return_derivative != 0 {
            builder = builder.return_derivative();
        }
        if !max_extrapolation_distance.is_nan() {
            builder = builder.max_extrapolation_distance(max_extrapolation_distance);
        }
        if !max_neighbor_distance.is_nan() {
            builder = builder.max_neighbor_distance(max_neighbor_distance);
        }
        let query = match builder.build() {
            Ok(query) => query,
            Err(error) => return predict_error_result(&error.to_string()),
        };
        let output = match processor.predict_window(new_x, &query) {
            Ok(output) => output,
            Err(error) => return predict_error_result(&error.to_string()),
        };
        CppPredictResult {
            n: output.y.len(),
            y: shared_parse::vec_to_raw_ptr(output.y),
            standard_errors: shared_parse::opt_vec_to_raw_ptr(output.standard_errors),
            confidence_lower: shared_parse::opt_vec_to_raw_ptr(output.confidence_lower),
            confidence_upper: shared_parse::opt_vec_to_raw_ptr(output.confidence_upper),
            prediction_lower: shared_parse::opt_vec_to_raw_ptr(output.prediction_lower),
            prediction_upper: shared_parse::opt_vec_to_raw_ptr(output.prediction_upper),
            derivative: shared_parse::opt_vec_to_raw_ptr(output.derivative),
            dimensions: model.dimensions as c_int,
            error: ptr::null_mut(),
        }
    })) {
        Ok(result) => result,
        Err(_) => predict_error_result(shared_parse::panic_fallback_message()),
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
            shared_parse::free_raw_f64_buffer(out.gradient, out.gradient_len);
            out.gradient = ptr::null_mut();
            out.gradient_len = 0;
            shared_parse::free_raw_c_string(out.error);
            out.error = ptr::null_mut();
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
        let n = r.n;
        let cv_n = r.cv_scores_len;

        shared_parse::free_raw_f64_buffer(r.x, n * r.dimensions.max(1) as usize);
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
        if !r.predict_handle.is_null() {
            drop(Box::from_raw(r.predict_handle));
        }
        *r = CppLoessResult::default();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cv_seed_preserves_high_bits() {
        let mut model = CppLoess {
            builder: Some(LoessBuilder::new()),
            cv_fractions: None,
            cv_method: None,
            cv_k: 5,
            cell: None,
            interpolation_vertices: None,
            boundary_degree_fallback: None,
            cv_seed: None,
        };
        let seed = (1u64 << 40) + 42;
        unsafe { cpp_loess_set_cv_seed(&mut model, seed) };
        assert_eq!(model.cv_seed, Some(seed));
        assert_eq!(
            std::mem::size_of::<usize>(),
            std::mem::size_of_val(&CppLoessResult::default().n)
        );
    }

    #[test]
    fn result_free_releases_retained_handle_and_is_idempotent() {
        let native = fastLoess::prelude::Loess::new()
            .fraction(1.0)
            .iterations(0)
            .surface_mode("direct")
            .retain_model(true)
            .build()
            .unwrap()
            .fit(&[0.0, 1.0, 2.0, 3.0][..], &[0.0, 1.1, 2.0, 3.1][..])
            .unwrap();
        let retained = Arc::downgrade(native.predict_state.as_ref().unwrap());
        let mut result = CppLoessResult::from(native);
        assert!(retained.upgrade().is_some());
        unsafe { cpp_loess_free_result(&mut result) };
        assert!(retained.upgrade().is_none());
        assert!(result.x.is_null());
        assert!(result.predict_handle.is_null());
        unsafe { cpp_loess_free_result(&mut result) };
    }

    #[test]
    fn result_free_cleans_zero_length_errors() {
        let mut result = error_result("test error");
        unsafe { cpp_loess_free_result(&mut result) };
        assert!(result.error.is_null());
        unsafe { cpp_loess_free_result(&mut result) };
        let mut prediction = CppPredictResult {
            error: shared_parse::into_raw_error_c_string("test prediction error"),
            ..Default::default()
        };
        unsafe { cpp_predict_free_result(&mut prediction) };
        assert!(prediction.error.is_null());
        unsafe { cpp_predict_free_result(&mut prediction) };
    }
}
