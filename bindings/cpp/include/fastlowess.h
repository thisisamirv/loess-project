#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

/// Opaque handle to a Loess batch model.
struct fastloess_CppLoess;

/// Opaque handle to a Loess online model.
struct fastloess_CppOnlineLoess;

/// Opaque handle to a Loess streaming model.
struct fastloess_CppStreamingLoess;

/// Result struct that can be passed across FFI boundary.
/// All arrays are allocated by Rust and must be freed by Rust.
struct fastloess_CppLoessResult {
  /// Sorted x values (length = n)
  double *x;
  /// Smoothed y values (length = n)
  double *y;
  /// Number of data points
  unsigned long n;
  /// Standard errors (NULL if not computed)
  double *standard_errors;
  /// Lower confidence bounds (NULL if not computed)
  double *confidence_lower;
  /// Upper confidence bounds (NULL if not computed)
  double *confidence_upper;
  /// Lower prediction bounds (NULL if not computed)
  double *prediction_lower;
  /// Upper prediction bounds (NULL if not computed)
  double *prediction_upper;
  /// Residuals (NULL if not computed)
  double *residuals;
  /// Robustness weights (NULL if not computed)
  double *robustness_weights;
  /// Fraction used for smoothing
  double fraction_used;
  /// Number of iterations performed (-1 if not available)
  int iterations_used;
  /// Diagnostics (NaN if not computed)
  double rmse;
  double mae;
  double r_squared;
  double aic;
  double aicc;
  double effective_df;
  double residual_sd;
  /// Error message (NULL if no error)
  char *error;
};

extern "C" {

/// C++ wrapper constructor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null. Arrays must be valid.
fastloess_CppLoess *cpp_loess_new(double fraction,
                                     int iterations,
                                     double delta,
                                     const char *weight_function,
                                     const char *robustness_method,
                                     const char *scaling_method,
                                     const char *boundary_policy,
                                     double confidence_intervals,
                                     double prediction_intervals,
                                     int return_diagnostics,
                                     int return_residuals,
                                     int return_robustness_weights,
                                     const char *zero_weight_fallback,
                                     double auto_converge,
                                     const double *cv_fractions,
                                     unsigned long cv_fractions_len,
                                     const char *cv_method,
                                     int cv_k,
                                     int parallel);

/// Fit the batch model.
///
/// # Safety
/// `ptr` must be a valid CppLoess pointer. `x` and `y` must be valid arrays of length `n`.
fastloess_CppLoessResult cpp_loess_fit(fastloess_CppLoess *ptr,
                                          const double *x,
                                          const double *y,
                                          unsigned long n);

/// Free batch model.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `cpp_loess_new` or null.
void cpp_loess_free(fastloess_CppLoess *ptr);

/// Create a new Streaming Loess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
fastloess_CppStreamingLoess *cpp_streaming_new(double fraction,
                                                 int iterations,
                                                 double delta,
                                                 const char *weight_function,
                                                 const char *robustness_method,
                                                 const char *scaling_method,
                                                 const char *boundary_policy,
                                                 int return_diagnostics,
                                                 int return_residuals,
                                                 int return_robustness_weights,
                                                 const char *zero_weight_fallback,
                                                 double auto_converge,
                                                 int parallel,
                                                 int chunk_size,
                                                 int overlap,
                                                 const char *merge_strategy);

/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be valid. `x` and `y` must be valid arrays of length `n`.
fastloess_CppLoessResult cpp_streaming_process(fastloess_CppStreamingLoess *ptr,
                                                 const double *x,
                                                 const double *y,
                                                 unsigned long n);

/// Finalize the streaming process.
///
/// # Safety
/// `ptr` must be valid.
fastloess_CppLoessResult cpp_streaming_finalize(fastloess_CppStreamingLoess *ptr);

/// Free streaming model.
///
/// # Safety
/// `ptr` must be valid or null.
void cpp_streaming_free(fastloess_CppStreamingLoess *ptr);

/// Create a new Online Loess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
fastloess_CppOnlineLoess *cpp_online_new(double fraction,
                                           int iterations,
                                           double delta,
                                           const char *weight_function,
                                           const char *robustness_method,
                                           const char *scaling_method,
                                           const char *boundary_policy,
                                           int return_robustness_weights,
                                           const char *zero_weight_fallback,
                                           double auto_converge,
                                           int parallel,
                                           int window_capacity,
                                           int min_points,
                                           const char *update_mode);

/// Add points to online model.
///
/// # Safety
/// `ptr` must be valid. `x` and `y` must be valid arrays of length `n`.
fastloess_CppLoessResult cpp_online_add_points(fastloess_CppOnlineLoess *ptr,
                                                 const double *x,
                                                 const double *y,
                                                 unsigned long n);

/// Free online model.
///
/// # Safety
/// `ptr` must be valid or null.
void cpp_online_free(fastloess_CppOnlineLoess *ptr);

/// Free a CppLoessResult.
///
/// # Safety
/// `result` must be a valid pointer to a CppLoessResult struct.
void cpp_loess_free_result(fastloess_CppLoessResult *result);

}  // extern "C"
