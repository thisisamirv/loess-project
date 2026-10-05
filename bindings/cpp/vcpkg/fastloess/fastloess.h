#include <stddef.h>
#include <stdint.h>

#ifndef FASTLOESS_H
#define FASTLOESS_H

struct fastloess_CppLoess;

struct fastloess_CppOnlineLoess;

struct fastloess_CppStreamingLoess;

struct fastloess_CppLoessResult {
  /// x values, in the same order as the input (length = n)
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
  /// Hat-matrix statistics (NaN / NULL if not computed; set return_se = 1 to
  /// enable)
  double enp;
  double trace_hat;
  double delta1;
  double delta2;
  double residual_scale;
  /// Per-point leverage / hat-matrix diagonal (NULL if not computed, length =
  /// n)
  double *leverage;
  /// Number of predictor dimensions used
  int dimensions;
  /// Cross-validation scores (NULL if not computed, length = cv_scores_len)
  double *cv_scores;
  unsigned long cv_scores_len;
  /// Error message (NULL if no error)
  char *error;
};

struct fastloess_CppOnlineOutput {
  int has_value;
  double y;
  double standard_error;
  double residual;
  double robustness_weight;
  int iterations_used;
  char *error;
};

extern "C" {

const char *cpp_last_error_message();

/// C++ wrapper constructor.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null. Arrays must be
/// valid.
fastloess_CppLoess *cpp_loess_new(
    double fraction, int iterations, const char *weight_function,
    const char *robustness_method, const char *scaling_method,
    const char *boundary_policy, double confidence_intervals,
    double prediction_intervals, int return_diagnostics, int return_residuals,
    int return_robustness_weights, const char *zero_weight_fallback,
    double auto_converge, const double *cv_fractions,
    unsigned long cv_fractions_len, const char *cv_method, int cv_k,
    int parallel, const char *degree, int dimensions,
    const char *distance_metric, const char *surface_mode, int return_se,
    int return_sorted, double cell, int interpolation_vertices,
    int boundary_degree_fallback, const double *weighted_metric_weights,
    unsigned long weighted_metric_weights_len, const char *missing);

/// Set CV seed for reproducible K-fold splits.
///
/// # Safety
/// ptr must be valid.
void cpp_loess_set_cv_seed(fastloess_CppLoess *ptr, unsigned long seed);

/// Set cell tuning parameter for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
void cpp_streaming_set_cell(fastloess_CppStreamingLoess *ptr, double cell);

/// Set number of interpolation vertices for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
void cpp_streaming_set_interpolation_vertices(fastloess_CppStreamingLoess *ptr,
                                              unsigned long vertices);

/// Enable or disable boundary degree fallback for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
void cpp_streaming_set_boundary_degree_fallback(
    fastloess_CppStreamingLoess *ptr, int enabled);

/// Set confidence interval level for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
void cpp_streaming_set_confidence_intervals(fastloess_CppStreamingLoess *ptr,
                                            double level);

/// Set prediction interval level for a model.
///
/// # Safety
/// `ptr` must be a valid `CppStreamingLoess` pointer.
void cpp_streaming_set_prediction_intervals(fastloess_CppStreamingLoess *ptr,
                                            double level);

/// Set cell tuning parameter for a model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
void cpp_online_set_cell(fastloess_CppOnlineLoess *ptr, double cell);

/// Set number of interpolation vertices for a model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
void cpp_online_set_interpolation_vertices(fastloess_CppOnlineLoess *ptr,
                                           unsigned long vertices);

/// Enable or disable boundary degree fallback for a model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
void cpp_online_set_boundary_degree_fallback(fastloess_CppOnlineLoess *ptr,
                                             int enabled);

/// Set confidence interval level for a model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
void cpp_online_set_confidence_intervals(fastloess_CppOnlineLoess *ptr,
                                         double level);

/// Set prediction interval level for a model.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
void cpp_online_set_prediction_intervals(fastloess_CppOnlineLoess *ptr,
                                         double level);

/// Fit the model.
///
/// # Safety
/// `ptr` must be a valid CppLoess pointer. `x_values` must be a valid array of
/// length `x_n`
/// (= n_observations * dimensions), `y_values` must be a valid array of length
/// `y_n` (= n_observations).
fastloess_CppLoessResult
cpp_loess_fit(fastloess_CppLoess *ptr, const double *x_values,
              unsigned long x_n, const double *y_values, unsigned long y_n,
              const double *custom_weights, unsigned long custom_weights_n);

/// Free model.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `cpp_loess_new` or null.
void cpp_loess_free(fastloess_CppLoess *ptr);

/// Create a new Streaming Loess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
fastloess_CppStreamingLoess *cpp_streaming_new(
    double fraction, int iterations, const char *weight_function,
    const char *robustness_method, const char *scaling_method,
    const char *boundary_policy, int return_diagnostics, int return_residuals,
    int return_robustness_weights, const char *zero_weight_fallback,
    double auto_converge, int parallel, int chunk_size, int overlap,
    const char *merge_strategy, const char *degree, int dimensions,
    const char *distance_metric, const char *surface_mode, double cell,
    int interpolation_vertices, int boundary_degree_fallback,
    const double *weighted_metric_weights,
    unsigned long weighted_metric_weights_len, const char *missing);

/// Process a chunk of data.
///
/// # Safety
/// `ptr` must be valid. `x_values` must be a valid array of length `x_n` (=
/// n_observations * dimensions), `y_values` must be a valid array of length
/// `y_n` (= n_observations).
fastloess_CppLoessResult cpp_streaming_process(fastloess_CppStreamingLoess *ptr,
                                               const double *x_values,
                                               unsigned long x_n,
                                               const double *y_values,
                                               unsigned long y_n);

/// Finalize the streaming process.
///
/// # Safety
/// `ptr` must be valid.
fastloess_CppLoessResult
cpp_streaming_finalize(fastloess_CppStreamingLoess *ptr);

/// Free model.
///
/// # Safety
/// `ptr` must be valid or null.
void cpp_streaming_free(fastloess_CppStreamingLoess *ptr);

/// Create a new Online Loess model.
///
/// # Safety
/// Pointers must be valid null-terminated strings or null.
fastloess_CppOnlineLoess *
cpp_online_new(double fraction, int iterations, const char *weight_function,
               const char *robustness_method, const char *scaling_method,
               const char *boundary_policy, int return_robustness_weights,
               const char *zero_weight_fallback, double auto_converge,
               int window_capacity, int min_points, const char *update_mode,
               const char *degree, int dimensions, const char *distance_metric,
               const char *surface_mode, double cell,
               int interpolation_vertices, int boundary_degree_fallback,
               const double *weighted_metric_weights,
               unsigned long weighted_metric_weights_len, const char *missing);

/// Add a single point to the model and return its smoothed value.
/// `has_value = 0` in the result means the window is still filling.
///
/// # Safety
/// `ptr` must be a valid `CppOnlineLoess` pointer.
fastloess_CppOnlineOutput cpp_online_add_point(fastloess_CppOnlineLoess *ptr,
                                               double x, double y);

/// Free the error string in a CppOnlineOutput (call only when error != NULL).
///
/// # Safety
/// `output` must be a valid pointer and `output->error` must have been
/// allocated by Rust.
void cpp_online_free_output(fastloess_CppOnlineOutput *output);

/// Free model.
///
/// # Safety
/// `ptr` must be valid or null.
void cpp_online_free(fastloess_CppOnlineLoess *ptr);

/// Free a CppLoessResult.
///
/// # Safety
/// `result` must be a valid pointer to a CppLoessResult struct.
void cpp_loess_free_result(fastloess_CppLoessResult *result);

} // extern "C"

#endif // FASTLOESS_H
