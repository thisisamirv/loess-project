#include <stddef.h>

typedef struct fastloess_GoLoess fastloess_GoLoess;

typedef struct fastloess_GoOnlineLoess fastloess_GoOnlineLoess;

typedef struct fastloess_GoPredictHandle fastloess_GoPredictHandle;

typedef struct fastloess_GoStreamingLoess fastloess_GoStreamingLoess;

typedef struct fastloess_GoLoessResult {
  /**
   * x values, in the same order as the input (length = n)
   */
  double *x;
  /**
   * Smoothed y values (length = n)
   */
  double *y;
  /**
   * Number of data points
   */
  size_t n;
  /**
   * Standard errors (NULL if not computed)
   */
  double *standard_errors;
  /**
   * Lower confidence bounds (NULL if not computed)
   */
  double *confidence_lower;
  /**
   * Upper confidence bounds (NULL if not computed)
   */
  double *confidence_upper;
  /**
   * Lower prediction bounds (NULL if not computed)
   */
  double *prediction_lower;
  /**
   * Upper prediction bounds (NULL if not computed)
   */
  double *prediction_upper;
  /**
   * Residuals (NULL if not computed)
   */
  double *residuals;
  /**
   * Robustness weights (NULL if not computed)
   */
  double *robustness_weights;
  /**
   * Local fit's gradient at each point, `dimensions` values per point, flattened
   * (NULL if not requested; only takes effect when surface_mode = "direct")
   */
  double *gradient;
  /**
   * Fraction used for smoothing
   */
  double fraction_used;
  /**
   * Number of iterations performed (-1 if not available)
   */
  int iterations_used;
  /**
   * Diagnostics (NaN if not computed)
   */
  double rmse;
  double mae;
  double r_squared;
  double aic;
  double aicc;
  double effective_df;
  double residual_sd;
  /**
   * Hat-matrix statistics (NaN / NULL if not computed; set return_se = 1 to enable)
   */
  double enp;
  double trace_hat;
  double delta1;
  double delta2;
  double residual_scale;
  /**
   * Per-point leverage / hat-matrix diagonal (NULL if not computed, length = n)
   */
  double *leverage;
  /**
   * Number of predictor dimensions used
   */
  int dimensions;
  /**
   * Cross-validation scores (NULL if not computed, length = cv_scores_len)
   */
  double *cv_scores;
  size_t cv_scores_len;
  /**
   * Opaque handle for `go_predict()`, non-NULL only if `retain_model` was set to 1.
   * Must eventually be freed via `go_predict_handle_free`.
   */
  struct fastloess_GoPredictHandle *predict_handle;
  /**
   * Error message (NULL if no error)
   */
  char *error;
} fastloess_GoLoessResult;

typedef struct fastloess_GoPredictResult {
  /**
   * Predicted y values, one per query point (length = n)
   */
  double *y;
  /**
   * Number of query points
   */
  size_t n;
  /**
   * Standard errors (NULL if not requested)
   */
  double *standard_errors;
  /**
   * Lower confidence bounds (NULL if not requested)
   */
  double *confidence_lower;
  /**
   * Upper confidence bounds (NULL if not requested)
   */
  double *confidence_upper;
  /**
   * Lower prediction bounds (NULL if not requested)
   */
  double *prediction_lower;
  /**
   * Upper prediction bounds (NULL if not requested)
   */
  double *prediction_upper;
  /**
   * Local fit's gradient at each query point, `dimensions` values per point,
   * flattened (NULL if not requested)
   */
  double *derivative;
  /**
   * Number of predictor dimensions (needed to know `derivative`'s true length,
   * `dimensions * n`, when freeing it)
   */
  int dimensions;
  /**
   * Error message (NULL if no error)
   */
  char *error;
} fastloess_GoPredictResult;

typedef struct fastloess_GoOnlineOutput {
  int has_value;
  double y;
  double standard_error;
  double residual;
  double robustness_weight;
  int iterations_used;
  /**
   * Confidence interval lower bound (`update_mode="full"` only, if requested)
   */
  double confidence_lower;
  /**
   * Confidence interval upper bound (`update_mode="full"` only, if requested)
   */
  double confidence_upper;
  /**
   * Prediction interval lower bound (`update_mode="full"` only, if requested)
   */
  double prediction_lower;
  /**
   * Prediction interval upper bound (`update_mode="full"` only, if requested)
   */
  double prediction_upper;
  /**
   * Latest point's local fit gradient, `dimensions` values, NULL if not requested
   */
  double *gradient;
  /**
   * Number of predictor dimensions (needed to know `gradient`'s true length)
   */
  int dimensions;
  char *error;
} fastloess_GoOnlineOutput;

const char *go_last_error_message(void);

/**
 * Go wrapper constructor.
 *
 * # Safety
 * Pointers must be valid null-terminated strings or null. Arrays must be valid.
 */
struct fastloess_GoLoess *go_loess_new(double fraction,
                                       int iterations,
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
                                       size_t cv_fractions_len,
                                       const char *cv_method,
                                       int cv_k,
                                       int parallel,
                                       const char *degree,
                                       int dimensions,
                                       const char *distance_metric,
                                       const char *surface_mode,
                                       int return_se,
                                       int return_sorted,
                                       double cell,
                                       int interpolation_vertices,
                                       int boundary_degree_fallback,
                                       const double *weighted_metric_weights,
                                       size_t weighted_metric_weights_len,
                                       const char *missing,
                                       int retain_model,
                                       int return_gradient);

/**
 * Set CV seed for reproducible K-fold splits.
 *
 * # Safety
 * ptr must be valid.
 */
void go_loess_set_cv_seed(struct fastloess_GoLoess *ptr, unsigned long long seed);

/**
 * Fit the model.
 *
 * # Safety
 * `ptr` must be a valid GoLoess pointer. `x_values` must be a valid array of length `x_n`
 * (= n_observations * dimensions), `y_values` must be a valid array of length `y_n` (= n_observations).
 */
struct fastloess_GoLoessResult go_loess_fit(struct fastloess_GoLoess *ptr,
                                            const double *x_values,
                                            size_t x_n,
                                            const double *y_values,
                                            size_t y_n,
                                            const double *custom_weights,
                                            size_t custom_weights_n);

/**
 * Free model.
 *
 * # Safety
 * `ptr` must be a valid pointer returned by `go_loess_new` or null.
 */
void go_loess_free(struct fastloess_GoLoess *ptr);

/**
 * Evaluate a fitted model (retained via `retain_model = 1`) at out-of-sample query
 * points not in the training set.
 *
 * # Safety
 * `handle` must be a valid pointer returned via `GoLoessResult::predict_handle`.
 * `new_x` must be a valid array of length `new_x_len`. `extrapolation` must be a
 * valid null-terminated string or null (defaults to "clamp").
 */
struct fastloess_GoPredictResult go_predict(struct fastloess_GoPredictHandle *handle,
                                            const double *new_x,
                                            size_t new_x_len,
                                            int return_se,
                                            double confidence_level,
                                            double prediction_level,
                                            int return_derivative,
                                            const char *extrapolation,
                                            double max_extrapolation_distance,
                                            double max_neighbor_distance);

/**
 * Free a GoPredictResult's heap-allocated buffers.
 *
 * # Safety
 * `result` must be a valid pointer to a GoPredictResult struct.
 */
void go_predict_free_result(struct fastloess_GoPredictResult *result);

/**
 * Free a `GoPredictHandle` returned via `GoLoessResult::predict_handle`.
 *
 * # Safety
 * `ptr` must be a valid pointer returned via `GoLoessResult::predict_handle`, or null.
 */
void go_predict_handle_free(struct fastloess_GoPredictHandle *ptr);

/**
 * Create a new Streaming Loess model.
 *
 * # Safety
 * Pointers must be valid null-terminated strings or null.
 */
struct fastloess_GoStreamingLoess *go_streaming_new(double fraction,
                                                    int iterations,
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
                                                    const char *merge_strategy,
                                                    const char *degree,
                                                    int dimensions,
                                                    const char *distance_metric,
                                                    const char *surface_mode,
                                                    double cell,
                                                    int interpolation_vertices,
                                                    int boundary_degree_fallback,
                                                    const double *weighted_metric_weights,
                                                    size_t weighted_metric_weights_len,
                                                    const char *missing,
                                                    int return_gradient,
                                                    double confidence_intervals,
                                                    double prediction_intervals,
                                                    int return_se);

/**
 * Process a chunk of data.
 *
 * # Safety
 * `ptr` must be valid. `x_values` must be a valid array of length `x_n` (= n_observations * dimensions),
 * `y_values` must be a valid array of length `y_n` (= n_observations).
 */
struct fastloess_GoLoessResult go_streaming_process(struct fastloess_GoStreamingLoess *ptr,
                                                    const double *x_values,
                                                    size_t x_n,
                                                    const double *y_values,
                                                    size_t y_n);

/**
 * Finalize the streaming process.
 *
 * # Safety
 * `ptr` must be valid.
 */
struct fastloess_GoLoessResult go_streaming_finalize(struct fastloess_GoStreamingLoess *ptr);

/**
 * Free model.
 *
 * # Safety
 * `ptr` must be valid or null.
 */
void go_streaming_free(struct fastloess_GoStreamingLoess *ptr);

/**
 * Create a new Online Loess model.
 *
 * # Safety
 * Pointers must be valid null-terminated strings or null.
 */
struct fastloess_GoOnlineLoess *go_online_new(double fraction,
                                              int iterations,
                                              const char *weight_function,
                                              const char *robustness_method,
                                              const char *scaling_method,
                                              const char *boundary_policy,
                                              int return_robustness_weights,
                                              const char *zero_weight_fallback,
                                              double auto_converge,
                                              int window_capacity,
                                              int min_points,
                                              const char *update_mode,
                                              const char *degree,
                                              int dimensions,
                                              const char *distance_metric,
                                              const char *surface_mode,
                                              double cell,
                                              int interpolation_vertices,
                                              int boundary_degree_fallback,
                                              const double *weighted_metric_weights,
                                              size_t weighted_metric_weights_len,
                                              const char *missing,
                                              int return_gradient,
                                              double confidence_intervals,
                                              double prediction_intervals,
                                              int return_se);

/**
 * Add a single point to the model and return its smoothed value.
 * `has_value = 0` in the result means the window is still filling.
 *
 * # Safety
 * `ptr` must be a valid `GoOnlineLoess` pointer.
 */
struct fastloess_GoOnlineOutput go_online_add_point(struct fastloess_GoOnlineLoess *ptr,
                                                    double x,
                                                    double y);

/**
 * Free the error field in a GoOnlineOutput (call only when error != NULL).
 *
 * # Safety
 * `output` must be a valid pointer and `output->error` must have been allocated by Rust.
 */
void go_online_free_output(struct fastloess_GoOnlineOutput *output);

/**
 * Free model.
 *
 * # Safety
 * `ptr` must be valid or null.
 */
void go_online_free(struct fastloess_GoOnlineLoess *ptr);

/**
 * Free a GoLoessResult.
 *
 * # Safety
 * `result` must be a valid pointer to a GoLoessResult struct.
 */
void go_loess_free_result(struct fastloess_GoLoessResult *result);
