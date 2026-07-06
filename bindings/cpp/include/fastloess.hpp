/**
 * @file fastloess.hpp
 * @brief C++ wrapper for fastLoess library
 *
 * Provides idiomatic C++ access to LOESS smoothing with RAII,
 * exceptions, and STL container support.
 */

#ifndef FASTLOESS_HPP
#define FASTLOESS_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Include the C header
#include "fastloess.h"

namespace fastloess {

namespace detail {
constexpr double k_default_fraction = 0.67;
constexpr int k_default_cv_k = 5;
constexpr int k_default_chunk_size = 5000;
constexpr int k_default_window_capacity = 1000;
} // namespace detail

/**
 * @brief Exception thrown when LOESS operation fails.
 */
class LoessError : public std::runtime_error {
public:
  explicit LoessError(const std::string &message)
      : std::runtime_error(message) {}
};

/**
 * @brief A result type that holds either a value or an error.
 * Mimics std::expected (C++23) behavior.
 */
template <typename T> class Expected {
public:
  // Success constructor
  Expected(T val) : val_(std::move(val)), has_val_(true) {}

  // Error constructor
  struct ErrorTag {};
  static Expected makeError(std::string msg) {
    return Expected(std::move(msg), ErrorTag{});
  }

  bool hasValue() const { return has_val_; }

  explicit operator bool() const { return has_val_; }

  T &value() & {
    if (!has_val_) {
      throw LoessError(err_);
    }
    return val_;
  }

  const T &value() const & {
    if (!has_val_) {
      throw LoessError(err_);
    }
    return val_;
  }

  T &&value() && {
    if (!has_val_) {
      throw LoessError(err_);
    }
    return std::move(val_);
  }

  const std::string &error() const {
    if (has_val_) {
      throw LoessError("Bad expected access: has value");
    }
    return err_;
  }

private:
  Expected(std::string err, ErrorTag /*error_tag*/)
      : err_(std::move(err)), has_val_(false) {}

  // We store both to avoid manual union management, relying on T's cheap
  // default ctor. LoessResult's default ctor is cheap (zero-init).
  T val_;
  std::string err_;
  bool has_val_;
};

/**
 * @brief Options for configuring LOESS smoothing.
 */
struct LoessOptions {
  double fraction = detail::k_default_fraction; ///< Smoothing fraction (0, 1]
  int iterations = 3;                           ///< Robustness iterations

  std::string weight_function = "tricube";
  std::string robustness_method = "bisquare";
  std::string scaling_method = "mad"; ///< mad, mar, mean
  std::string boundary_policy = "extend";
  std::string zero_weight_fallback = "use_local_mean";

  double confidence_intervals = NAN; ///< Confidence level (NaN = disabled)
  double prediction_intervals = NAN; ///< Prediction level (NaN = disabled)
  double auto_converge = NAN;        ///< Auto-convergence threshold

  bool return_diagnostics = false;
  bool return_residuals = false;
  bool return_robustness_weights = false;
  bool return_se = false; ///< Compute standard errors and hat-matrix statistics
  bool parallel = false;

  // LOESS-specific options
  std::string degree =
      "linear";       ///< constant, linear, quadratic, cubic, quartic
  int dimensions = 1; ///< Number of predictor dimensions
  std::string distance_metric =
      "normalized"; ///< euclidean, normalized, manhattan, chebyshev
  std::string surface_mode = "interpolation"; ///< direct, interpolation

  // Cross-validation options
  std::vector<double> cv_fractions;
  std::string cv_method = "kfold";
  int cv_k = detail::k_default_cv_k;

  // Advanced / tuning options
  /// Per-dimension weights for the \"weighted\" distance metric.
  std::vector<double> weighted_metric_weights;
  /// Cell size tuning parameter for the interpolation grid (NaN = library
  /// default).
  double cell = NAN;
  /// Number of interpolation vertices (0 = library default).
  int interpolation_vertices = 0;
  /// -1 = unset (library default), 0 = false, 1 = true.
  int boundary_degree_fallback = -1;
  /// Seed for cross-validation RNG (0 = unset / random).
  uint64_t cv_seed = 0;
};

/**
 * @brief Options for streaming LOESS.
 */
struct StreamingOptions : public LoessOptions {
  int chunk_size = detail::k_default_chunk_size;
  int overlap = -1;                        ///< -1 for auto
  std::string merge_strategy = "weighted"; ///< average, weighted, first, last
};

/**
 * @brief Options for online LOESS.
 */
struct OnlineOptions : public LoessOptions {
  int window_capacity = detail::k_default_window_capacity;
  int min_points = 2;
  std::string update_mode = "full";
};

/**
 * @brief Diagnostics from LOESS fitting.
 */
class Diagnostics {
public:
  Diagnostics() = default;

  explicit Diagnostics(const fastloess_CppLoessResult &result)
      : rmse_(result.rmse), mae_(result.mae), r_squared_(result.r_squared),
        aic_(result.aic), aicc_(result.aicc),
        effective_df_(result.effective_df), residual_sd_(result.residual_sd) {}

  bool hasValue() const { return !std::isnan(rmse_); }

  double rmse() const { return rmse_; }
  double mae() const { return mae_; }
  double rSquared() const { return r_squared_; }
  double aic() const { return aic_; }
  double aicc() const { return aicc_; }
  double effectiveDf() const { return effective_df_; }
  double residualSd() const { return residual_sd_; }

private:
  double rmse_ = NAN;
  double mae_ = NAN;
  double r_squared_ = NAN;
  double aic_ = NAN;
  double aicc_ = NAN;
  double effective_df_ = NAN;
  double residual_sd_ = NAN;
};

/**
 * @brief Result of LOESS smoothing operation.
 *
 * RAII wrapper that automatically frees the underlying C result.
 */
class LoessResult {
public:
  LoessResult() = default;

  explicit LoessResult(const fastloess_CppLoessResult &c_result)
      : result_(c_result) {}

  ~LoessResult() {
    if (result_.n > 0) {
      cpp_loess_free_result(&result_);
    }
  }

  // Move-only
  LoessResult(const LoessResult &) = delete;
  LoessResult &operator=(const LoessResult &) = delete;

  LoessResult(LoessResult &&other) noexcept : result_(other.result_) {
    other.result_ = fastloess_CppLoessResult{};
  }

  LoessResult &operator=(LoessResult &&other) noexcept {
    if (this != &other) {
      if (result_.n > 0) {
        cpp_loess_free_result(&result_);
      }
      result_ = other.result_;
      other.result_ = fastloess_CppLoessResult{};
    }
    return *this;
  }

  /// Number of data points
  size_t size() const { return static_cast<size_t>(result_.n); }

  /// Check if result is valid
  bool valid() const { return result_.n > 0 && result_.error == nullptr; }

  /// Get error message (empty if no error)
  std::string error() const {
    return result_.error != nullptr ? std::string(result_.error) : "";
  }

  /// Access x value at index
  double xValue(size_t index) const { return result_.x[index]; }

  /// Access smoothed y value at index
  double yValue(size_t index) const { return result_.y[index]; }

  /// Get x values as vector
  std::vector<double> xVector() const {
    return std::vector<double>(result_.x, result_.x + result_.n);
  }

  /// Get smoothed y values as vector
  std::vector<double> yVector() const {
    return std::vector<double>(result_.y, result_.y + result_.n);
  }

  /// Get residuals (empty if not computed)
  std::vector<double> residuals() const {
    if (result_.residuals != nullptr) {
      return std::vector<double>(result_.residuals,
                                 result_.residuals + result_.n);
    }
    return {};
  }

  /// Get standard errors (empty if not computed)
  std::vector<double> standardErrors() const {
    if (result_.standard_errors != nullptr) {
      return std::vector<double>(result_.standard_errors,
                                 result_.standard_errors + result_.n);
    }
    return {};
  }

  /// Get confidence interval lower bounds
  std::vector<double> confidenceLower() const {
    if (result_.confidence_lower != nullptr) {
      return std::vector<double>(result_.confidence_lower,
                                 result_.confidence_lower + result_.n);
    }
    return {};
  }

  /// Get confidence interval upper bounds
  std::vector<double> confidenceUpper() const {
    if (result_.confidence_upper != nullptr) {
      return std::vector<double>(result_.confidence_upper,
                                 result_.confidence_upper + result_.n);
    }
    return {};
  }

  /// Get prediction interval lower bounds
  std::vector<double> predictionLower() const {
    if (result_.prediction_lower != nullptr) {
      return std::vector<double>(result_.prediction_lower,
                                 result_.prediction_lower + result_.n);
    }
    return {};
  }

  /// Get prediction interval upper bounds
  std::vector<double> predictionUpper() const {
    if (result_.prediction_upper != nullptr) {
      return std::vector<double>(result_.prediction_upper,
                                 result_.prediction_upper + result_.n);
    }
    return {};
  }

  /// Get robustness weights (empty if not computed)
  std::vector<double> robustnessWeights() const {
    if (result_.robustness_weights != nullptr) {
      return std::vector<double>(result_.robustness_weights,
                                 result_.robustness_weights + result_.n);
    }
    return {};
  }

  /// Fraction used for smoothing
  double fractionUsed() const { return result_.fraction_used; }

  /// Number of iterations performed (-1 if not available)
  int iterationsUsed() const { return result_.iterations_used; }

  /// Number of predictor dimensions used
  int dimensions() const { return result_.dimensions; }

  /// Equivalent number of parameters / ENP (NaN if not computed)
  double enp() const { return result_.enp; }

  /// Trace of hat matrix (NaN if not computed)
  double traceHat() const { return result_.trace_hat; }

  /// Delta1 for SE/CI computation (NaN if not computed)
  double delta1() const { return result_.delta1; }

  /// Delta2 for SE/CI computation (NaN if not computed)
  double delta2() const { return result_.delta2; }

  /// Residual scale estimate (NaN if not computed)
  double residualScale() const { return result_.residual_scale; }

  /// Per-point leverage / hat-matrix diagonal (empty if not computed)
  std::vector<double> leverage() const {
    if (result_.leverage != nullptr) {
      return std::vector<double>(result_.leverage,
                                 result_.leverage + result_.n);
    }
    return {};
  }

  /// Cross-validation scores per tested fraction (empty if CV not performed)
  std::vector<double> cvScores() const {
    if (result_.cv_scores != nullptr && result_.cv_scores_len > 0) {
      return std::vector<double>(result_.cv_scores,
                                 result_.cv_scores + result_.cv_scores_len);
    }
    return {};
  }

  /// Get diagnostics
  Diagnostics diagnostics() const { return Diagnostics(result_); }

private:
  fastloess_CppLoessResult result_ = {};
};

/**
 * @brief Batch LOESS model.
 */
class Loess {
public:
  explicit Loess(const LoessOptions &options = {}) {
    ptr_ = cpp_loess_new(
        options.fraction, options.iterations, options.weight_function.c_str(),
        options.robustness_method.c_str(), options.scaling_method.c_str(),
        options.boundary_policy.c_str(), options.confidence_intervals,
        options.prediction_intervals, options.return_diagnostics ? 1 : 0,
        options.return_residuals ? 1 : 0,
        options.return_robustness_weights ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.cv_fractions.empty() ? nullptr : options.cv_fractions.data(),
        static_cast<unsigned long>(options.cv_fractions.size()),
        options.cv_method.c_str(), options.cv_k, options.parallel ? 1 : 0,
        options.degree.c_str(), options.dimensions,
        options.weighted_metric_weights.empty()
            ? options.distance_metric.c_str()
            : nullptr,
        options.surface_mode.c_str(), options.return_se ? 1 : 0);
    if (!options.weighted_metric_weights.empty()) {
      cpp_loess_set_weighted_metric(
          ptr_, options.weighted_metric_weights.data(),
          static_cast<unsigned long>(options.weighted_metric_weights.size()));
    }
    if (!std::isnan(options.cell)) {
      cpp_loess_set_cell(ptr_, options.cell);
    }
    if (options.interpolation_vertices > 0) {
      cpp_loess_set_interpolation_vertices(
          ptr_, static_cast<unsigned long>(options.interpolation_vertices));
    }
    if (options.boundary_degree_fallback >= 0) {
      cpp_loess_set_boundary_degree_fallback(ptr_,
                                             options.boundary_degree_fallback);
    }
    if (options.cv_seed > 0) {
      cpp_loess_set_cv_seed(ptr_, static_cast<unsigned long>(options.cv_seed));
    }
  }

  ~Loess() {
    if (ptr_ != nullptr) {
      cpp_loess_free(ptr_);
    }
  }

  // Non-copyable
  Loess(const Loess &) = delete;
  Loess &operator=(const Loess &) = delete;

  // Move-able
  Loess(Loess &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  Loess &operator=(Loess &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cpp_loess_free(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  /// @param custom_weights Per-observation weights (empty = no weights). Each
  /// weight
  ///   multiplies the local kernel weight: w_ij = custom_weights[j] * K(d_ij/h)
  ///   * rob_j. Analogous to the `weights` argument in R's `stats::loess`.
  Expected<LoessResult> fit(const std::vector<double> &x_values,
                            const std::vector<double> &y_values,
                            const std::vector<double> &custom_weights = {}) {
    if (y_values.empty() || x_values.empty() ||
        x_values.size() % y_values.size() != 0) {
      return Expected<LoessResult>::makeError(
          "x length must be a non-zero multiple of y length");
    }
    if (x_values.empty()) {
      return Expected<LoessResult>::makeError(
          "Input arrays must not be empty");
    }
    if (!custom_weights.empty()) {
      cpp_loess_set_custom_weights(
          ptr_, custom_weights.data(),
          static_cast<unsigned long>(custom_weights.size()));
    }

    auto result = cpp_loess_fit(
        ptr_, x_values.data(), static_cast<unsigned long>(x_values.size()),
        y_values.data(), static_cast<unsigned long>(y_values.size()));

    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_loess_free_result(&result);
      return Expected<LoessResult>::makeError(error_msg);
    }

    return Expected<LoessResult>(LoessResult(result));
  }

private:
  fastloess_CppLoess *ptr_ = nullptr;
};

/**
 * @brief Streaming LOESS model.
 */
class StreamingLoess {
public:
  explicit StreamingLoess(const StreamingOptions &options = {}) {
    ptr_ = cpp_streaming_new(
        options.fraction, options.iterations, options.weight_function.c_str(),
        options.robustness_method.c_str(), options.scaling_method.c_str(),
        options.boundary_policy.c_str(), options.return_diagnostics ? 1 : 0,
        options.return_residuals ? 1 : 0,
        options.return_robustness_weights ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.parallel ? 1 : 0, options.chunk_size, options.overlap,
        options.merge_strategy.c_str(), options.degree.c_str(),
        options.dimensions,
        options.weighted_metric_weights.empty()
            ? options.distance_metric.c_str()
            : nullptr,
        options.surface_mode.c_str(), options.return_se ? 1 : 0);
    if (!options.weighted_metric_weights.empty()) {
      cpp_streaming_set_weighted_metric(
          ptr_, options.weighted_metric_weights.data(),
          static_cast<unsigned long>(options.weighted_metric_weights.size()));
    }
    if (!std::isnan(options.cell)) {
      cpp_streaming_set_cell(ptr_, options.cell);
    }
    if (options.interpolation_vertices > 0) {
      cpp_streaming_set_interpolation_vertices(
          ptr_, static_cast<unsigned long>(options.interpolation_vertices));
    }
    if (options.boundary_degree_fallback >= 0) {
      cpp_streaming_set_boundary_degree_fallback(
          ptr_, options.boundary_degree_fallback);
    }
    if (!std::isnan(options.confidence_intervals)) {
      cpp_streaming_set_confidence_intervals(ptr_,
                                             options.confidence_intervals);
    }
    if (!std::isnan(options.prediction_intervals)) {
      cpp_streaming_set_prediction_intervals(ptr_,
                                             options.prediction_intervals);
    }
  }

  ~StreamingLoess() {
    if (ptr_ != nullptr) {
      cpp_streaming_free(ptr_);
    }
  }

  StreamingLoess(const StreamingLoess &) = delete;
  StreamingLoess &operator=(const StreamingLoess &) = delete;
  StreamingLoess(StreamingLoess &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  StreamingLoess &operator=(StreamingLoess &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cpp_streaming_free(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  Expected<LoessResult> processChunk(const std::vector<double> &x_values,
                                     const std::vector<double> &y_values) {
    if (expect_finalized_) {
      return Expected<LoessResult>::makeError("Model already finalized");
    }
    if (y_values.empty() || x_values.empty() ||
        x_values.size() % y_values.size() != 0) {
      return Expected<LoessResult>::makeError("x and y length mismatch");
    }

    auto result = cpp_streaming_process(
        ptr_, x_values.data(), static_cast<unsigned long>(x_values.size()),
        y_values.data(), static_cast<unsigned long>(y_values.size()));

    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_loess_free_result(&result);
      return Expected<LoessResult>::makeError(error_msg);
    }
    return Expected<LoessResult>(LoessResult(result));
  }

  Expected<LoessResult> finalize() {
    if (expect_finalized_) {
      return Expected<LoessResult>::makeError("Model already finalized");
    }
    expect_finalized_ = true;

    auto result = cpp_streaming_finalize(ptr_);
    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_loess_free_result(&result);
      return Expected<LoessResult>::makeError(error_msg);
    }
    return Expected<LoessResult>(LoessResult(result));
  }

private:
  fastloess_CppStreamingLoess *ptr_ = nullptr;
  bool expect_finalized_ = false;
};

/**
 * @brief Online LOESS model.
 */
class OnlineLoess {
public:
  explicit OnlineLoess(const OnlineOptions &options = {}) {
    ptr_ = cpp_online_new(
        options.fraction, options.iterations, options.weight_function.c_str(),
        options.robustness_method.c_str(), options.scaling_method.c_str(),
        options.boundary_policy.c_str(),
        options.return_robustness_weights ? 1 : 0,
        options.return_diagnostics ? 1 : 0, options.return_residuals ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.parallel ? 1 : 0, options.window_capacity, options.min_points,
        options.update_mode.c_str(), options.degree.c_str(), options.dimensions,
        options.weighted_metric_weights.empty()
            ? options.distance_metric.c_str()
            : nullptr,
        options.surface_mode.c_str(), options.return_se ? 1 : 0);
    if (!options.weighted_metric_weights.empty()) {
      cpp_online_set_weighted_metric(
          ptr_, options.weighted_metric_weights.data(),
          static_cast<unsigned long>(options.weighted_metric_weights.size()));
    }
    if (!std::isnan(options.cell)) {
      cpp_online_set_cell(ptr_, options.cell);
    }
    if (options.interpolation_vertices > 0) {
      cpp_online_set_interpolation_vertices(
          ptr_, static_cast<unsigned long>(options.interpolation_vertices));
    }
    if (options.boundary_degree_fallback >= 0) {
      cpp_online_set_boundary_degree_fallback(ptr_,
                                              options.boundary_degree_fallback);
    }
    if (!std::isnan(options.confidence_intervals)) {
      cpp_online_set_confidence_intervals(ptr_, options.confidence_intervals);
    }
    if (!std::isnan(options.prediction_intervals)) {
      cpp_online_set_prediction_intervals(ptr_, options.prediction_intervals);
    }
  }

  ~OnlineLoess() {
    if (ptr_ != nullptr) {
      cpp_online_free(ptr_);
    }
  }

  OnlineLoess(const OnlineLoess &) = delete;
  OnlineLoess &operator=(const OnlineLoess &) = delete;
  OnlineLoess(OnlineLoess &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  OnlineLoess &operator=(OnlineLoess &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cpp_online_free(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  Expected<LoessResult> addPoints(const std::vector<double> &x_values,
                                  const std::vector<double> &y_values) {
    if (y_values.empty() || x_values.empty() ||
        x_values.size() % y_values.size() != 0) {
      return Expected<LoessResult>::makeError("x and y length mismatch");
    }

    auto result = cpp_online_add_points(
        ptr_, x_values.data(), static_cast<unsigned long>(x_values.size()),
        y_values.data(), static_cast<unsigned long>(y_values.size()));

    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_loess_free_result(&result);
      return Expected<LoessResult>::makeError(error_msg);
    }
    return Expected<LoessResult>(LoessResult(result));
  }

private:
  fastloess_CppOnlineLoess *ptr_ = nullptr;
};

} // namespace fastloess

#endif // FASTLOESS_HPP
