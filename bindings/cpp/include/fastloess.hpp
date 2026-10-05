/**
 * @file fastloess.hpp
 * @brief C++ wrapper for fastLoess library
 *
 * Provides idiomatic C++ access to LOESS smoothing with RAII,
 * exceptions, and STL container support.
 */

#ifndef FASTLOESS_HPP
#define FASTLOESS_HPP

#ifdef _MSVC_LANG
#if _MSVC_LANG < 201703L
#error "fastloess.hpp requires C++17 or later"
#endif
#elif __cplusplus < 201703L
#error "fastloess.hpp requires C++17 or later"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Include the C header
#include "fastloess.h"
#include "fastloess_version.h" // IWYU pragma: export

namespace fastloess {

namespace detail {
constexpr double k_default_fraction = 0.67;
constexpr int k_default_cv_k = 5;
constexpr int k_default_chunk_size = 5000;
/// Sentinel meaning "use the library default" (chunk_size / 10, clamped to
/// [1, chunk_size - 10]); negative values are never sent to the FFI layer.
constexpr int k_default_overlap = -1;
constexpr int k_default_window_capacity = 1000;
constexpr int k_default_min_points = 2;
} // namespace detail

struct CVOptions {
  std::vector<double> fractions;
  std::string method = "kfold";
  int k = detail::k_default_cv_k;
};

struct IntervalsOptions {
  double confidence = NAN;
  double prediction = NAN;
};

inline bool hasOutput(const std::vector<std::string> &outputs,
                      const char *name) {
  return std::find(outputs.begin(), outputs.end(), name) != outputs.end();
}

/**
 * @brief Exception thrown when LOESS operation fails.
 */
class LoessError : public std::runtime_error {
public:
  explicit LoessError(const std::string &message)
      : std::runtime_error(message) {}
};

inline void validateOutputs(const std::vector<std::string> &outputs,
                            std::initializer_list<const char *> allowed) {
  for (const auto &output : outputs) {
    const auto *const found =
        std::find_if(allowed.begin(), allowed.end(),
                     [&output](const char *name) { return output == name; });
    if (found == allowed.end()) {
      throw LoessError("Unknown or unsupported output: " + output);
    }
  }
}

/**
 * @brief A result type that holds either a value or an error.
 * Mimics std::expected (C++23) behavior.
 */
template <typename T> class Expected {
public:
  // Success constructor
  explicit Expected(T val) : val_(std::move(val)), has_val_(true) {}

  // Error constructor
  struct ErrorTag {};
  static Expected make_error(std::string msg) {
    return Expected(std::move(msg), ErrorTag{});
  }

  bool has_value() const { return has_val_; }

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
  Expected(std::string err, ErrorTag error_tag)
      : err_(std::move(err)), has_val_(false) {
    static_cast<void>(error_tag);
  }

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

  IntervalsOptions intervals;
  double auto_converge = NAN; ///< Auto-convergence threshold

  /// Optional result components: "diagnostics", "residuals", "weights",
  /// "gradient" (or "derivative"), "se", and "sorted".
  std::vector<std::string> outputs;
  CVOptions cv;
  bool parallel = true;

  // LOESS-specific options
  std::string degree =
      "linear";       ///< constant, linear, quadratic, cubic, quartic
  int dimensions = 1; ///< Number of predictor dimensions
  std::string distance_metric =
      "normalized"; ///< euclidean, normalized, manhattan, chebyshev
  std::string surface_mode = "interpolation"; ///< direct, interpolation

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
  std::optional<uint64_t> seed;
  /// Policy for non-finite (NaN/Inf) values in input data ("error", "drop").
  std::string missing = "error";
  /// Retain the fitted model's training data, enabling
  /// `LoessResult::predict_model()`.
  bool retain_model = false;
};

/**
 * @brief Options for streaming LOESS.
 */
struct StreamingOptions : public LoessOptions {
  int chunk_size = detail::k_default_chunk_size;
  /// Negative (the default) means "use the library default"
  /// (chunk_size / 10, clamped to [1, chunk_size - 10]).
  int overlap = detail::k_default_overlap;
  std::string merge_strategy =
      "weighted_average"; ///< weighted_average, average, take_first, take_last
};

/**
 * @brief Options for online LOESS.
 *
 * Identical fields to LoessOptions but has no `parallel` option: online
 * LOESS processes one point at a time and always runs sequentially.
 * Cross-validation and diagnostics/residuals are Batch-only (or
 * Batch/Streaming-only) and have no equivalent here.
 * `intervals` and `outputs = {"se"}` require
 * `update_mode == "full"`.
 */
struct OnlineOptions {
  double fraction = detail::k_default_fraction;
  int iterations = 0;
  std::string weight_function = "tricube";
  std::string robustness_method = "bisquare";
  std::string scaling_method = "mad";
  std::string boundary_policy = "extend";
  std::string zero_weight_fallback = "use_local_mean";
  double auto_converge = NAN;
  /// Optional output components: "weights", "gradient" (or "derivative"),
  /// and/or "se". "se" requires `update_mode == "full"`.
  std::vector<std::string> outputs;
  IntervalsOptions intervals;
  std::string degree = "linear";
  int dimensions = 1;
  std::string distance_metric = "normalized";
  std::string surface_mode = "interpolation";
  std::vector<double> weighted_metric_weights;
  double cell = NAN;
  int interpolation_vertices = 0;
  int boundary_degree_fallback = -1;
  /// Policy for non-finite (NaN/Inf) \`x\`/\`y\` values passed to \`add_point\`
  /// ("error", "drop").
  std::string missing = "error";
  // Online-specific fields
  int window_capacity = detail::k_default_window_capacity;
  int min_points = detail::k_default_min_points;
  std::string update_mode = "incremental";
};

/**
 * @brief Diagnostics from LOESS fitting.
 */
class Diagnostics {
public:
  Diagnostics() = default;

  explicit Diagnostics(const fastloess_CppLoessResult &result)
      : rmse_(optional_metric(result.rmse)), mae_(optional_metric(result.mae)),
        r_squared_(optional_metric(result.r_squared)),
        aic_(optional_metric(result.aic)), aicc_(optional_metric(result.aicc)),
        effective_df_(optional_metric(result.effective_df)),
        residual_sd_(optional_metric(result.residual_sd)) {}

  explicit Diagnostics(const fastloess_CppOnlineDiagnostics &result)
      : rmse_(optional_metric(result.rmse)), mae_(optional_metric(result.mae)),
        r_squared_(optional_metric(result.r_squared)),
        aic_(optional_metric(result.aic)), aicc_(optional_metric(result.aicc)),
        effective_df_(optional_metric(result.effective_df)),
        residual_sd_(optional_metric(result.residual_sd)) {}

  bool has_value() const { return rmse_.has_value(); }

  std::optional<double> rmse() const { return rmse_; }
  std::optional<double> mae() const { return mae_; }
  std::optional<double> r_squared() const { return r_squared_; }
  std::optional<double> aic() const { return aic_; }
  std::optional<double> aicc() const { return aicc_; }
  std::optional<double> effective_df() const { return effective_df_; }
  std::optional<double> residual_sd() const { return residual_sd_; }

private:
  static std::optional<double> optional_metric(double value) {
    return std::isnan(value) ? std::nullopt : std::optional<double>(value);
  }

  std::optional<double> rmse_;
  std::optional<double> mae_;
  std::optional<double> r_squared_;
  std::optional<double> aic_;
  std::optional<double> aicc_;
  std::optional<double> effective_df_;
  std::optional<double> residual_sd_;
};

/**
 * @brief Options for `PredictModel::predict()`.
 */
struct PredictOptions {
  /// Optional prediction components: "se" and/or "gradient" ("derivative").
  std::vector<std::string> outputs;
  IntervalsOptions intervals;
  /// Behavior for query points outside the training range ("clamp", "linear",
  /// "error").
  std::string extrapolation = "clamp";
  /// Under "linear" extrapolation, the maximum allowed distance beyond the
  /// training boundary before predict() errors instead of returning an
  /// unbounded value (NaN = disabled).
  double max_extrapolation_distance = NAN;
  /// Maximum allowed distance to the farthest point in a query's neighbor
  /// window before predict() errors, catching in-range-but-sparse query points
  /// (NaN = disabled).
  double max_neighbor_distance = NAN;
};

/**
 * @brief Result of `PredictModel::predict()`.
 *
 * RAII wrapper that automatically frees the underlying C result.
 */
class PredictResult {
public:
  PredictResult() = default;

  explicit PredictResult(const fastloess_CppPredictResult &c_result)
      : result_(c_result) {}

  ~PredictResult() { cpp_predict_free_result(&result_); }

  // Move-only
  PredictResult(const PredictResult &) = delete;
  PredictResult &operator=(const PredictResult &) = delete;

  PredictResult(PredictResult &&other) noexcept : result_(other.result_) {
    other.result_ = fastloess_CppPredictResult{};
  }

  PredictResult &operator=(PredictResult &&other) noexcept {
    if (this != &other) {
      cpp_predict_free_result(&result_);
      result_ = other.result_;
      other.result_ = fastloess_CppPredictResult{};
    }
    return *this;
  }

  /// Number of query points
  size_t size() const { return static_cast<size_t>(result_.n); }

  /// Check if result is valid
  bool valid() const { return result_.n > 0 && result_.error == nullptr; }

  /// Get error message (empty if no error)
  std::string error() const {
    return result_.error != nullptr ? std::string(result_.error) : "";
  }

  /// Predicted y values, one per query point
  std::vector<double> y() const {
    if (result_.n == 0 || result_.y == nullptr) {
      return {};
    }
    return std::vector<double>(result_.y, result_.y + result_.n);
  }

  /// Standard errors (empty if not requested)
  std::vector<double> standard_errors() const {
    if (result_.standard_errors != nullptr) {
      return std::vector<double>(result_.standard_errors,
                                 result_.standard_errors + result_.n);
    }
    return {};
  }

  /// Lower confidence interval bounds (empty if not requested)
  std::vector<double> confidence_lower() const {
    if (result_.confidence_lower != nullptr) {
      return std::vector<double>(result_.confidence_lower,
                                 result_.confidence_lower + result_.n);
    }
    return {};
  }

  /// Upper confidence interval bounds (empty if not requested)
  std::vector<double> confidence_upper() const {
    if (result_.confidence_upper != nullptr) {
      return std::vector<double>(result_.confidence_upper,
                                 result_.confidence_upper + result_.n);
    }
    return {};
  }

  /// Lower prediction interval bounds (empty if not requested)
  std::vector<double> prediction_lower() const {
    if (result_.prediction_lower != nullptr) {
      return std::vector<double>(result_.prediction_lower,
                                 result_.prediction_lower + result_.n);
    }
    return {};
  }

  /// Upper prediction interval bounds (empty if not requested)
  std::vector<double> prediction_upper() const {
    if (result_.prediction_upper != nullptr) {
      return std::vector<double>(result_.prediction_upper,
                                 result_.prediction_upper + result_.n);
    }
    return {};
  }

  /// Local fit's gradient at each query point, `dimensions()` values per
  /// point, flattened (empty if not requested)
  std::vector<double> derivative() const {
    if (result_.derivative != nullptr) {
      const size_t count = static_cast<size_t>(result_.n) *
                           static_cast<size_t>(std::max(result_.dimensions, 1));
      return std::vector<double>(result_.derivative,
                                 result_.derivative + count);
    }
    return {};
  }

private:
  fastloess_CppPredictResult result_ = {};
};

/**
 * @brief Retained fitted-model state enabling out-of-sample `predict()`.
 *
 * Obtained via `LoessResult::predict_model()`, only available when
 * `LoessOptions::retain_model` was set to `true` before `fit()`.
 */
class PredictModel {
public:
  PredictModel() = default;

  explicit PredictModel(fastloess_CppPredictHandle *handle) : ptr_(handle) {}

  ~PredictModel() {
    if (ptr_ != nullptr) {
      cpp_predict_handle_free(ptr_);
    }
  }

  // Non-copyable
  PredictModel(const PredictModel &) = delete;
  PredictModel &operator=(const PredictModel &) = delete;

  // Move-able
  PredictModel(PredictModel &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  PredictModel &operator=(PredictModel &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cpp_predict_handle_free(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  /// True if this model was actually retained (`retain_model` was set).
  bool valid() const { return ptr_ != nullptr; }

  /// Evaluate the fitted model at out-of-sample query points not in the
  /// training set (flattened, `dimensions` values per point).
  PredictResult predict(const std::vector<double> &new_x,
                        const PredictOptions &options = {}) const {
    validateOutputs(options.outputs, {"se", "gradient", "derivative"});
    auto result = cpp_predict(
        ptr_, new_x.data(), static_cast<size_t>(new_x.size()),
        hasOutput(options.outputs, "se") ? 1 : 0, options.intervals.confidence,
        options.intervals.prediction,
        (hasOutput(options.outputs, "gradient") ||
         hasOutput(options.outputs, "derivative"))
            ? 1
            : 0,
        options.extrapolation.c_str(), options.max_extrapolation_distance,
        options.max_neighbor_distance);
    return PredictResult(result);
  }

private:
  fastloess_CppPredictHandle *ptr_ = nullptr;
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

  ~LoessResult() { cpp_loess_free_result(&result_); }

  // Move-only
  LoessResult(const LoessResult &) = delete;
  LoessResult &operator=(const LoessResult &) = delete;

  LoessResult(LoessResult &&other) noexcept : result_(other.result_) {
    other.result_ = fastloess_CppLoessResult{};
  }

  LoessResult &operator=(LoessResult &&other) noexcept {
    if (this != &other) {
      cpp_loess_free_result(&result_);
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
  double x_value(size_t index) const {
    if (result_.x == nullptr || index >= size() * static_cast<size_t>(std::max(
                                                      result_.dimensions, 1))) {
      throw std::out_of_range("LOESS x index out of range");
    }
    return result_.x[index];
  }

  /// Access smoothed y value at index
  double y_value(size_t index) const {
    if (result_.y == nullptr || index >= size()) {
      throw std::out_of_range("LOESS y index out of range");
    }
    return result_.y[index];
  }

  /// Get x values as vector
  std::vector<double> x_vector() const {
    if (result_.n == 0 || result_.x == nullptr) {
      return {};
    }
    const size_t count =
        size() * static_cast<size_t>(std::max(result_.dimensions, 1));
    return std::vector<double>(result_.x, result_.x + count);
  }

  /// Get smoothed y values as vector
  std::vector<double> y_vector() const {
    if (result_.n == 0 || result_.y == nullptr) {
      return {};
    }
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
  std::vector<double> standard_errors() const {
    if (result_.standard_errors != nullptr) {
      return std::vector<double>(result_.standard_errors,
                                 result_.standard_errors + result_.n);
    }
    return {};
  }

  /// Get confidence interval lower bounds
  std::vector<double> confidence_lower() const {
    if (result_.confidence_lower != nullptr) {
      return std::vector<double>(result_.confidence_lower,
                                 result_.confidence_lower + result_.n);
    }
    return {};
  }

  /// Get confidence interval upper bounds
  std::vector<double> confidence_upper() const {
    if (result_.confidence_upper != nullptr) {
      return std::vector<double>(result_.confidence_upper,
                                 result_.confidence_upper + result_.n);
    }
    return {};
  }

  /// Get prediction interval lower bounds
  std::vector<double> prediction_lower() const {
    if (result_.prediction_lower != nullptr) {
      return std::vector<double>(result_.prediction_lower,
                                 result_.prediction_lower + result_.n);
    }
    return {};
  }

  /// Get prediction interval upper bounds
  std::vector<double> prediction_upper() const {
    if (result_.prediction_upper != nullptr) {
      return std::vector<double>(result_.prediction_upper,
                                 result_.prediction_upper + result_.n);
    }
    return {};
  }

  /// Get robustness weights (empty if not computed)
  std::vector<double> robustness_weights() const {
    if (result_.robustness_weights != nullptr) {
      return std::vector<double>(result_.robustness_weights,
                                 result_.robustness_weights + result_.n);
    }
    return {};
  }

  /// Get the per-point local fit gradient (flattened, `dimensions` values per
  /// point) (empty if not computed)
  std::vector<double> gradient() const {
    if (result_.gradient != nullptr) {
      const size_t count = static_cast<size_t>(result_.n) *
                           static_cast<size_t>(std::max(result_.dimensions, 1));
      return std::vector<double>(result_.gradient, result_.gradient + count);
    }
    return {};
  }

  /// Fraction used for smoothing
  double fraction_used() const { return result_.fraction_used; }

  /// Number of iterations performed (-1 if not available)
  int iterations_used() const { return result_.iterations_used; }

  /// Number of predictor dimensions used
  int dimensions() const { return result_.dimensions; }

  /// Equivalent number of parameters / ENP (NaN if not computed)
  double enp() const { return valid() ? result_.enp : NAN; }

  /// Trace of hat matrix (NaN if not computed)
  double trace_hat() const { return valid() ? result_.trace_hat : NAN; }

  /// Delta1 for SE/CI computation (NaN if not computed)
  double delta1() const { return valid() ? result_.delta1 : NAN; }

  /// Delta2 for SE/CI computation (NaN if not computed)
  double delta2() const { return valid() ? result_.delta2 : NAN; }

  /// Residual scale estimate (NaN if not computed)
  double residual_scale() const {
    return valid() ? result_.residual_scale : NAN;
  }

  /// Per-point leverage / hat-matrix diagonal (empty if not computed)
  std::vector<double> leverage() const {
    if (result_.leverage != nullptr) {
      return std::vector<double>(result_.leverage,
                                 result_.leverage + result_.n);
    }
    return {};
  }

  /// Cross-validation scores per tested fraction (empty if CV not performed)
  std::vector<double> cv_scores() const {
    if (result_.cv_scores != nullptr && result_.cv_scores_len > 0) {
      return std::vector<double>(result_.cv_scores,
                                 result_.cv_scores + result_.cv_scores_len);
    }
    return {};
  }

  /// Get diagnostics
  Diagnostics diagnostics() const {
    return valid() ? Diagnostics(result_) : Diagnostics();
  }

  /// Extract the retained predict model (only available when
  /// `LoessOptions::retain_model` was set to `true` before `fit()`). Transfers
  /// ownership: subsequent calls return an invalid (empty) `PredictModel`.
  /// Check `PredictModel::valid()` before use.
  PredictModel predict_model() {
    auto *handle = result_.predict_handle;
    result_.predict_handle = nullptr;
    return PredictModel(handle);
  }

private:
  fastloess_CppLoessResult result_ = {};
};

/**
 * @brief Batch LOESS model.
 */
class Loess {
public:
  explicit Loess(const LoessOptions &options = {}) {
    validateOutputs(options.outputs,
                    {"diagnostics", "residuals", "weights", "gradient",
                     "derivative", "se", "sorted"});
    const auto &cv_fractions = options.cv.fractions;
    const auto &cv_method = options.cv.method;
    const int cv_k = options.cv.k;
    ptr_ = cpp_loess_new(
        options.fraction, options.iterations, options.weight_function.c_str(),
        options.robustness_method.c_str(), options.scaling_method.c_str(),
        options.boundary_policy.c_str(), options.intervals.confidence,
        options.intervals.prediction,
        hasOutput(options.outputs, "diagnostics") ? 1 : 0,
        hasOutput(options.outputs, "residuals") ? 1 : 0,
        hasOutput(options.outputs, "weights") ? 1 : 0,
        (hasOutput(options.outputs, "gradient") ||
         hasOutput(options.outputs, "derivative"))
            ? 1
            : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        cv_fractions.empty() ? nullptr : cv_fractions.data(),
        static_cast<size_t>(cv_fractions.size()), cv_method.c_str(), cv_k,
        options.parallel ? 1 : 0, options.degree.c_str(), options.dimensions,
        options.distance_metric.c_str(), options.surface_mode.c_str(),
        hasOutput(options.outputs, "se") ? 1 : 0,
        hasOutput(options.outputs, "sorted") ? 1 : 0, options.cell,
        options.interpolation_vertices, options.boundary_degree_fallback,
        options.weighted_metric_weights.empty()
            ? nullptr
            : options.weighted_metric_weights.data(),
        static_cast<size_t>(options.weighted_metric_weights.size()),
        options.missing.c_str(), options.retain_model ? 1 : 0);
    if (ptr_ == nullptr) {
      throw LoessError(cpp_last_error_message());
    }
    if (options.seed.has_value()) {
      cpp_loess_set_cv_seed(ptr_, *options.seed);
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

  /// @param x_values Predictor values, flattened row-major across
  /// dimensions (length must be a non-zero multiple of `y_values.size()`).
  /// @param y_values Response values.
  /// @param custom_weights Per-observation weights (empty = no weights). Each
  /// weight multiplies the local kernel weight:
  /// w_ij = custom_weights[j] * K(d_ij/h) * rob_j. Analogous to the
  /// `weights` argument in R's `stats::loess`.
  Expected<LoessResult> fit(const std::vector<double> &x_values,
                            const std::vector<double> &y_values,
                            const std::vector<double> &custom_weights = {}) {
    if (y_values.empty() || x_values.empty() ||
        x_values.size() % y_values.size() != 0) {
      return Expected<LoessResult>::make_error(
          "x length must be a non-zero multiple of y length");
    }
    auto result = cpp_loess_fit(
        ptr_, x_values.data(), static_cast<size_t>(x_values.size()),
        y_values.data(), static_cast<size_t>(y_values.size()),
        custom_weights.empty() ? nullptr : custom_weights.data(),
        static_cast<size_t>(custom_weights.size()));

    LoessResult owned_result(result);
    if (result.error != nullptr) {
      return Expected<LoessResult>::make_error(owned_result.error());
    }

    return Expected<LoessResult>(std::move(owned_result));
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
    validateOutputs(options.outputs, {"diagnostics", "residuals", "weights",
                                      "gradient", "derivative", "se"});
    if (!options.cv.fractions.empty() || options.cv.method != "kfold" ||
        options.cv.k != detail::k_default_cv_k || options.seed.has_value() ||
        options.retain_model) {
      throw LoessError("StreamingLoess does not support Batch-only CV, seed, "
                       "or retained-model options");
    }
    ptr_ = cpp_streaming_new(
        options.fraction, options.iterations, options.weight_function.c_str(),
        options.robustness_method.c_str(), options.scaling_method.c_str(),
        options.boundary_policy.c_str(),
        hasOutput(options.outputs, "diagnostics") ? 1 : 0,
        hasOutput(options.outputs, "residuals") ? 1 : 0,
        hasOutput(options.outputs, "weights") ? 1 : 0,
        (hasOutput(options.outputs, "gradient") ||
         hasOutput(options.outputs, "derivative"))
            ? 1
            : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.parallel ? 1 : 0, options.chunk_size, options.overlap,
        options.merge_strategy.c_str(), options.degree.c_str(),
        options.dimensions, options.distance_metric.c_str(),
        options.surface_mode.c_str(), options.cell,
        options.interpolation_vertices, options.boundary_degree_fallback,
        options.weighted_metric_weights.empty()
            ? nullptr
            : options.weighted_metric_weights.data(),
        static_cast<size_t>(options.weighted_metric_weights.size()),
        options.missing.c_str(), options.intervals.confidence,
        options.intervals.prediction, hasOutput(options.outputs, "se") ? 1 : 0);
    if (ptr_ == nullptr) {
      throw LoessError(cpp_last_error_message());
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

  Expected<LoessResult> process_chunk(const std::vector<double> &x_values,
                                      const std::vector<double> &y_values) {
    if (expect_finalized_) {
      return Expected<LoessResult>::make_error("Model already finalized");
    }
    if (y_values.empty() || x_values.empty() ||
        x_values.size() % y_values.size() != 0) {
      return Expected<LoessResult>::make_error("x and y length mismatch");
    }

    auto result = cpp_streaming_process(
        ptr_, x_values.data(), static_cast<size_t>(x_values.size()),
        y_values.data(), static_cast<size_t>(y_values.size()));

    LoessResult owned_result(result);
    if (result.error != nullptr) {
      return Expected<LoessResult>::make_error(owned_result.error());
    }
    return Expected<LoessResult>(std::move(owned_result));
  }

  Expected<LoessResult>
  process_chunk_weighted(const std::vector<double> &x_values,
                         const std::vector<double> &y_values,
                         const std::vector<double> &custom_weights) {
    if (expect_finalized_) {
      return Expected<LoessResult>::make_error("Model already finalized");
    }
    if (y_values.empty() || x_values.empty() ||
        x_values.size() % y_values.size() != 0) {
      return Expected<LoessResult>::make_error("x and y length mismatch");
    }
    auto result = cpp_streaming_process_weighted(
        ptr_, x_values.data(), static_cast<size_t>(x_values.size()),
        y_values.data(), static_cast<size_t>(y_values.size()),
        custom_weights.data(), static_cast<size_t>(custom_weights.size()));
    LoessResult owned_result(result);
    if (result.error != nullptr) {
      return Expected<LoessResult>::make_error(owned_result.error());
    }
    return Expected<LoessResult>(std::move(owned_result));
  }

  Expected<LoessResult> finalize() {
    if (expect_finalized_) {
      return Expected<LoessResult>::make_error("Model already finalized");
    }
    expect_finalized_ = true;

    auto result = cpp_streaming_finalize(ptr_);
    LoessResult owned_result(result);
    if (result.error != nullptr) {
      return Expected<LoessResult>::make_error(owned_result.error());
    }
    return Expected<LoessResult>(std::move(owned_result));
  }

private:
  fastloess_CppStreamingLoess *ptr_ = nullptr;
  bool expect_finalized_ = false;
};

/**
 * @brief Result of a single online update step.
 *
 * Call has_value() to check if the window is ready.  When false, the window
 * is still filling and no smoothed estimate is available yet.  All optional
 * fields are NaN when not computed.
 */
class OnlineOutput {
public:
  /// True when the window has enough points to produce a smoothed estimate.
  bool has_value() const { return has_value_; }

  /// Smoothed value for the latest point (valid only when has_value() == true).
  double y() const { return y_; }

  /// Standard error (NaN if not computed).
  double standard_error() const { return standard_error_; }

  /// Residual y − smoothed (NaN if not computed).
  double residual() const { return residual_; }

  /// Robustness weight for the latest point (NaN if not computed).
  double robustness_weight() const { return robustness_weight_; }

  /// Number of robustness iterations performed (−1 if not applicable).
  int iterations_used() const { return iterations_used_; }

  /// Confidence interval lower bound (`update_mode == "full"` only, NaN if
  /// not computed).
  double confidence_lower() const { return confidence_lower_; }

  /// Confidence interval upper bound (`update_mode == "full"` only, NaN if
  /// not computed).
  double confidence_upper() const { return confidence_upper_; }

  /// Prediction interval lower bound (`update_mode == "full"` only, NaN if
  /// not computed).
  double prediction_lower() const { return prediction_lower_; }

  /// Prediction interval upper bound (`update_mode == "full"` only, NaN if
  /// not computed).
  double prediction_upper() const { return prediction_upper_; }

  /// Local fit gradient (`dimensions` values) for the latest point (empty if
  /// not computed).
  const std::vector<double> &gradient() const { return gradient_; }

private:
  friend class OnlineLoess;
  template <typename U> friend class Expected;
  OnlineOutput() =
      default; ///< Constructs an empty (has_value==false) instance.
  explicit OnlineOutput(const fastloess_CppOnlineOutput &raw)
      : has_value_(raw.has_value != 0), y_(raw.y),
        standard_error_(raw.standard_error), residual_(raw.residual),
        robustness_weight_(raw.robustness_weight),
        iterations_used_(raw.iterations_used),
        confidence_lower_(raw.confidence_lower),
        confidence_upper_(raw.confidence_upper),
        prediction_lower_(raw.prediction_lower),
        prediction_upper_(raw.prediction_upper),
        gradient_(raw.gradient != nullptr
                      ? std::vector<double>(raw.gradient,
                                            raw.gradient + raw.gradient_len)
                      : std::vector<double>()) {}

  bool has_value_ = false;
  double y_ = 0.0;
  double standard_error_ = std::numeric_limits<double>::quiet_NaN();
  double residual_ = std::numeric_limits<double>::quiet_NaN();
  double robustness_weight_ = std::numeric_limits<double>::quiet_NaN();
  int iterations_used_ = -1;
  double confidence_lower_ = std::numeric_limits<double>::quiet_NaN();
  double confidence_upper_ = std::numeric_limits<double>::quiet_NaN();
  double prediction_lower_ = std::numeric_limits<double>::quiet_NaN();
  double prediction_upper_ = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> gradient_;
};

/**
 * @brief Online LOESS model.
 */
class OnlineLoess {
public:
  explicit OnlineLoess(const OnlineOptions &options = {}) {
    validateOutputs(options.outputs,
                    {"weights", "gradient", "derivative", "se"});
    ptr_ = cpp_online_new(
        options.fraction, options.iterations, options.weight_function.c_str(),
        options.robustness_method.c_str(), options.scaling_method.c_str(),
        options.boundary_policy.c_str(),
        hasOutput(options.outputs, "weights") ? 1 : 0,
        (hasOutput(options.outputs, "gradient") ||
         hasOutput(options.outputs, "derivative"))
            ? 1
            : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.window_capacity, options.min_points,
        options.update_mode.c_str(), options.degree.c_str(), options.dimensions,
        options.distance_metric.c_str(), options.surface_mode.c_str(),
        options.cell, options.interpolation_vertices,
        options.boundary_degree_fallback,
        options.weighted_metric_weights.empty()
            ? nullptr
            : options.weighted_metric_weights.data(),
        static_cast<size_t>(options.weighted_metric_weights.size()),
        options.missing.c_str(), options.intervals.confidence,
        options.intervals.prediction, hasOutput(options.outputs, "se") ? 1 : 0);
    if (ptr_ == nullptr) {
      throw LoessError(cpp_last_error_message());
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

  Expected<OnlineOutput> add_point(double x, double y) {
    return wrap_output(cpp_online_add_point(ptr_, x, y));
  }

  Expected<OnlineOutput> add_point(double x, double y, double weight) {
    return wrap_output(cpp_online_add_point_weighted(ptr_, x, y, weight));
  }

  /// Add a point with one coordinate per configured predictor dimension.
  Expected<OnlineOutput> add_point(const std::vector<double> &x, double y) {
    return wrap_output(cpp_online_add_point_nd(
        ptr_, x.data(), static_cast<size_t>(x.size()), y));
  }

  Expected<OnlineOutput> add_point(const std::vector<double> &x, double y,
                                   double weight) {
    return wrap_output(cpp_online_add_point_nd_weighted(
        ptr_, x.data(), static_cast<size_t>(x.size()), y, weight));
  }

  Expected<std::optional<Diagnostics>> window_diagnostics() const {
    auto raw = cpp_online_window_diagnostics(ptr_);
    if (raw.error != nullptr) {
      const std::string message(raw.error);
      cpp_online_free_diagnostics(&raw);
      return Expected<std::optional<Diagnostics>>::make_error(message);
    }
    if (raw.has_value == 0) {
      cpp_online_free_diagnostics(&raw);
      return Expected<std::optional<Diagnostics>>(std::nullopt);
    }
    Diagnostics diagnostics(raw);
    cpp_online_free_diagnostics(&raw);
    return Expected<std::optional<Diagnostics>>(diagnostics);
  }

  Expected<PredictResult>
  predict_window(const std::vector<double> &new_x,
                 const PredictOptions &options = {}) const {
    validateOutputs(options.outputs, {"se", "gradient", "derivative"});
    auto raw = cpp_online_predict_window(
        ptr_, new_x.data(), static_cast<size_t>(new_x.size()),
        hasOutput(options.outputs, "se") ? 1 : 0, options.intervals.confidence,
        options.intervals.prediction,
        (hasOutput(options.outputs, "gradient") ||
         hasOutput(options.outputs, "derivative"))
            ? 1
            : 0,
        options.extrapolation.c_str(), options.max_extrapolation_distance,
        options.max_neighbor_distance);
    PredictResult result(raw);
    if (raw.error != nullptr) {
      return Expected<PredictResult>::make_error(result.error());
    }
    return Expected<PredictResult>(std::move(result));
  }

private:
  static Expected<OnlineOutput> wrap_output(fastloess_CppOnlineOutput raw) {
    const std::unique_ptr<fastloess_CppOnlineOutput,
                          decltype(&cpp_online_free_output)>
        guard(&raw, cpp_online_free_output);

    if (raw.error != nullptr) {
      const std::string error_msg(raw.error);
      return Expected<OnlineOutput>::make_error(error_msg);
    }
    OnlineOutput out(raw);
    return Expected<OnlineOutput>(std::move(out));
  }

  fastloess_CppOnlineLoess *ptr_ = nullptr;
};

} // namespace fastloess

#endif // FASTLOESS_HPP
