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
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Include the C header
#include "fastloess.h"

namespace fastloess {

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
  static Expected make_error(std::string msg) {
    return Expected(std::move(msg), ErrorTag{});
  }

  bool has_value() const { return has_val_; }
  explicit operator bool() const { return has_val_; }

  T &value() & {
    if (!has_val_)
      throw LoessError(err_);
    return val_;
  }

  const T &value() const & {
    if (!has_val_)
      throw LoessError(err_);
    return val_;
  }

  T &&value() && {
    if (!has_val_)
      throw LoessError(err_);
    return std::move(val_);
  }

  const std::string &error() const {
    if (has_val_)
      throw LoessError("Bad expected access: has value");
    return err_;
  }

private:
  Expected(std::string err, ErrorTag) : err_(std::move(err)), has_val_(false) {}

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
  double fraction = 0.67; ///< Smoothing fraction (0, 1]
  int iterations = 3;     ///< Robustness iterations
  double delta = NAN;     ///< Interpolation threshold (NaN = auto)

  std::string weight_function = "tricube";
  std::string robustness_method = "bisquare";
  std::string scaling_method = "mad";
  std::string boundary_policy = "extend";
  std::string zero_weight_fallback = "use_local_mean";

  double confidence_intervals = NAN; ///< Confidence level (NaN = disabled)
  double prediction_intervals = NAN; ///< Prediction level (NaN = disabled)
  double auto_converge = NAN;        ///< Auto-convergence threshold

  bool return_diagnostics = false;
  bool return_residuals = false;
  bool return_robustness_weights = false;
  bool parallel = false;

  // Cross-validation options
  std::vector<double> cv_fractions;
  std::string cv_method = "kfold";
  int cv_k = 5;
};

/**
 * @brief Options for streaming LOESS.
 */
struct StreamingOptions : public LoessOptions {
  int chunk_size = 5000;
  int overlap = -1;                        ///< -1 for auto
  std::string merge_strategy = "weighted"; ///< average, weighted, first, last
};

/**
 * @brief Options for online LOESS.
 */
struct OnlineOptions : public LoessOptions {
  int window_capacity = 1000;
  int min_points = 2;
  std::string update_mode = "full";
};

/**
 * @brief Diagnostics from LOESS fitting.
 */
struct Diagnostics {
  double rmse = NAN;
  double mae = NAN;
  double r_squared = NAN;
  double aic = NAN;
  double aicc = NAN;
  double effective_df = NAN;
  double residual_sd = NAN;

  bool has_value() const { return !std::isnan(rmse); }
};

/**
 * @brief Result of LOESS smoothing operation.
 *
 * RAII wrapper that automatically frees the underlying C result.
 */
class LoessResult {
public:
  LoessResult() = default;

  explicit LoessResult(fastloess_CppLoessResult &&c_result)
      : result_(std::move(c_result)) {
    c_result = fastloess_CppLoessResult{}; // Clear moved-from result
  }

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
    return result_.error ? std::string(result_.error) : "";
  }

  /// Access x value at index
  double x(size_t i) const { return result_.x[i]; }

  /// Access smoothed y value at index
  double y(size_t i) const { return result_.y[i]; }

  /// Get x values as vector
  std::vector<double> x_vector() const {
    return std::vector<double>(result_.x, result_.x + result_.n);
  }

  /// Get smoothed y values as vector
  std::vector<double> y_vector() const {
    return std::vector<double>(result_.y, result_.y + result_.n);
  }

  /// Get residuals (empty if not computed)
  std::vector<double> residuals() const {
    if (result_.residuals) {
      return std::vector<double>(result_.residuals,
                                 result_.residuals + result_.n);
    }
    return {};
  }

  /// Get standard errors (empty if not computed)
  std::vector<double> standard_errors() const {
    if (result_.standard_errors) {
      return std::vector<double>(result_.standard_errors,
                                 result_.standard_errors + result_.n);
    }
    return {};
  }

  /// Get confidence interval lower bounds
  std::vector<double> confidence_lower() const {
    if (result_.confidence_lower) {
      return std::vector<double>(result_.confidence_lower,
                                 result_.confidence_lower + result_.n);
    }
    return {};
  }

  /// Get confidence interval upper bounds
  std::vector<double> confidence_upper() const {
    if (result_.confidence_upper) {
      return std::vector<double>(result_.confidence_upper,
                                 result_.confidence_upper + result_.n);
    }
    return {};
  }

  /// Get prediction interval lower bounds
  std::vector<double> prediction_lower() const {
    if (result_.prediction_lower) {
      return std::vector<double>(result_.prediction_lower,
                                 result_.prediction_lower + result_.n);
    }
    return {};
  }

  /// Get prediction interval upper bounds
  std::vector<double> prediction_upper() const {
    if (result_.prediction_upper) {
      return std::vector<double>(result_.prediction_upper,
                                 result_.prediction_upper + result_.n);
    }
    return {};
  }

  /// Get robustness weights (empty if not computed)
  std::vector<double> robustness_weights() const {
    if (result_.robustness_weights) {
      return std::vector<double>(result_.robustness_weights,
                                 result_.robustness_weights + result_.n);
    }
    return {};
  }

  /// Fraction used for smoothing
  double fraction_used() const { return result_.fraction_used; }

  /// Number of iterations performed (-1 if not available)
  int iterations_used() const { return result_.iterations_used; }

  /// Get diagnostics
  Diagnostics diagnostics() const {
    return Diagnostics{result_.rmse,       result_.mae,  result_.r_squared,
                       result_.aic,        result_.aicc, result_.effective_df,
                       result_.residual_sd};
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
    ptr_ = cpp_loess_new(
        options.fraction, options.iterations, options.delta,
        options.weight_function.c_str(), options.robustness_method.c_str(),
        options.scaling_method.c_str(), options.boundary_policy.c_str(),
        options.confidence_intervals, options.prediction_intervals,
        options.return_diagnostics ? 1 : 0, options.return_residuals ? 1 : 0,
        options.return_robustness_weights ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.cv_fractions.empty() ? nullptr : options.cv_fractions.data(),
        options.cv_fractions.size(), options.cv_method.c_str(), options.cv_k,
        options.parallel ? 1 : 0);
  }

  ~Loess() {
    if (ptr_) {
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
      if (ptr_)
        cpp_loess_free(ptr_);
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  Expected<LoessResult> fit(const std::vector<double> &x,
                             const std::vector<double> &y) {
    if (x.size() != y.size()) {
      return Expected<LoessResult>::make_error(
          "x and y must have the same length");
    }
    if (x.empty()) {
      return Expected<LoessResult>::make_error(
          "Input arrays must not be empty");
    }

    auto result = cpp_loess_fit(ptr_, x.data(), y.data(), x.size());

    if (result.error != nullptr) {
      std::string error_msg(result.error);
      cpp_loess_free_result(&result);
      return Expected<LoessResult>::make_error(error_msg);
    }

    return Expected<LoessResult>(LoessResult(std::move(result)));
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
        options.fraction, options.iterations, options.delta,
        options.weight_function.c_str(), options.robustness_method.c_str(),
        options.scaling_method.c_str(), options.boundary_policy.c_str(),
        options.return_diagnostics ? 1 : 0, options.return_residuals ? 1 : 0,
        options.return_robustness_weights ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.parallel ? 1 : 0, options.chunk_size, options.overlap,
        options.merge_strategy.c_str());
  }

  ~StreamingLoess() {
    if (ptr_) {
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
      if (ptr_)
        cpp_streaming_free(ptr_);
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  Expected<LoessResult> process_chunk(const std::vector<double> &x,
                                       const std::vector<double> &y) {
    if (expect_finalized_)
      return Expected<LoessResult>::make_error("Model already finalized");
    if (x.size() != y.size())
      return Expected<LoessResult>::make_error("x and y length mismatch");

    auto result = cpp_streaming_process(ptr_, x.data(), y.data(), x.size());

    if (result.error != nullptr) {
      std::string error_msg(result.error);
      cpp_loess_free_result(&result);
      return Expected<LoessResult>::make_error(error_msg);
    }
    return Expected<LoessResult>(LoessResult(std::move(result)));
  }

  Expected<LoessResult> finalize() {
    if (expect_finalized_)
      return Expected<LoessResult>::make_error("Model already finalized");
    expect_finalized_ = true;

    auto result = cpp_streaming_finalize(ptr_);
    if (result.error != nullptr) {
      std::string error_msg(result.error);
      cpp_loess_free_result(&result);
      return Expected<LoessResult>::make_error(error_msg);
    }
    return Expected<LoessResult>(LoessResult(std::move(result)));
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
        options.fraction, options.iterations, options.delta,
        options.weight_function.c_str(), options.robustness_method.c_str(),
        options.scaling_method.c_str(), options.boundary_policy.c_str(),
        options.return_robustness_weights ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.parallel ? 1 : 0, options.window_capacity, options.min_points,
        options.update_mode.c_str());
  }

  ~OnlineLoess() {
    if (ptr_) {
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
      if (ptr_)
        cpp_online_free(ptr_);
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  Expected<LoessResult> add_points(const std::vector<double> &x,
                                    const std::vector<double> &y) {
    if (x.size() != y.size())
      return Expected<LoessResult>::make_error("x and y length mismatch");

    auto result = cpp_online_add_points(ptr_, x.data(), y.data(), x.size());

    if (result.error != nullptr) {
      std::string error_msg(result.error);
      cpp_loess_free_result(&result);
      return Expected<LoessResult>::make_error(error_msg);
    }
    return Expected<LoessResult>(LoessResult(std::move(result)));
  }

private:
  fastloess_CppOnlineLoess *ptr_ = nullptr;
};

} // namespace fastloess

#endif // FASTLOESS_HPP
