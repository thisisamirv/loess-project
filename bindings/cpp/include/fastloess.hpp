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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Include the C header
extern "C" {
#include "fastloess.h"
}

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
  int overlap = -1; ///< -1 for auto
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
 * @brief Perform batch LOESS smoothing.
 *
 * @param x Independent variable values
 * @param y Dependent variable values
 * @param options Smoothing options
 * @return LoessResult containing smoothed values
 * @throws LoessError if smoothing fails
 */
inline LoessResult smooth(const std::vector<double> &x,
                           const std::vector<double> &y,
                           const LoessOptions &options = {}) {
  if (x.size() != y.size()) {
    throw LoessError("x and y must have the same length");
  }
  if (x.empty()) {
    throw LoessError("Input arrays must not be empty");
  }

  fastloess_CppLoessResult result = cpp_loess_smooth(
      x.data(), y.data(), x.size(), options.fraction, options.iterations,
      options.delta, options.weight_function.c_str(),
      options.robustness_method.c_str(), options.scaling_method.c_str(),
      options.boundary_policy.c_str(), options.confidence_intervals,
      options.prediction_intervals, options.return_diagnostics ? 1 : 0,
      options.return_residuals ? 1 : 0,
      options.return_robustness_weights ? 1 : 0,
      options.zero_weight_fallback.c_str(), options.auto_converge,
      options.cv_fractions.empty() ? nullptr : options.cv_fractions.data(),
      options.cv_fractions.size(), options.cv_method.c_str(), options.cv_k,
      options.parallel ? 1 : 0);

  if (result.error != nullptr) {
    std::string error_msg(result.error);
    cpp_loess_free_result(&result);
    throw LoessError(error_msg);
  }

  return LoessResult(std::move(result));
}

/**
 * @brief Perform streaming LOESS for large datasets.
 */
inline LoessResult streaming(const std::vector<double> &x,
                              const std::vector<double> &y,
                              const StreamingOptions &options = {}) {
  if (x.size() != y.size()) {
    throw LoessError("x and y must have the same length");
  }
  if (x.empty()) {
    throw LoessError("Input arrays must not be empty");
  }

  fastloess_CppLoessResult result = cpp_loess_streaming(
      x.data(), y.data(), x.size(), options.fraction, options.chunk_size,
      options.overlap, options.iterations, options.delta,
      options.weight_function.c_str(), options.robustness_method.c_str(),
      options.scaling_method.c_str(), options.boundary_policy.c_str(),
      options.auto_converge, options.return_diagnostics ? 1 : 0,
      options.return_residuals ? 1 : 0,
      options.return_robustness_weights ? 1 : 0,
      options.zero_weight_fallback.c_str(), options.merge_strategy.c_str(),
      options.parallel ? 1 : 0);

  if (result.error != nullptr) {
    std::string error_msg(result.error);
    cpp_loess_free_result(&result);
    throw LoessError(error_msg);
  }

  return LoessResult(std::move(result));
}

/**
 * @brief Perform online LOESS with sliding window.
 */
inline LoessResult online(const std::vector<double> &x,
                           const std::vector<double> &y,
                           const OnlineOptions &options = {}) {
  if (x.size() != y.size()) {
    throw LoessError("x and y must have the same length");
  }
  if (x.empty()) {
    throw LoessError("Input arrays must not be empty");
  }

  fastloess_CppLoessResult result = cpp_loess_online(
      x.data(), y.data(), x.size(), options.fraction, options.window_capacity,
      options.min_points, options.iterations, options.delta,
      options.weight_function.c_str(), options.robustness_method.c_str(),
      options.scaling_method.c_str(), options.boundary_policy.c_str(),
      options.update_mode.c_str(), options.auto_converge,
      options.return_robustness_weights ? 1 : 0,
      options.zero_weight_fallback.c_str(), options.parallel ? 1 : 0);

  if (result.error != nullptr) {
    std::string error_msg(result.error);
    cpp_loess_free_result(&result);
    throw LoessError(error_msg);
  }

  return LoessResult(std::move(result));
}

} // namespace fastloess

#endif // FASTLOESS_HPP
