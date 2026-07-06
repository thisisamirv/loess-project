/**
 * @file batch_smoothing.cpp
 * @brief fastloess Batch Smoothing Example
 *
 * This example demonstrates batch LOESS smoothing features:
 * - Basic smoothing with different parameters
 * - Robustness iterations for outlier handling
 * - Confidence and prediction intervals
 * - Diagnostics and cross-validation
 *
 * The Loess class is the primary interface for
 * processing complete datasets that fit in memory.
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "../../bindings/cpp/include/fastloess.hpp"

namespace {

constexpr size_t k_default_point_count = 100;
constexpr unsigned int k_random_seed = 42;
constexpr double k_noise_std_dev = 1.5;
constexpr double k_outlier_magnitude_min = 10.0;
constexpr double k_outlier_magnitude_max = 20.0;
constexpr double k_x_range_max = 50.0;
constexpr double k_trend_slope = 0.5;
constexpr double k_seasonal_amplitude = 5.0;
constexpr double k_seasonal_frequency = 0.5;
constexpr size_t k_outlier_divisor = 10;
constexpr double k_basic_fraction = 0.05;
constexpr double k_confidence_level = 0.95;
constexpr double k_linear_range_max = 10.0;
constexpr size_t k_linear_point_count = 50;
constexpr double k_linear_slope = 2.0;
constexpr double k_linear_intercept = 1.0;
constexpr double k_boundary_fraction = 0.6;
constexpr double k_auto_converge_tol = 1e-3;
constexpr double k_cv_fraction_low = 0.1;
constexpr double k_cv_fraction_mid = 0.2;
constexpr int k_cv_k = 3;

struct Data {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> y_true;
};

Data generateSampleData(size_t point_count = k_default_point_count) {
  Data data;
  data.x.resize(point_count);
  data.y.resize(point_count);
  data.y_true.resize(point_count);

  std::seed_seq generator_seed = {k_random_seed, k_random_seed, k_random_seed,
                                  k_random_seed};
  std::mt19937 generator(generator_seed);
  std::normal_distribution<> noise(0.0, k_noise_std_dev);
  std::uniform_real_distribution<> outlier_magnitude(k_outlier_magnitude_min,
                                                     k_outlier_magnitude_max);
  std::uniform_int_distribution<> outlier_sign(0, 1);

  for (size_t point_index = 0; point_index < point_count; ++point_index) {
    data.x[point_index] = static_cast<double>(point_index) * k_x_range_max /
                          static_cast<double>(point_count - 1);

    data.y_true[point_index] =
        (k_trend_slope * data.x[point_index]) +
        (k_seasonal_amplitude *
         std::sin(data.x[point_index] * k_seasonal_frequency));

    data.y[point_index] = data.y_true[point_index] + noise(generator);
  }

  const size_t outlier_count = point_count / k_outlier_divisor;
  std::uniform_int_distribution<size_t> outlier_index(0, point_count - 1);

  for (size_t outlier_number = 0; outlier_number < outlier_count;
       ++outlier_number) {
    const size_t point_index = outlier_index(generator);
    double outlier_value = outlier_magnitude(generator);
    if (outlier_sign(generator) == 0) {
      outlier_value = -outlier_value;
    }
    data.y[point_index] += outlier_value;
  }

  return data;
}

} // namespace

int main() {
  try {
    std::cout << "=== fastloess Batch Smoothing Example ===\n";

    // 1. Generate Data
    auto data = generateSampleData(k_default_point_count);
    std::cout << "Generated " << data.x.size() << " data points with outliers."
              << '\n';

    // 2. Basic Smoothing (Default parameters)
    std::cout << "Running basic smoothing...\n";
    fastloess::LoessOptions basic_opts;
    basic_opts.fraction = k_basic_fraction;
    basic_opts.iterations = 0;
    fastloess::Loess model_basic(basic_opts);
    auto res_basic = model_basic.fit(data.x, data.y).value();

    // 3. Robust Smoothing (IRLS)
    std::cout << "Running robust smoothing (3 iterations)...\n";
    fastloess::LoessOptions robust_opts;
    robust_opts.fraction = k_basic_fraction;
    robust_opts.iterations = 3;
    robust_opts.robustness_method = "bisquare";
    robust_opts.return_robustness_weights = true;

    fastloess::Loess model_robust(robust_opts);
    auto res_robust = model_robust.fit(data.x, data.y).value();

    // 4. Uncertainty Quantification
    std::cout << "Computing confidence and prediction intervals..." << '\n';
    fastloess::LoessOptions interval_opts;
    interval_opts.fraction = k_basic_fraction;
    interval_opts.confidence_intervals = k_confidence_level;
    interval_opts.prediction_intervals = k_confidence_level;
    interval_opts.return_diagnostics = true;

    fastloess::Loess model_intervals(interval_opts);
    auto res_intervals = model_intervals.fit(data.x, data.y).value();

    // 5. Cross-Validation for optimal fraction
    std::cout << "Running cross-validation to find optimal fraction..." << '\n';

    // Manual CV search
    const std::vector<double> fractions = {k_basic_fraction, 0.1, 0.2, 0.4};
    double best_fraction = 0.0;
    double min_rmse = std::numeric_limits<double>::max();

    for (const double fraction : fractions) {
      fastloess::LoessOptions cv_opts;
      cv_opts.fraction = fraction;
      cv_opts.return_diagnostics = true;
      fastloess::Loess model(cv_opts);
      auto res_exp = model.fit(data.x, data.y);

      // Use non-throwing interface
      if (res_exp.hasValue()) {
        auto &res = res_exp.value();
        if (res.diagnostics().hasValue()) {
          const double rmse = res.diagnostics().rmse();
          if (rmse < min_rmse) {
            min_rmse = rmse;
            best_fraction = fraction;
          }
        }
      }
    }
    std::cout << "Optimal fraction found (manual CV): " << best_fraction
              << '\n';

    // Diagnostics Printout
    if (res_intervals.diagnostics().hasValue()) {
      const auto diag = res_intervals.diagnostics();
      std::cout << "\nFit Statistics (Intervals Model):\n";
      std::cout << " - R^2:   " << diag.rSquared() << '\n';
      std::cout << " - RMSE: " << diag.rmse() << '\n';
      std::cout << " - MAE:  " << diag.mae() << '\n';
    }

    // 6. Boundary Policy Comparison
    std::cout << "\nDemonstrating boundary policy effects on linear data..."
              << '\n';
    std::vector<double> linear_x(k_linear_point_count);
    std::vector<double> linear_y(k_linear_point_count);
    for (size_t point_index = 0; point_index < k_linear_point_count;
         ++point_index) {
      linear_x[point_index] = static_cast<double>(point_index) *
                              k_linear_range_max /
                              static_cast<double>(k_linear_point_count - 1);
      linear_y[point_index] =
          (k_linear_slope * linear_x[point_index]) + k_linear_intercept;
    }

    fastloess::LoessOptions opt_ext;
    opt_ext.fraction = k_boundary_fraction;
    opt_ext.boundary_policy = "extend";
    auto r_ext = fastloess::Loess(opt_ext).fit(linear_x, linear_y).value();

    fastloess::LoessOptions opt_ref;
    opt_ref.fraction = k_boundary_fraction;
    opt_ref.boundary_policy = "reflect";
    auto r_ref = fastloess::Loess(opt_ref).fit(linear_x, linear_y).value();

    fastloess::LoessOptions opt_zero;
    opt_zero.fraction = k_boundary_fraction;
    opt_zero.boundary_policy = "zero";
    auto r_zr = fastloess::Loess(opt_zero).fit(linear_x, linear_y).value();

    std::cout << "Boundary policy comparison:\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " - Extend (Default): first=" << r_ext.yValue(0)
              << ", last=" << r_ext.yValue(k_linear_point_count - 1) << '\n';
    std::cout << " - Reflect:          first=" << r_ref.yValue(0)
              << ", last=" << r_ref.yValue(k_linear_point_count - 1) << '\n';
    std::cout << " - Zero:             first=" << r_zr.yValue(0)
              << ", last=" << r_zr.yValue(k_linear_point_count - 1) << '\n';

    // 7. Boundary policy: noboundary
    std::cout << "\n--- Boundary Policy: noboundary ---\n";
    fastloess::LoessOptions nb_opts;
    nb_opts.fraction = k_boundary_fraction;
    nb_opts.boundary_policy = "noboundary";
    auto nb_res = fastloess::Loess(nb_opts).fit(linear_x, linear_y).value();
    std::cout << " - noboundary first=" << nb_res.yValue(0)
              << ", last=" << nb_res.yValue(k_linear_point_count - 1) << '\n';

    // 8. Weight function variants
    std::cout << "\n--- Weight Function Variants ---\n";
    for (const char *wfn : {"tricube", "epanechnikov", "gaussian", "uniform",
                            "biweight", "triangle", "cosine"}) {
      fastloess::LoessOptions wf_opts;
      wf_opts.fraction = k_basic_fraction;
      wf_opts.weight_function = wfn;
      auto wf_res = fastloess::Loess(wf_opts).fit(data.x, data.y).value();
      std::cout << "  weight_function=" << wfn << "  y[0]=" << wf_res.yValue(0)
                << '\n';
    }

    // 9. Polynomial degrees
    std::cout << "\n--- Polynomial Degrees ---\n";
    for (const char *deg : {"constant", "linear", "quadratic"}) {
      fastloess::LoessOptions deg_opts;
      deg_opts.fraction = k_basic_fraction;
      deg_opts.degree = deg;
      auto deg_res = fastloess::Loess(deg_opts).fit(data.x, data.y).value();
      std::cout << "  degree=" << deg << "  y[0]=" << deg_res.yValue(0)
                << '\n';
    }

    // 10. Scaling methods
    std::cout << "\n--- Scaling Methods ---\n";
    for (const char *scl : {"mad", "mar", "mean"}) {
      fastloess::LoessOptions scl_opts;
      scl_opts.fraction = k_basic_fraction;
      scl_opts.scaling_method = scl;
      auto scl_res = fastloess::Loess(scl_opts).fit(data.x, data.y).value();
      std::cout << "  scaling_method=" << scl << "  y[0]=" << scl_res.yValue(0)
                << '\n';
    }

    // 11. Distance metrics
    std::cout << "\n--- Distance Metrics ---\n";
    for (const char *met :
         {"euclidean", "normalized", "manhattan", "chebyshev"}) {
      fastloess::LoessOptions met_opts;
      met_opts.fraction = k_basic_fraction;
      met_opts.distance_metric = met;
      auto met_res = fastloess::Loess(met_opts).fit(data.x, data.y).value();
      std::cout << "  distance_metric=" << met
                << "  y[0]=" << met_res.yValue(0) << '\n';
    }

    // 12. Robustness method variants (bisquare shown above; add huber + talwar)
    std::cout << "\n--- Robustness Method Variants ---\n";
    for (const char *rob : {"huber", "talwar"}) {
      fastloess::LoessOptions rob_opts;
      rob_opts.fraction = k_basic_fraction;
      rob_opts.iterations = 2;
      rob_opts.robustness_method = rob;
      auto rob_res = fastloess::Loess(rob_opts).fit(data.x, data.y).value();
      std::cout << "  robustness_method=" << rob
                << "  y[0]=" << rob_res.yValue(0) << '\n';
    }

    // 13. Surface mode "direct" + return_se: standard errors and hat-matrix
    // stats
    std::cout << "\n--- Surface Mode: direct + Standard Errors ---\n";
    fastloess::LoessOptions se_opts;
    se_opts.fraction = k_basic_fraction;
    se_opts.surface_mode = "direct";
    se_opts.return_se = true;
    auto se_res = fastloess::Loess(se_opts).fit(data.x, data.y).value();
    std::cout << "  enp:           " << se_res.enp() << '\n';
    std::cout << "  traceHat:      " << se_res.traceHat() << '\n';
    std::cout << "  delta1:        " << se_res.delta1() << '\n';
    std::cout << "  delta2:        " << se_res.delta2() << '\n';
    std::cout << "  residualScale: " << se_res.residualScale() << '\n';
    {
      const auto std_errors = se_res.standardErrors();
      if (!std_errors.empty()) {
        std::cout << "  se[0]:         " << std_errors[0] << '\n';
      }
      const auto leverage_vals = se_res.leverage();
      if (!leverage_vals.empty()) {
        std::cout << "  leverage[0]:   " << leverage_vals[0] << '\n';
      }
    }

    // 14. return_residuals + return_robustness_weights + result metadata
    std::cout
        << "\n--- Residuals, Robustness Weights, and Result Metadata ---\n";
    fastloess::LoessOptions meta_opts;
    meta_opts.fraction = k_basic_fraction;
    meta_opts.iterations = 2;
    meta_opts.robustness_method = "huber";
    meta_opts.return_residuals = true;
    meta_opts.return_robustness_weights = true;
    auto meta_res = fastloess::Loess(meta_opts).fit(data.x, data.y).value();
    std::cout << "  fractionUsed:    " << meta_res.fractionUsed() << '\n';
    std::cout << "  iterationsUsed:  " << meta_res.iterationsUsed() << '\n';
    std::cout << "  dimensions:      " << meta_res.dimensions() << '\n';
    std::cout << "  valid():         " << meta_res.valid() << '\n';
    std::cout << "  xVector().size() " << meta_res.xVector().size() << '\n';
    std::cout << "  yVector().size() " << meta_res.yVector().size() << '\n';
    {
      const auto residuals = meta_res.residuals();
      if (!residuals.empty()) {
        std::cout << "  residuals[0]:    " << residuals[0] << '\n';
      }
      const auto rob_weights = meta_res.robustnessWeights();
      if (!rob_weights.empty()) {
        std::cout << "  robWeight[0]:    " << rob_weights[0] << '\n';
      }
    }

    // 15. Auto-convergence
    std::cout << "\n--- Auto-Convergence ---\n";
    fastloess::LoessOptions conv_opts;
    conv_opts.fraction = k_basic_fraction;
    conv_opts.auto_converge = k_auto_converge_tol;
    auto conv_res = fastloess::Loess(conv_opts).fit(data.x, data.y).value();
    std::cout << "  auto_converge=" << k_auto_converge_tol
              << "  iterationsUsed=" << conv_res.iterationsUsed() << '\n';

    // 16. Zero-weight fallback options
    std::cout << "\n--- Zero-Weight Fallback ---\n";
    for (const char *zfb :
         {"use_local_mean", "return_original", "return_none"}) {
      fastloess::LoessOptions zfb_opts;
      zfb_opts.fraction = k_basic_fraction;
      zfb_opts.zero_weight_fallback = zfb;
      auto zfb_res = fastloess::Loess(zfb_opts).fit(data.x, data.y).value();
      std::cout << "  zero_weight_fallback=" << zfb
                << "  y[0]=" << zfb_res.yValue(0) << '\n';
    }

    // 17. Built-in cross-validation (cv_fractions, cv_method, cv_k)
    std::cout << "\n--- Built-in Cross-Validation ---\n";
    fastloess::LoessOptions cv2_opts;
    cv2_opts.cv_fractions = {k_cv_fraction_low, k_basic_fraction,
                             k_cv_fraction_mid};
    cv2_opts.cv_method = "kfold";
    cv2_opts.cv_k = k_cv_k;
    auto cv2_res = fastloess::Loess(cv2_opts).fit(data.x, data.y).value();
    std::cout << "  CV-selected fraction: " << cv2_res.fractionUsed() << '\n';

    // 18. Parallel smoothing
    std::cout << "\n--- Parallel Smoothing ---\n";
    fastloess::LoessOptions par_opts;
    par_opts.fraction = k_basic_fraction;
    par_opts.parallel = true;
    auto par_res = fastloess::Loess(par_opts).fit(data.x, data.y).value();
    std::cout << "  parallel result size: " << par_res.size() << '\n';

    // 19. Expected<> error path: hasValue() / error()
    std::cout << "\n--- Expected<> Error Path ---\n";
    {
      const fastloess::LoessOptions err_opts;
      const std::vector<double> short_x = {1.0, 2.0, 3.0};
      const std::vector<double> short_y = {1.0, 2.0};
      auto err_exp = fastloess::Loess(err_opts).fit(short_x, short_y);
      if (!err_exp.hasValue()) {
        std::cout << "  hasValue()=false, error: " << err_exp.error() << '\n';
      }
    }

    // 20. Full diagnostics (aic, aicc, effectiveDf, residualSd, hasValue)
    std::cout << "\n--- Full Diagnostics ---\n";
    fastloess::LoessOptions full_diag_opts;
    full_diag_opts.fraction = k_basic_fraction;
    full_diag_opts.return_diagnostics = true;
    auto full_diag_res =
        fastloess::Loess(full_diag_opts).fit(data.x, data.y).value();
    const auto full_diag = full_diag_res.diagnostics();
    if (full_diag.hasValue()) {
      std::cout << "  aic:          " << full_diag.aic() << '\n';
      std::cout << "  aicc:         " << full_diag.aicc() << '\n';
      std::cout << "  effectiveDf:  " << full_diag.effectiveDf() << '\n';
      std::cout << "  residualSd:   " << full_diag.residualSd() << '\n';
    }

    std::cout << "\n=== Batch Smoothing Example Complete ===\n";

  } catch (const std::exception &exception) {
    std::fputs("Error: ", stderr);
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
  }
  return 0;
}
