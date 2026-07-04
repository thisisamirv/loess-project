#include "../../bindings/cpp/include/fastloess.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace fastloess;

namespace {

// ── Named constants ────────────────────────────────────────────────────────
constexpr double k_default_epsilon = 1e-10;
constexpr double k_fraction_half = 0.5;
constexpr double k_fraction_third = 0.3;
constexpr double k_fraction_seventh = 0.7;
constexpr double k_fraction_tenth = 0.1;
constexpr double k_confidence_level = 0.95;
constexpr double k_auto_converge_tol = 1e-3;
constexpr double k_domain_end_ten = 10.0;
constexpr double k_domain_end_hundred = 100.0;
constexpr double k_domain_end_thousand = 1000.0;
constexpr double k_linear_slope = 2.0;
constexpr double k_linear_intercept = 1.0;
constexpr size_t k_small_count = 5;
constexpr size_t k_twenty_count = 20;
constexpr size_t k_hundred_count = 100;
constexpr size_t k_two_hundred_count = 200;
constexpr size_t k_two_thousand_count = 2000;
constexpr size_t k_chunk_small = 1000;
constexpr size_t k_chunk_large = 5000;
constexpr size_t k_window_capacity = 10;
constexpr size_t k_min_points_online = 3;
constexpr size_t k_thirty_count = 30;
constexpr size_t k_chunk_half = 15;
constexpr size_t k_mismatch_x_count = 3;
constexpr size_t k_mismatch_y_count = 2;
constexpr int k_iterations3 = 3;
constexpr int k_iterations2 = 2;
constexpr int k_cv_k = 3;
constexpr int k_overlap_size = 3;

// ── Test fixture data ──────────────────────────────────────────────────────
// Constexpr arrays: literals in constexpr initializers are not magic numbers.
constexpr std::array<double, k_small_count> k_simple_x = {1.0, 2.0, 3.0, 4.0,
                                                          5.0};
constexpr std::array<double, k_small_count> k_simple_y_noisy = {2.0, 4.1, 5.9,
                                                                8.2, 9.8};
constexpr std::array<double, k_small_count> k_simple_y_outlier = {
    2.0, 4.1, 100.0, 8.2, 9.8};
constexpr std::array<double, k_small_count> k_reuse_x2 = {10.0, 20.0, 30.0,
                                                          40.0, 50.0};
constexpr std::array<double, k_small_count> k_reuse_y2 = {20.0, 40.0, 60.0,
                                                          80.0, 100.0};
constexpr std::array<double, k_mismatch_x_count> k_mismatch_x_arr = {1.0, 2.0,
                                                                     3.0};
constexpr std::array<double, k_mismatch_y_count> k_mismatch_y_arr = {2.0, 4.0};

// ── Assert helpers ─────────────────────────────────────────────────────────
bool isApprox(double lhs, double rhs, double epsilon = k_default_epsilon) {
  if (std::isnan(lhs) && std::isnan(rhs)) {
    return true;
  }
  return std::abs(lhs - rhs) < epsilon;
}

void assertApprox(double lhs, double rhs, const std::string &msg = "") {
  if (!isApprox(lhs, rhs)) {
    std::cerr << "Assertion failed: " << lhs << " != " << rhs << " " << msg
              << '\n';
    std::exit(1);
  }
}

void assertApprox(double lhs, double rhs, double epsilon) {
  if (!isApprox(lhs, rhs, epsilon)) {
    std::cerr << "Assertion failed: " << lhs << " != " << rhs
              << " (eps=" << epsilon << ")" << '\n';
    std::exit(1);
  }
}

void assertTrue(bool cond, const std::string &msg = "") {
  if (!cond) {
    std::cerr << "Assertion failed " << msg << '\n';
    std::exit(1);
  }
}

// ── Batch LOESS tests ──────────────────────────────────────────────────────
void testBasicSmooth() {
  std::cout << "Running testBasicSmooth...\n";
  const std::vector<double> x_vals(k_simple_x.begin(), k_simple_x.end());
  const std::vector<double> y_vals(k_simple_y_noisy.begin(),
                                   k_simple_y_noisy.end());

  LoessOptions opts;
  opts.fraction = k_fraction_half;
  Loess loess(opts);
  auto result = loess.fit(x_vals, y_vals).value();

  assertTrue(result.valid(), "Result should be valid");
  assertTrue(result.yVector().size() == k_small_count,
             "Output length mismatch");
  assertTrue(result.xVector().size() == k_small_count, "X length mismatch");
  assertApprox(result.fractionUsed(), k_fraction_half);
}

void testBasicSmoothSerial() {
  std::cout << "Running testBasicSmoothSerial...\n";
  const std::vector<double> x_vals(k_simple_x.begin(), k_simple_x.end());
  const std::vector<double> y_vals(k_simple_y_noisy.begin(),
                                   k_simple_y_noisy.end());

  LoessOptions opts;
  opts.fraction = k_fraction_half;
  opts.parallel = false;
  Loess loess(opts);
  auto result = loess.fit(x_vals, y_vals).value();

  assertTrue(result.valid());
  assertTrue(result.yVector().size() == k_small_count);
}

void testLoessWithDiagnostics() {
  std::cout << "Running testLoessWithDiagnostics...\n";
  const std::vector<double> x_vals(k_simple_x.begin(), k_simple_x.end());
  const std::vector<double> y_vals(k_simple_y_noisy.begin(),
                                   k_simple_y_noisy.end());

  LoessOptions opts;
  opts.fraction = k_fraction_half;
  opts.return_diagnostics = true;
  Loess loess(opts);
  auto result = loess.fit(x_vals, y_vals).value();

  auto diag = result.diagnostics();
  assertTrue(diag.rmse() >= 0, "RMSE negative");
  assertTrue(diag.mae() >= 0, "MAE negative");
  assertTrue(diag.rSquared() >= 0 && diag.rSquared() <= 1, "R2 out of range");
}

void testLoessWithResiduals() {
  std::cout << "Running testLoessWithResiduals...\n";
  const std::vector<double> x_vals(k_simple_x.begin(), k_simple_x.end());
  const std::vector<double> y_vals(k_simple_y_noisy.begin(),
                                   k_simple_y_noisy.end());

  LoessOptions opts;
  opts.fraction = k_fraction_half;
  opts.return_residuals = true;
  Loess loess(opts);
  auto result = loess.fit(x_vals, y_vals).value();

  assertTrue(result.residuals().size() == k_small_count, "Residuals missing");
}

void testLoessWithRobustnessWeights() {
  std::cout << "Running testLoessWithRobustnessWeights...\n";
  const std::vector<double> x_vals(k_simple_x.begin(), k_simple_x.end());
  const std::vector<double> y_vals(k_simple_y_outlier.begin(),
                                   k_simple_y_outlier.end());

  LoessOptions opts;
  opts.fraction = k_fraction_seventh;
  opts.iterations = k_iterations3;
  opts.return_robustness_weights = true;
  Loess loess(opts);
  auto result = loess.fit(x_vals, y_vals).value();

  auto weights = result.robustnessWeights();
  assertTrue(weights.size() == k_small_count);
  for (const double weight : weights) {
    assertTrue(weight >= 0 && weight <= 1, "Weight out of range");
  }
}

void testLoessWithConfidenceIntervals() {
  std::cout << "Running testLoessWithConfidenceIntervals...\n";
  std::vector<double> x_vals(k_twenty_count);
  std::vector<double> y_vals(k_twenty_count);
  const double x_step =
      k_domain_end_ten / static_cast<double>(k_twenty_count - 1);
  for (size_t idx = 0; idx < k_twenty_count; ++idx) {
    x_vals[idx] = static_cast<double>(idx) * x_step;
    y_vals[idx] = k_linear_slope * x_vals[idx];
  }

  LoessOptions opts;
  opts.fraction = k_fraction_half;
  opts.confidence_intervals = k_confidence_level;
  Loess loess(opts);
  auto result = loess.fit(x_vals, y_vals).value();

  auto conf_lower = result.confidenceLower();
  auto conf_upper = result.confidenceUpper();
  assertTrue(conf_lower.size() == k_twenty_count);
  assertTrue(conf_upper.size() == k_twenty_count);
  for (size_t idx = 0; idx < k_twenty_count; ++idx) {
    assertTrue(conf_lower[idx] <= conf_upper[idx], "Lower > Upper confidence");
  }
}

void testLoessWithPredictionIntervals() {
  std::cout << "Running testLoessWithPredictionIntervals...\n";
  std::vector<double> x_vals(k_twenty_count);
  std::vector<double> y_vals(k_twenty_count);
  const double x_step =
      k_domain_end_ten / static_cast<double>(k_twenty_count - 1);
  for (size_t idx = 0; idx < k_twenty_count; ++idx) {
    x_vals[idx] = static_cast<double>(idx) * x_step;
    y_vals[idx] = k_linear_slope * x_vals[idx];
  }

  LoessOptions opts;
  opts.fraction = k_fraction_half;
  opts.prediction_intervals = k_confidence_level;
  Loess loess(opts);
  auto result = loess.fit(x_vals, y_vals).value();

  assertTrue(result.predictionLower().size() == k_twenty_count);
  assertTrue(result.predictionUpper().size() == k_twenty_count);
}

void testLoessReuse() {
  std::cout << "Running testLoessReuse...\n";
  const std::vector<double> x_vals1(k_simple_x.begin(), k_simple_x.end());
  const std::vector<double> y_vals1(k_simple_y_noisy.begin(),
                                    k_simple_y_noisy.end());
  const std::vector<double> x_vals2(k_reuse_x2.begin(), k_reuse_x2.end());
  const std::vector<double> y_vals2(k_reuse_y2.begin(), k_reuse_y2.end());

  LoessOptions opts;
  opts.fraction = k_fraction_half;
  opts.return_diagnostics = true;
  Loess loess(opts);

  auto result1 = loess.fit(x_vals1, y_vals1).value();
  auto result2 = loess.fit(x_vals2, y_vals2).value();

  assertTrue(result1.yVector().size() == k_small_count);
  assertTrue(result2.yVector().size() == k_small_count);
}

// ── Streaming LOESS tests ──────────────────────────────────────────────────
void testStreamingReturnsAllPoints() {
  std::cout << "Running testStreamingReturnsAllPoints...\n";
  std::vector<double> x_vals(k_hundred_count);
  std::vector<double> y_vals(k_hundred_count);
  const double x_step =
      k_domain_end_hundred / static_cast<double>(k_hundred_count - 1);
  for (size_t idx = 0; idx < k_hundred_count; ++idx) {
    x_vals[idx] = static_cast<double>(idx) * x_step;
    y_vals[idx] = (k_linear_slope * x_vals[idx]) + k_linear_intercept;
  }

  StreamingOptions opts;
  opts.fraction = k_fraction_third;
  opts.chunk_size = k_chunk_large; // > k_hundred_count
  StreamingLoess stream(opts);

  auto val1 = stream.processChunk(x_vals, y_vals).value();
  auto val2 = stream.finalize().value();

  assertTrue(val1.yVector().size() + val2.yVector().size() == k_hundred_count,
             "Total points mismatch");
}

void testStreamingBasic() {
  std::cout << "Running testStreamingBasic...\n";
  std::vector<double> x_vals(k_two_thousand_count);
  std::vector<double> y_vals(k_two_thousand_count);
  const double x_step =
      k_domain_end_thousand / static_cast<double>(k_two_thousand_count - 1);
  for (size_t idx = 0; idx < k_two_thousand_count; ++idx) {
    x_vals[idx] = static_cast<double>(idx) * x_step;
    y_vals[idx] = std::sin(x_vals[idx] / k_domain_end_hundred);
  }

  StreamingOptions opts;
  opts.fraction = k_fraction_tenth;
  opts.chunk_size = k_chunk_small;
  StreamingLoess stream(opts);

  auto chunk_result = stream.processChunk(x_vals, y_vals).value();
  auto final_result = stream.finalize().value();
  (void)chunk_result;
  (void)final_result;
}

void testStreamingAccuracy() {
  std::cout << "Running testStreamingAccuracy...\n";
  std::vector<double> x_vals(k_two_hundred_count);
  std::vector<double> y_vals(k_two_hundred_count);
  const double x_step =
      k_domain_end_hundred / static_cast<double>(k_two_hundred_count - 1);
  for (size_t idx = 0; idx < k_two_hundred_count; ++idx) {
    x_vals[idx] = static_cast<double>(idx) * x_step;
    y_vals[idx] = (k_linear_slope * x_vals[idx]) + k_linear_intercept;
  }

  // Streaming
  StreamingOptions sopts;
  sopts.fraction = k_fraction_half;
  sopts.chunk_size = k_chunk_small;
  StreamingLoess stream(sopts);
  auto val1 = stream.processChunk(x_vals, y_vals).value();
  auto val2 = stream.finalize().value();

  std::vector<double> stream_y;
  auto y_vec1 = val1.yVector();
  stream_y.insert(stream_y.end(), y_vec1.begin(), y_vec1.end());
  auto y_vec2 = val2.yVector();
  stream_y.insert(stream_y.end(), y_vec2.begin(), y_vec2.end());

  // Batch
  LoessOptions bopts;
  bopts.fraction = k_fraction_half;
  Loess batch(bopts);
  auto bres = batch.fit(x_vals, y_vals).value();
  auto batch_y = bres.yVector();

  assertTrue(stream_y.size() == batch_y.size());
  for (size_t idx = 0; idx < k_two_hundred_count; ++idx) {
    assertApprox(stream_y[idx], batch_y[idx], k_default_epsilon);
  }
}

// ── Online LOESS tests ─────────────────────────────────────────────────────
void testOnlineBasic() {
  std::cout << "Running testOnlineBasic...\n";
  std::vector<double> x_vals(k_window_capacity);
  std::vector<double> y_vals(k_window_capacity);
  for (size_t idx = 0; idx < k_window_capacity; ++idx) {
    x_vals[idx] = static_cast<double>(idx) + 1.0;
    y_vals[idx] = k_linear_slope * x_vals[idx];
  }

  OnlineOptions opts;
  opts.fraction = k_fraction_half;
  opts.window_capacity = k_window_capacity;
  opts.min_points = k_min_points_online;
  OnlineLoess online(opts);

  int points_out = 0;
  for (size_t idx = 0; idx < x_vals.size(); ++idx) {
    const std::vector<double> x_val = {x_vals[idx]};
    const std::vector<double> y_val = {y_vals[idx]};
    auto res = online.addPoints(x_val, y_val).value();
    if (!res.yVector().empty()) {
      points_out++;
    }
  }
  assertTrue(points_out > 0);
}

// ── Error handling tests ───────────────────────────────────────────────────
void testMismatchedLengths() {
  std::cout << "Running testMismatchedLengths...\n";
  const std::vector<double> x_vals(k_mismatch_x_arr.begin(),
                                   k_mismatch_x_arr.end());
  const std::vector<double> y_vals(k_mismatch_y_arr.begin(),
                                   k_mismatch_y_arr.end());

  const LoessOptions opts;
  Loess loess(opts);
  try {
    loess.fit(x_vals, y_vals).value();
    assertTrue(false, "Should have thrown");
  } catch (const std::exception &err) {
    (void)err; // expected: mismatched lengths throw
  }

  // Also test checking hasValue()
  auto res = loess.fit(x_vals, y_vals);
  assertTrue(!res.hasValue());
  assertTrue(!res.error().empty());
}

// ── Parameter coverage tests ───────────────────────────────────────────────

// Helper: 30-point linear data y = 2x+1
std::pair<std::vector<double>, std::vector<double>> makeLinear30() {
  std::vector<double> x_vals(k_thirty_count);
  std::vector<double> y_vals(k_thirty_count);
  for (size_t i = 0; i < k_thirty_count; ++i) {
    x_vals[i] = static_cast<double>(i);
    y_vals[i] = (k_linear_slope * x_vals[i]) + k_linear_intercept;
  }
  return {x_vals, y_vals};
}

void testLoessScalingMethods() {
  std::cout << "Running testLoessScalingMethods...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  for (const char *method : {"mad", "mar", "mean"}) {
    LoessOptions opts;
    opts.fraction = k_fraction_half;
    opts.scaling_method = method;
    Loess loess(opts);
    auto res = loess.fit(x_vals, y_vals).value();
    assertTrue(res.yVector().size() == k_thirty_count, method);
  }
}

void testLoessBoundaryPolicies() {
  std::cout << "Running testLoessBoundaryPolicies...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  for (const char *policy : {"extend", "reflect", "zero", "noboundary"}) {
    LoessOptions opts;
    opts.fraction = k_fraction_half;
    opts.boundary_policy = policy;
    Loess loess(opts);
    auto res = loess.fit(x_vals, y_vals).value();
    assertTrue(res.yVector().size() == k_thirty_count, policy);
  }
}

void testLoessZeroWeightFallback() {
  std::cout << "Running testLoessZeroWeightFallback...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  for (const char *fallback_name :
       {"use_local_mean", "return_original", "return_none"}) {
    LoessOptions opts;
    opts.fraction = k_fraction_half;
    opts.zero_weight_fallback = fallback_name;
    Loess loess(opts);
    auto res = loess.fit(x_vals, y_vals).value();
    assertTrue(res.yVector().size() == k_thirty_count, fallback_name);
  }
}

void testLoessAutoConverge() {
  std::cout << "Running testLoessAutoConverge...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  LoessOptions opts;
  opts.fraction = k_fraction_half;
  opts.auto_converge = k_auto_converge_tol;
  Loess loess(opts);
  auto res = loess.fit(x_vals, y_vals).value();
  assertTrue(res.yVector().size() == k_thirty_count);
}

void testLoessPolynomialDegrees() {
  std::cout << "Running testLoessPolynomialDegrees...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  for (const char *deg : {"constant", "linear", "quadratic"}) {
    LoessOptions opts;
    opts.fraction = k_fraction_half;
    opts.degree = deg;
    Loess loess(opts);
    auto res = loess.fit(x_vals, y_vals).value();
    assertTrue(res.yVector().size() == k_thirty_count, deg);
  }
}

void testLoessDistanceMetrics() {
  std::cout << "Running testLoessDistanceMetrics...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  for (const char *metric : {"euclidean", "manhattan", "chebyshev"}) {
    LoessOptions opts;
    opts.fraction = k_fraction_half;
    opts.distance_metric = metric;
    Loess loess(opts);
    auto res = loess.fit(x_vals, y_vals).value();
    assertTrue(res.yVector().size() == k_thirty_count, metric);
  }
}

void testLoessSurfaceModeAndReturnSe() {
  std::cout << "Running testLoessSurfaceModeAndReturnSe...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  LoessOptions opts;
  opts.fraction = k_fraction_half;
  opts.surface_mode = "direct";
  opts.return_se = true;
  Loess loess(opts);
  auto res = loess.fit(x_vals, y_vals).value();
  assertTrue(res.yVector().size() == k_thirty_count);
  assertTrue(!std::isnan(res.enp()), "enp should be set with return_se+direct");
  auto std_errors = res.standardErrors();
  assertTrue(std_errors.size() == k_thirty_count,
             "Standard errors should be populated");
}

void testLoessWeightFunctions() {
  std::cout << "Running testLoessWeightFunctions...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  for (const char *weight_fn : {"epanechnikov", "gaussian", "uniform",
                                "biweight", "triangle", "cosine"}) {
    LoessOptions opts;
    opts.fraction = k_fraction_half;
    opts.weight_function = weight_fn;
    Loess loess(opts);
    auto res = loess.fit(x_vals, y_vals).value();
    assertTrue(res.yVector().size() == k_thirty_count, weight_fn);
  }
}

void testLoessRobustnessMethods() {
  std::cout << "Running testLoessRobustnessMethods...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  for (const char *robustness_m : {"huber", "talwar"}) {
    LoessOptions opts;
    opts.fraction = k_fraction_half;
    opts.iterations = k_iterations2;
    opts.robustness_method = robustness_m;
    Loess loess(opts);
    auto res = loess.fit(x_vals, y_vals).value();
    assertTrue(res.yVector().size() == k_thirty_count, robustness_m);
  }
}

void testLoessCrossValidation() {
  std::cout << "Running testLoessCrossValidation...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  LoessOptions opts;
  opts.cv_fractions = {k_fraction_third, k_fraction_half, k_fraction_seventh};
  opts.cv_method = "kfold";
  opts.cv_k = k_cv_k;
  Loess loess(opts);
  auto res = loess.fit(x_vals, y_vals).value();
  assertTrue(res.valid());
}

void testStreamingMergeStrategies() {
  std::cout << "Running testStreamingMergeStrategies...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  for (const char *merge_strat : {"average", "first", "last"}) {
    StreamingOptions opts;
    opts.fraction = k_fraction_half;
    opts.chunk_size = k_chunk_half;
    opts.merge_strategy = merge_strat;
    StreamingLoess stream(opts);
    auto chunk_res = stream.processChunk(x_vals, y_vals).value();
    auto final_res = stream.finalize().value();
    assertTrue(chunk_res.yVector().size() + final_res.yVector().size() ==
                   k_thirty_count,
               merge_strat);
  }
}

void testStreamingOverlapAndParams() {
  std::cout << "Running testStreamingOverlapAndParams...\n";
  auto data = makeLinear30();
  auto x_vals = data.first;
  auto y_vals = data.second;
  StreamingOptions opts;
  opts.fraction = k_fraction_half;
  opts.chunk_size = k_chunk_half;
  opts.overlap = k_overlap_size;
  opts.scaling_method = "mean";
  opts.boundary_policy = "reflect";
  opts.degree = "quadratic";
  opts.distance_metric = "manhattan";
  opts.surface_mode = "direct";
  opts.return_se = true;
  StreamingLoess stream(opts);
  auto chunk_res = stream.processChunk(x_vals, y_vals).value();
  auto final_res = stream.finalize().value();
  assertTrue(chunk_res.yVector().size() + final_res.yVector().size() ==
             k_thirty_count);
}

void testOnlineUpdateModeAndParams() {
  std::cout << "Running testOnlineUpdateModeAndParams...\n";
  OnlineOptions opts;
  opts.fraction = k_fraction_half;
  opts.window_capacity = k_window_capacity;
  opts.min_points = k_min_points_online;
  opts.update_mode = "incremental";
  opts.zero_weight_fallback = "return_original";
  opts.scaling_method = "mar";
  opts.boundary_policy = "zero";
  opts.degree = "quadratic";
  opts.distance_metric = "chebyshev";
  OnlineLoess online(opts);
  std::vector<double> x_vals(k_window_capacity);
  std::vector<double> y_vals(k_window_capacity);
  for (size_t i = 0; i < k_window_capacity; ++i) {
    x_vals[i] = static_cast<double>(i);
    y_vals[i] = k_linear_slope * x_vals[i];
  }
  auto res = online.addPoints(x_vals, y_vals).value();
  assertTrue(res.valid() ||
             res.yVector().empty()); // may be empty until min_points reached
}

} // namespace

// std::ios_base::failure can theoretically propagate through stream I/O even
// though it never does with default exception masks (goodbit).  Suppress the
// clang-tidy warning rather than adding no-op exceptions() calls that
// themselves trigger the same diagnostic on MSVC headers.
// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
  try {
    testBasicSmooth();
    testBasicSmoothSerial();
    testLoessWithDiagnostics();
    testLoessWithResiduals();
    testLoessWithRobustnessWeights();
    testLoessWithConfidenceIntervals();
    testLoessWithPredictionIntervals();
    testLoessReuse();

    testStreamingReturnsAllPoints();
    testStreamingBasic();
    testStreamingAccuracy();

    testOnlineBasic();

    testMismatchedLengths();

    testLoessScalingMethods();
    testLoessBoundaryPolicies();
    testLoessZeroWeightFallback();
    testLoessAutoConverge();
    testLoessPolynomialDegrees();
    testLoessDistanceMetrics();
    testLoessSurfaceModeAndReturnSe();
    testLoessWeightFunctions();
    testLoessRobustnessMethods();
    testLoessCrossValidation();
    testStreamingMergeStrategies();
    testStreamingOverlapAndParams();
    testOnlineUpdateModeAndParams();

    std::cout << "All C++ tests passed!\n";
  } catch (const std::exception &err) {
    std::cerr << "Test failed with exception: " << err.what() << '\n';
    return 1;
  } catch (...) {
    return 1;
  }
  return 0;
}
