#include "../../bindings/cpp/include/fastloess.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace fastloess;

namespace {

// ── Named constants ────────────────────────────────────────────────────────
constexpr double kDefaultEpsilon = 1e-10;
constexpr double kFractionHalf = 0.5;
constexpr double kFractionThird = 0.3;
constexpr double kFractionSeventh = 0.7;
constexpr double kFractionTenth = 0.1;
constexpr double kConfidenceLevel = 0.95;
constexpr size_t kSmallCount = 5;
constexpr size_t kTwentyCount = 20;
constexpr size_t kHundredCount = 100;
constexpr size_t kTwoHundredCount = 200;
constexpr size_t kTwoThousandCount = 2000;
constexpr size_t kChunkSmall = 1000;
constexpr size_t kChunkLarge = 5000;
constexpr size_t kWindowCapacity = 10;
constexpr size_t kMinPointsOnline = 3;
constexpr int kIterations3 = 3;

// ── Assert helpers ─────────────────────────────────────────────────────────
bool isApprox(double lhs, double rhs, double epsilon = kDefaultEpsilon) {
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
  std::vector<double> xVals = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> yVals = {2.0, 4.1, 5.9, 8.2, 9.8};

  LoessOptions opts;
  opts.fraction = kFractionHalf;
  Loess loess(opts);
  auto result = loess.fit(xVals, yVals).value();

  assertTrue(result.valid(), "Result should be valid");
  assertTrue(result.yVector().size() == kSmallCount, "Output length mismatch");
  assertTrue(result.xVector().size() == kSmallCount, "X length mismatch");
  assertApprox(result.fractionUsed(), kFractionHalf);
}

void testBasicSmoothSerial() {
  std::cout << "Running testBasicSmoothSerial...\n";
  std::vector<double> xVals = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> yVals = {2.0, 4.1, 5.9, 8.2, 9.8};

  LoessOptions opts;
  opts.fraction = kFractionHalf;
  opts.parallel = false;
  Loess loess(opts);
  auto result = loess.fit(xVals, yVals).value();

  assertTrue(result.valid());
  assertTrue(result.yVector().size() == kSmallCount);
}

void testLoessWithDiagnostics() {
  std::cout << "Running testLoessWithDiagnostics...\n";
  std::vector<double> xVals = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> yVals = {2.0, 4.1, 5.9, 8.2, 9.8};

  LoessOptions opts;
  opts.fraction = kFractionHalf;
  opts.return_diagnostics = true;
  Loess loess(opts);
  auto result = loess.fit(xVals, yVals).value();

  auto diag = result.diagnostics();
  assertTrue(diag.rmse() >= 0, "RMSE negative");
  assertTrue(diag.mae() >= 0, "MAE negative");
  assertTrue(diag.rSquared() >= 0 && diag.rSquared() <= 1, "R2 out of range");
}

void testLoessWithResiduals() {
  std::cout << "Running testLoessWithResiduals...\n";
  std::vector<double> xVals = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> yVals = {2.0, 4.1, 5.9, 8.2, 9.8};

  LoessOptions opts;
  opts.fraction = kFractionHalf;
  opts.return_residuals = true;
  Loess loess(opts);
  auto result = loess.fit(xVals, yVals).value();

  assertTrue(result.residuals().size() == kSmallCount, "Residuals missing");
}

void testLoessWithRobustnessWeights() {
  std::cout << "Running testLoessWithRobustnessWeights...\n";
  std::vector<double> xVals = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> yVals = {2.0, 4.1, 100.0, 8.2, 9.8}; // Outlier

  LoessOptions opts;
  opts.fraction = kFractionSeventh;
  opts.iterations = kIterations3;
  opts.return_robustness_weights = true;
  Loess loess(opts);
  auto result = loess.fit(xVals, yVals).value();

  auto weights = result.robustnessWeights();
  assertTrue(weights.size() == kSmallCount);
  for (double weight : weights) {
    assertTrue(weight >= 0 && weight <= 1, "Weight out of range");
  }
}

void testLoessWithConfidenceIntervals() {
  std::cout << "Running testLoessWithConfidenceIntervals...\n";
  std::vector<double> xVals(kTwentyCount);
  std::vector<double> yVals(kTwentyCount);
  for (size_t idx = 0; idx < kTwentyCount; ++idx) {
    xVals[idx] = static_cast<double>(idx) * (10.0 / 19.0);
    yVals[idx] = 2 * xVals[idx]; // simple linear
  }

  LoessOptions opts;
  opts.fraction = kFractionHalf;
  opts.confidence_intervals = kConfidenceLevel;
  Loess loess(opts);
  auto result = loess.fit(xVals, yVals).value();

  auto confLower = result.confidenceLower();
  auto confUpper = result.confidenceUpper();
  assertTrue(confLower.size() == kTwentyCount);
  assertTrue(confUpper.size() == kTwentyCount);
  for (size_t idx = 0; idx < kTwentyCount; ++idx) {
    assertTrue(confLower[idx] <= confUpper[idx], "Lower > Upper confidence");
  }
}

void testLoessWithPredictionIntervals() {
  std::cout << "Running testLoessWithPredictionIntervals...\n";
  std::vector<double> xVals(kTwentyCount);
  std::vector<double> yVals(kTwentyCount);
  for (size_t idx = 0; idx < kTwentyCount; ++idx) {
    xVals[idx] = static_cast<double>(idx) * (10.0 / 19.0);
    yVals[idx] = 2 * xVals[idx];
  }

  LoessOptions opts;
  opts.fraction = kFractionHalf;
  opts.prediction_intervals = kConfidenceLevel;
  Loess loess(opts);
  auto result = loess.fit(xVals, yVals).value();

  assertTrue(result.predictionLower().size() == kTwentyCount);
  assertTrue(result.predictionUpper().size() == kTwentyCount);
}

void testLoessReuse() {
  std::cout << "Running testLoessReuse...\n";
  std::vector<double> xVals1 = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> yVals1 = {2.0, 4.1, 5.9, 8.2, 9.8};
  std::vector<double> xVals2 = {10.0, 20.0, 30.0, 40.0, 50.0};
  std::vector<double> yVals2 = {20.0, 40.0, 60.0, 80.0, 100.0};

  LoessOptions opts;
  opts.fraction = kFractionHalf;
  opts.return_diagnostics = true;
  Loess loess(opts);

  auto res1 = loess.fit(xVals1, yVals1).value();
  auto res2 = loess.fit(xVals2, yVals2).value();

  assertTrue(res1.yVector().size() == kSmallCount);
  assertTrue(res2.yVector().size() == kSmallCount);
}

// ── Streaming LOESS tests ──────────────────────────────────────────────────
void testStreamingReturnsAllPoints() {
  std::cout << "Running testStreamingReturnsAllPoints...\n";
  std::vector<double> xVals(kHundredCount);
  std::vector<double> yVals(kHundredCount);
  for (size_t idx = 0; idx < kHundredCount; ++idx) {
    xVals[idx] = static_cast<double>(idx) * (100.0 / 99.0);
    yVals[idx] = (2 * xVals[idx]) + 1;
  }

  StreamingOptions opts;
  opts.fraction = kFractionThird;
  opts.chunk_size = kChunkLarge; // > kHundredCount
  StreamingLoess stream(opts);

  auto val1 = stream.processChunk(xVals, yVals).value();
  auto val2 = stream.finalize().value();

  assertTrue(val1.yVector().size() + val2.yVector().size() == kHundredCount,
             "Total points mismatch");
}

void testStreamingBasic() {
  std::cout << "Running testStreamingBasic...\n";
  std::vector<double> xVals(kTwoThousandCount);
  std::vector<double> yVals(kTwoThousandCount);
  for (size_t idx = 0; idx < kTwoThousandCount; ++idx) {
    xVals[idx] = static_cast<double>(idx) * (1000.0 / 1999.0);
    yVals[idx] = std::sin(xVals[idx] / 100.0);
  }

  StreamingOptions opts;
  opts.fraction = kFractionTenth;
  opts.chunk_size = kChunkSmall;
  StreamingLoess stream(opts);

  auto chunkResult = stream.processChunk(xVals, yVals).value();
  auto finalResult = stream.finalize().value();
  (void)chunkResult;
  (void)finalResult;
}

void testStreamingAccuracy() {
  std::cout << "Running testStreamingAccuracy...\n";
  std::vector<double> xVals(kTwoHundredCount);
  std::vector<double> yVals(kTwoHundredCount);
  for (size_t idx = 0; idx < kTwoHundredCount; ++idx) {
    xVals[idx] = static_cast<double>(idx) * (100.0 / 199.0);
    yVals[idx] = (2 * xVals[idx]) + 1;
  }

  // Streaming
  StreamingOptions sopts;
  sopts.fraction = kFractionHalf;
  sopts.chunk_size = kChunkSmall;
  StreamingLoess stream(sopts);
  auto val1 = stream.processChunk(xVals, yVals).value();
  auto val2 = stream.finalize().value();

  std::vector<double> streamY;
  auto yVec1 = val1.yVector();
  streamY.insert(streamY.end(), yVec1.begin(), yVec1.end());
  auto yVec2 = val2.yVector();
  streamY.insert(streamY.end(), yVec2.begin(), yVec2.end());

  // Batch
  LoessOptions bopts;
  bopts.fraction = kFractionHalf;
  Loess batch(bopts);
  auto bres = batch.fit(xVals, yVals).value();
  auto batchY = bres.yVector();

  assertTrue(streamY.size() == batchY.size());
  for (size_t idx = 0; idx < kTwoHundredCount; ++idx) {
    assertApprox(streamY[idx], batchY[idx], kDefaultEpsilon);
  }
}

// ── Online LOESS tests ─────────────────────────────────────────────────────
void testOnlineBasic() {
  std::cout << "Running testOnlineBasic...\n";
  std::vector<double> xVals = {1.0, 2.0, 3.0, 4.0, 5.0,
                               6.0, 7.0, 8.0, 9.0, 10.0};
  std::vector<double> yVals = {2.0,  4.0,  6.0,  8.0,  10.0,
                               12.0, 14.0, 16.0, 18.0, 20.0};

  OnlineOptions opts;
  opts.fraction = kFractionHalf;
  opts.window_capacity = kWindowCapacity;
  opts.min_points = kMinPointsOnline;
  OnlineLoess online(opts);

  int pointsOut = 0;
  for (size_t idx = 0; idx < xVals.size(); ++idx) {
    std::vector<double> xVal = {xVals[idx]};
    std::vector<double> yVal = {yVals[idx]};
    auto res = online.addPoints(xVal, yVal).value();
    if (!res.yVector().empty()) {
      pointsOut++;
    }
  }
  assertTrue(pointsOut > 0);
}

// ── Error handling tests ───────────────────────────────────────────────────
void testMismatchedLengths() {
  std::cout << "Running testMismatchedLengths...\n";
  std::vector<double> xVals = {1.0, 2.0, 3.0};
  std::vector<double> yVals = {2.0, 4.0};

  LoessOptions opts;
  Loess loess(opts);
  try {
    loess.fit(xVals, yVals).value();
    assertTrue(false, "Should have thrown");
  } catch (const std::exception &err) {
    (void)err; // expected: mismatched lengths throw
  }

  // Also test checking hasValue()
  auto res = loess.fit(xVals, yVals);
  assertTrue(!res.hasValue());
  assertTrue(!res.error().empty());
}

} // namespace

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

    std::cout << "All C++ tests passed!\n";
  } catch (const std::exception &err) {
    std::cerr << "Test failed with exception: " << err.what() << '\n';
    return 1;
  }
  return 0;
}
