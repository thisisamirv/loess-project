/**
 * @file streaming_smoothing.cpp
 * @brief Streaming LOESS smoothing example
 *
 * Demonstrates chunk-based processing for large datasets.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

#include "../../bindings/cpp/include/fastloess.hpp"

namespace {

constexpr size_t k_point_count = 10000;
constexpr unsigned int k_random_seed = 42;
constexpr double k_noise_std_dev = 0.5;
constexpr double k_sine_divisor = 10.0;
constexpr double k_scale_divisor = 50.0;
constexpr double k_fraction = 0.1;
constexpr int k_chunk_size = 1000;
constexpr int k_overlap = 100;
constexpr size_t k_progress_interval = 2000;

} // namespace

int main() {
  try {
    std::cout << "=== Streaming LOESS Smoothing Example ===\n";

    // Generate large synthetic dataset
    const size_t point_count = k_point_count;
    std::vector<double> x_values(point_count);
    std::vector<double> y_values(point_count);

    std::seed_seq generator_seed = {k_random_seed, k_random_seed, k_random_seed,
                                    k_random_seed};
    std::mt19937 generator(generator_seed);
    std::normal_distribution<> noise(0.0, k_noise_std_dev);

    for (size_t point_index = 0; point_index < point_count; ++point_index) {
      x_values[point_index] = static_cast<double>(point_index) / k_overlap;
      y_values[point_index] =
          ((std::sin(x_values[point_index] / k_sine_divisor) *
            x_values[point_index]) /
           k_scale_divisor) +
          noise(generator);
    }

    std::cout << "Generated " << point_count << " data points\n";

    // Streaming smoothing
    fastloess::StreamingOptions opts;
    opts.fraction = k_fraction;
    opts.iterations = 2;
    opts.chunk_size = k_chunk_size;
    opts.overlap = k_overlap;
    opts.return_diagnostics = true;

    std::cout << "\nProcessing with chunk_size=" << opts.chunk_size
              << ", overlap=" << opts.overlap << '\n';

    fastloess::StreamingLoess model(opts);

    std::cout << "\nProcessing data in chunks...\n";

    const size_t chunk_size = static_cast<size_t>(opts.chunk_size);
    size_t total_processed = 0;

    for (size_t chunk_start = 0; chunk_start < point_count;
         chunk_start += chunk_size) {
      const size_t current_chunk_len =
          std::min(chunk_size, point_count - chunk_start);
      std::vector<double> x_chunk(current_chunk_len);
      std::vector<double> y_chunk(current_chunk_len);

      std::copy_n(x_values.begin() + static_cast<std::ptrdiff_t>(chunk_start),
                  static_cast<std::ptrdiff_t>(current_chunk_len),
                  x_chunk.begin());
      std::copy_n(y_values.begin() + static_cast<std::ptrdiff_t>(chunk_start),
                  static_cast<std::ptrdiff_t>(current_chunk_len),
                  y_chunk.begin());

      auto res = model.process_chunk(x_chunk, y_chunk).value();
      total_processed += res.size();

      if (chunk_start % k_progress_interval == 0) {
        std::cout << "  Processed " << chunk_start << " points...\n";
      }
    }

    auto final_res = model.finalize().value();
    total_processed += final_res.size();

    std::cout << "\nStreaming completed:\n";
    std::cout << "  Total points smoothed: " << total_processed << '\n';

    // Show sample of final results
    if (final_res.size() > 0) {
      std::cout << "\nSample from final chunk:\n";
      std::cout << "  x=" << final_res.x_value(0)
                << " y=" << final_res.y_value(0) << '\n';
    }

    // Merge strategy variants
    std::cout << "\n--- Merge Strategy Variants ---\n";
    for (const char *strat : {"weighted", "average", "first", "last"}) {
      fastloess::StreamingOptions ms_opts;
      ms_opts.fraction = k_fraction;
      ms_opts.chunk_size = k_chunk_size;
      ms_opts.merge_strategy = strat;
      fastloess::StreamingLoess ms_model(ms_opts);
      const std::vector<double> x_s(x_values.begin(),
                                    x_values.begin() + k_overlap);
      const std::vector<double> y_s(y_values.begin(),
                                    y_values.begin() + k_overlap);
      auto ms_r1 = ms_model.process_chunk(x_s, y_s).value();
      auto ms_r2 = ms_model.finalize().value();
      std::cout << "  merge_strategy=" << strat
                << "  total=" << ms_r1.size() + ms_r2.size() << '\n';
    }

    // Advanced inherited options: degree, scaling_method, distance_metric,
    // surface_mode, return_se, return_residuals, zero_weight_fallback
    std::cout << "\n--- Advanced Streaming Options ---\n";
    {
      fastloess::StreamingOptions adv_opts;
      adv_opts.fraction = k_fraction;
      adv_opts.chunk_size = k_chunk_size;
      adv_opts.degree = "quadratic";
      adv_opts.scaling_method = "mean";
      adv_opts.distance_metric = "euclidean";
      adv_opts.surface_mode = "direct";
      adv_opts.return_se = true;
      adv_opts.return_residuals = true;
      adv_opts.zero_weight_fallback = "return_original";
      fastloess::StreamingLoess adv_model(adv_opts);
      const std::vector<double> a_x(
          x_values.begin(),
          x_values.begin() + static_cast<std::ptrdiff_t>(k_chunk_size));
      const std::vector<double> a_y(
          y_values.begin(),
          y_values.begin() + static_cast<std::ptrdiff_t>(k_chunk_size));
      auto adv_r1 = adv_model.process_chunk(a_x, a_y).value();
      auto adv_r2 = adv_model.finalize().value();
      std::cout << "  total points: " << adv_r1.size() + adv_r2.size() << '\n';
    }

    std::cout << "\n=== Example completed successfully ===\n";

  } catch (const std::exception &exception) {
    std::fputs("Error: ", stderr);
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
  }
  return 0;
}
