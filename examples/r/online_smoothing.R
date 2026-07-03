#!/usr/bin/env Rscript
# =============================================================================
# rfastloess Online Smoothing - Comprehensive Examples
#
#  1. Basic incremental processing
#  2. Real-time sensor data simulation
#  3. Outlier handling in online mode
#  4. Window size comparison
#  5. Memory-bounded processing (embedded systems)
#  6. Sliding window behavior
#  7. Benchmark (sequential online)
#  8. Update modes (Full vs Incremental) and min_points
#  9. Advanced online options
# =============================================================================

library(rfastloess)

# ── Example 1: Basic Incremental Processing ──────────────────────────────────
example_1_basic_streaming <- function() {
    cat("Example 1: Basic Incremental Processing\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    y <- c(3.1, 5.0, 7.2, 8.9, 11.1, 13.0, 15.2, 16.8, 19.1, 21.0)

    model <- OnlineLoess(fraction = 0.5, iterations = 2L,
                         window_capacity = 5L, return_robustness_weights = FALSE)
    result <- model$add_points(x, y)

    cat(sprintf("  %8s %12s %12s\n", "X", "Y_obs", "Y_smooth"))
    for (i in seq_along(result$y)) {
        cat(sprintf("  %8.2f %12.2f %12.2f\n", result$x[i], y[i], result$y[i]))
    }
    cat("\n")
}

# ── Example 2: Real-Time Sensor Data Simulation ───────────────────────────────
example_2_sensor_data_simulation <- function() {
    cat("Example 2: Real-Time Sensor Data Simulation\n")
    cat("  Simulating temperature sensor with noise...\n")

    n <- 24L
    hours <- as.numeric(0:(n - 1))
    temp <- 20 + 5 * sin(hours * pi / 12) + ((hours * 7) %% 11) * 0.3 - 1.5

    model <- OnlineLoess(fraction = 0.4, iterations = 3L,
                         robustness_method = "bisquare",
                         window_capacity = 12L)
    result <- model$add_points(hours, temp)

    cat(sprintf("  %6s %12s %12s\n", "Hour", "Raw", "Smoothed"))
    for (i in seq_along(result$y)) {
        cat(sprintf("  %6.0f %10.2f degC %10.2f degC\n",
            result$x[i], temp[i], result$y[i]))
    }
    cat("\n")
}

# ── Example 3: Outlier Handling in Online Mode ────────────────────────────────
example_3_outlier_handling <- function() {
    cat("Example 3: Outlier Handling in Online Mode\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    y <- c(2.0, 4.1, 5.9, 25.0, 10.1, 12.0, 14.1, 50.0, 18.0, 20.1)

    for (method in c("bisquare", "talwar")) {
        model <- OnlineLoess(fraction = 0.5, iterations = 5L,
                             robustness_method = method,
                             window_capacity = 6L)
        result <- model$add_points(x, y)
        cat(sprintf("  %s: [%s]\n", method, paste(round(result$y, 1), collapse = ", ")))
    }
    cat("\n")
}

# ── Example 4: Window Size Comparison ────────────────────────────────────────
example_4_window_size_comparison <- function() {
    cat("Example 4: Window Size Comparison\n")

    x <- as.numeric(1:20)
    y <- 2 * x + sin(x * 0.5) * 3

    for (w in c(5L, 10L, 15L)) {
        model <- OnlineLoess(fraction = 0.5, iterations = 2L, window_capacity = w)
        result <- model$add_points(x, y)
        last5 <- tail(result$y, 5)
        cat(sprintf("  window_capacity=%d: last 5 = [%s]\n",
            w, paste(round(last5, 2), collapse = ", ")))
    }
    cat("\n")
}

# ── Example 5: Memory-Bounded Processing ──────────────────────────────────────
example_5_memory_bounded_processing <- function() {
    cat("Example 5: Memory-Bounded Processing (Embedded Systems)\n")

    total <- 1000L
    x <- as.numeric(0:(total - 1))
    y <- 2 * x + sin(x * 0.1) * 5 + ((0:(total - 1)) %% 7 - 3) * 0.5

    model <- OnlineLoess(fraction = 0.3, iterations = 1L, window_capacity = 20L)
    result <- model$add_points(x, y)

    n_out <- length(result$y)
    for (milestone in c(200L, 400L, 600L, 800L, 1000L)) {
        if (milestone <= n_out) {
            cat(sprintf("  Processed: %4d pts | smoothed=%.2f\n",
                milestone, result$y[milestone]))
        }
    }
    cat(sprintf("  Total: %d, final smoothed: %.2f\n", n_out, tail(result$y, 1)))
    cat("  Memory: constant (window=20)\n\n")
}

# ── Example 6: Sliding Window Behavior ───────────────────────────────────────
example_6_sliding_window_behavior <- function() {
    cat("Example 6: Sliding Window Behavior\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8)
    y <- c(2, 4, 6, 8, 10, 12, 14, 16)

    model <- OnlineLoess(fraction = 0.6, iterations = 0L, window_capacity = 4L)
    result <- model$add_points(x, y)

    cat(sprintf("  %4s %6s %8s %10s %-22s\n", "Pt", "X", "Y", "Smoothed", "Status"))
    for (i in seq_along(x)) {
        if (i <= length(result$y)) {
            cat(sprintf("  %4d %6.0f %8.0f %10.2f %-22s\n",
                i, x[i], y[i], result$y[i], "Window full (sliding)"))
        } else {
            cat(sprintf("  %4d %6.0f %8.0f %10s %-22s\n",
                i, x[i], y[i], "-", sprintf("Filling (%d/4)", i)))
        }
    }
    cat("  Output starts after window fills (4 pts), then slides.\n\n")
}

# ── Example 7: Benchmark (Sequential Online) ──────────────────────────────────
example_7_benchmark <- function() {
    cat("Example 7: Benchmark (Sequential Online)\n")

    n <- 1000L
    x <- as.numeric(0:(n - 1))
    y <- sin(x * 0.1) + cos(x * 0.01)

    t0 <- proc.time()["elapsed"]
    model <- OnlineLoess(fraction = 0.5, iterations = 3L, window_capacity = 10L)
    result <- model$add_points(x, y)
    elapsed_ms <- (proc.time()["elapsed"] - t0) * 1000

    cat(sprintf("  %d pts processed in %.2fms (window_capacity=10)\n\n",
        length(result$y), elapsed_ms))
}

# ── Example 8: Update Modes (Full vs Incremental) and min_points ───────────────
example_8_update_modes <- function() {
    cat("Example 8: Update Modes (Full vs Incremental) and min_points\n")

    x <- as.numeric(0:29)
    y <- 2 * x + 1

    for (mode in c("full", "incremental")) {
        model <- OnlineLoess(
            fraction = 0.5, iterations = 2L,
            update_mode = mode, min_points = 5L,
            window_capacity = 15L
        )
        result <- model$add_points(x, y)
        cat(sprintf("  %s: %d pts emitted (out of %d)\n", mode, length(result$y), length(x)))
    }

    # Show fraction_used and iterations_used from the result
    model <- OnlineLoess(fraction = 0.5, iterations = 2L,
                         window_capacity = 10L, min_points = 3L)
    result <- model$add_points(x, y)
    cat(sprintf("  last smoothed: %.3f\n", tail(result$y, 1)))
    cat(sprintf("  fraction_used: %g\n", result$fraction_used))
    if (!is.null(result$iterations_used))
        cat(sprintf("  iterations_used: %d\n", result$iterations_used))
    cat("\n")
}

# ── Example 9: Advanced Online Options ────────────────────────────────────────
example_9_advanced_online_options <- function() {
    cat("Example 9: Advanced Online Options\n")

    x <- as.numeric(0:29)
    y <- 2 * x + 1

    model <- OnlineLoess(
        fraction = 0.5, iterations = 2L,
        degree = "quadratic",
        scaling_method = "mar",
        boundary_policy = "reflect",
        zero_weight_fallback = "return_original",
        distance_metric = "chebyshev",
        auto_converge = 1e-3,
        return_robustness_weights = TRUE,
        min_points = 5L,
        window_capacity = 15L
    )

    result <- model$add_points(x, y)
    cat(sprintf("  emitted: %d\n", length(result$y)))
    cat(sprintf("  last smoothed: %.3f\n", tail(result$y, 1)))
    cat(sprintf("  fraction_used: %g\n", result$fraction_used))
    cat("\n")
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
    cat(strrep("=", 60), "\n")
    cat("rfastloess Online Smoothing - Comprehensive Examples\n")
    cat(strrep("=", 60), "\n\n")

    example_1_basic_streaming()
    example_2_sensor_data_simulation()
    example_3_outlier_handling()
    example_4_window_size_comparison()
    example_5_memory_bounded_processing()
    example_6_sliding_window_behavior()
    example_7_benchmark()
    example_8_update_modes()
    example_9_advanced_online_options()

    cat("=== Online Smoothing Examples Complete ===\n")
}

if (sys.nframe() == 0) main()
