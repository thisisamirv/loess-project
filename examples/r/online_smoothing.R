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

# Helper: feed all (x, y) pairs through add_point one at a time.
# Returns a named list with $smoothed (numeric vector,
# NA where window not full).
add_all_points <- function(model, x, y) {
    results <- lapply(seq_along(x), function(i) model$add_point(x[i], y[i]))
    list(
        smoothed = sapply(
            results, function(r) if (is.null(r)) NA_real_ else r$smoothed
        )
    )
}

# ── Example 1: Basic Incremental Processing ──────────────────────────────────
example_1_basic_streaming <- function() {
    cat("Example 1: Basic Incremental Processing\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    y <- c(3.1, 5.0, 7.2, 8.9, 11.1, 13.0, 15.2, 16.8, 19.1, 21.0)

    model <- OnlineLoess(
        fraction = 0.5, iterations = 2L,
        window_capacity = 5L,
        return_robustness_weights = FALSE
    )
    result <- add_all_points(model, x, y)

    cat(sprintf("  %8s %12s %12s\n", "X", "Y_obs", "Y_smooth"))
    for (i in seq_along(y)) {
        smoothed <- if (is.na(result$smoothed[i])) {
            "(buffering)"
        } else {
            sprintf("%.2f", result$smoothed[i])
        }
        cat(sprintf("  %8.2f %12.2f %12s\n", x[i], y[i], smoothed))
    }
    cat("\n")
}

# ── Example 2: Real-Time Sensor Data Simulation ───────────────────────────────
example_2_sensor_simulation <- function() {
    cat("Example 2: Real-Time Sensor Data Simulation\n")
    cat("  Simulating temperature sensor with noise...\n")

    n <- 24L
    hours <- as.numeric(0:(n - 1))
    temp <- 20 + 5 * sin(hours * pi / 12) + ((hours * 7) %% 11) * 0.3 - 1.5

    model <- OnlineLoess(
        fraction = 0.4, iterations = 3L,
        robustness_method = "bisquare",
        window_capacity = 12L
    )
    result <- add_all_points(model, hours, temp)

    cat(sprintf("  %6s %12s %12s\n", "Hour", "Raw", "Smoothed"))
    for (i in seq_along(hours)) {
        smoothed <- if (is.na(result$smoothed[i])) {
            "(warming up)"
        } else {
            sprintf("%.2f degC", result$smoothed[i])
        }
        cat(sprintf("  %6.0f %10.2f degC %s\n", hours[i], temp[i], smoothed))
    }
    cat("\n")
}

# ── Example 3: Outlier Handling in Online Mode ────────────────────────────────
example_3_outlier_handling <- function() {
    cat("Example 3: Outlier Handling in Online Mode\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    y <- c(2.0, 4.1, 5.9, 25.0, 10.1, 12.0, 14.1, 50.0, 18.0, 20.1)

    for (method in c("bisquare", "talwar")) {
        model <- OnlineLoess(
            fraction = 0.5, iterations = 5L,
            robustness_method = method,
            window_capacity = 6L
        )
        result <- add_all_points(model, x, y)
        valid <- result$smoothed[!is.na(result$smoothed)]
        cat(sprintf(
            "  %s: [%s]\n", method, paste(round(valid, 1), collapse = ", ")
        ))
    }
    cat("\n")
}

# ── Example 4: Window Size Comparison ────────────────────────────────────────
example_4_window_comparison <- function() {
    cat("Example 4: Window Size Comparison\n")

    x <- as.numeric(1:20)
    y <- 2 * x + sin(x * 0.5) * 3

    for (w in c(5L, 10L, 15L)) {
        model <- OnlineLoess(
            fraction = 0.5, iterations = 2L,
            window_capacity = w
        )
        result <- add_all_points(model, x, y)
        valid <- result$smoothed[!is.na(result$smoothed)]
        last5 <- tail(valid, 5)
        cat(sprintf(
            "  window_capacity=%d: last 5 = [%s]\n",
            w, paste(round(last5, 2), collapse = ", ")
        ))
    }
    cat("\n")
}

# ── Example 5: Memory-Bounded Processing ──────────────────────────────────────
example_5_memory_bounded <- function() {
    cat("Example 5: Memory-Bounded Processing (Embedded Systems)\n")

    total <- 1000L
    x <- as.numeric(0:(total - 1))
    y <- 2 * x + sin(x * 0.1) * 5 + ((0:(total - 1)) %% 7 - 3) * 0.5

    model <- OnlineLoess(fraction = 0.3, iterations = 1L, window_capacity = 20L)
    result <- add_all_points(model, x, y)

    valid_smoothed <- result$smoothed[!is.na(result$smoothed)]
    n_out <- length(valid_smoothed)
    for (milestone in c(200L, 400L, 600L, 800L, 1000L)) {
        if (milestone <= n_out) {
            cat(sprintf(
                "  Processed: %4d pts | smoothed=%.2f\n",
                milestone, valid_smoothed[milestone]
            ))
        }
    }
    cat(sprintf(
        "  Total smoothed: %d, final: %.2f\n",
        n_out, tail(valid_smoothed, 1)
    ))
    cat("  Memory: constant (window=20)\n\n")
}

# ── Example 6: Sliding Window Behavior ───────────────────────────────────────
example_6_sliding_window <- function() {
    cat("Example 6: Sliding Window Behavior\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8)
    y <- c(2, 4, 6, 8, 10, 12, 14, 16)

    model <- OnlineLoess(fraction = 0.6, iterations = 0L, window_capacity = 4L)
    result <- add_all_points(model, x, y)

    cat(sprintf(
        "  %4s %6s %8s %10s %-22s\n",
        "Pt", "X", "Y", "Smoothed", "Status"
    ))
    for (i in seq_along(x)) {
        if (!is.na(result$smoothed[i])) {
            cat(sprintf(
                "  %4d %6.0f %8.0f %10.2f %-22s\n",
                i, x[i], y[i], result$smoothed[i], "Window full (sliding)"
            ))
        } else {
            cat(sprintf(
                "  %4d %6.0f %8.0f %10s %-22s\n",
                i, x[i], y[i], "-", sprintf("Filling (%d/4)", i)
            ))
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
    result <- add_all_points(model, x, y)
    elapsed_ms <- (proc.time()["elapsed"] - t0) * 1000

    valid <- result$smoothed[!is.na(result$smoothed)]
    cat(sprintf(
        "  %d pts processed in %.2fms (window_capacity=10)\n\n",
        length(valid), elapsed_ms
    ))
}

# ── Example 8: Update Modes (Full vs Incremental) and min_points ──────────────
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
        result <- add_all_points(model, x, y)
        valid <- result$smoothed[!is.na(result$smoothed)]
        cat(sprintf(
            "  %s: %d pts emitted (out of %d)\n",
            mode, length(valid), length(x)
        ))
    }

    # Show last smoothed value
    model <- OnlineLoess(
        fraction = 0.5, iterations = 2L,
        window_capacity = 10L, min_points = 3L
    )
    result <- add_all_points(model, x, y)
    valid <- result$smoothed[!is.na(result$smoothed)]
    if (length(valid) > 0) {
        cat(sprintf("  last smoothed: %.3f\n", tail(valid, 1)))
    }
    cat("\n")
}

# ── Example 9: Advanced Online Options ────────────────────────────────────────
example_9_online_options <- function() {
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

    result <- add_all_points(model, x, y)
    valid <- result$smoothed[!is.na(result$smoothed)]
    cat(sprintf("  emitted: %d\n", length(valid)))
    if (length(valid) > 0) {
        cat(sprintf("  last smoothed: %.3f\n", tail(valid, 1)))
    }
    cat("\n")
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
    cat(strrep("=", 60), "\n")
    cat("rfastloess Online Smoothing - Comprehensive Examples\n")
    cat(strrep("=", 60), "\n\n")

    example_1_basic_streaming()
    example_2_sensor_simulation()
    example_3_outlier_handling()
    example_4_window_comparison()
    example_5_memory_bounded()
    example_6_sliding_window()
    example_7_benchmark()
    example_8_update_modes()
    example_9_online_options()

    cat("=== Online Smoothing Examples Complete ===\n")
}

if (sys.nframe() == 0) main()
