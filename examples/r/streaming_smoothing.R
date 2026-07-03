#!/usr/bin/env Rscript
# =============================================================================
# rfastloess Streaming Smoothing - Comprehensive Examples
#
#  1. Basic chunked processing
#  2. Chunk size comparison
#  3. Overlap strategies
#  4. Large dataset processing
#  5. Outlier handling in streaming mode
#  6. File-based streaming simulation
#  7. Benchmark (sequential streaming)
#  8. Merge strategies
#  9. Advanced streaming options
# =============================================================================

library(rfastloess)

make_linear <- function(n) {
    list(x = as.numeric(0:(n - 1)), y = 2 * as.numeric(0:(n - 1)) + 1)
}

process_all <- function(model, x, y, chunk_size, overlap) {
    # Feed full-size chunks; let finalize() handle remaining data.
    # Avoids panic when tail chunk would be smaller than overlap.
    step <- chunk_size - overlap
    n <- length(x)
    result_x <- numeric(0)
    result_y <- numeric(0)

    start <- 1L
    while (start + chunk_size - 1L <= n) {
        end <- start + chunk_size - 1L
        res <- model$process_chunk(x[start:end], y[start:end])
        result_x <- c(result_x, res$x)
        result_y <- c(result_y, res$y)
        start <- start + step
    }
    fin <- model$finalize()
    list(x = c(result_x, fin$x), y = c(result_y, fin$y))
}

# ── Example 1: Basic Chunked Processing ──────────────────────────────────────
example_1_basic_chunked_processing <- function() {
    cat("Example 1: Basic Chunked Processing\n")

    n <- 50L
    d <- make_linear(n)
    chunk_size <- 15L; overlap <- 5L

    model <- StreamingLoess(
        fraction = 0.5, iterations = 2L,
        chunk_size = chunk_size, overlap = overlap,
        return_residuals = TRUE
    )

    cat(sprintf("  Dataset: %d pts, chunk=%d, overlap=%d\n", n, chunk_size, overlap))

    result_x <- numeric(0); result_y <- numeric(0)
    ci <- 0L
    start <- 1L
    while (start + chunk_size - 1L <= n) {
        end <- start + chunk_size - 1L
        res <- model$process_chunk(d$x[start:end], d$y[start:end])
        if (length(res$x) > 0) {
            result_x <- c(result_x, res$x)
            result_y <- c(result_y, res$y)
            cat(sprintf("  Chunk %d: %d pts (x: %.0f..%.0f)\n",
                ci, length(res$x), res$x[1], res$x[length(res$x)]))
        }
        ci <- ci + 1L
        start <- start + chunk_size - overlap
    }
    fin <- model$finalize()
    if (length(fin$x) > 0) {
        result_x <- c(result_x, fin$x); result_y <- c(result_y, fin$y)
        cat(sprintf("  Finalize: %d remaining pts\n", length(fin$x)))
    }
    cat(sprintf("  Total: %d/%d\n\n", length(result_y), n))
}

# ── Example 2: Chunk Size Comparison ─────────────────────────────────────────
example_2_chunk_size_comparison <- function() {
    cat("Example 2: Chunk Size Comparison\n")

    n <- 100L
    d <- make_linear(n)

    configs <- list(
        list(cs = 20L, ov = 5L,  label = "Small"),
        list(cs = 50L, ov = 10L, label = "Medium"),
        list(cs = 80L, ov = 15L, label = "Large")
    )

    for (cfg in configs) {
        model <- StreamingLoess(
            fraction = 0.5, iterations = 1L,
            chunk_size = cfg$cs, overlap = cfg$ov
        )
        chunks <- 0L; total <- 0L
        start <- 1L
        while (start + cfg$cs - 1L <= n) {
            end <- start + cfg$cs - 1L
            res <- model$process_chunk(d$x[start:end], d$y[start:end])
            if (length(res$x) > 0) { chunks <- chunks + 1L; total <- total + length(res$x) }
            start <- start + cfg$cs - cfg$ov
        }
        fin <- model$finalize()
        if (length(fin$x) > 0) { chunks <- chunks + 1L; total <- total + length(fin$x) }
        cat(sprintf("  %s (size=%d, overlap=%d): chunks=%d, total=%d\n",
            cfg$label, cfg$cs, cfg$ov, chunks, total))
    }
    cat("\n")
}

# ── Example 3: Overlap Strategies ────────────────────────────────────────────
example_3_overlap_strategies <- function() {
    cat("Example 3: Overlap Strategies\n")

    n <- 100L
    d <- make_linear(n)
    cs <- 40L

    for (pair in list(c(0L, "No overlap"), c(10L, "10-pt overlap"), c(20L, "20-pt overlap"))) {
        ov <- as.integer(pair[1]); label <- pair[2]
        model <- StreamingLoess(fraction = 0.5, chunk_size = cs, overlap = ov)
        total <- 0L
        start <- 1L
        while (start + cs - 1L <= n) {
            end <- start + cs - 1L
            res <- model$process_chunk(d$x[start:end], d$y[start:end])
            total <- total + length(res$x)
            start <- start + cs - ov
        }
        total <- total + length(model$finalize()$x)
        cat(sprintf("  %s: total output=%d\n", label, total))
    }
    cat("\n")
}

# ── Example 4: Large Dataset Processing ──────────────────────────────────────
example_4_large_dataset_processing <- function() {
    cat("Example 4: Large Dataset Processing\n")

    n <- 10000L
    x <- as.numeric(0:(n - 1))
    y <- sin(x * 0.01) + x * 0.001

    cs <- 500L; ov <- 50L
    model <- StreamingLoess(fraction = 0.05, iterations = 2L, chunk_size = cs, overlap = ov)

    total <- 0L; step <- cs - ov; start <- 1L
    while (start + cs - 1L <= n) {
        end <- start + cs - 1L
        res <- model$process_chunk(x[start:end], y[start:end])
        total <- total + length(res$x)
        if (total > 0L && total %% 2000L < step) cat(sprintf("  Progress: ~%d pts smoothed\n", total))
        start <- start + step
    }
    total <- total + length(model$finalize()$x)
    cat(sprintf("  Total: %d/%d, memory: constant (chunk=%d)\n\n", total, n, cs))
}

# ── Example 5: Outlier Handling in Streaming Mode ─────────────────────────────
example_5_outlier_handling <- function() {
    cat("Example 5: Outlier Handling in Streaming Mode\n")

    n <- 100L
    x <- as.numeric(0:(n - 1))
    y <- 2 * x + 1 + sin(x * 0.2) * 2
    y[c(26, 51, 76)] <- y[c(26, 51, 76)] + 50  # Outliers (1-indexed)

    for (method in c("bisquare", "huber", "talwar")) {
        model <- StreamingLoess(
            fraction = 0.5, iterations = 5L,
            robustness_method = method,
            chunk_size = 30L, overlap = 10L,
            return_residuals = TRUE
        )
        large <- 0L; start <- 1L
        while (start + 29L <= n) {
            end <- start + 29L
            res <- model$process_chunk(x[start:end], y[start:end])
            if (!is.null(res$residuals)) large <- large + sum(abs(res$residuals) > 10)
            start <- start + 20L
        }
        fin <- model$finalize()
        if (!is.null(fin$residuals)) large <- large + sum(abs(fin$residuals) > 10)
        cat(sprintf("  %s: pts with |residual|>10: %d\n", method, large))
    }
    cat("\n")
}

# ── Example 6: File-Based Streaming Simulation ───────────────────────────────
example_6_file_simulation <- function() {
    cat("Example 6: File-Based Streaming Simulation\n")
    cat("  Simulating: input.csv -> Smooth -> output.csv\n")

    total_lines <- 200L; cs <- 50L; ov <- 10L
    model <- StreamingLoess(
        fraction = 0.5, iterations = 2L, chunk_size = cs, overlap = ov,
        return_residuals = TRUE
    )

    out_count <- 0L
    n_chunks <- ceiling(total_lines / (cs - ov))
    for (ci in seq_len(n_chunks)) {
        start_line <- (ci - 1L) * (cs - ov)
        end_line   <- min(start_line + cs - 1L, total_lines - 1L)
        xc <- as.numeric(start_line:end_line)
        yc <- 2 * xc + 1 + sin(xc * 0.1) * 3

        cat(sprintf("  Reading chunk %d (lines %d..%d)\n", ci - 1L, start_line, end_line))
        res <- model$process_chunk(xc, yc)
        if (length(res$x) > 0) {
            out_count <- out_count + length(res$x)
            cat(sprintf("    -> Writing %d smoothed pts (total: %d)\n", length(res$x), out_count))
        }
    }
    fin <- model$finalize()
    if (length(fin$x) > 0) {
        out_count <- out_count + length(fin$x)
        cat(sprintf("  Finalizing: %d remaining pts\n", length(fin$x)))
    }
    cat(sprintf("  Input: %d, Output: %d\n\n", total_lines, out_count))
}

# ── Example 7: Benchmark (Sequential Streaming) ───────────────────────────────
example_7_benchmark <- function() {
    cat("Example 7: Benchmark (Sequential Streaming)\n")

    n <- 1000L; cs <- 100L; ov <- 10L
    model <- StreamingLoess(fraction = 0.5, iterations = 3L, chunk_size = cs, overlap = ov)

    t0 <- proc.time()["elapsed"]
    total <- 0L; start <- 1L
    while (start + cs - 1L <= n) {
        end <- start + cs - 1L
        xc <- as.numeric((start - 1L):(end - 1L))
        yc <- sin(xc * 0.1) + cos(xc * 0.01)
        total <- total + length(model$process_chunk(xc, yc)$x)
        start <- start + cs - ov
    }
    total <- total + length(model$finalize()$x)
    elapsed_ms <- (proc.time()["elapsed"] - t0) * 1000

    cat(sprintf("  %d pts in %.2fms (chunk=%d, overlap=%d)\n\n", total, elapsed_ms, cs, ov))
}

# ── Example 8: Merge Strategies ──────────────────────────────────────────────
example_8_merge_strategies <- function() {
    cat("Example 8: Merge Strategies\n")

    n <- 50L
    d <- make_linear(n)

    for (strategy in c("average", "weighted_average", "take_first", "take_last")) {
        model <- StreamingLoess(
            fraction = 0.5, iterations = 2L,
            chunk_size = 20L, overlap = 5L,
            merge_strategy = strategy
        )
        total <- 0L; start <- 1L
        while (start + 19L <= n) {
            end <- start + 19L
            total <- total + length(model$process_chunk(d$x[start:end], d$y[start:end])$x)
            start <- start + 15L
        }
        total <- total + length(model$finalize()$x)
        cat(sprintf("  %s: total=%d\n", strategy, total))
    }
    cat("\n")
}

# ── Example 9: Advanced Streaming Options ─────────────────────────────────────
example_9_advanced_options <- function() {
    cat("Example 9: Advanced Streaming Options\n")

    n <- 50L
    d <- make_linear(n)

    model <- StreamingLoess(
        fraction = 0.5, iterations = 2L,
        degree = "quadratic",
        scaling_method = "mar",
        boundary_policy = "reflect",
        zero_weight_fallback = "return_original",
        distance_metric = "manhattan",
        surface_mode = "direct",
        return_se = TRUE,
        return_diagnostics = TRUE,
        return_robustness_weights = TRUE,
        auto_converge = 1e-3,
        chunk_size = 20L, overlap = 5L
    )

    total <- 0L; start <- 1L
    while (start + 19L <= n) {
        end <- start + 19L
        total <- total + length(model$process_chunk(d$x[start:end], d$y[start:end])$x)
        start <- start + 15L
    }
    fin <- model$finalize()
    total <- total + length(fin$x)

    cat(sprintf("  total pts: %d\n", total))
    if (!is.null(fin$standard_errors) && length(fin$standard_errors) > 0)
        cat(sprintf("  standard_errors[1]: %.4f\n", fin$standard_errors[1]))
    if (!is.null(fin$diagnostics)) {
        cat(sprintf("  diagnostics$rmse: %.3f\n", fin$diagnostics$rmse))
        cat(sprintf("  diagnostics$r_squared: %.3f\n", fin$diagnostics$r_squared))
        if (!is.nan(fin$diagnostics$aic)) cat(sprintf("  diagnostics$aic: %.3f\n", fin$diagnostics$aic))
    }
    if (!is.null(fin$robustness_weights) && length(fin$robustness_weights) > 0)
        cat(sprintf("  robustness_weights[1]: %.4f\n", fin$robustness_weights[1]))
    cat("\n")
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
    cat(strrep("=", 60), "\n")
    cat("rfastloess Streaming Smoothing - Comprehensive Examples\n")
    cat(strrep("=", 60), "\n\n")

    example_1_basic_chunked_processing()
    example_2_chunk_size_comparison()
    example_3_overlap_strategies()
    example_4_large_dataset_processing()
    example_5_outlier_handling()
    example_6_file_simulation()
    example_7_benchmark()
    example_8_merge_strategies()
    example_9_advanced_options()

    cat("=== Streaming Smoothing Examples Complete ===\n")
}

if (sys.nframe() == 0) main()
