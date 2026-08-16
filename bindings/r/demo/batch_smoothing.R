#!/usr/bin/env Rscript
# =============================================================================
# rfastloess Batch Smoothing - Comprehensive Examples
#
#  1. Basic smoothing
#  2. Robust smoothing with outliers
#  3. Uncertainty quantification (confidence/prediction intervals)
#  4. Cross-validation (K-Fold)
#  5. Complete diagnostic analysis
#  6. Different weight functions (kernels)
#  7. Robustness methods comparison
#  8. Benchmark
#  9. Scaling methods (MAR, MAD, Mean)
# 10. Boundary policies
# 11. Zero-weight fallback strategies
# 12. Polynomial degrees + iterations_used
# 13. Distance metrics
# 14. Surface modes and standard errors
# 15. Additional weight functions
# 16. LOOCV and auto-converge
# 17. Interpolation tuning (surface_mode effects)
# =============================================================================

library(rfastloess)

make_linear <- function(n) {
    list(x = as.numeric(0:(n - 1)), y = 2 * as.numeric(0:(n - 1)) + 1)
}

# ── Example 1: Basic Smoothing ──────────────────────────────────────────────
example_1_basic_smoothing <- function() {
    cat("Example 1: Basic Smoothing\n")

    x <- c(1, 2, 3, 4, 5)
    y <- c(2.0, 4.1, 5.9, 8.2, 9.8)

    result <- Loess(fraction = 0.5, iterations = 3L)$fit(x, y)

    cat(sprintf("  fraction_used=%g\n", result$fraction_used))
    cat("  Smoothed:", paste(round(result$y, 3), collapse = ", "), "\n\n")
}

# ── Example 2: Robust Smoothing with Outliers ────────────────────────────────
example_2_robust_with_outliers <- function() {
    cat("Example 2: Robust Smoothing with Outliers\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8)
    y <- c(2.1, 4.0, 5.9, 25.0, 10.1, 12.0, 14.1, 15.9) # 25.0 is outlier

    result <- Loess(
        fraction = 0.5, iterations = 5L,
        robustness_method = "bisquare",
        return_robustness_weights = TRUE,
        return_residuals = TRUE
    )$fit(x, y)

    if (!is.null(result$robustness_weights)) {
        for (i in seq_along(result$robustness_weights)) {
            if (result$robustness_weights[i] < 0.5) {
                cat(sprintf(
                    "  Outlier at index %d (y=%.1f): weight=%.3f\n",
                    i, y[i], result$robustness_weights[i]
                ))
            }
        }
    }
    cat("  Smoothed:", paste(round(result$y, 2), collapse = ", "), "\n\n")
}

# ── Example 3: Uncertainty Quantification ───────────────────────────────────
example_3_uncertainty_quant <- function() {
    cat("Example 3: Uncertainty Quantification\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8)
    y <- c(2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7)

    result <- Loess(
        fraction = 0.5, iterations = 3L,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95
    )$fit(x, y)

    cat("  x  y_smooth  conf_low  conf_high  pred_low  pred_high\n")
    for (i in seq_along(result$y)) {
        cat(sprintf(
            "  %d  %.4f  %.4f  %.4f  %.4f  %.4f\n",
            result$x[i], result$y[i],
            result$confidence_lower[i], result$confidence_upper[i],
            result$prediction_lower[i], result$prediction_upper[i]
        ))
    }
    cat("\n")
}

# ── Example 4: Cross-Validation ──────────────────────────────────────────────
example_4_cross_validation <- function() {
    cat("Example 4: Cross-Validation for Parameter Selection\n")

    x <- 1:20
    y <- 2 * x + 1 + sin(x * 0.5)

    result <- Loess(
        cv_fractions = c(0.2, 0.3, 0.5, 0.7),
        cv_method = "kfold", cv_k = 5L,
        iterations = 2L,
        return_diagnostics = TRUE
    )$fit(x, y)

    cat(sprintf("  Selected fraction: %g\n", result$fraction_used))
    if (!is.null(result$cv_scores)) {
        fracs <- c(0.2, 0.3, 0.5, 0.7)
        cat("  CV Scores (RMSE per fraction):\n")
        for (i in seq_along(fracs)) {
            cat(sprintf(
                "    fraction=%.1f: %.4f\n",
                fracs[i], result$cv_scores[i]
            ))
        }
    }
    cat("\n")
}

# ── Example 5: Complete Diagnostic Analysis ──────────────────────────────────
example_5_complete_diagnostics <- function() {
    cat("Example 5: Complete Diagnostic Analysis\n")

    x <- c(1, 2, 3, 4, 5, 6, 7, 8)
    y <- c(2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7)

    result <- Loess(
        fraction = 0.5, iterations = 3L,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95,
        return_diagnostics = TRUE,
        return_residuals = TRUE,
        return_robustness_weights = TRUE
    )$fit(x, y)

    if (!is.null(result$diagnostics)) {
        d <- result$diagnostics
        cat("  Diagnostics:\n")
        cat(sprintf("    RMSE:        %.6f\n", d$rmse))
        cat(sprintf("    MAE:         %.6f\n", d$mae))
        cat(sprintf("    R²:          %.6f\n", d$r_squared))
        cat(sprintf("    Residual SD: %.6f\n", d$residual_sd))
        if (!is.nan(d$aic)) cat(sprintf("    AIC:         %.2f\n", d$aic))
        if (!is.nan(d$aicc)) cat(sprintf("    AICc:        %.2f\n", d$aicc))
        if (!is.nan(d$effective_df)) {
            cat(sprintf("    Eff. DF:     %.2f\n", d$effective_df))
        }
    }
    cat(sprintf("  smoothed[1]: %.5f\n", result$y[1]))
    if (!is.null(result$residuals)) {
        cat(sprintf("  residuals[1]: %.5f\n", result$residuals[1]))
    }
    if (!is.null(result$robustness_weights)) {
        cat(sprintf("  rob_weight[1]: %.4f\n", result$robustness_weights[1]))
    }
    cat("\n")
}

# ── Example 6: Different Weight Functions (Kernels) ──────────────────────────
example_6_different_kernels <- function() {
    cat("Example 6: Different Weight Functions (Kernels)\n")

    x <- c(1, 2, 3, 4, 5)
    y <- c(2.0, 4.1, 5.9, 8.2, 9.8)

    for (kernel in c("tricube", "epanechnikov", "gaussian", "biweight")) {
        result <- Loess(fraction = 0.5, weight_function = kernel)$fit(x, y)
        cat(sprintf(
            "  %s: [%s]\n", kernel,
            paste(round(result$y, 3), collapse = ", ")
        ))
    }
    cat("\n")
}

# ── Example 7: Robustness Methods Comparison ─────────────────────────────────
example_7_robustness_methods <- function() {
    cat("Example 7: Robustness Methods Comparison\n")

    x <- c(1, 2, 3, 4, 5)
    y <- c(2.0, 4.1, 20.0, 8.2, 9.8) # 20.0 is an outlier

    for (method in c("bisquare", "huber", "talwar")) {
        result <- Loess(
            fraction = 0.5, iterations = 5L,
            robustness_method = method,
            return_robustness_weights = TRUE
        )$fit(x, y)
        cat(sprintf("  %s:\n", method))
        cat(sprintf(
            "    Smoothed: [%s]\n",
            paste(round(result$y, 2), collapse = ", ")
        ))
        if (!is.null(result$robustness_weights)) {
            cat(sprintf(
                "    Weights:  [%s]\n",
                paste(round(result$robustness_weights, 3),
                    collapse = ", "
                )
            ))
        }
    }
    cat("\n")
}

# ── Example 8: Benchmark ─────────────────────────────────────────────────────
example_8_benchmark <- function() {
    cat("Example 8: Benchmark\n")

    n <- 1000L
    x <- as.numeric(0:(n - 1))
    y <- sin(x * 0.1) + cos(x * 0.01)

    t0 <- proc.time()["elapsed"]
    result <- Loess(parallel = TRUE)$fit(x, y)
    elapsed_ms <- (proc.time()["elapsed"] - t0) * 1000

    cat(sprintf("  %d points in %.2fms\n", n, elapsed_ms))
    cat(sprintf(
        "  fraction_used=%g, y[1]=%.4f\n",
        result$fraction_used, result$y[1]
    ))
    cat("\n")
}

# ── Example 9: Scaling Methods (MAR, MAD, Mean) ──────────────────────────────
example_9_scaling_methods <- function() {
    cat("Example 9: Scaling Methods\n")

    d <- make_linear(20)

    for (method in c("mar", "mad", "mean")) {
        result <- Loess(fraction = 0.5, scaling_method = method)$fit(d$x, d$y)
        cat(sprintf("  %s: y[1]=%.3f\n", method, result$y[1]))
    }
    cat("\n")
}

# ── Example 10: Boundary Policies ────────────────────────────────────────────
example_10_boundary_policies <- function() {
    cat("Example 10: Boundary Policies\n")

    d <- make_linear(30)

    for (policy in c("extend", "reflect", "zero", "noboundary")) {
        result <- Loess(fraction = 0.5, boundary_policy = policy)$fit(d$x, d$y)
        n <- length(result$y)
        cat(sprintf(
            "  %s: first=%.2f, last=%.2f\n",
            policy, result$y[1], result$y[n]
        ))
    }
    cat("\n")
}

# ── Example 11: Zero-Weight Fallback Strategies ───────────────────────────────
example_11_zero_wt_fallback <- function() {
    cat("Example 11: Zero-Weight Fallback Strategies\n")

    d <- make_linear(20)

    for (fb in c("use_local_mean", "return_original", "return_none")) {
        result <- Loess(fraction = 0.5, zero_weight_fallback = fb)$fit(d$x, d$y)
        cat(sprintf("  %s: y[1]=%.3f\n", fb, result$y[1]))
    }
    cat("\n")
}

# ── Example 12: Polynomial Degrees + iterations_used ──────────────────────────
example_12_polynomial_degrees <- function() {
    cat("Example 12: Polynomial Degrees\n")

    d <- make_linear(30)

    for (deg in c("constant", "linear", "quadratic", "cubic", "quartic")) {
        result <- Loess(
            fraction = 0.5, iterations = 2L,
            degree = deg
        )$fit(d$x, d$y)
        iter_used <- result$iterations_used
        if (is.null(iter_used)) iter_used <- "NULL"
        cat(sprintf(
            "  %s: y[1]=%.3f, iterations_used=%s\n",
            deg, result$y[1], iter_used
        ))
    }
    cat("\n")
}

# ── Example 13: Distance Metrics ─────────────────────────────────────────────
example_13_distance_metrics <- function() {
    cat("Example 13: Distance Metrics\n")

    d <- make_linear(20)

    for (metric in c("euclidean", "normalized", "manhattan", "chebyshev")) {
        result <- Loess(fraction = 0.5, distance_metric = metric)$fit(d$x, d$y)
        cat(sprintf("  %s: y[1]=%.3f\n", metric, result$y[1]))
    }

    # Minkowski with custom p via "minkowski:p" format
    result_mink <- Loess(
        fraction = 0.5,
        distance_metric = "minkowski:3"
    )$fit(d$x, d$y)
    cat(sprintf("  minkowski(p=3): y[1]=%.3f\n", result_mink$y[1]))
    cat("\n")
}

# ── Example 14: Surface Modes and Standard Errors ────────────────────────────
example_14_surface_modes_se <- function() {
    cat("Example 14: Surface Modes and Standard Errors\n")

    d <- make_linear(30)

    # Direct surface — fits every point; SE fields fully populated
    r_direct <- Loess(
        fraction = 0.5, surface_mode = "direct",
        return_se = TRUE,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95
    )$fit(d$x, d$y)

    cat("  surface_mode=direct:\n")
    cat(sprintf(
        "    confidence_lower non-null: %s\n",
        !is.null(r_direct$confidence_lower)
    ))
    cat(sprintf(
        "    prediction_lower non-null: %s\n",
        !is.null(r_direct$prediction_lower)
    ))
    if (!is.null(r_direct$standard_errors)) {
        cat(sprintf(
            "    standard_errors[1]: %.4f\n",
            r_direct$standard_errors[1]
        ))
    }
    if (!is.null(r_direct$enp)) cat(sprintf("    enp: %.3f\n", r_direct$enp))
    if (!is.null(r_direct$trace_hat)) {
        cat(sprintf("    trace_hat: %.3f\n", r_direct$trace_hat))
    }
    if (!is.null(r_direct$delta1)) {
        cat(sprintf("    delta1: %.3f\n", r_direct$delta1))
    }
    if (!is.null(r_direct$delta2)) {
        cat(sprintf("    delta2: %.3f\n", r_direct$delta2))
    }
    if (!is.null(r_direct$residual_scale)) {
        cat(sprintf("    residual_scale: %.4f\n", r_direct$residual_scale))
    }
    if (!is.null(r_direct$leverage)) {
        cat(sprintf("    leverage[1]: %.4f\n", r_direct$leverage[1]))
    }

    # Interpolation surface — faster, approximate
    r_interp <- Loess(
        fraction = 0.5, surface_mode = "interpolation",
        return_se = TRUE
    )$fit(d$x, d$y)

    cat("  surface_mode=interpolation:\n")
    cat(sprintf("    y[1]: %.3f\n", r_interp$y[1]))
    if (!is.null(r_interp$standard_errors)) {
        cat(sprintf(
            "    standard_errors[1]: %.4f\n",
            r_interp$standard_errors[1]
        ))
    }
    cat("\n")
}

# ── Example 15: Additional Weight Functions (Uniform, Triangle, Cosine) ───────
example_15_additional_kernels <- function() {
    cat("Example 15: Additional Weight Functions (Uniform, Triangle, Cosine)\n")

    x <- c(1, 2, 3, 4, 5)
    y <- c(2.0, 4.1, 5.9, 8.2, 9.8)

    for (kernel in c("uniform", "triangle", "cosine")) {
        result <- Loess(fraction = 0.5, weight_function = kernel)$fit(x, y)
        cat(sprintf(
            "  %s: [%s]\n", kernel,
            paste(round(result$y, 3), collapse = ", ")
        ))
    }
    cat("\n")
}

# ── Example 16: LOOCV, K-Fold, and Auto-Converge ─────────────────────────────
example_16_loocv_auto_conv <- function() {
    cat("Example 16: LOOCV, K-Fold, and Auto-Converge\n")

    x <- 1:20
    y <- 2 * x + 1 + sin(x * 0.5)

    # Leave-one-out cross-validation
    r_loocv <- Loess(
        cv_fractions = c(0.3, 0.5, 0.7),
        cv_method = "loocv"
    )$fit(x, y)
    cat(sprintf("  LOOCV selected fraction: %g\n", r_loocv$fraction_used))
    if (!is.null(r_loocv$cv_scores)) {
        cat(sprintf(
            "  LOOCV scores: [%s]\n",
            paste(round(r_loocv$cv_scores, 4), collapse = ", ")
        ))
    }

    # K-Fold cross-validation
    r_kfold <- Loess(
        cv_fractions = c(0.2, 0.4, 0.6),
        cv_method = "kfold", cv_k = 5L
    )$fit(x, y)
    cat(sprintf("  KFold(k=5) selected fraction: %g\n", r_kfold$fraction_used))
    if (!is.null(r_kfold$cv_scores)) {
        cat(sprintf(
            "  KFold scores: [%s]\n",
            paste(round(r_kfold$cv_scores, 4), collapse = ", ")
        ))
    }

    # Auto-converge: stop robustness iterations when change < tolerance
    r_ac <- Loess(fraction = 0.5, auto_converge = 1e-4)$fit(x, y)
    iter_used_ac <- r_ac$iterations_used
    if (is.null(iter_used_ac)) iter_used_ac <- "NULL"
    cat(sprintf(
        "  auto_converge=1e-4: iterations_used=%s\n",
        iter_used_ac
    ))
    cat("\n")
}

# ── Example 17: Interpolation Tuning (surface_mode effects) ──────────────────
example_17_interp_tuning <- function() {
    cat("Example 17: Interpolation Tuning (surface_mode effects)\n")

    n <- 50L
    d <- make_linear(n)

    # Default (interpolation) — fastest, uses a spatial grid
    r_interp <- Loess(
        fraction = 0.5,
        surface_mode = "interpolation"
    )$fit(d$x, d$y)
    cat(sprintf(
        "  interpolation: y[1]=%.3f, y[%d]=%.3f\n",
        r_interp$y[1], n, r_interp$y[n]
    ))

    # Direct — fits every point exactly, more accurate but slower
    r_direct <- Loess(fraction = 0.5, surface_mode = "direct")$fit(d$x, d$y)
    cat(sprintf(
        "  direct:        y[1]=%.3f, y[%d]=%.3f\n",
        r_direct$y[1], n, r_direct$y[n]
    ))

    # Fraction sweep with direct surface
    for (frac in c(0.2, 0.5, 0.8)) {
        r <- Loess(fraction = frac, surface_mode = "direct")$fit(d$x, d$y)
        cat(sprintf("  direct fraction=%.1f: y[1]=%.3f\n", frac, r$y[1]))
    }

    # Interpolation + SE for hat-matrix statistics
    r_se <- Loess(
        fraction = 0.5, surface_mode = "interpolation",
        return_se = TRUE
    )$fit(d$x, d$y)
    if (!is.null(r_se$enp)) {
        cat(sprintf("  interpolation+SE enp: %.3f\n", r_se$enp))
    }
    cat("\n")
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
    cat(strrep("=", 60), "\n")
    cat("rfastloess Batch Smoothing - Comprehensive Examples\n")
    cat(strrep("=", 60), "\n\n")

    example_1_basic_smoothing()
    example_2_robust_with_outliers()
    example_3_uncertainty_quant()
    example_4_cross_validation()
    example_5_complete_diagnostics()
    example_6_different_kernels()
    example_7_robustness_methods()
    example_8_benchmark()
    example_9_scaling_methods()
    example_10_boundary_policies()
    example_11_zero_wt_fallback()
    example_12_polynomial_degrees()
    example_13_distance_metrics()
    example_14_surface_modes_se()
    example_15_additional_kernels()
    example_16_loocv_auto_conv()
    example_17_interp_tuning()

    cat("=== Batch Smoothing Examples Complete ===\n")
}

if (sys.nframe() == 0) main()
