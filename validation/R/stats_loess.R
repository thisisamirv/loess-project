#!/usr/bin/env Rscript
# R LOESS Validation Script
# Generates reference outputs for validation using R's loess() function.
#
# R's loess() supports:
# - span: fraction of data in local neighborhood (0, 1]
# - degree: 1 (linear) or 2 (quadratic)
# - family: "gaussian" (no robustness) or "symmetric" (robust/bisquare)
# - loess.control(surface, iterations)
#
# These outputs are compared against loess-rs to validate implementation accuracy.

library(jsonlite)

OUTPUT_DIR <- "output/r/"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

run_scenario <- function(
    name,
    x,
    y,
    frac,
    iter,
    degree = 1,
    notes = "",
    ...
) {
    cat(sprintf("Running scenario: %s\n", name))

    # Robustness: family = "symmetric" enables bisquare robustness weights.
    # When iter = 0, use family = "gaussian" (no robustness iterations).
    family <- if (iter > 0) "symmetric" else "gaussian"

    args <- list(...)
    surface_arg <- if (!is.null(args$surface)) args$surface else "interpolate"
    r_surface <- if (surface_arg == "direct") "direct" else "interpolate"

    df <- data.frame(x = x, y = y)
    # loess.control iterations only applies when family = "symmetric"
    ctrl_iter <- if (iter > 0) iter else 4
    ctrl <- loess.control(surface = r_surface, iterations = ctrl_iter)

    fit <- loess(y ~ x, data = df, span = frac, degree = degree,
                 family = family, control = ctrl)
    fitted <- fit$fitted

    data <- list(
        name = name,
        notes = notes,
        input = list(x = x, y = y),
        params = list(
            fraction = frac,
            degree = degree,
            iterations = iter,
            extra = list(...)
        ),
        result = list(fitted = fitted)
    )

    path <- file.path(OUTPUT_DIR, paste0(name, ".json"))
    write_json(data, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

generate_data <- function(
    n = 100,
    kind = "linear",
    noise = 0.0,
    range_min = 0.0,
    range_max = 1.0,
    outlier_ratio = 0.0
) {
    # Fixed seed for reproducibility
    set.seed(42)

    x <- seq(range_min, range_max, length.out = n)

    y <- switch(kind,
        "linear"    = 2 * x + 1,
        "quadratic" = x^2,
        "sine"      = sin(4 * x),
        "step"      = ifelse(x < (range_min + range_max) / 2, 0.0, 1.0),
        "constant"  = rep(5.0, n),
        x
    )

    # Add noise
    if (noise > 0) {
        y <- y + rnorm(n, 0, noise)
    }

    # Add outliers
    if (outlier_ratio > 0) {
        n_out <- as.integer(n * outlier_ratio)
        indices <- sample(n, n_out, replace = FALSE)
        y[indices] <- y[indices] + 10.0 # Significant outlier
    }

    list(x = x, y = y)
}

main <- function() {
    # 01. Tiny Linear
    data <- generate_data(n = 10, kind = "linear")
    run_scenario("01_tiny_linear", data$x, data$y, frac = 0.8, iter = 0)

    # 02. Quadratic Degree 2 (degree=2 fits quadratic data exactly)
    data <- generate_data(n = 50, kind = "quadratic", noise = 0.02)
    run_scenario("02_quadratic_deg2", data$x, data$y,
        frac = 0.4, iter = 0, degree = 2)

    # 03. Sine Standard
    data <- generate_data(n = 100, kind = "sine", noise = 0.1)
    run_scenario("03_sine_standard", data$x, data$y, frac = 0.3, iter = 0)

    # 04. Sine Robust
    data <- generate_data(n = 100, kind = "sine", outlier_ratio = 0.05)
    run_scenario("04_sine_robust", data$x, data$y, frac = 0.3, iter = 4)

    # 05. Degree 2 (Quadratic LOESS -- unique to loess, not available in lowess)
    data <- generate_data(n = 100, kind = "quadratic", noise = 0.05)
    run_scenario("05_degree2", data$x, data$y,
        frac = 0.4, iter = 0, degree = 2)

    # 06. Large Scale
    data <- generate_data(n = 500, kind = "sine")
    run_scenario("06_large_scale", data$x, data$y, frac = 0.1, iter = 0)

    # 07. High Smoothness
    data <- generate_data(n = 100, kind = "linear", noise = 0.5)
    run_scenario("07_high_smoothness", data$x, data$y, frac = 0.9, iter = 0)

    # 08. Low Smoothness (direct surface -- exact computation at all points)
    data <- generate_data(n = 100, kind = "sine")
    run_scenario("08_low_smoothness", data$x, data$y,
        frac = 0.1, iter = 0,
        surface = "direct"
    )

    # 09. Sine Degree 2 (quadratic local fit on sine data)
    data <- generate_data(n = 100, kind = "sine", noise = 0.1)
    run_scenario("09_sine_degree2", data$x, data$y,
        frac = 0.3, iter = 0, degree = 2)

    # 10. Constant Function
    data <- generate_data(n = 50, kind = "constant")
    run_scenario("10_constant", data$x, data$y, frac = 0.5, iter = 0)

    # 11. Step Function
    data <- generate_data(n = 100, kind = "step")
    run_scenario("11_step_func", data$x, data$y, frac = 0.4, iter = 0)

    # 12. End-effects Left
    data <- generate_data(n = 50, kind = "linear", noise = 0.1)
    run_scenario("12_end_effects_left", data$x, data$y,
        frac = 0.3, iter = 0,
        notes = "Check left boundary"
    )

    # 13. End-effects Right (same data, just naming)
    run_scenario("13_end_effects_right", data$x, data$y,
        frac = 0.3, iter = 0,
        notes = "Check right boundary"
    )

    # 14. Sparse Data
    data <- generate_data(
        n = 20,
        range_max = 100.0,
        kind = "linear",
        noise = 1.0
    )
    run_scenario("14_sparse_data", data$x, data$y, frac = 0.6, iter = 0)

    # 15. Dense Data (direct surface)
    data <- generate_data(n = 500, kind = "sine", noise = 0.1)
    run_scenario("15_dense_data", data$x, data$y,
        frac = 0.05, iter = 0,
        surface = "direct"
    )

    # 16. Degree 2 Robust (quadratic + bisquare robustness)
    data <- generate_data(n = 100, kind = "sine", outlier_ratio = 0.05)
    run_scenario("16_degree2_robust", data$x, data$y,
        frac = 0.3, iter = 4, degree = 2)

    # 17. Degree 2 Direct (exact quadratic fit at all points)
    data <- generate_data(n = 100, kind = "sine")
    run_scenario("17_degree2_direct", data$x, data$y,
        frac = 0.2, iter = 0, degree = 2,
        surface = "direct"
    )

    # 18. Iter 2 Check
    data <- generate_data(n = 100, kind = "sine", outlier_ratio = 0.05)
    run_scenario("18_iter_2", data$x, data$y, frac = 0.4, iter = 2)

    # 19. Interpolate Exact
    data <- generate_data(n = 50, kind = "linear")
    run_scenario("19_interpolate_exact", data$x, data$y, frac = 0.5, iter = 0)

    # 20. Zero Variance
    data <- generate_data(n = 10, kind = "constant")
    run_scenario("20_zero_variance", data$x, data$y, frac = 0.5, iter = 0)

    cat("\nAll supported loess scenarios completed successfully!\n")
}

main()
