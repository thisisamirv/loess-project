#' @srrstats {G5.4a} Analytic correctness cases.
#' @srrstats {G5.5} Fixed random seeds are used for generated data.
#' @srrstats {G5.9a} Machine-epsilon-scale noise does not meaningfully change
#'   the fitted output.
#' @srrstats {RE7.0, RE7.0a} Noiseless exact predictor relationships, including
#'   repeated and constant predictor values, are handled gracefully.
#' @srrstats {RE7.1, RE7.1a} Noiseless exact predictor-response relationships
#'   reproduce their truth and are no slower than noisy equivalents.

test_that("G5.9a machine-epsilon noise does not change LOESS output", {
    set.seed(42)
    x <- as.double(seq(-3, 3, length.out = 80))
    y <- as.double(sin(x) * x + rnorm(length(x), sd = 0.25))
    model <- Loess(
        fraction = 0.5,
        iterations = 2L,
        surface_mode = "direct",
        parallel = FALSE
    )

    baseline <- fit(model, x, y)$y
    perturbed <- fit(model, x, y + .Machine$double.eps * max(1, max(abs(y))))$y

    expect_equal(perturbed, baseline, tolerance = 1e-10)
})

test_that("RE7.0 and RE7.0a handle degenerate predictors", {
    x_repeated <- rep(seq(0, 5, length.out = 10), each = 3)
    y_repeated <- sin(x_repeated) + 0.1
    expect_no_error({
        result <- fit(
            Loess(fraction = 0.4, parallel = FALSE),
            as.double(x_repeated),
            as.double(y_repeated)
        )
    })
    expect_true(all(is.finite(result$y)))

    x_constant <- rep(3, 30)
    y_variable <- seq(1, 30)
    expect_no_error({
        result_constant <- fit(
            Loess(fraction = 0.5, parallel = FALSE),
            as.double(x_constant),
            as.double(y_variable)
        )
    })
    expect_true(all(is.finite(result_constant$y)))
})

test_that("RE7.1 and RE7.1a reproduce noiseless relationships", {
    x_linear <- as.double(seq(-2, 2, length.out = 60))
    y_linear <- 2.5 * x_linear - 1.25
    linear_model <- Loess(
        fraction = 1.0,
        iterations = 0L,
        surface_mode = "direct",
        boundary_policy = "noboundary",
        parallel = FALSE
    )
    linear_result <- fit(linear_model, x_linear, y_linear)
    expect_equal(linear_result$y, y_linear, tolerance = 1e-10)

    x_constant <- as.double(seq(-2, 2, length.out = 40))
    y_constant <- rep(4.25, length(x_constant))
    constant_result <- fit(
        Loess(fraction = 0.5, iterations = 0L, parallel = FALSE),
        x_constant,
        y_constant
    )
    expect_equal(constant_result$y, y_constant, tolerance = 1e-10)

    diagnostic_result <- fit(
        Loess(
            fraction = 1.0,
            iterations = 0L,
            surface_mode = "direct",
            boundary_policy = "noboundary",
            return_diagnostics = TRUE,
            parallel = FALSE
        ),
        x_linear,
        y_linear
    )
    expect_equal(diagnostic_result$diagnostics$r_squared, 1.0, tolerance = 1e-10)
    expect_equal(diagnostic_result$diagnostics$rmse, 0.0, tolerance = 1e-12)

    set.seed(99)
    y_noisy <- y_linear + rnorm(length(y_linear), sd = 0.05)
    fit(linear_model, x_linear, y_linear)
    fit(linear_model, x_linear, y_noisy)
    exact_time <- min(replicate(
        3L,
        system.time(fit(linear_model, x_linear, y_linear))["elapsed"]
    ))
    noisy_time <- min(replicate(
        3L,
        system.time(fit(linear_model, x_linear, y_noisy))["elapsed"]
    ))
    expect_lte(exact_time, noisy_time * 2 + 0.05)
})
