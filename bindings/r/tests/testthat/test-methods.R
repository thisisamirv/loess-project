#' @srrstats {G5.2, G5.2a, G5.2b} Error/warning tests for print/plot methods.
#' @srrstats {G5.3} No NA/NaN in print method outputs.
#' @srrstats {G5.5} Fixed random seeds in tests using set.seed().
#' @srrstats {RE4.17, RE4.18} Print method tests verify S3 dispatch.
#' @srrstats {RE6.0, RE6.2} Plot method tests verify output.

test_that("print.Loess outputs correct fields", {
    model <- Loess(fraction = 0.3, iterations = 2L)
    out <- capture.output(print(model))
    expect_true(any(grepl("Loess Model", out, fixed = TRUE)))
    expect_true(any(grepl("Fraction", out, fixed = TRUE)))
    expect_true(any(grepl("Iterations", out, fixed = TRUE)))
    expect_true(any(grepl("Weight Function", out, fixed = TRUE)))
    expect_true(any(grepl("Parallel", out, fixed = TRUE)))
    # print returns x invisibly
    expect_identical(print(model), model)
})

test_that("print.LoessResult outputs basic fields", {
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    result <- fit(Loess(fraction = 0.3), x, y)
    out <- capture.output(print(result))
    expect_true(any(grepl("LoessResult", out, fixed = TRUE)))
    expect_true(any(grepl("Points", out, fixed = TRUE)))
    expect_true(any(grepl("Fraction Used", out, fixed = TRUE)))
    expect_identical(print(result), result)
})

test_that("print.LoessResult shows iterations_used when present", {
    # Use a mock object to unconditionally exercise the optional branch
    mock <- structure(
        list(
            x = as.double(1:10),
            y = as.double(1:10),
            fraction_used = 0.3,
            iterations_used = 3L,
            cv_scores = NULL
        ),
        class = "LoessResult"
    )
    out <- capture.output(print(mock))
    expect_true(any(grepl("Iterations Used", out, fixed = TRUE)))
})

test_that("print.LoessResult shows cv_scores when present", {
    set.seed(42)
    x <- seq(0, 10, length.out = 100)
    y <- sin(x) + rnorm(100, 0, 0.2)
    result <- fit(
        Loess(
            cv = cv_opts(fractions = c(0.2, 0.3, 0.5), method = "kfold", k = 5L)
        ),
        x,
        y
    )
    out <- capture.output(print(result))
    expect_true(any(grepl("CV Scores", out, fixed = TRUE)))
    expect_true(any(grepl("3 scores", out, fixed = TRUE)))
    expect_false(any(grepl("3 folds", out, fixed = TRUE)))
})

test_that("print.StreamingLoess outputs correct fields", {
    model <- StreamingLoess(fraction = 0.3, chunk_size = 50L)
    out <- capture.output(print(model))
    expect_true(any(grepl("StreamingLoess Model", out, fixed = TRUE)))
    expect_true(any(grepl("Fraction", out, fixed = TRUE)))
    expect_true(any(grepl("Chunk Size", out, fixed = TRUE)))
    expect_true(any(grepl("Parallel", out, fixed = TRUE)))
    expect_identical(print(model), model)
})

test_that("print.OnlineLoess outputs correct fields", {
    model <- OnlineLoess(
        fraction = 0.2,
        window_capacity = 20L,
        update_mode = "full"
    )
    out <- capture.output(print(model))
    expect_true(any(grepl("OnlineLoess Model", out, fixed = TRUE)))
    expect_true(any(grepl("Fraction", out, fixed = TRUE)))
    expect_true(any(grepl("Window Capacity", out, fixed = TRUE)))
    expect_true(any(grepl("Min Points", out, fixed = TRUE)))
    expect_true(any(grepl("Update Mode:.*full", out)))
    expect_identical(print(model), model)
})

test_that("plot.LoessResult runs without error", {
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    result <- fit(Loess(fraction = 0.3), x, y)
    expect_no_error(plot(result))
})

test_that("plot.LoessResult explains its multivariate limitation", {
    result <- structure(
        list(x = as.double(1:4), y = as.double(1:2), dimensions = 2L),
        class = "LoessResult"
    )

    expect_error(plot(result), "supports only one-dimensional fits")
})

test_that("plot.LoessResult draws confidence interval lines when present", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.2)
    result <- fit(
        Loess(fraction = 0.5, intervals = intervals_opts(confidence = 0.95)),
        x,
        y
    )
    expect_no_error(plot(result, main = "With CI"))
})

test_that("online methods reject malformed inputs and unused arguments", {
    model <- OnlineLoess(window_capacity = 20L)
    for (value in list(NULL, "bad", 1i, matrix(1))) {
        expect_error(add_point(model, value, 1), "coordinate vector")
    }
    expect_error(add_point(model, c(1, 2), 1), "exactly 1 values")
    for (value in list(NULL, "bad", 1i, c(1, 2), matrix(1))) {
        expect_error(add_point(model, 1, value), "single numeric value")
    }
    invalid_weights <- list(
        NULL, "bad", 1i, c(1, 2), matrix(1), NA_real_, NaN, Inf, -1
    )
    for (weight in invalid_weights) {
        expect_error(
            add_point(model, 1, 1, weight = weight),
            "single finite non-negative numeric value"
        )
    }
    expect_error(add_point(model, 1, 1, typo = 1), "unused arguments")
    expect_error(window_diagnostics(model, typo = 1), "unused arguments")
    expect_error(predict_window(model, 1, typo = 1), "unused arguments")
    expect_null(add_point(model, 1L, 3L, weight = 0L))
    expect_type(add_point(model, 2L, 5L, weight = 1L)$y, "double")
})

test_that("retained prediction validates query shapes and gradient aliases", {
    model <- Loess(
        fraction = 0.8,
        iterations = 0L,
        surface_mode = "direct",
        retain_model = TRUE
    )
    invisible(fit(model, 1:20, 2 * (1:20) + 1))
    expect_error(predict(model, numeric()), "non-empty")
    expect_error(predict(model, matrix(1)), "numeric vector")
    expect_error(predict(model, 1, typo = 1), "unused arguments")
    multivariate <- Loess(dimensions = 2L)
    expect_error(predict(multivariate, 1:3), "multiple of dimensions")
    gradient <- predict(model, c(10, 11), outputs = "gradient")
    derivative <- predict(model, c(10, 11), outputs = "derivative")
    expect_equal(gradient$derivative, c(2, 2), tolerance = 1e-10)
    expect_identical(gradient, derivative)
    expect_false("derivative" %in% names(predict(model, 5)))
})
