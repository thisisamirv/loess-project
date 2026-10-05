#' @srrstats {G5.4} Correctness tests for streaming/chunked mode.
#' @srrstats {G5.5} Fixed random seeds.
#' @srrstats {G5.7} Parallel vs serial performance comparison.
#' @srrstats {G5.8} Edge cases: small data, chunk > data.
#' @srrstats {RE3.0} Diagnostics in streaming mode tested.
# Helper to simulate bulk streaming
bulk_stream <- function(x, y, ...) {
    sl <- StreamingLoess(...)
    res <- process_chunk(sl, as.double(x), as.double(y))
    fin <- finalize(sl)
    # Merge results
    list(
        x = c(res$x, fin$x),
        y = c(res$y, fin$y),
        diagnostics = if (!is.null(fin$diagnostics)) fin$diagnostics else NULL
    )
}

test_that("StreamingLoess basic functionality works", {
    set.seed(42)
    x <- seq(0, 10, length.out = 1000)
    y <- sin(x) + rnorm(1000, sd = 0.1)

    result <- bulk_stream(x, y, fraction = 0.3, chunk_size = 200, overlap = 20)

    expect_type(result, "list")
    expect_length(result$x, length(x))
    expect_length(result$y, length(y))
})

test_that("StreamingLoess preserves matrix predictor row order", {
    set.seed(1)
    predictors <- cbind(runif(60), runif(60))
    responses <- 3 * predictors[, 1] + 7 * predictors[, 2]
    model <- StreamingLoess(
        fraction = 1,
        chunk_size = 60L,
        overlap = 0L,
        iterations = 0L,
        dimensions = 2L,
        surface_mode = "direct"
    )

    result <- process_chunk(model, predictors, responses)

    expect_equal(result$y, responses, tolerance = 1e-10)
})

test_that("StreamingLoess custom weights downweight outliers", {
    x <- as.double(0:9)
    y <- 2 * x + 1
    y[6] <- 100
    weights <- rep(1, length(y))
    weights[6] <- 0

    options <- list(
        fraction = 1,
        chunk_size = 10,
        overlap = 0,
        iterations = 0,
        surface_mode = "direct"
    )
    weighted <- do.call(StreamingLoess, options)
    plain <- do.call(StreamingLoess, options)
    weighted_result <- process_chunk(weighted, x, y, custom_weights = weights)
    plain_result <- process_chunk(plain, x, y)

    expect_lt(abs(weighted_result$y[6] - 11), abs(plain_result$y[6] - 11))
    expect_error(process_chunk(weighted, x, y, custom_weights = 1), "one numeric value per observation")
})

test_that("StreamingLoess handles different chunk sizes", {
    set.seed(42)
    x <- seq(0, 10, length.out = 500)
    y <- sin(x) + rnorm(500, sd = 0.1)

    result_small <- bulk_stream(x, y, fraction = 0.3, chunk_size = 100)
    result_large <- bulk_stream(x, y, fraction = 0.3, chunk_size = 250)

    expect_length(result_small$y, length(y))
    expect_length(result_large$y, length(y))

    # Results should be similar
    expect_equal(result_small$y, result_large$y, tolerance = 0.1)
})

test_that("StreamingLoess overlap parameter works", {
    set.seed(42)
    x <- seq(0, 10, length.out = 500)
    y <- sin(x) + rnorm(500, sd = 0.1)

    result_no_overlap <- bulk_stream(
        x,
        y,
        fraction = 0.3,
        chunk_size = 100,
        overlap = 0
    )
    result_overlap <- bulk_stream(
        x,
        y,
        fraction = 0.3,
        chunk_size = 100,
        overlap = 20
    )

    expect_length(result_no_overlap$y, length(y))
    expect_length(result_overlap$y, length(y))
})

test_that("StreamingLoess diagnostics work", {
    set.seed(42)
    x <- seq(0, 10, length.out = 500)
    y <- 2 * x + rnorm(500, sd = 0.5)

    result <- bulk_stream(
        x,
        y,
        fraction = 0.3,
        chunk_size = 100,
        outputs = "diagnostics"
    )

    expect_true("diagnostics" %in% names(result))
    expect_type(result$diagnostics, "list")
})

test_that("StreamingLoess handles edge cases", {
    # Small dataset
    x <- 1:50
    y <- sin(x / 10)
    result <- bulk_stream(x, y, fraction = 0.3, chunk_size = 20)
    expect_length(result$y, 50)

    # Chunk size larger than data
    result2 <- bulk_stream(x, y, fraction = 0.3, chunk_size = 100)
    expect_length(result2$y, 50)
})

test_that("StreamingLoess parallel execution works", {
    set.seed(42)
    x <- seq(0, 10, length.out = 1000)
    y <- sin(x) + rnorm(1000, sd = 0.1)

    result_serial <- bulk_stream(
        x,
        y,
        fraction = 0.3,
        chunk_size = 200,
        parallel = FALSE
    )
    result_parallel <- bulk_stream(
        x,
        y,
        fraction = 0.3,
        chunk_size = 200,
        parallel = TRUE
    )

    # Results should be nearly identical
    expect_equal(result_serial$y, result_parallel$y, tolerance = 1e-8)
})

# ---- Parameter coverage ----

test_that("StreamingLoess: merge_strategy variants", {
    set.seed(42)
    x <- as.double(seq(0, 10, length.out = 500))
    y <- sin(x) + rnorm(500, sd = 0.1)

    for (ms in c("average", "weighted_average", "take_first", "take_last")) {
        result <- bulk_stream(
            x,
            y,
            fraction = 0.3,
            chunk_size = 200,
            merge_strategy = ms
        )
        expect_length(result$y, length(y))
    }
})

test_that("StreamingLoess: zero_weight_fallback", {
    x <- as.double(seq(0, 10, length.out = 200))
    y <- sin(x)
    result <- bulk_stream(
        x,
        y,
        fraction = 0.3,
        chunk_size = 100,
        zero_weight_fallback = "return_original"
    )
    expect_length(result$y, length(y))
})

test_that("StreamingLoess: missing = \"drop\" removes non-finite rows", {
    x <- as.double(1:10)
    y <- as.double(c(2, 4, NaN, 8, 10, 12, 14, 16, 18, 20))
    result <- bulk_stream(
        x,
        y,
        fraction = 0.5,
        chunk_size = 10,
        missing = "drop"
    )
    expect_length(result$y, length(y) - 1)
})

test_that("StreamingLoess: return_residuals", {
    x <- as.double(seq(0, 10, length.out = 200))
    y <- sin(x)
    sl <- StreamingLoess(
        fraction = 0.3,
        chunk_size = 100,
        outputs = "residuals"
    )
    process_chunk(sl, x, y)
    fin <- finalize(sl)
    expect_type(fin, "list")
})

test_that("StreamingLoess: degree, distance_metric, surface_mode, return_se", {
    x <- as.double(seq(0, 10, length.out = 200))
    y <- sin(x)
    result <- bulk_stream(
        x,
        y,
        fraction = 0.3,
        chunk_size = 100,
        degree = "quadratic",
        distance_metric = "minkowski:3",
        surface_mode = "direct",
        outputs = "se"
    )
    expect_length(result$y, length(y))
})

test_that("StreamingLoess: scaling_method, boundary_policy, auto_converge", {
    x <- as.double(seq(0, 10, length.out = 200))
    y <- sin(x)
    result <- bulk_stream(
        x,
        y,
        fraction = 0.3,
        chunk_size = 100,
        scaling_method = "mean",
        boundary_policy = "reflect",
        auto_converge = 1e-3,
        outputs = "weights"
    )
    expect_length(result$y, length(y))
})

test_that("StreamingLoess: return_se", {
    x <- as.double(seq(0, 100, length.out = 200))
    y <- sin(x / 10)
    sl <- StreamingLoess(fraction = 0.3, chunk_size = 100, outputs = "se")
    chunk_result <- process_chunk(sl, x, y)
    expect_false(is.null(chunk_result$standard_errors))
    expect_null(chunk_result$confidence_lower)
})

test_that("StreamingLoess: confidence_intervals and prediction_intervals", {
    x <- as.double(seq(0, 100, length.out = 200))
    y <- sin(x / 10)
    sl <- StreamingLoess(
        fraction = 0.3,
        chunk_size = 100,
        intervals = intervals_opts(confidence = 0.95, prediction = 0.95)
    )
    chunk_result <- process_chunk(sl, x, y)
    expect_false(is.null(chunk_result$confidence_lower))
    expect_false(is.null(chunk_result$prediction_lower))
    expect_true(all(
        chunk_result$confidence_lower <= chunk_result$confidence_upper
    ))
    ci_width <- chunk_result$confidence_upper - chunk_result$confidence_lower
    pi_width <- chunk_result$prediction_upper - chunk_result$prediction_lower
    expect_true(all(pi_width >= ci_width - 1e-9))
})
