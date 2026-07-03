#' @srrstats {G5.4} Correctness tests for streaming/chunked mode.
#' @srrstats {G5.5} Fixed random seeds.
#' @srrstats {G5.7} Parallel vs serial performance comparison.
#' @srrstats {G5.8} Edge cases: small data, chunk > data.
#' @srrstats {RE3.0} Diagnostics in streaming mode tested.
# Helper to simulate bulk streaming
bulk_stream <- function(x, y, ...) {
    sl <- StreamingLoess(...)
    res <- sl$process_chunk(as.double(x), as.double(y))
    fin <- sl$finalize()
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

    result <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 200, overlap = 20
    )

    expect_type(result, "list")
    expect_length(result$x, length(x))
    expect_length(result$y, length(y))
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

    result_no_overlap <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100, overlap = 0
    )
    result_overlap <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100, overlap = 20
    )

    expect_length(result_no_overlap$y, length(y))
    expect_length(result_overlap$y, length(y))
})

test_that("StreamingLoess diagnostics work", {
    set.seed(42)
    x <- seq(0, 10, length.out = 500)
    y <- 2 * x + rnorm(500, sd = 0.5)

    result <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100,
        return_diagnostics = TRUE
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

    result_serial <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 200, parallel = FALSE
    )
    result_parallel <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 200, parallel = TRUE
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
        result <- bulk_stream(x, y,
            fraction = 0.3, chunk_size = 200, merge_strategy = ms
        )
        expect_length(result$y, length(y))
    }
})

test_that("StreamingLoess: zero_weight_fallback", {
    x <- as.double(seq(0, 10, length.out = 200))
    y <- sin(x)
    result <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100, zero_weight_fallback = "return_original"
    )
    expect_length(result$y, length(y))
})

test_that("StreamingLoess: return_residuals", {
    x <- as.double(seq(0, 10, length.out = 200))
    y <- sin(x)
    sl <- StreamingLoess(fraction = 0.3, chunk_size = 100, return_residuals = TRUE)
    sl$process_chunk(x, y)
    fin <- sl$finalize()
    expect_type(fin, "list")
})

test_that("StreamingLoess: degree, distance_metric, surface_mode, return_se", {
    x <- as.double(seq(0, 10, length.out = 200))
    y <- sin(x)
    result <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100,
        degree = "quadratic", distance_metric = "minkowski:3",
        surface_mode = "direct", return_se = TRUE
    )
    expect_length(result$y, length(y))
})

test_that("StreamingLoess: scaling_method, boundary_policy, auto_converge, return_robustness_weights", {
    x <- as.double(seq(0, 10, length.out = 200))
    y <- sin(x)
    result <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100,
        scaling_method = "mean", boundary_policy = "reflect",
        auto_converge = 1e-3, return_robustness_weights = TRUE
    )
    expect_length(result$y, length(y))
})
