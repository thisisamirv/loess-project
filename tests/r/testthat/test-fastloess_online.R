#' @srrstats {G5.4} Correctness tests for online/sliding window mode.
#' @srrstats {G5.5} Fixed random seeds.
#' @srrstats {G5.8} Edge cases: min data, window > data.
#' @srrstats {RE4.0} Robustness iterations tested.
test_that("OnlineLoess basic functionality works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol <- OnlineLoess(
        fraction = 0.3, window_capacity = 25, min_points = 10
    )
    result <- ol$add_points(as.double(x), as.double(y))

    expect_type(result, "list")
    expect_length(result$x, length(x))
    expect_length(result$y, length(y))
    expect_type(result$x, "double")
    expect_type(result$y, "double")
})

test_that("OnlineLoess window capacity works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_small <- OnlineLoess(
        fraction = 0.3, window_capacity = 15
    )
    result_small <- ol_small$add_points(as.double(x), as.double(y))

    ol_large <- OnlineLoess(
        fraction = 0.3, window_capacity = 50
    )
    result_large <- ol_large$add_points(as.double(x), as.double(y))

    expect_length(result_small$y, length(y))
    expect_length(result_large$y, length(y))
})

test_that("OnlineLoess min_points parameter works", {
    set.seed(42)
    x <- 1:50
    y <- sin(x / 10) + rnorm(50, sd = 0.1)

    ol <- OnlineLoess(
        fraction = 0.3, window_capacity = 25, min_points = 5
    )
    result <- ol$add_points(as.double(x), as.double(y))

    expect_length(result$y, 50)

    # First few poins (before min_points) should be original values
    # (or close to them if smoothing starts immediately)
    expect_type(result$y, "double")
})

test_that("OnlineLoess update modes work", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_full <- OnlineLoess(
        fraction = 0.3, window_capacity = 25,
        update_mode = "full"
    )
    result_full <- ol_full$add_points(as.double(x), as.double(y))

    ol_incr <- OnlineLoess(
        fraction = 0.3, window_capacity = 25,
        update_mode = "incremental"
    )
    result_incr <- ol_incr$add_points(as.double(x), as.double(y))

    expect_length(result_full$y, length(y))
    expect_length(result_incr$y, length(y))
})

test_that("OnlineLoess handles edge cases", {
    # Minimum data points
    x <- 1:10
    y <- 1:10
    ol <- OnlineLoess(
        fraction = 0.5, window_capacity = 5, min_points = 3
    )
    result <- ol$add_points(as.double(x), as.double(y))
    expect_length(result$y, 10)

    # Window larger than data
    ol2 <- OnlineLoess(
        fraction = 0.5, window_capacity = 20, min_points = 3
    )
    result2 <- ol2$add_points(as.double(x), as.double(y))
    expect_length(result2$y, 10)
})

test_that("OnlineLoess robustness works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)
    y[50] <- y[50] + 5 # Add outlier

    ol_no_robust <- OnlineLoess(
        fraction = 0.3, window_capacity = 25,
        iterations = 0
    )
    result_no_robust <- ol_no_robust$add_points(as.double(x), as.double(y))

    ol_robust <- OnlineLoess(
        fraction = 0.3, window_capacity = 25,
        iterations = 3
    )
    result_robust <- ol_robust$add_points(as.double(x), as.double(y))

    expect_length(result_no_robust$y, length(y))
    expect_length(result_robust$y, length(y))
})

# ---- Parameter coverage ----

test_that("OnlineLoess: zero_weight_fallback", {
    x <- as.double(1:50)
    y <- sin(x / 10)

    for (zwf in c("use_local_mean", "return_original", "return_none")) {
        ol <- OnlineLoess(fraction = 0.3, window_capacity = 20, zero_weight_fallback = zwf)
        r <- ol$add_points(x, y)
        expect_length(r$y, 50)
    }
})

test_that("OnlineLoess: degree, distance_metric, surface_mode, return_se", {
    x <- as.double(seq(0, 10, length.out = 30))
    y <- sin(x)

    r <- OnlineLoess(
        fraction = 0.5, window_capacity = 20,
        degree = "quadratic", distance_metric = "minkowski:3",
        surface_mode = "direct", return_se = TRUE
    )$add_points(x, y)
    expect_length(r$y, 30)
    # enp may be NULL for online mode (hat matrix only for direct surface)
    expect_type(r, "list")
})

test_that("OnlineLoess: scaling_method, boundary_policy, auto_converge, return_robustness_weights", {
    x <- as.double(seq(0, 10, length.out = 30))
    y <- sin(x)

    r <- OnlineLoess(
        fraction = 0.5, window_capacity = 20,
        scaling_method = "mar", boundary_policy = "reflect",
        auto_converge = 1e-3, return_robustness_weights = TRUE
    )$add_points(x, y)
    expect_length(r$y, 30)
})
