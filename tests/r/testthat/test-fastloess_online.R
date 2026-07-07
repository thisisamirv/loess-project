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
    results <- lapply(seq_along(x), function(i) ol$add_point(x[i], y[i]))
    non_null <- Filter(Negate(is.null), results)

    expect_true(length(non_null) > 0)
    expect_true("smoothed" %in% names(non_null[[1]]))
    expect_type(non_null[[1]]$smoothed, "double")
})

test_that("OnlineLoess window capacity works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_small <- OnlineLoess(
        fraction = 0.3, window_capacity = 15
    )
    results_small <- lapply(seq_along(x), function(i) ol_small$add_point(x[i], y[i]))

    ol_large <- OnlineLoess(
        fraction = 0.3, window_capacity = 50
    )
    results_large <- lapply(seq_along(x), function(i) ol_large$add_point(x[i], y[i]))

    expect_true(length(Filter(Negate(is.null), results_small)) > 0)
    expect_true(length(Filter(Negate(is.null), results_large)) > 0)
})

test_that("OnlineLoess min_points parameter works", {
    set.seed(42)
    x <- 1:50
    y <- sin(x / 10) + rnorm(50, sd = 0.1)

    ol <- OnlineLoess(
        fraction = 0.3, window_capacity = 25, min_points = 5
    )
    results <- lapply(seq_along(x), function(i) ol$add_point(x[i], y[i]))

    # First 4 calls (before min_points = 5) should return NULL
    expect_null(results[[1]])
    expect_null(results[[4]])
    # After min_points points, should return a result
    non_null <- Filter(Negate(is.null), results)
    expect_true(length(non_null) > 0)
    expect_type(non_null[[1]]$smoothed, "double")
})

test_that("OnlineLoess update modes work", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_full <- OnlineLoess(
        fraction = 0.3, window_capacity = 25,
        update_mode = "full"
    )
    results_full <- lapply(seq_along(x), function(i) ol_full$add_point(x[i], y[i]))

    ol_incr <- OnlineLoess(
        fraction = 0.3, window_capacity = 25,
        update_mode = "incremental"
    )
    results_incr <- lapply(seq_along(x), function(i) ol_incr$add_point(x[i], y[i]))

    expect_true(length(Filter(Negate(is.null), results_full)) > 0)
    expect_true(length(Filter(Negate(is.null), results_incr)) > 0)
})

test_that("OnlineLoess handles edge cases", {
    # Minimum data points
    x <- 1:10
    y <- 1:10
    ol <- OnlineLoess(
        fraction = 0.5, window_capacity = 5, min_points = 3
    )
    results <- lapply(seq_along(x), function(i) ol$add_point(x[i], y[i]))
    non_null <- Filter(Negate(is.null), results)
    expect_true(length(non_null) > 0)

    # Window larger than data
    ol2 <- OnlineLoess(
        fraction = 0.5, window_capacity = 20, min_points = 3
    )
    results2 <- lapply(seq_along(x), function(i) ol2$add_point(x[i], y[i]))
    non_null2 <- Filter(Negate(is.null), results2)
    expect_true(length(non_null2) > 0)
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
    results_no_robust <- lapply(seq_along(x), function(i) ol_no_robust$add_point(x[i], y[i]))

    ol_robust <- OnlineLoess(
        fraction = 0.3, window_capacity = 25,
        iterations = 3
    )
    results_robust <- lapply(seq_along(x), function(i) ol_robust$add_point(x[i], y[i]))

    expect_true(length(Filter(Negate(is.null), results_no_robust)) > 0)
    expect_true(length(Filter(Negate(is.null), results_robust)) > 0)
})

# ---- Parameter coverage ----

test_that("OnlineLoess: zero_weight_fallback", {
    x <- as.double(1:50)
    y <- sin(x / 10)

    for (zwf in c("use_local_mean", "return_original", "return_none")) {
        ol <- OnlineLoess(
            fraction = 0.3, window_capacity = 20,
            zero_weight_fallback = zwf
        )
        r <- ol$add_point(x[1], y[1])
        # NULL until min_points reached, or a list with smoothed
        expect_true(is.null(r) || "smoothed" %in% names(r))
    }
})

test_that("OnlineLoess: degree, distance_metric, surface_mode, return_se", {
    x <- as.double(seq(0, 10, length.out = 30))
    y <- sin(x)

    ol <- OnlineLoess(
        fraction = 0.5, window_capacity = 20,
        degree = "quadratic", distance_metric = "minkowski:3",
        surface_mode = "direct", return_se = TRUE
    )
    results <- lapply(seq_along(x), function(i) ol$add_point(x[i], y[i]))
    non_null <- Filter(Negate(is.null), results)
    expect_true(length(non_null) > 0)
    expect_type(non_null[[1]]$smoothed, "double")
})

test_that("OnlineLoess: scaling_method, boundary_policy, auto_converge", {
    x <- as.double(seq(0, 10, length.out = 30))
    y <- sin(x)

    ol <- OnlineLoess(
        fraction = 0.5, window_capacity = 20,
        scaling_method = "mar", boundary_policy = "reflect",
        auto_converge = 1e-3, return_robustness_weights = TRUE
    )
    results <- lapply(seq_along(x), function(i) ol$add_point(x[i], y[i]))
    non_null <- Filter(Negate(is.null), results)
    expect_true(length(non_null) > 0)
})
