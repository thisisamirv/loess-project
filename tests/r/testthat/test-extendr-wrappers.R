RLoess <- getFromNamespace("RLoess", "rfastloess")
RStreamingLoess <- getFromNamespace("RStreamingLoess", "rfastloess")
ROnlineLoess <- getFromNamespace("ROnlineLoess", "rfastloess")

test_that("RLoess generated accessors dispatch fit methods", {
    null_value <- Nullable(NULL)
    handle <- RLoess$new(
        0.3, 1L, "tricube", "bisquare", "mad", "extend",
        null_value, null_value, FALSE, FALSE, FALSE, "use_local_mean",
        null_value, null_value, "kfold", 5L, FALSE,
        "linear", 1L, "normalized",
        null_value, "interpolation", FALSE,
        null_value, null_value, null_value, null_value
    )

    x <- as.double(1:10)
    y <- as.double(2 * x)
    result_dollar <- handle$fit(x, y)
    result_bracket <- handle[["fit"]](x, y)

    expect_length(result_dollar$y, length(y))
    expect_identical(result_dollar$y, result_bracket$y)
})


test_that("RStreamingLoess generated accessors dispatch chunked methods", {
    null_value <- Nullable(NULL)
    handle <- RStreamingLoess$new(
        0.3, 10L, null_value, 1L, "tricube", "bisquare", "mad",
        "extend", "use_local_mean", null_value, FALSE, FALSE, FALSE,
        "average", FALSE, "linear", 1L, "normalized",
        null_value, "interpolation", FALSE,
        null_value, null_value, null_value, null_value, null_value
    )

    x <- as.double(1:10)
    y <- as.double(sin(x))
    chunk_result <- handle$process_chunk(x[1:5], y[1:5])
    final_result <- handle[["finalize"]]()

    expect_type(chunk_result, "list")
    expect_type(final_result, "list")
})


test_that("ROnlineLoess generated accessors dispatch add_point", {
    null_value <- Nullable(NULL)
    handle <- ROnlineLoess$new(
        fraction = 0.3,
        window_capacity = 20L,
        min_points = 3L,
        iterations = 1L,
        weight_function = "tricube",
        robustness_method = "bisquare",
        scaling_method = "mad",
        boundary_policy = "extend",
        zero_weight_fallback = "use_local_mean",
        update_mode = "incremental",
        auto_converge = null_value,
        return_robustness_weights = FALSE,
        return_diagnostics = FALSE,
        return_residuals = FALSE,
        parallel = FALSE,
        degree = "linear",
        dimensions = 1L,
        distance_metric = "normalized",
        weighted_metric_weights = null_value,
        surface_mode = "interpolation",
        return_se = FALSE,
        confidence_intervals = null_value,
        prediction_intervals = null_value,
        cell = null_value,
        interpolation_vertices = null_value,
        boundary_degree_fallback = null_value
    )

    x <- as.double(1:10)
    y <- as.double(cos(x))

    results <- lapply(seq_along(x), function(i) handle$add_point(x[i], y[i]))
    non_null <- Filter(Negate(is.null), results)

    expect_gt(length(non_null), 0)
    has_smoothed <- vapply(
        non_null, function(r) "smoothed" %in% names(r), logical(1)
    )
    expect_true(all(has_smoothed))
})
