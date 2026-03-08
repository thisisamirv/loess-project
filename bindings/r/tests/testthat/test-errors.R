#' @srrstats {G5.2, G5.2a, G5.2b} Error and tests for all input validation.
#' @srrstats {G5.8, G5.8a, G5.8b} Edge condition tests for invalid inputs.
test_that("Loess rejects invalid inputs", {
    # Invalid fraction
    expect_error(
        Loess(fraction = -0.1),
        "fraction must be between 0 and 1"
    )
    expect_error(
        Loess(fraction = 1.5),
        "fraction must be between 0 and 1"
    )

    # Invalid iterations
    expect_error(
        Loess(iterations = -1),
        "iterations must be a non-negative integer"
    )

    # Mismatched lengths at fit time
    expect_error(
        Loess(fraction = 0.5)$fit(as.double(1:10), as.double(1:5)),
        "x and y must have the same length"
    )
})

test_that("OnlineLoess rejects invalid inputs", {
    # Invalid parameters
    expect_error(
        OnlineLoess(fraction = -0.1),
        "fraction must be between 0 and 1"
    )
    expect_error(
        OnlineLoess(window_capacity = 0),
        "window_capacity must be a positive integer"
    )
    expect_error(
        OnlineLoess(min_points = -1),
        "min_points must be a non-negative integer"
    )

    # Mismatched lengths at add_points time
    ol <- OnlineLoess(fraction = 0.5)
    expect_error(
        ol$add_points(as.double(1:10), as.double(1:5)),
        "x and y must have the same length"
    )
})

test_that("StreamingLoess rejects invalid inputs", {
    # Invalid parameters
    expect_error(
        StreamingLoess(fraction = -0.1),
        "fraction must be between 0 and 1"
    )
    expect_error(
        StreamingLoess(chunk_size = 0),
        "chunk_size must be a positive integer"
    )

    # Mismatched lengths at process_chunk time
    sl <- StreamingLoess(fraction = 0.5)
    expect_error(
        sl$process_chunk(as.double(1:10), as.double(1:5)),
        "x and y must have the same length"
    )
})
