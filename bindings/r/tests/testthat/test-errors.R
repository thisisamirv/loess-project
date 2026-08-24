#' @srrstats {G5.2, G5.2a, G5.2b} Error and tests for all input validation.
#' @srrstats {G5.8, G5.8a, G5.8b} Edge condition tests for invalid inputs.
test_that("Loess rejects invalid inputs", {
    # Unnamed positional arguments are forbidden
    expect_error(
        Loess(0.5, 3),
        "All arguments after 'fraction' must be named"
    )

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
        fit(Loess(fraction = 0.5), as.double(1:7), as.double(1:5)),
        "must match y"
    )

    # Extra ... args rejected by fit.Loess
    expect_error(
        fit(Loess(fraction = 0.5), as.double(1:10), as.double(1:10), extra = 1),
        "unused arguments"
    )

    # Matrix x with wrong column count rejected by fit.Loess
    expect_error(
        fit(
            Loess(fraction = 0.5, dimensions = 1L),
            matrix(as.double(1:20), nrow = 10, ncol = 2),
            as.double(1:10)
        ),
        "dimensions"
    )
})

test_that("OnlineLoess rejects invalid inputs", {
    # Unnamed positional arguments are forbidden
    expect_error(
        OnlineLoess(0.5, 100),
        "All arguments after 'min_points' must be named"
    )

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

    # add_point accepts scalar x and y without error
    ol <- OnlineLoess(fraction = 0.5)
    result <- add_point(ol, 1.0, 2.0)
    expect_true(is.null(result) || "y" %in% names(result))

    # Extra ... args rejected by add_point.OnlineLoess
    expect_error(
        add_point(OnlineLoess(fraction = 0.5), 1.0, 2.0, extra = 1),
        "unused arguments"
    )
})

test_that("StreamingLoess rejects invalid inputs", {
    # Unnamed positional arguments are forbidden
    expect_error(
        StreamingLoess(0.5, 100),
        "All arguments after 'chunk_size' must be named"
    )

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
        process_chunk(sl, as.double(1:7), as.double(1:5)),
        "must match y"
    )

    # Extra ... args rejected by process_chunk.StreamingLoess
    expect_error(
        process_chunk(
            StreamingLoess(fraction = 0.5),
            as.double(1:10), as.double(1:10),
            extra = 1
        ),
        "unused arguments"
    )

    # Matrix x with wrong column count rejected by process_chunk.StreamingLoess
    expect_error(
        process_chunk(
            StreamingLoess(fraction = 0.5, dimensions = 1L),
            matrix(as.double(1:20), nrow = 10, ncol = 2),
            as.double(1:10)
        ),
        "dimensions"
    )

    # Extra ... args rejected by finalize.StreamingLoess
    sl2 <- StreamingLoess(fraction = 0.5)
    process_chunk(sl2, as.double(1:10), as.double(1:10))
    expect_error(finalize(sl2, extra = 1), "unused arguments")
})
