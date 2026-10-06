test_that("matches stats::loess for randomized inputs (property-based)", {
    property <- function(xy, fraction, degree) {
        x <- loess_property_x(xy[[1]])
        y <- xy[[2]]

        expect_true(check_stats_loess(
            x,
            y,
            fraction,
            degree = degree,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
            len = c(12L, 40L)
        ),
        fraction = quickcheck::double_bounded(0.4, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        property = property,
        tests = 200L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess for randomized sorted output", {
    property <- function(xy, fraction, degree) {
        x <- loess_property_x(xy[[1]])
        y <- xy[[2]]

        expect_true(check_stats_loess(
            x,
            y,
            fraction,
            degree = degree,
            sorted = TRUE,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
            len = c(12L, 40L)
        ),
        fraction = quickcheck::double_bounded(0.4, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        property = property,
        tests = 200L,
        shrinks = 0L,
        discards = 1000L
    )
})

# `loess_property_x()` builds strictly increasing x, so the comparisons above
# never see ties. Collapsing the draw onto a few levels exercises the tied-x
# neighbourhoods instead.
test_that("matches stats::loess for tied x-values (property-based)", {
    property <- function(xy, levels, fraction, degree, iterations) {
        x <- round(xy[[1]] / (200 / levels))

        expect_true(check_stats_loess(
            x,
            xy[[2]],
            fraction,
            degree = degree,
            iterations = iterations,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
            len = c(12L, 40L)
        ),
        levels = quickcheck::integer_bounded(3L, 12L, len = 1L),
        fraction = quickcheck::double_bounded(0.4, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 8L, len = 1L),
        property = property,
        tests = 200L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess when the interpolation threshold is zero", {
    order_values <- c(
        -95.852866,
        50.239543,
        44.266116,
        38.704759,
        63.911146,
        -14.960172,
        80.152979,
        7.994468,
        99.341495,
        -97.587895,
        -86.512836,
        -30.204539
    )
    responses <- c(
        22.357882,
        25.303123,
        -5.544972,
        -50.281210,
        78.202536,
        52.206016,
        -89.353151,
        -63.280238,
        87.488326,
        66.498669,
        15.548199,
        37.319462
    )
    predictors <- loess_property_x(order_values)
    fraction <- 0.412400163523853

    expect_equal(floor(length(predictors) * fraction * 0.2), 0)
    for (degree in 1:2) {
        expect_true(check_stats_loess(
            predictors,
            responses,
            fraction,
            degree = degree,
            surface = "interpolate",
            boundary_degree_fallback = FALSE,
            tolerance = 1e-10
        ))
    }
})

# Also compare R's interpolated surface (local fits at kd-tree vertices blended
# with cubic Hermite bases) on randomized inputs.
# `boundary_degree_fallback = FALSE` selects R's behaviour at vertices outside
# the data range; the package default reduces those to linear fits instead.
test_that("matches stats::loess on the interpolated surface", {
    property <- function(xy, fraction, degree) {
        x <- loess_property_x(xy[[1]])
        y <- xy[[2]]

        expect_true(check_stats_loess(
            x,
            y,
            fraction,
            degree = degree,
            surface = "interpolate",
            boundary_degree_fallback = FALSE,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
            len = c(12L, 40L)
        ),
        fraction = quickcheck::double_bounded(0.4, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        property = property,
        tests = 200L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess for randomized case-weighted fits", {
    property <- function(
        samples,
        fraction,
        degree,
        iterations,
        interpolate,
        parallel,
        zero_weight
    ) {
        predictors <- loess_property_x(samples[[1]])
        weights <- samples[[3]]
        if (zero_weight) {
            weights[1L] <- 0
        }

        expect_true(check_stats_loess(
            predictors,
            samples[[2]],
            fraction,
            degree = degree,
            iterations = iterations,
            custom_weights = weights,
            surface = if (interpolate) "interpolate" else "direct",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100),
            quickcheck::double_bounded(-100, 100),
            quickcheck::double_bounded(0.05, 5),
            len = c(24L, 48L)
        ),
        fraction = quickcheck::double_bounded(0.6, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 4L, len = 1L),
        interpolate = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        zero_weight = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess for unevenly spaced and scaled predictors", {
    property <- function(
        samples,
        fraction,
        degree,
        exponent,
        offset,
        interpolate,
        parallel
    ) {
        predictors <- offset + samples[[1]] * 10^exponent

        expect_true(check_stats_loess(
            predictors,
            samples[[2]],
            fraction,
            degree = degree,
            surface = if (interpolate) "interpolate" else "direct",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-5, 5),
            quickcheck::double_bounded(-100, 100),
            len = c(24L, 48L)
        ),
        fraction = quickcheck::double_bounded(0.6, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        exponent = quickcheck::integer_bounded(-4L, 4L, len = 1L),
        offset = quickcheck::double_bounded(-10, 10, len = 1L),
        interpolate = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess for two-dimensional predictors", {
    property <- function(
        samples,
        fraction,
        degree,
        scale,
        interpolate,
        parallel
    ) {
        predictors <- cbind(samples[[1]], samples[[2]] * scale)

        expect_true(check_stats_loess(
            predictors,
            samples[[3]],
            fraction,
            degree = degree,
            surface = if (interpolate) "interpolate" else "direct",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-5, 5),
            quickcheck::double_bounded(-5, 5),
            quickcheck::double_bounded(-100, 100),
            len = c(36L, 60L)
        ),
        fraction = quickcheck::double_bounded(0.65, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        scale = quickcheck::double_bounded(0.01, 100, len = 1L),
        interpolate = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess near span boundaries with varied cell sizes", {
    property <- function(
        samples,
        neighbors,
        direction,
        degree,
        cell,
        parallel
    ) {
        predictors <- loess_property_x(samples[[1]])
        fraction <- (neighbors + direction * 1e-8) / length(predictors)

        expect_true(check_stats_loess(
            predictors,
            samples[[2]],
            fraction,
            degree = degree,
            cell = cell,
            surface = "interpolate",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100),
            quickcheck::double_bounded(-100, 100),
            len = c(12L, 40L)
        ),
        neighbors = quickcheck::integer_bounded(5L, 10L, len = 1L),
        direction = quickcheck::integer_bounded(-1L, 1L, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        cell = quickcheck::double_bounded(0.05, 0.7, len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess predictions at interior and boundary queries", {
    property <- function(
        samples,
        fraction,
        degree,
        position,
        interpolate,
        parallel
    ) {
        predictors <- loess_property_x(samples[[1]])
        lower <- min(predictors)
        upper <- max(predictors)
        anchor <- predictors[1L + floor(position * (length(predictors) - 1L))]
        queries <- pmax(
            lower,
            pmin(
                upper,
                c(
                    lower,
                    upper,
                    lower + position * (upper - lower),
                    anchor,
                    anchor - (upper - lower) * 1e-8,
                    anchor + (upper - lower) * 1e-8
                )
            )
        )

        expect_true(check_stats_loess(
            predictors,
            samples[[2]],
            fraction,
            degree = degree,
            new_x = queries,
            surface = if (interpolate) "interpolate" else "direct",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100),
            quickcheck::double_bounded(-10, 10),
            len = c(24L, 48L)
        ),
        fraction = quickcheck::double_bounded(0.6, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        position = quickcheck::double_bounded(0, 1, len = 1L),
        interpolate = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess for three- and four-dimensional fits", {
    property <- function(
        samples,
        dimensions,
        fraction,
        degree,
        scale,
        interpolate,
        parallel
    ) {
        predictors <- do.call(cbind, samples[seq_len(dimensions)])
        predictors[, 2L] <- predictors[, 2L] * scale
        responses <- samples[[5]] + predictors[, 1L] * samples[[2]]

        expect_true(check_stats_loess(
            predictors,
            responses,
            fraction,
            degree = degree,
            surface = if (interpolate) "interpolate" else "direct",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-3, 3),
            quickcheck::double_bounded(-3, 3),
            quickcheck::double_bounded(-3, 3),
            quickcheck::double_bounded(-3, 3),
            quickcheck::double_bounded(-10, 10),
            len = c(96L, 128L)
        ),
        dimensions = quickcheck::integer_bounded(3L, 4L, len = 1L),
        fraction = quickcheck::double_bounded(0.65, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        scale = quickcheck::double_bounded(0.01, 100, len = 1L),
        interpolate = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 50L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess for weighted robust multidimensional fits", {
    property <- function(
        samples,
        fraction,
        degree,
        iterations,
        interpolate,
        parallel
    ) {
        predictors <- cbind(samples[[1]], samples[[2]])
        responses <- samples[[3]]
        responses[1L] <- responses[1L] + 20

        expect_true(check_stats_loess(
            predictors,
            responses,
            fraction,
            degree = degree,
            iterations = iterations,
            custom_weights = samples[[4]],
            surface = if (interpolate) "interpolate" else "direct",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-3, 3),
            quickcheck::double_bounded(-3, 3),
            quickcheck::double_bounded(-5, 5),
            quickcheck::double_bounded(0.1, 3),
            len = c(48L, 72L)
        ),
        fraction = quickcheck::double_bounded(0.75, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        iterations = quickcheck::integer_bounded(1L, 6L, len = 1L),
        interpolate = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 50L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess prediction standard errors", {
    property <- function(samples, fraction, position, interpolate, parallel) {
        predictors <- loess_property_x(samples[[1]])
        queries <- c(
            min(predictors),
            max(predictors),
            min(predictors) + position * diff(range(predictors))
        )

        expect_true(check_stats_loess(
            predictors,
            samples[[2]],
            fraction,
            degree = 1L,
            new_x = queries,
            se = TRUE,
            surface = if (interpolate) "interpolate" else "direct",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100),
            quickcheck::double_bounded(-5, 5),
            len = c(48L, 80L)
        ),
        fraction = quickcheck::double_bounded(0.6, 1.0, len = 1L),
        position = quickcheck::double_bounded(0, 1, len = 1L),
        interpolate = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("compares finite stats::loess fits on degenerate predictor geometry", {
    property <- function(samples, geometry, fraction, degree, parallel) {
        predictors <- switch(
            geometry,
            rep(2, length(samples[[1]])),
            cbind(samples[[1]], 2 * samples[[1]]),
            cbind(round(samples[[1]]), round(samples[[1]])^2)
        )

        expect_true(check_stats_loess(
            predictors,
            samples[[2]],
            fraction,
            degree = degree,
            surface = "direct",
            parallel = parallel,
            compare_singular = TRUE,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-3, 3),
            quickcheck::double_bounded(-5, 5),
            len = c(40L, 64L)
        ),
        geometry = quickcheck::integer_bounded(1L, 3L, len = 1L),
        fraction = quickcheck::double_bounded(0.8, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess across extreme predictor and response scales", {
    property <- function(
        samples,
        fraction,
        degree,
        exponent,
        translation,
        response_exponent,
        interpolate,
        parallel
    ) {
        scale <- 10^exponent
        predictors <- (loess_property_x(samples[[1]]) + translation) * scale
        responses <- samples[[2]] * 10^response_exponent

        expect_true(check_stats_loess(
            predictors,
            responses,
            fraction,
            degree = degree,
            surface = if (interpolate) "interpolate" else "direct",
            boundary_degree_fallback = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100),
            quickcheck::double_bounded(-5, 5),
            len = c(32L, 64L)
        ),
        fraction = quickcheck::double_bounded(0.65, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        exponent = quickcheck::integer_bounded(-12L, 12L, len = 1L),
        translation = quickcheck::double_bounded(-1e6, 1e6, len = 1L),
        response_exponent = quickcheck::integer_bounded(-4L, 8L, len = 1L),
        interpolate = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::loess for randomized robust fits with outliers", {
    property <- function(
        n,
        seed,
        fraction,
        degree,
        iterations,
        spike_position,
        spike_magnitude,
        spike_negative
    ) {
        set.seed(seed)
        x <- as.double(seq(-5, 5, length.out = n)[sample.int(n)])
        y <- as.double(sin(x) + rnorm(n, sd = 0.2))
        spike_index <- min(n, floor(spike_position * n) + 1L)
        spike_value <- if (spike_negative) -spike_magnitude else spike_magnitude
        y[spike_index] <- y[spike_index] + spike_value

        expect_true(check_stats_loess(
            x,
            y,
            fraction,
            degree = degree,
            iterations = iterations,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        n = quickcheck::integer_bounded(20L, 40L, len = 1L),
        seed = quickcheck::integer_bounded(1L, 100000L, len = 1L),
        fraction = quickcheck::double_bounded(0.4, 1.0, len = 1L),
        degree = quickcheck::integer_bounded(1L, 2L, len = 1L),
        iterations = quickcheck::integer_bounded(1L, 12L, len = 1L),
        spike_position = quickcheck::double_bounded(0, 1, len = 1L),
        spike_magnitude = quickcheck::double_bounded(2, 10, len = 1L),
        spike_negative = quickcheck::logical_(len = 1L),
        property = property,
        tests = 50L,
        discards = 1000L
    )
})

test_that("matches stats::loess for fixed long-run robust fits", {
    set.seed(912)
    x <- as.double(seq(-5, 5, length.out = 48L))
    y <- as.double(sin(x) + rnorm(length(x), sd = 0.2))
    y[c(11L, 32L)] <- y[c(11L, 32L)] + c(5, -4)

    for (iterations in c(12L, 24L)) {
        reference <- stats::loess(
            y ~ x,
            data = data.frame(x = x, y = y),
            span = 0.6,
            degree = 2L,
            family = "symmetric",
            control = stats::loess.control(
                surface = "direct",
                iterations = iterations
            )
        )
        result <- fit(
            Loess(
                fraction = 0.6,
                degree = "quadratic",
                iterations = iterations,
                scaling_method = "mar",
                boundary_policy = "noboundary",
                surface_mode = "direct",
                parallel = FALSE
            ),
            x,
            y
        )

        expect_equal(result$y, reference$fitted, tolerance = 1e-10)
    }
})

test_that("matches initial stats::loess fits for sparse one-spike responses", {
    property <- function(
        x,
        spike_position,
        spike_magnitude,
        spike_negative,
        fraction
    ) {
        x <- loess_property_x(x)

        spike_index <- min(length(x), floor(spike_position * length(x)) + 1L)
        spike_value <- if (spike_negative) -spike_magnitude else spike_magnitude
        y <- numeric(length(x))
        y[spike_index] <- spike_value

        expect_true(check_stats_loess(
            x,
            y,
            fraction,
            sorted = TRUE,
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        x = quickcheck::double_bounded(-100, 100, len = c(12L, 40L)),
        spike_position = quickcheck::double_bounded(0, 1, len = 1L),
        spike_magnitude = quickcheck::double_bounded(1e-4, 100, len = 1L),
        spike_negative = quickcheck::logical_(len = 1L),
        fraction = quickcheck::double_bounded(0.4, 1.0, len = 1L),
        property = property,
        tests = 200L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("accounts explicitly for reference comparisons and discards", {
    expect_gt(loess_reference_counts$compared, 0L)
    message(
        "Reference comparisons: ",
        loess_reference_counts$compared,
        "; discarded: ",
        loess_reference_counts$discarded,
        "; finite singular fits compared: ",
        loess_reference_counts$singular,
        "; captured failures: ",
        length(loess_reference_counts$failures)
    )
})
