#' @srrstats {G5.4, G5.4b} Reference comparisons against `stats::loess`, generalized to randomized inputs.
#' @srrstats {G5.10} Property-based tests run in the standard suite. `quickcheck` is a Suggests test dependency, not a runtime dependency.
#' @noRd

loess_property_x <- function(order_values) {
    sorted_values <- sort(as.double(order_values))
    value_range <- diff(range(sorted_values))
    gaps <- 1 + diff(sorted_values) / max(1, value_range)
    sorted_x <- c(0, cumsum(gaps))
    sorted_x <- 10 * sorted_x / max(sorted_x) - 5

    ranks <- integer(length(order_values))
    ranks[order(order_values)] <- seq_along(order_values)
    as.double(sorted_x[ranks])
}

#' Check whether `Loess()` matches a direct `stats::loess()` fit.
#'
#' The comparisons use direct surfaces and no boundary padding on both
#' implementations. Gaussian fits are used at zero iterations; positive
#' iteration counts use symmetric robust fitting and MAR residual scaling.
#' `stats::loess.control()` requires a positive `iterations` value even when
#' `family = "gaussian"`, where robust reweighting is disabled.
#' @noRd
check_stats_loess <- function(
    x,
    y,
    fraction,
    degree = 2L,
    iterations = 0L,
    sorted = FALSE,
    tolerance = 1e-10
) {
    x <- as.double(x)
    y <- as.double(y)
    ord <- order(x)
    degree_name <- if (degree == 1L) "linear" else "quadratic"
    family <- if (iterations == 0L) "gaussian" else "symmetric"

    reference <- stats::loess(
        y ~ x,
        data = data.frame(x = x, y = y),
        span = fraction,
        degree = degree,
        family = family,
        control = stats::loess.control(
            surface = "direct",
            iterations = max(1L, as.integer(iterations))
        )
    )
    result <- fit(
        Loess(
            fraction = fraction,
            degree = degree_name,
            iterations = as.integer(iterations),
            boundary_policy = "noboundary",
            scaling_method = "mar",
            surface_mode = "direct",
            outputs = if (sorted) "sorted" else NULL,
            parallel = FALSE
        ),
        x,
        y
    )

    expected_x <- if (sorted) x[ord] else x
    expected_y <- if (sorted) reference$fitted[ord] else reference$fitted
    scale <- max(1, abs(result$y), abs(expected_y))
    isTRUE(all.equal(result$x, expected_x, tolerance = 0)) &&
        max(abs(result$y - expected_y)) <= tolerance * scale &&
        identical(result$fraction_used, fraction)
}

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
