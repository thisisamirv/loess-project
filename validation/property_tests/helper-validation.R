#' @srrstats {G5.4, G5.4b} Reference comparisons against `stats::loess`,
#'   generalized to randomized inputs.
#' @srrstats {G5.10} Property-based tests run through `make validate`.
#'   `quickcheck` is a validation-only dependency.
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

#' Check whether `Loess()` matches a `stats::loess()` fit.
#'
#' The comparisons use no boundary padding on both implementations. Gaussian
#' fits are used at zero iterations; positive iteration counts use symmetric
#' robust fitting and MAR residual scaling.
#' `stats::loess.control()` requires a positive `iterations` value even when
#' `family = "gaussian"`, where robust reweighting is disabled.
#'
#' @param surface `"direct"` or `"interpolate"`, passed to both sides.
#' @param boundary_degree_fallback Passed through to `Loess()`. `FALSE` selects
#'   R's behaviour at interpolation vertices outside the data range.
#' @noRd
check_stats_loess <- function(
    x,
    y,
    fraction,
    degree = 2L,
    iterations = 0L,
    sorted = FALSE,
    tolerance = 1e-10,
    surface = "direct",
    boundary_degree_fallback = NULL
) {
    x <- as.double(x)
    y <- as.double(y)
    ord <- order(x)
    degree_name <- if (degree == 1L) "linear" else "quadratic"
    family <- if (iterations == 0L) "gaussian" else "symmetric"

    warning_log <- new.env(parent = emptyenv())
    warning_log$messages <- character(0)
    reference <- withCallingHandlers(
        stats::loess(
            y ~ x,
            data = data.frame(x = x, y = y),
            span = fraction,
            degree = degree,
            family = family,
            control = stats::loess.control(
                surface = surface,
                iterations = max(1L, as.integer(iterations))
            )
        ),
        warning = function(w) {
            warning_log$messages <- c(
                warning_log$messages,
                conditionMessage(w)
            )
            invokeRestart("muffleWarning")
        }
    )

    # Tied x can collapse a local neighbourhood to zero width. `stats::loess()`
    # then warns, falls back to a pseudoinverse and returns 0, so it provides no
    # well-defined value to compare against and the case is skipped.
    degenerate <- paste(
        "zero-width neighborhood",
        "pseudoinverse",
        "condition number",
        "singular",
        sep = "|"
    )
    if (any(grepl(degenerate, warning_log$messages))) {
        return(TRUE)
    }

    result <- fit(
        Loess(
            fraction = fraction,
            degree = degree_name,
            iterations = as.integer(iterations),
            boundary_policy = "noboundary",
            scaling_method = "mar",
            surface_mode = surface,
            boundary_degree_fallback = boundary_degree_fallback,
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
