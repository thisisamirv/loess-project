#' @srrstats {G5.4, G5.4b} Reference comparisons against `stats::loess`,
#'   generalized to randomized inputs.
#' @srrstats {G5.10} Property-based tests run through `make validate`.
#'   `quickcheck` is a validation-only dependency.
#' @noRd

loess_reference_counts <- new.env(parent = emptyenv())
loess_reference_counts$compared <- 0L
loess_reference_counts$discarded <- 0L
loess_reference_counts$singular <- 0L
loess_reference_counts$failures <- list()

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
#' @param custom_weights Optional case weights, passed to both implementations.
#' @param cell Interpolation cell size, passed to both implementations.
#' @param parallel Whether to exercise the parallel Rust implementation.
#' @param new_x Optional out-of-sample query points, with one predictor per column.
#' @param se Whether to compare prediction standard errors.
#' @param compare_singular Whether to compare finite pseudoinverse reference fits.
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
    boundary_degree_fallback = NULL,
    custom_weights = NULL,
    cell = 0.2,
    parallel = FALSE,
    new_x = NULL,
    se = FALSE,
    compare_singular = FALSE
) {
    predictors <- if (is.matrix(x)) x else matrix(as.double(x), ncol = 1L)
    storage.mode(predictors) <- "double"
    y <- as.double(y)
    dimensions <- ncol(predictors)
    reference_data <- as.data.frame(predictors)
    names(reference_data) <- paste0("predictor", seq_len(dimensions))
    reference_formula <- reformulate(
        names(reference_data),
        response = "response"
    )
    reference_data$response <- y
    ord <- order(predictors[, 1L])
    degree_name <- if (degree == 1L) "linear" else "quadratic"
    family <- if (iterations == 0L) "gaussian" else "symmetric"
    finish <- function(passed, component) {
        if (!isTRUE(passed)) {
            loess_reference_counts$failures[[
                length(loess_reference_counts$failures) + 1L
            ]] <- list(
                component = component,
                arguments = list(
                    x = predictors,
                    y = y,
                    fraction = fraction,
                    degree = degree,
                    iterations = iterations,
                    sorted = sorted,
                    tolerance = tolerance,
                    surface = surface,
                    boundary_degree_fallback = boundary_degree_fallback,
                    custom_weights = custom_weights,
                    cell = cell,
                    parallel = parallel,
                    new_x = new_x,
                    se = se,
                    compare_singular = compare_singular
                )
            )
        }
        isTRUE(passed)
    }

    warning_log <- new.env(parent = emptyenv())
    warning_log$messages <- character(0)
    reference <- withCallingHandlers(
        stats::loess(
            reference_formula,
            data = reference_data,
            weights = custom_weights,
            span = fraction,
            degree = degree,
            family = family,
            control = stats::loess.control(
                surface = surface,
                cell = cell,
                statistics = if (se) "exact" else "approximate",
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

    singular <- any(grepl(
        "pseudoinverse|condition number|singular",
        warning_log$messages
    ))
    undefined <- any(grepl(
        "zero-width neighborhood|all data on boundary",
        warning_log$messages
    )) ||
        any(!is.finite(reference$fitted))
    if (undefined || (singular && !compare_singular)) {
        loess_reference_counts$discarded <- loess_reference_counts$discarded +
            1L
        hedgehog::discard()
    }
    loess_reference_counts$compared <- loess_reference_counts$compared + 1L
    if (singular) {
        loess_reference_counts$singular <- loess_reference_counts$singular + 1L
    }

    model <- Loess(
        fraction = fraction,
        degree = degree_name,
        dimensions = dimensions,
        iterations = as.integer(iterations),
        boundary_policy = "noboundary",
        scaling_method = "mar",
        surface_mode = surface,
        cell = cell,
        boundary_degree_fallback = boundary_degree_fallback,
        outputs = if (sorted) "sorted" else NULL,
        parallel = parallel,
        retain_model = !is.null(new_x) || se
    )
    result <- fit(
        model,
        predictors,
        y,
        custom_weights = custom_weights
    )

    expected_predictors <- if (sorted) {
        predictors[ord, , drop = FALSE]
    } else {
        predictors
    }
    expected_x <- as.double(t(expected_predictors))
    expected_y <- if (sorted) reference$fitted[ord] else reference$fitted
    scale <- max(1, abs(result$y), abs(expected_y))
    matches <- isTRUE(all.equal(result$x, expected_x, tolerance = 0)) &&
        length(result$y) == length(expected_y) &&
        all(is.finite(result$y)) &&
        max(abs(result$y - expected_y)) <= tolerance * scale &&
        identical(result$fraction_used, fraction)
    if (!matches || (is.null(new_x) && !se)) {
        return(finish(matches, "fitted values"))
    }

    queries <- if (is.null(new_x)) {
        predictors
    } else if (is.matrix(new_x)) {
        new_x
    } else {
        matrix(as.double(new_x), ncol = dimensions)
    }
    query_data <- as.data.frame(queries)
    names(query_data) <- names(reference_data)[seq_len(dimensions)]
    reference_prediction <- suppressWarnings(predict(
        reference,
        query_data,
        se = se
    ))
    expected_prediction <- if (se) {
        reference_prediction$fit
    } else {
        reference_prediction
    }
    if (
        any(!is.finite(expected_prediction)) ||
            (se && any(!is.finite(reference_prediction$se.fit)))
    ) {
        loess_reference_counts$discarded <- loess_reference_counts$discarded +
            1L
        hedgehog::discard()
    }
    prediction <- predict(
        model,
        as.double(t(queries)),
        outputs = if (se) "se" else NULL
    )
    prediction_scale <- max(1, abs(prediction$y), abs(expected_prediction))
    prediction_matches <- length(prediction$y) == length(expected_prediction) &&
        all(is.finite(prediction$y)) &&
        max(abs(prediction$y - expected_prediction)) <=
            tolerance * prediction_scale
    if (!prediction_matches || !se) {
        return(finish(prediction_matches, "predictions"))
    }
    error_scale <- max(
        1,
        abs(prediction$standard_errors),
        abs(reference_prediction$se.fit)
    )
    finish(
        length(prediction$standard_errors) ==
            length(reference_prediction$se.fit) &&
            all(is.finite(prediction$standard_errors)) &&
            max(abs(
                prediction$standard_errors - reference_prediction$se.fit
            )) <=
                tolerance * error_scale,
        "prediction standard errors"
    )
}
