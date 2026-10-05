#' Print Loess Model
#'
#' @srrstats {G1.3} S3 print methods for model objects.
#' @srrstats {RE4.17, RE4.18} Print and summary S3 methods implemented.
#' @srrstats {RE1.4} LOESS assumptions documented in vignette and README.
#'
#' @param x A Loess object.
#' @param ... Additional arguments (ignored).
#' @return The input object `x`, invisibly.
#' @examples
#' model <- Loess(fraction = 0.3)
#' print(model)
#' @export
print.Loess <- function(x, ...) {
    cat("<Loess Model>\n")
    cat("  Fraction:         ", x$params$fraction, "\n")
    cat("  Iterations:       ", x$params$iterations, "\n")
    cat("  Weight Function:  ", x$params$weight_function, "\n")
    cat("  Parallel:         ", x$params$parallel, "\n")
    invisible(x)
}

#' Print Loess Result
#'
#' @param x A LoessResult object.
#' @param ... Additional arguments (ignored).
#' @return The input object `x`, invisibly.
#' @examples
#' x <- seq(0, 10, length.out = 50)
#' y <- sin(x) + rnorm(50, 0, 0.1)
#' model <- Loess(fraction = 0.3)
#' result <- fit(model, x, y)
#' print(result)
#' @export
print.LoessResult <- function(x, ...) {
    cat("<LoessResult>\n")
    cat("  Points:           ", length(x$x), "\n")
    cat("  Fraction Used:    ", x$fraction_used, "\n")
    if (!is.null(x$iterations_used)) {
        cat("  Iterations Used:  ", x$iterations_used, "\n")
    }
    if (!is.null(x$cv_scores)) {
        cat("  CV Scores:        ", length(x$cv_scores), "folds\n")
    }
    invisible(x)
}

#' Plot Loess Result
#'
#' @param x A LoessResult object.
#' @param main Plot title.
#' @param ... Additional arguments passed to plot() and lines().
#' @srrstats {RE6.0} Default S3 plot method implemented.
#' @srrstats {RE6.2} Plot shows fitted values with confidence intervals.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Loess(fraction = 0.2)
#' res <- fit(model, x, y)
#' plot(res)
#' @return NULL, invisibly. Called for side effects (plotting).
#' @importFrom graphics lines
#' @export
plot.LoessResult <- function(x, main = "LOESS Fit", ...) {
    # Plot the smoothed curve
    plot(
        x$x,
        x$y,
        type = "l",
        col = "blue",
        lwd = 2,
        xlab = "x",
        ylab = "Fitted",
        main = main,
        ...
    )

    # If confidence intervals exist, plot them
    if (!is.null(x$confidence_lower)) {
        lines(x$x, x$confidence_lower, lty = 2, col = "gray")
        lines(x$x, x$confidence_upper, lty = 2, col = "gray")
    }
}

#' Print StreamingLoess Model
#'
#' @param x A StreamingLoess object.
#' @param ... Additional arguments.
#' @return The input object `x`, invisibly.
#' @examples
#' model <- StreamingLoess(fraction = 0.3, chunk_size = 50L)
#' print(model)
#' @export
print.StreamingLoess <- function(x, ...) {
    cat("<StreamingLoess Model>\n")
    cat("  Fraction:         ", x$params$fraction, "\n")
    cat("  Chunk Size:       ", x$params$chunk_size, "\n")
    cat("  Parallel:         ", x$params$parallel, "\n")
    invisible(x)
}

#' Print OnlineLoess Model
#'
#' @param x An OnlineLoess object.
#' @param ... Additional arguments.
#' @return The input object `x`, invisibly.
#' @examples
#' model <- OnlineLoess(fraction = 0.2, window_capacity = 20L)
#' print(model)
#' @export
print.OnlineLoess <- function(x, ...) {
    cat("<OnlineLoess Model>\n")
    cat("  Fraction:         ", x$params$fraction, "\n")
    cat("  Window Capacity:  ", x$params$window_capacity, "\n")
    cat("  Min Points:       ", x$params$min_points, "\n")
    cat("  Update Mode:      ", x$params$update_mode, "\n")
    invisible(x)
}

#' Fit a LOESS model to data
#'
#' @param model A \code{Loess} object.
#' @param x Numeric vector of predictor values.
#' @param y Numeric vector of response values.
#' @param custom_weights Optional numeric vector of non-negative per-observation
#'   weights. \code{NULL} (default) applies no custom weighting.
#' @param ... Must be empty.
#' @return A \code{LoessResult} object.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Loess(fraction = 0.2)
#' fit(model, x, y)
#' @export
fit <- function(model, ...) UseMethod("fit")

#' @rdname fit
#' @export
fit.Loess <- function(model, x, y, custom_weights = NULL, ...) {
    if (...length() > 0L) {
        stop("unused arguments (...)")
    }
    if (is.matrix(x) && ncol(x) != model$params$dimensions) {
        fmt <- paste0(
            "x is a %d\u00d7%d matrix but model has dimensions=%d;",
            " use Loess(dimensions=%dL)"
        )
        stop(sprintf(fmt, nrow(x), ncol(x), model$params$dimensions, ncol(x)))
    }
    validated_args <- validate_common_args(
        x,
        y,
        model$params$fraction,
        model$params$iterations
    )
    if (!is.null(custom_weights)) {
        validate_numeric_vector(custom_weights, "custom_weights")
        if (length(custom_weights) != length(y)) {
            stop("custom_weights must have the same length as y", call. = FALSE)
        }
        if (any(!is.finite(custom_weights)) || any(custom_weights < 0)) {
            stop(
                "custom_weights must be finite and non-negative",
                call. = FALSE
            )
        }
        custom_weights <- as.double(custom_weights)
    }
    model$handle$fit(validated_args$x, validated_args$y, custom_weights)
}

#' Predict from a fitted LOESS model at out-of-sample points
#'
#' @param object A \code{Loess} object, fitted (via \code{\link{fit}}) with
#'   \code{retain_model = TRUE} passed to \code{\link{Loess}}.
#' @param new_x Numeric vector of out-of-sample query points (flattened,
#'   \code{dimensions} values per point).
#' @param intervals Grouped coverage levels from \code{\link{intervals_opts}}.
#' @param outputs Optional character vector selecting \code{"se"},
#'   \code{"gradient"}, or \code{"derivative"}. \code{NULL} (default)
#'   selects no optional components.
#' @param extrapolation Behavior for query points outside the training range:
#'   \code{"clamp"} (default), \code{"linear"}, or \code{"error"}.
#' @param max_extrapolation_distance Under \code{"linear"} extrapolation, the
#'   maximum allowed distance beyond the training boundary before
#'   \code{predict} errors instead of returning an unbounded value.
#'   \code{NULL} (default) disables the cap.
#' @param max_neighbor_distance Maximum allowed distance to the farthest point
#'   in a query's neighbor window before \code{predict} errors, catching
#'   in-range-but-sparse query points. \code{NULL} (default) disables the cap.
#' @param ... Must be empty.
#' @return A list with a \code{y} element (predicted values) and optional
#'   \code{standard_errors}/\code{confidence_lower}/\code{confidence_upper}/
#'   \code{prediction_lower}/\code{prediction_upper}/\code{derivative} elements.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Loess(fraction = 0.2, retain_model = TRUE)
#' fit(model, x, y)
#' predict(model, c(2.5, 7.5))
#' @export
predict.Loess <- function(
    object,
    new_x,
    intervals = NULL,
    extrapolation = "clamp",
    max_extrapolation_distance = NULL,
    max_neighbor_distance = NULL,
    outputs = NULL,
    ...
) {
    if (...length() > 0L) {
        stop("unused arguments (...)")
    }
    validate_numeric_vector(new_x, "new_x")
    if (
        length(new_x) == 0L ||
            length(new_x) %% object$params$dimensions != 0L
    ) {
        stop(
            "new_x must be non-empty and its length must be a multiple of dimensions",
            call. = FALSE
        )
    }
    flags <- parse_outputs_flags(outputs, c("se", "gradient", "derivative"))
    interval_options <- parse_intervals_options(intervals)
    return_se <- flags[["se"]]
    return_derivative <- flags[["gradient"]] || flags[["derivative"]]
    object$handle$predict(
        as.double(new_x),
        as.logical(return_se),
        coerce_nullable(interval_options$confidence)[[1]],
        coerce_nullable(interval_options$prediction)[[1]],
        as.logical(
            return_derivative
        ),
        as.character(extrapolation),
        coerce_nullable(max_extrapolation_distance)[[1]],
        coerce_nullable(max_neighbor_distance)[[1]]
    )
}

#' Process a data chunk through a streaming LOESS model
#'
#' @param model A \code{StreamingLoess} object.
#' @param x Numeric vector of x values.
#' @param y Numeric vector of y values.
#' @param custom_weights Optional numeric case weight per observation.
#' @param ... Must be empty.
#' @return A \code{LoessResult} for this chunk.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- StreamingLoess(fraction = 0.2, chunk_size = 50L)
#' process_chunk(model, x[1:50], y[1:50])
#' @export
process_chunk <- function(model, ...) UseMethod("process_chunk")

#' @rdname process_chunk
#' @export
process_chunk.StreamingLoess <- function(model, x, y, custom_weights = NULL, ...) {
    if (...length() > 0L) {
        stop("unused arguments (...)")
    }
    if (is.matrix(x) && ncol(x) != model$params$dimensions) {
        fmt <- paste0(
            "x is a %d\u00d7%d matrix but model has dimensions=%d;",
            " use StreamingLoess(dimensions=%dL)"
        )
        stop(sprintf(fmt, nrow(x), ncol(x), model$params$dimensions, ncol(x)))
    }
    args <- validate_common_args(
        x,
        y,
        model$params$fraction,
        model$params$iterations
    )
    if (is.null(custom_weights)) {
        model$handle$process_chunk(args$x, args$y)
    } else {
        if (!is.numeric(custom_weights) || is.complex(custom_weights) ||
            length(custom_weights) != length(args$y)) {
            stop("custom_weights must have one numeric value per observation", call. = FALSE)
        }
        model$handle$process_chunk_weighted(args$x, args$y, as.double(custom_weights))
    }
}

#' Finalize a streaming LOESS model
#'
#' @param model A \code{StreamingLoess} object.
#' @param ... Must be empty.
#' @return A \code{LoessResult} combining all processed chunks.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- StreamingLoess(fraction = 0.2, chunk_size = 50L)
#' invisible(process_chunk(model, x[1:50], y[1:50]))
#' finalize(model)
#' @export
finalize <- function(model, ...) UseMethod("finalize")

#' @export
finalize.StreamingLoess <- function(model, ...) {
    if (...length() > 0L) {
        stop("unused arguments (...)")
    }
    model$handle$finalize()
}

#' Add a single point to an online LOESS model
#'
#' @param model An \code{OnlineLoess} object.
#' @param x A numeric coordinate vector with one value per configured
#'   dimension. For one-dimensional models, a scalar is also accepted.
#' @param y A single numeric y value.
#' @param weight Finite non-negative case weight for this observation; defaults to 1.
#' @param ... Must be empty.
#' @return An online result list, or \code{NULL} if fewer than
#'   \code{min_points} have been added.
#' @examples
#' model <- OnlineLoess(fraction = 0.2, window_capacity = 20L, min_points = 2L)
#' invisible(add_point(model, 1.0, 0.5))
#' add_point(model, 2.0, 0.6)
#' @export
add_point <- function(model, ...) UseMethod("add_point")

#' @rdname add_point
#' @export
add_point.OnlineLoess <- function(model, x, y, weight = 1.0, ...) {
    if (...length() > 0L) {
        stop("unused arguments (...)")
    }
    if (
        !is.numeric(x) || is.complex(x) || !length(x) || !is.null(dim(x))
    ) {
        stop("x must be a non-empty numeric coordinate vector", call. = FALSE)
    }
    if (length(x) != model$params$dimensions) {
        stop(
            sprintf("x must have exactly %d values", model$params$dimensions),
            call. = FALSE
        )
    }
    if (
        !is.numeric(y) || is.complex(y) || length(y) != 1L || !is.null(dim(y))
    ) {
        stop("y must be a single numeric value", call. = FALSE)
    }
    if (!is.numeric(weight) || is.complex(weight) || length(weight) != 1L ||
        !is.finite(weight) || weight < 0 || !is.null(dim(weight))) {
        stop("weight must be a single finite non-negative numeric value", call. = FALSE)
    }
    model$handle$add_point_weighted(as.double(x), as.double(y), as.double(weight))
}

#' Compute diagnostics for the current Online window
#'
#' @param model An OnlineLoess object.
#' @param ... Must be empty.
#' @return A list of goodness-of-fit metrics, or `NULL` until the window
#'   reaches `min_points`.
#' @export
window_diagnostics <- function(model, ...) UseMethod("window_diagnostics")

#' @export
window_diagnostics.OnlineLoess <- function(model, ...) {
    if (...length() > 0L) {
        stop("unused arguments (...)", call. = FALSE)
    }
    model$handle$window_diagnostics()
}

#' Predict from the current Online window
#'
#' @param model An OnlineLoess object.
#' @param new_x Numeric query points (flattened, one coordinate per dimension).
#' @param outputs Optional character vector selecting `"se"` and/or
#'   `"gradient"` (alias `"derivative"`).
#' @param intervals Grouped confidence and prediction coverage levels.
#' @param extrapolation Behavior outside the window's predictor bounds:
#'   `"clamp"` (default), `"linear"`, or `"error"`.
#' @param max_extrapolation_distance Optional cap for linear extrapolation.
#' @param max_neighbor_distance Optional cap for sparse-neighborhood predictions.
#' @param ... Must be empty.
#' @return A list containing predicted values and requested optional outputs.
#' @export
predict_window <- function(model, ...) UseMethod("predict_window")

#' @export
predict_window.OnlineLoess <- function(
    model,
    new_x,
    outputs = NULL,
    intervals = NULL,
    extrapolation = "clamp",
    max_extrapolation_distance = NULL,
    max_neighbor_distance = NULL,
    ...
) {
    if (...length() > 0L) {
        stop("unused arguments (...)", call. = FALSE)
    }
    if (!is.numeric(new_x) || is.complex(new_x) || !length(new_x)) {
        stop("new_x must be a non-empty numeric vector", call. = FALSE)
    }
    flags <- parse_outputs_flags(outputs, c("se", "gradient", "derivative"))
    interval_options <- parse_intervals_options(intervals)
    model$handle$predict_window(
        as.double(new_x),
        names(flags)[flags],
        coerce_nullable(interval_options$confidence)[[1]],
        coerce_nullable(interval_options$prediction)[[1]],
        as.character(extrapolation),
        coerce_nullable(max_extrapolation_distance)[[1]],
        coerce_nullable(max_neighbor_distance)[[1]]
    )
}
