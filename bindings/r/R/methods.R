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
#' @srrstats {RE6.1} Plot method returns the input object invisibly.
#' @srrstats {RE6.2} Plot shows fitted values with confidence intervals.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Loess(fraction = 0.2)
#' res <- fit(model, x, y)
#' plot(res)
#' @return The input object \code{x}, invisibly.
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

    invisible(x)
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
#' @param ... Not used.
#' @return A \code{LoessResult} object.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Loess(fraction = 0.2)
#' result <- fit(model, x, y)
#' @export
fit <- function(model, ...) UseMethod("fit")

#' @export
fit.Loess <- function(model, x, y, custom_weights = NULL, ...) {
    validated_args <- validate_common_args(
        x,
        y,
        model$params$fraction,
        model$params$iterations
    )
    model$handle$fit(validated_args$x, validated_args$y, custom_weights)
}

#' Process a data chunk through a streaming LOESS model
#'
#' @param model A \code{StreamingLoess} object.
#' @param x Numeric vector of x values.
#' @param y Numeric vector of y values.
#' @param ... Not used.
#' @return A \code{LoessResult} for this chunk.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- StreamingLoess(fraction = 0.2, chunk_size = 50L)
#' res <- process_chunk(model, x[1:50], y[1:50])
#' @export
process_chunk <- function(model, ...) UseMethod("process_chunk")

#' @export
process_chunk.StreamingLoess <- function(model, x, y, ...) {
    args <- validate_common_args(
        x,
        y,
        model$params$fraction,
        model$params$iterations
    )
    model$handle$process_chunk(args$x, args$y)
}

#' Finalize a streaming LOESS model
#'
#' @param model A \code{StreamingLoess} object.
#' @param ... Not used.
#' @return A \code{LoessResult} combining all processed chunks.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- StreamingLoess(fraction = 0.2, chunk_size = 50L)
#' invisible(process_chunk(model, x[1:50], y[1:50]))
#' final <- finalize(model)
#' @export
finalize <- function(model, ...) UseMethod("finalize")

#' @export
finalize.StreamingLoess <- function(model, ...) {
    model$handle$finalize()
}

#' Add a single point to an online LOESS model
#'
#' @param model An \code{OnlineLoess} object.
#' @param x A single numeric x value.
#' @param y A single numeric y value.
#' @param ... Not used.
#' @return An online result list, or \code{NULL} if fewer than
#'   \code{min_points} have been added.
#' @examples
#' model <- OnlineLoess(fraction = 0.2, window_capacity = 20L)
#' result <- add_point(model, 1.0, 0.5)
#' @export
add_point <- function(model, ...) UseMethod("add_point")

#' @export
add_point.OnlineLoess <- function(model, x, y, ...) {
    model$handle$add_point(as.double(x), as.double(y))
}
