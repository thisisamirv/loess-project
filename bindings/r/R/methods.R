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
#' result <- model$fit(x, y)
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
#' res <- model$fit(x, y)
#' plot(res)
#' @return NULL, invisibly. Called for side effects (plotting).
#' @importFrom graphics lines
#' @export
plot.LoessResult <- function(x, main = "LOESS Fit", ...) {
    # Plot the smoothed curve
    plot(
        x$x, x$y,
        type = "l", col = "blue", lwd = 2,
        xlab = "x", ylab = "Fitted",
        main = main, ...
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
