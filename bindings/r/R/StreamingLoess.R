#' LOESS Streaming Smoothing
#'
#' @description
#' Create a stateful LOESS model for streaming data.
#'
#' @srrstats {G2.0} Input validation for fraction, chunk_size.
#' @srrstats {G1.6} Memory-efficient streaming for large datasets.
#'
#' @inheritParams Loess
#' @param ... Not used; forces all subsequent arguments to be named.
#' @param chunk_size Number of data points per processing chunk.
#' @param overlap Number of overlapping points between consecutive chunks.
#' @param merge_strategy Strategy for reconciling overlapping chunk regions:
#'   \code{"weighted_average"} (default; alias: \code{"weighted"}),
#'   \code{"average"} (alias: \code{"mean"}),
#'   \code{"take_first"} (alias: \code{"first"}), or
#'   \code{"take_last"} (alias: \code{"last"}).
#'
#' @return A StreamingLoess object.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- StreamingLoess(fraction = 0.2, chunk_size = 50)
#' res1 <- process_chunk(model, x[1:50], y[1:50])
#' res2 <- process_chunk(model, x[51:100], y[51:100])
#' final <- finalize(model)
#' @export
StreamingLoess <- function(
    fraction = 0.67,
    chunk_size = 5000L,
    ...,
    overlap = NULL,
    iterations = 3L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    auto_converge = NULL,
    return_diagnostics = FALSE,
    return_residuals = FALSE,
    return_robustness_weights = FALSE,
    zero_weight_fallback = "use_local_mean",
    merge_strategy = "weighted_average",
    parallel = TRUE,
    degree = "linear",
    dimensions = 1L,
    distance_metric = "normalized",
    surface_mode = "interpolation",
    return_se = FALSE,
    confidence_intervals = NULL,
    prediction_intervals = NULL,
    weighted_metric_weights = NULL,
    cell = NULL,
    interpolation_vertices = NULL,
    boundary_degree_fallback = NULL
) {
    if (...length() > 0) stop("All arguments after 'chunk_size' must be named.", call. = FALSE)
    validate_params(fraction = fraction, chunk_size = chunk_size)
    handle <- do.call(RStreamingLoess$new, env_args(streaming_params))
    .make_streaming_loess(handle, fraction, chunk_size, iterations, parallel)
}

.make_streaming_loess <- function(
    handle, fraction, chunk_size, iterations, parallel
) {
    structure(
        list(
            handle = handle,
            params = list(
                fraction = fraction,
                chunk_size = chunk_size,
                iterations = iterations,
                parallel = parallel
            )
        ),
        class = "StreamingLoess"
    )
}
