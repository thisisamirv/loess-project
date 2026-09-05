#' LOESS Streaming Smoothing
#'
#' @description
#' Create a stateful LOESS model for streaming data. Processes data in
#' fixed-size chunks with configurable overlap: results for each chunk are
#' returned by \code{\link{process_chunk}}, and \code{\link{finalize}}
#' flushes any remaining buffered points after the last chunk.
#'
#' @details
#' Best suited for datasets over 100,000 points, memory-constrained
#' environments, or batch processing pipelines. For smaller datasets that fit
#' in memory, see \code{\link{Loess}}; for point-by-point real-time data,
#' see \code{\link{OnlineLoess}}.
#'
#' Overlapping regions between chunks are reconciled via \code{merge_strategy}:
#'
#' | Strategy | Behavior |
#' | --- | --- |
#' | \code{"average"} | Arithmetic mean of both estimates |
#' | \code{"weighted_average"} | Distance-weighted blend (default) |
#' | \code{"take_first"} | Keep left-chunk estimate |
#' | \code{"take_last"} | Keep right-chunk estimate |
#'
#' @srrstats {G2.0} Input validation for fraction, chunk_size, overlap.
#' @srrstats {RE2.0} Kernel, robustness, boundary, and scaling configurable.
#'
#' @inheritParams Loess
#' @param chunk_size Number of data points per processing chunk, at least 10.
#'   Default: 5000.
#' @param overlap Number of overlapping points between consecutive chunks,
#'   less than \code{chunk_size}. \code{NULL} (default) uses
#'   \code{chunk_size / 10} (clamped to \code{[1, chunk_size - 10]}).
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
#' finalize(model)
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
    zero_weight_fallback = "use_local_mean",
    auto_converge = NULL,
    return_diagnostics = FALSE,
    return_residuals = FALSE,
    return_robustness_weights = FALSE,
    merge_strategy = "weighted_average",
    parallel = TRUE,
    degree = "linear",
    dimensions = 1L,
    distance_metric = "normalized",
    surface_mode = "interpolation",
    weighted_metric_weights = NULL,
    cell = NULL,
    interpolation_vertices = NULL,
    boundary_degree_fallback = NULL,
    missing = "error"
) {
    reject_extra_positional_args(sys.call(), "chunk_size")
    validate_params(fraction = fraction, chunk_size = chunk_size)
    handle <- do.call(RStreamingLoess$new, env_args(streaming_params))

    structure(
        list(
            handle = handle,
            params = list(
                fraction = fraction,
                chunk_size = chunk_size,
                iterations = iterations,
                parallel = parallel,
                dimensions = dimensions
            )
        ),
        class = "StreamingLoess"
    )
}
