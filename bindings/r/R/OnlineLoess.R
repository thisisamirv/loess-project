#' LOESS Online Smoothing
#'
#' @description
#' Create a stateful LOESS model for real-time online data. Maintains a
#' sliding window and processes each incoming point immediately via
#' \code{\link{add_point}}.
#'
#' @details
#' Best suited when data arrives incrementally (e.g. sensors or streams),
#' real-time smoothed values are needed, or memory is fixed. For datasets
#' that fit in memory, see \code{\link{Loess}}; for large batches processed
#' in chunks, see \code{\link{StreamingLoess}}.
#'
#' @srrstats {G2.0} Input validation for fraction, window_capacity, min_points.
#' @srrstats {G1.6} Sliding window for incremental updates.
#'
#' @inheritParams Loess
#' @param outputs Optional character vector selecting \code{"weights"},
#'   \code{"gradient"} (or \code{"derivative"}), and \code{"se"}.
#'   Combined with individual flags; \code{"se"} requires full update mode.
#' @param window_capacity Maximum number of points kept in the sliding
#'   window, at least 3. Default: 1000.
#' @param min_points Minimum number of points required before smoothing
#'   begins, between 2 and \code{window_capacity}. Default: 2.
#' @param update_mode Window update strategy: \code{"incremental"} (default;
#'   alias: \code{"single"}) updates only the newest point;
#'   \code{"full"} (alias: \code{"resmooth"}) re-smooths all window points
#'   after each addition.
#' @param missing Policy for a non-finite (NaN/Inf) \code{x} or \code{y} value
#'   passed to \code{\link{add_point}}: \code{"error"} (default) raises an
#'   error, \code{"drop"} silently ignores the point (returns \code{NULL})
#'   instead of adding it to the window.
#' @param confidence_intervals Confidence level for confidence intervals (e.g.
#'   \code{0.95}). Only computed under \code{update_mode = "full"} — raises an
#'   error at construction if set (or \code{return_se}/
#'   \code{prediction_intervals} is set) while \code{update_mode} is left at
#'   its default \code{"incremental"}.
#'   \code{NULL} (default) disables confidence intervals.
#' @param prediction_intervals Confidence level for prediction intervals; same
#'   \code{update_mode = "full"} requirement as \code{confidence_intervals}.
#'   \code{NULL} (default) disables prediction intervals.
#' @param return_se Include the standard error for the latest point in the
#'   result. Same \code{update_mode = "full"} requirement as
#'   \code{confidence_intervals}. Default: \code{FALSE}.
#'
#' @return An OnlineLoess object.
#' @examples
#' model <- OnlineLoess(fraction = 0.2, window_capacity = 20)
#' x <- 1:50
#' y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
#' smoothed <- numeric(0)
#' for (i in seq_along(x)) {
#'     result <- add_point(model, x[i], y[i])
#'     if (!is.null(result)) smoothed <- c(smoothed, result$y)
#' }
#' head(smoothed, 5)
#' @export
OnlineLoess <- function(
    fraction = 0.67,
    window_capacity = 1000L,
    min_points = 2L,
    ...,
    iterations = 0L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    zero_weight_fallback = "use_local_mean",
    update_mode = "incremental",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    return_gradient = FALSE,
    confidence_intervals = NULL,
    prediction_intervals = NULL,
    return_se = FALSE,
    degree = "linear",
    dimensions = 1L,
    distance_metric = "normalized",
    surface_mode = "interpolation",
    weighted_metric_weights = NULL,
    cell = NULL,
    interpolation_vertices = NULL,
    boundary_degree_fallback = NULL,
    missing = "error",
    outputs = NULL
) {
    reject_extra_positional_args(sys.call(), "min_points")
    validate_params(
        fraction = fraction,
        window_capacity = window_capacity,
        min_points = min_points
    )
    flags <- parse_outputs_flags(
        outputs, c("weights", "gradient", "derivative", "se")
    )
    return_robustness_weights <- return_robustness_weights || flags[["weights"]]
    return_gradient <- return_gradient ||
        flags[["gradient"]] || flags[["derivative"]]
    return_se <- return_se || flags[["se"]]
    handle <- do.call(ROnlineLoess$new, env_args(online_params))

    structure(
        list(
            handle = handle,
            params = list(
                fraction = fraction,
                window_capacity = window_capacity,
                min_points = min_points,
                iterations = iterations
            )
        ),
        class = "OnlineLoess"
    )
}
