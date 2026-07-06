#' LOESS Online Smoothing
#'
#' @description
#' Create a stateful LOESS model for real-time online data.
#'
#' @srrstats {G2.0} Input validation for fraction, window_capacity, min_points.
#' @srrstats {G1.6} Sliding window for incremental updates.
#'
#' @inheritParams Loess
#' @param window_capacity Maximum number of points kept in the sliding window.
#' @param min_points Minimum number of points required before smoothing begins.
#' @param update_mode Window update strategy: \code{"full"} (default) re-smooths
#'   all window points after each addition, \code{"incremental"} updates only the
#'   newest point.
#' @param confidence_intervals Confidence level for confidence intervals
#'   (e.g., 0.95). \code{NULL} (default) disables confidence intervals.
#' @param prediction_intervals Confidence level for prediction intervals
#'   (e.g., 0.95). \code{NULL} (default) disables prediction intervals.
#'
#' @return An OnlineLoess object.
#' @examples
#' model <- OnlineLoess(fraction = 0.2, window_capacity = 20)
#' x <- 1:50
#' y <- sin(x * 0.1) + rnorm(50, 0, 0.1)
#' result <- model$add_points(x, y)
#' plot(x, y)
#' lines(x, result$y, col = "red")
#' @export
OnlineLoess <- function(
    fraction = 0.2,
    window_capacity = 100L,
    min_points = 2L,
    iterations = 3L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    update_mode = "full",
    auto_converge = NULL,
    return_robustness_weights = FALSE,
    return_diagnostics = FALSE,
    return_residuals = FALSE,
    zero_weight_fallback = "use_local_mean",
    parallel = FALSE,
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
    validate_params(
        fraction = fraction, window_capacity = window_capacity,
        min_points = min_points
    )
    handle <- do.call(ROnlineLoess$new, env_args(online_params))

    structure(
        list(
            handle = handle,
            add_points = function(x, y) {
                args <- validate_common_args(x, y, fraction, iterations)
                handle$add_points(args$x, args$y)
            },
            params = list(
                fraction = fraction,
                window_capacity = window_capacity,
                min_points = min_points,
                iterations = iterations,
                parallel = parallel
            )
        ),
        class = "OnlineLoess"
    )
}
