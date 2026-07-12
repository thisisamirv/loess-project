#' LOESS Batch Smoothing
#'
#' @description
#' Create a stateful LOESS model for batch smoothing.
#'
#' @srrstats {G2.0} Input validation for fraction and iterations.
#' @srrstats {G2.1} Parameter bounds checking (fraction 0-1, iterations >= 0).
#' @srrstats {RE2.0} Kernel, robustness, boundary, and scaling configurable.
#' @srrstats {RE2.1, RE2.2} NA handling options available via Rust backend.
#' @srrstats {RE3.0, RE3.1} Convergence warnings; thresholds settable.
#' @srrstats {RE4.0, RE4.1} Model object returned; fitting deferred to $fit().
#' @srrstats {RE4.7} Convergence stats returned in result.
#' @srrstats {RE4.8, RE4.9, RE4.10} Response, fitted, residuals returned.
#' @srrstats {RE4.11} Goodness-of-fit metrics via return_diagnostics.
#' @srrstats {RE5.0} O(n) scaling documented in README.
#'
#' @param fraction Smoothing fraction (between 0 and 1).
#' @param iterations Number of robustness iterations (non-negative integer).
#'   Default: 3.
#' @param weight_function Kernel weight function. One of \code{"tricube"}
#'   (default), \code{"gaussian"}, \code{"uniform"}, \code{"cosine"},
#'   \code{"epanechnikov"}, \code{"biweight"}, \code{"triangle"}.
#' @param robustness_method Outlier downweighting method: \code{"bisquare"}
#'   (default), \code{"huber"}, or \code{"talwar"}.
#' @param scaling_method Residual scale estimation for robustness weights:
#'   \code{"mad"} (default), \code{"mar"}, or \code{"mean"}.
#' @param boundary_policy Boundary handling strategy: \code{"extend"}
#'   (default), \code{"reflect"}, \code{"zero"}, or \code{"noboundary"}.
#' @param auto_converge Convergence tolerance for early stopping of robustness
#'   iterations. \code{NULL} (default) disables early stopping.
#' @param return_diagnostics Logical; if \code{TRUE}, return fit-quality
#'   metrics (RMSE, MAE, R-squared, AIC, etc.). Default: \code{FALSE}.
#' @param return_residuals Logical; if \code{TRUE}, return residuals in the
#'   result. Default: \code{FALSE}.
#' @param return_robustness_weights Logical; if \code{TRUE}, return per-point
#'   robustness weights. Default: \code{FALSE}.
#' @param zero_weight_fallback Fallback policy when all robustness weights drop
#'   to zero: \code{"use_local_mean"} (default), \code{"return_original"}, or
#'   \code{"return_none"}.
#' @param parallel Logical; enable parallel processing. Default: \code{TRUE}.
#' @param degree Local polynomial degree: \code{"constant"}, \code{"linear"}
#'   (default), \code{"quadratic"}, \code{"cubic"}, or \code{"quartic"}.
#' @param dimensions Number of predictor dimensions. Default: 1.
#' @param distance_metric Distance metric for neighbourhood computation:
#'   \code{"normalized"} (default), \code{"euclidean"}, \code{"manhattan"},
#'   \code{"chebyshev"}, \code{"minkowski"}, or \code{"weighted"}.
#'   Use \code{"minkowski:p"} to set a custom \emph{p} value.
#' @param surface_mode Surface evaluation mode: \code{"interpolation"}
#'   (default) or \code{"direct"}.
#' @param return_se Logical; if \code{TRUE}, compute hat-matrix statistics
#'   (effective degrees of freedom, leverage, standard errors).
#'   Default: \code{FALSE}.
#' @param confidence_intervals Confidence level for confidence intervals
#'   (e.g., 0.95). \code{NULL} (default) disables confidence intervals.
#' @param prediction_intervals Confidence level for prediction intervals
#'   (e.g., 0.95). \code{NULL} (default) disables prediction intervals.
#' @param cv_fractions Numeric vector of candidate fractions for
#'   cross-validation. \code{NULL} (default) disables CV.
#' @param cv_method Cross-validation method: \code{"kfold"} (default) or
#'   \code{"loocv"}.
#' @param cv_k Number of folds for k-fold CV. Default: 5.
#' @param weighted_metric_weights Numeric vector of per-dimension weights used
#'   when \code{distance_metric = "weighted"}. Length must equal
#'   \code{dimensions}. \code{NULL} (default) uses equal weights.
#' @param cell Cell size tuning parameter for the interpolation grid.
#'   \code{NULL} (default) uses the library default.
#' @param interpolation_vertices Number of vertices in the interpolation grid.
#'   \code{NULL} (default) uses the library default.
#' @param boundary_degree_fallback Logical; if \code{TRUE}, fall back to lower
#'   polynomial degree at boundaries when fitting at the requested degree
#'   fails. \code{NULL} (default) uses the library default.
#' @param cv_seed Integer seed for the cross-validation random number
#'   generator. \code{NULL} (default) uses a random seed.
#'
#' @return A Loess object.
#' @examples
#' x <- seq(0, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, 0, 0.1)
#' model <- Loess(fraction = 0.2)
#' result <- model$fit(x, y)
#' plot(x, y)
#' lines(x, result$y, col = "red")
#' @export
Loess <- function(
    fraction = 0.67, iterations = 3L,
    weight_function = "tricube", robustness_method = "bisquare",
    scaling_method = "mad", boundary_policy = "extend",
    confidence_intervals = NULL, prediction_intervals = NULL,
    return_diagnostics = FALSE, return_residuals = FALSE,
    return_robustness_weights = FALSE, zero_weight_fallback = "use_local_mean",
    auto_converge = NULL, cv_fractions = NULL,
    cv_method = "kfold", cv_k = 5L,
    parallel = TRUE, degree = "linear",
    dimensions = 1L, distance_metric = "normalized",
    surface_mode = "interpolation", return_se = FALSE,
    weighted_metric_weights = NULL, cell = NULL,
    interpolation_vertices = NULL, boundary_degree_fallback = NULL,
    cv_seed = NULL
) {
    validate_params(fraction = fraction, iterations = iterations)
    handle <- do.call(RLoess$new, env_args(loess_params))

    # Return a wrapper that coerces inputs for methods
    structure(
        list(
            handle = handle,
            fit = function(x, y, custom_weights = NULL) {
                validated_args <- validate_common_args(
                    x, y, fraction, iterations
                )
                handle$fit(validated_args$x, validated_args$y, custom_weights)
            },
            params = list(
                fraction = fraction,
                iterations = iterations,
                weight_function = weight_function,
                robustness_method = robustness_method,
                scaling_method = scaling_method,
                parallel = parallel,
                degree = degree,
                dimensions = dimensions,
                distance_metric = distance_metric,
                surface_mode = surface_mode
            )
        ),
        class = "Loess"
    )
}

env_args <- function(param_names) {
    env <- parent.frame()
    result <- lapply(param_names, function(name) {
        val <- get(name, envir = env)
        type <- param_types[[name]]
        if (is.null(type)) {
            return(val)
        }
        switch(type,
            double = as.double(val),
            integer = as.integer(val),
            character = as.character(val),
            logical = as.logical(val),
            nullable = coerce_nullable(val)[[1]],
            val
        )
    })
    setNames(result, param_names)
}

#' Parameter names for each Loess constructor
#' @noRd
loess_params <- c(
    "fraction", "iterations", "weight_function", "robustness_method",
    "scaling_method", "boundary_policy", "confidence_intervals",
    "prediction_intervals", "return_diagnostics", "return_residuals",
    "return_robustness_weights", "zero_weight_fallback", "auto_converge",
    "cv_fractions", "cv_method", "cv_k", "parallel",
    "degree", "dimensions", "distance_metric", "surface_mode", "return_se",
    "weighted_metric_weights", "cell", "interpolation_vertices",
    "boundary_degree_fallback", "cv_seed"
)

online_params <- c(
    "fraction", "window_capacity", "min_points", "iterations",
    "weight_function", "robustness_method", "scaling_method",
    "boundary_policy", "zero_weight_fallback", "update_mode", "auto_converge",
    "return_robustness_weights", "return_diagnostics",
    "return_residuals", "parallel",
    "degree", "dimensions", "distance_metric", "surface_mode", "return_se",
    "confidence_intervals", "prediction_intervals",
    "weighted_metric_weights", "cell", "interpolation_vertices",
    "boundary_degree_fallback"
)

streaming_params <- c(
    "fraction", "chunk_size", "overlap", "iterations",
    "weight_function", "robustness_method", "scaling_method",
    "boundary_policy", "zero_weight_fallback", "auto_converge",
    "return_diagnostics", "return_residuals", "return_robustness_weights",
    "merge_strategy", "parallel",
    "degree", "dimensions", "distance_metric", "surface_mode", "return_se",
    "confidence_intervals", "prediction_intervals",
    "weighted_metric_weights", "cell", "interpolation_vertices",
    "boundary_degree_fallback"
)
