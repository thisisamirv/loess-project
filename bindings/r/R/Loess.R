#' LOESS Batch Smoothing
#'
#' @description
#' Create a stateful LOESS model for batch smoothing. This is the default
#' mode: it processes the entire dataset at once and supports every feature
#' (confidence/prediction intervals, cross-validation, GPU backend).
#'
#' @details
#' Best suited when the dataset fits in memory and you need intervals,
#' cross-validation, or diagnostics. For datasets that don't fit in memory or
#' arrive in chunks, see \code{\link{StreamingLoess}}; for point-by-point
#' real-time data, see \code{\link{OnlineLoess}}.
#'
#' `fraction` is the most important parameter: it controls the size of the
#' local neighbourhood used at each point.
#'
#' | Range | Effect | Use case |
#' | --- | --- | --- |
#' | 0.1-0.3 | Fine detail | Rapidly changing signals |
#' | 0.3-0.5 | Balanced | General purpose |
#' | 0.5-0.7 | Heavy smoothing | Noisy data |
#' | 0.7-1.0 | Very smooth | Trend extraction |
#'
#' @srrstats {G2.0} Input validation for fraction and iterations.
#' @srrstats {G2.1} Parameter bounds checking (fraction 0-1, iterations >= 0).
#' @srrstats {RE2.0} Kernel, robustness, boundary, and scaling configurable.
#' @srrstats {RE2.1, RE2.2} NA handling options available via Rust backend.
#' @srrstats {RE3.0, RE3.1} Convergence warnings; thresholds settable.
#' @srrstats {RE4.0, RE4.1} Model object returned; fitting via S3 generic fit().
#' @srrstats {RE4.7} Convergence stats returned in result.
#' @srrstats {RE4.8, RE4.9, RE4.10} Response, fitted, residuals returned.
#' @srrstats {RE4.11} Goodness-of-fit metrics via return_diagnostics.
#' @srrstats {RE5.0} O(n) scaling documented in README.
#'
#' @param ... Not used; forces all subsequent arguments to be named.
#' @param fraction Smoothing fraction, greater than 0 and up to 1. Default:
#'   0.67. See Details for guidance on choosing a value.
#' @param iterations Number of robustness iterations, between 0 and 1000
#'   (inclusive). Default: 3.
#' @param weight_function Kernel weight function. One of \code{"tricube"}
#'   (default), \code{"gaussian"},
#'   \code{"uniform"} (alias: \code{"boxcar"}),
#'   \code{"cosine"}, \code{"epanechnikov"},
#'   \code{"biweight"} (alias: \code{"bisquare"}), or
#'   \code{"triangle"} (alias: \code{"triangular"}).
#' @param robustness_method Outlier downweighting method: \code{"bisquare"}
#'   (default; alias: \code{"biweight"}), \code{"huber"}, or \code{"talwar"}.
#' @param scaling_method Residual scale estimation for robustness weights:
#'   \code{"mad"} (default; alias: \code{"median_absolute_deviation"}),
#'   \code{"mar"} (alias: \code{"median_absolute_residual"}), or
#'   \code{"mean"} (alias: \code{"mean_absolute_residual"}).
#' @param boundary_policy Boundary handling strategy: \code{"extend"}
#'   (default; alias: \code{"pad"}), \code{"reflect"} (alias:
#'   \code{"mirror"}), \code{"zero"}, or
#'   \code{"noboundary"} (alias: \code{"none"}).
#' @param auto_converge Convergence tolerance for early stopping of robustness
#'   iterations. \code{NULL} (default) disables early stopping.
#' @param return_diagnostics Logical; if \code{TRUE}, return fit-quality
#'   metrics (RMSE, MAE, R-squared, AIC, etc.). Default: \code{FALSE}.
#' @param return_residuals Logical; if \code{TRUE}, return residuals in the
#'   result. Default: \code{FALSE}.
#' @param return_robustness_weights Logical; if \code{TRUE}, return per-point
#'   robustness weights. Default: \code{FALSE}.
#' @param zero_weight_fallback Fallback policy when all robustness weights drop
#'   to zero: \code{"use_local_mean"} (default; aliases: \code{"local_mean"},
#'   \code{"mean"}), \code{"return_original"} (alias: \code{"original"}), or
#'   \code{"return_none"} (alias: \code{"none"}).
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
#' @param confidence_intervals Confidence level for confidence intervals,
#'   greater than 0 and less than 1 (e.g., 0.95). \code{NULL} (default)
#'   disables confidence intervals.
#' @param prediction_intervals Confidence level for prediction intervals,
#'   greater than 0 and less than 1 (e.g., 0.95). \code{NULL} (default)
#'   disables prediction intervals.
#' @param cv_fractions Numeric vector of candidate fractions for
#'   cross-validation. \code{NULL} (default) disables CV.
#' @param cv_method Cross-validation method: \code{"kfold"} (default) or
#'   \code{"loocv"}.
#' @param cv_k Number of folds for k-fold CV, at least 2. Default: 5.
#' @param weighted_metric_weights Numeric vector of per-dimension weights.
#'   Length must equal \code{dimensions}. Only used when
#'   \code{distance_metric = "weighted"}; setting
#'   \code{distance_metric = "weighted"} without providing this raises an
#'   error. \code{NULL} (default) has no effect unless
#'   \code{distance_metric = "weighted"} is set.
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
#' result <- fit(model, x, y)
#' plot(x, y)
#' lines(x, result$y, col = "red")
#' @export
Loess <- function(
    fraction = 0.67,
    ...,
    iterations = 3L,
    weight_function = "tricube",
    robustness_method = "bisquare",
    scaling_method = "mad",
    boundary_policy = "extend",
    confidence_intervals = NULL,
    prediction_intervals = NULL,
    return_diagnostics = FALSE,
    return_residuals = FALSE,
    return_robustness_weights = FALSE,
    zero_weight_fallback = "use_local_mean",
    auto_converge = NULL,
    cv_fractions = NULL,
    cv_method = "kfold",
    cv_k = 5L,
    parallel = TRUE,
    degree = "linear",
    dimensions = 1L,
    distance_metric = "normalized",
    surface_mode = "interpolation",
    return_se = FALSE,
    weighted_metric_weights = NULL,
    cell = NULL,
    interpolation_vertices = NULL,
    boundary_degree_fallback = NULL,
    cv_seed = NULL
) {
    reject_extra_positional_args(sys.call(), "fraction")
    validate_params(fraction = fraction, iterations = iterations)
    handle <- do.call(RLoess$new, env_args(loess_params))

    structure(
        list(
            handle = handle,
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
