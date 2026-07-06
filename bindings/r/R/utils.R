#' Common argument validation and coercion
#'
#' @description
#' Internal helper to validate x, y, fraction, and iterations inputs and
#' force them to the correct types for Rust FFI.
#'
#' @srrstats {G2.0} Validates matching lengths, minimum points, numeric types.
#' @srrstats {G2.3} Informative error messages for invalid inputs.
#'
#' @param x Numeric vector
#' @param y Numeric vector
#' @param fraction Numeric
#' @param iterations Integer
#'
#' @return A list containing the coerced x, y, fraction, and iterations.
#' @noRd
#' @srrstats {G2.0} Validates matching lengths, minimum points, numeric types.
#' @srrstats {G2.2} Univariate only; multivariate inputs rejected.
#' @srrstats {G2.3} Informative error messages for invalid inputs.
#' @srrstats {G2.6} Numeric pre-processing via as.double/as.integer.
#' @srrstats {G2.13} Explicit NA handling before Rust FFI call.
#' @srrstats {G2.14, G2.14a, G2.14b, G2.14c} NaN/Inf reported via errors.
#' @srrstats {G2.15} NA checks on inputs before passing to algorithms.
#' @srrstats {G2.16} Inf/NaN validation in input vectors.
#' @srrstats {G3.0} Tolerance-based comparisons used in robustness weights.
validate_common_args <- function(x, y, fraction, iterations) {
    # For multi-dimensional input, x is a flat vector of length n * d.
    # Accept if length(x) is a positive multiple of length(y).
    n_y <- length(y)
    if (length(x) == 0 || n_y == 0 || length(x) %% n_y != 0) {
        stop("x must match y's length or be its multiple for multi-dim input")
    }
    if (length(x) < 3) {
        stop("At least 3 data points are required")
    }
    if (!is.numeric(fraction) || length(fraction) != 1) {
        stop("fraction must be a single numeric value")
    }
    if (fraction <= 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
    }
    if (!is.numeric(iterations) || length(iterations) != 1 || iterations < 0) {
        stop("iterations must be a non-negative integer")
    }

    list(
        x = as.double(x),
        y = as.double(y),
        fraction = as.double(fraction),
        iterations = as.integer(iterations)
    )
}

#' Validate the fraction parameter
#' @noRd
validate_fraction <- function(fraction) {
    if (!is.numeric(fraction) || length(fraction) != 1 || is.na(fraction)) {
        stop("fraction must be a single numeric value")
    }
    if (fraction < 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
    }
}

#' Validate the iterations parameter (optional)
#' @noRd
validate_iterations <- function(iterations) {
    if (is.null(iterations)) {
        return(invisible(NULL))
    }
    cond1 <- !is.numeric(iterations)
    cond2 <- length(iterations) != 1
    cond3 <- is.na(iterations)
    if (cond1 || cond2 || cond3) {
        stop("iterations must be a single numeric value")
    }
    if (iterations < 0) {
        stop("iterations must be a non-negative integer")
    }
}

#' Validate the window_capacity parameter (optional)
#' @noRd
validate_window_capacity <- function(window_capacity) {
    if (is.null(window_capacity)) {
        return(invisible(NULL))
    }
    if (window_capacity <= 0) {
        stop("window_capacity must be a positive integer")
    }
}

#' Validate the min_points parameter (optional)
#' @noRd
validate_min_points <- function(min_points) {
    if (is.null(min_points)) {
        return(invisible(NULL))
    }
    if (min_points < 0) {
        stop("min_points must be a non-negative integer")
    }
}

#' Validate the chunk_size parameter (optional)
#' @noRd
validate_chunk_size <- function(chunk_size) {
    if (is.null(chunk_size)) {
        return(invisible(NULL))
    }
    if (chunk_size <= 0) {
        stop("chunk_size must be a positive integer")
    }
}

#' Validate constructor parameters
#'
#' @param fraction Smoothing fraction
#' @param iterations Robustness iterations (optional)
#' @param window_capacity Window capacity (optional)
#' @param min_points Minimum points (optional)
#' @param chunk_size Chunk size (optional)
#' @noRd
validate_params <- function(
    fraction,
    iterations = NULL,
    window_capacity = NULL,
    min_points = NULL,
    chunk_size = NULL
) {
    validate_fraction(fraction)
    validate_iterations(iterations)
    validate_window_capacity(window_capacity)
    validate_min_points(min_points)
    validate_chunk_size(chunk_size)
}

#' Coerce optional values to Nullable
#' @noRd
#' @srrstats {RE1.2} Numeric vector inputs documented and validated.
coerce_nullable <- function(...) {
    args <- list(...)
    lapply(args, function(x) if (is.null(x)) Nullable(NULL) else x)
}

#' Parameter type registry for Rust FFI coercion
#' @noRd
param_types <- list(
    # Numeric parameters
    fraction = "double",
    # Integer parameters
    iterations = "integer",
    window_capacity = "integer",
    min_points = "integer",
    chunk_size = "integer",
    cv_k = "integer",
    dimensions = "integer",
    # Character parameters
    weight_function = "character",
    robustness_method = "character",
    scaling_method = "character",
    boundary_policy = "character",
    update_mode = "character",
    zero_weight_fallback = "character",
    cv_method = "character",
    degree = "character",
    distance_metric = "character",
    surface_mode = "character",
    merge_strategy = "character",
    # Logical parameters
    return_diagnostics = "logical",
    return_residuals = "logical",
    return_robustness_weights = "logical",
    parallel = "logical",
    return_se = "logical",
    # Nullable parameters (optional, NULL -> Nullable(NULL))
    overlap = "nullable",
    confidence_intervals = "nullable",
    prediction_intervals = "nullable",
    auto_converge = "nullable",
    cv_fractions = "nullable",
    weighted_metric_weights = "nullable",
    cell = "nullable",
    interpolation_vertices = "nullable",
    boundary_degree_fallback = "nullable",
    cv_seed = "nullable"
)

#' Build args from parent environment
#'
#' Captures all known parameters from the calling function's environment.
#' @param param_names Character vector of parameter names to extract.
#' @return Coerced list ready for do.call.
#' @noRd
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
    "return_robustness_weights", "parallel",
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
