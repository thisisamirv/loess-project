#' @srrstats {G2.0, G2.2, G2.3} Validation tests for length, type, range.
#' @srrstats {G2.4} Type coercion verified in constructor tests.
#' @srrstats {G5.3} No NA/NaN in validated outputs.
#' @srrstats {G5.8, G5.8a, G5.8b, G5.8c, G5.8d} Edge condition tests.
# Tests targeting uncovered lines in utils.R:
#   validate_common_args (lines 18-39)
#   coerce_nullable (lines 77-78)
#   env_args: unknown-param passthrough (line 157)

validate_common_args <- getFromNamespace("validate_common_args", "rfastloess")
validate_params <- getFromNamespace("validate_params", "rfastloess")
coerce_nullable <- getFromNamespace("coerce_nullable", "rfastloess")
env_args <- getFromNamespace("env_args", "rfastloess")
validate_named_options <- getFromNamespace(
    "validate_named_options",
    "rfastloess"
)

# ── validate_common_args ────────────────────────────────────────────────────

test_that("validate_common_args rejects mismatched lengths", {
    expect_error(
        validate_common_args(1:3, 1:4, 0.5, 3),
        "must match y"
    )
})

test_that("validate_common_args rejects non-numeric and malformed inputs", {
    expect_error(
        validate_common_args(as.character(1:5), 1:5, 0.5, 3),
        "x must be numeric"
    )
    expect_error(
        validate_common_args(1:5, as.character(1:5), 0.5, 3),
        "y must be a numeric vector"
    )
    expect_error(
        validate_common_args(array(1:8, c(2, 2, 2)), 1:2, 0.5, 3),
        "vector or matrix"
    )
    expect_error(
        validate_common_args(matrix(1:6, nrow = 3), 1:2, 0.5, 3),
        "rows must match"
    )
})

test_that("validate_common_args rejects fewer than 2 points", {
    expect_error(
        validate_common_args(1, 1, 0.5, 3),
        "At least 2 data points are required"
    )
})

test_that("validate_common_args rejects non-numeric fraction", {
    expect_error(
        validate_common_args(1:5, 1:5, "a", 3),
        "fraction must be a single numeric value"
    )
})

test_that("validate_common_args rejects fraction out of range", {
    expect_error(
        validate_common_args(1:5, 1:5, 0, 3),
        "fraction must be between 0 and 1"
    )
    expect_error(
        validate_common_args(1:5, 1:5, 1.5, 3),
        "fraction must be between 0 and 1"
    )
})

test_that("validate_common_args rejects negative iterations", {
    expect_error(
        validate_common_args(1:5, 1:5, 0.5, -1),
        "iterations must be a non-negative integer"
    )
})

test_that("count validation rejects fractional and overflowing values", {
    expect_error(validate_params(0.5, iterations = 1.5), "whole number")
    expect_error(validate_params(0.5, iterations = Inf), "single numeric value")
    expect_error(
        validate_params(0.5, iterations = .Machine$integer.max + 1),
        "maximum supported integer"
    )
    expect_error(validate_params(0.5, window_capacity = 10.5), "whole number")
    expect_error(validate_params(0.5, overlap = 2.5), "whole number")
    expect_error(validate_params(0.5, dimensions = 1.5), "whole number")
    expect_error(
        validate_params(0.5, interpolation_vertices = 10.5),
        "whole number"
    )
})

test_that("grouped options require unique known names", {
    expect_error(
        validate_named_options(list(1), "fractions", "cv"),
        "named list"
    )
    expect_error(
        validate_named_options(
            setNames(list(1, 2), rep("fractions", 2)),
            "fractions",
            "cv"
        ),
        "Duplicate `cv` keys"
    )
    expect_error(
        validate_named_options(
            list(fractions = 1, typo = 2),
            "fractions",
            "cv"
        ),
        "Invalid `cv` key"
    )
})

test_that("validate_common_args returns coerced list on valid input", {
    result <- validate_common_args(1:5, 2:6, 0.5, 3)
    expect_type(result$x, "double")
    expect_type(result$y, "double")
    expect_type(result$fraction, "double")
    expect_type(result$iterations, "integer")
})

test_that("validate_params rejects NA scalar inputs", {
    expect_error(
        validate_params(NA_real_),
        "fraction must be a single numeric value"
    )
    expect_error(
        validate_params(0.5, iterations = NA_real_),
        "iterations must be a single numeric value"
    )
})

# ── coerce_nullable ─────────────────────────────────────────────────────────

test_that("coerce_nullable wraps NULL values", {
    result <- coerce_nullable(NULL, NULL)
    expect_null(result[[1]])
    expect_null(result[[2]])
})

test_that("coerce_nullable passes through non-NULL values unchanged", {
    result <- coerce_nullable(0.95, NULL)
    expect_identical(result[[1]], 0.95)
    expect_null(result[[2]])
})

# ── env_args: unknown-param passthrough (line 157) ──────────────────────────
# env_args returns val as-is when the param name is not in param_types.

test_that("env_args passes through unknown parameter names unchanged", {
    result <- local({
        my_unknown_param <- 42
        env_args("my_unknown_param")
    })
    expect_identical(result[[1]], 42)
})

test_that("env_args handles unknown types in param_types registry", {
    ns <- asNamespace("rfastloess")
    orig_types <- ns$param_types

    # Temporarily inject a dummy type
    new_types <- orig_types
    new_types[["dummy_type_param"]] <- "unhandled_switch_type"

    # assignInNamespace handles unlocking/relocking internally for namespaces
    utils::assignInNamespace("param_types", new_types, "rfastloess")
    on.exit(
        utils::assignInNamespace("param_types", orig_types, "rfastloess"),
        add = TRUE
    )

    result <- local({
        dummy_type_param <- "test_value"
        env_args("dummy_type_param")
    })

    expect_identical(result[[1]], "test_value")
})

# ── constructor-level coverage of env_args type branches ────────────────────

test_that("Loess constructor coerces all param types via env_args", {
    # Exercises double, integer, character, logical, nullable
    model <- Loess(
        fraction = 0.4,
        iterations = 2L,
        weight_function = "tricube",
        parallel = FALSE,
        intervals = intervals_opts(confidence = 0.95)
    )
    expect_s3_class(model, "Loess")
    expect_identical(model$params$fraction, 0.4)
    expect_identical(model$params$iterations, 2L)
})

test_that("StreamingLoess constructor coerces overlap via env_args", {
    model <- StreamingLoess(fraction = 0.3, chunk_size = 50L, overlap = NULL)
    expect_s3_class(model, "StreamingLoess")
})

test_that("OnlineLoess constructor coerces all param types via env_args", {
    model <- OnlineLoess(
        fraction = 0.2,
        window_capacity = 20L,
        min_points = 3L,
        update_mode = "incremental"
    )
    expect_s3_class(model, "OnlineLoess")
})

test_that("validation handles empty, matrix and flattened coordinates", {
    for (xy in list(list(numeric(), numeric()), list(numeric(), 1:2))) {
        expect_error(
            validate_common_args(xy[[1]], xy[[2]], 0.5, 0),
            "rows must match"
        )
    }
    coordinates <- matrix(1:6, nrow = 3)
    result <- validate_common_args(coordinates, 1:3, 0.5, 0)
    expect_identical(result$x, as.double(t(coordinates)))
    result <- validate_common_args(1:6, 1:3, 0.5, 1000)
    expect_identical(result$x, as.double(1:6))
    expect_identical(result$iterations, 1000L)
    for (value in list(NULL, "bad", c(0, 1), NA_real_, Inf, -1, 1.5, 1001)) {
        expect_error(
            validate_common_args(1:3, 1:3, 0.5, value),
            "iterations must be a non-negative integer"
        )
    }
})

test_that("constructor count validation checks upper and lower limits", {
    expect_error(validate_params(0.5, iterations = 1001), "0 and 1000")
    expect_error(validate_params(0.5, min_points = 1), "at least 2")
    expect_error(validate_params(0.5, window_capacity = 0), "positive integer")
    expect_error(validate_params(0.5, overlap = -1), "non-negative integer")
})

test_that("grouped options reject malformed names and containers", {
    for (options in list(1, list(1), setNames(list(1), NA_character_))) {
        expect_error(
            validate_named_options(options, "fractions", "cv"),
            "named list"
        )
    }
})

test_that("CV options validate fraction, method and fold boundaries", {
    expect_error(cv_opts(NULL), "candidate fractions")
    expect_error(cv_opts(numeric()), "non-empty")
    for (fractions in list(NA_real_, Inf, 0, -0.1, 1.1)) {
        expect_error(cv_opts(fractions), "finite values")
    }
    for (method in list(1, character(), c("kfold", "loocv"), NA_character_)) {
        expect_error(cv_opts(0.5, method = method), "single character")
    }
    for (method in c("kfold", "k_fold", "k-fold", "KFOLD")) {
        expect_error(cv_opts(0.5, method = method, k = 1), "at least 2 folds")
    }
    expect_identical(cv_opts(1, method = "loocv", k = 1)$k, 1L)
    expect_identical(cv_opts(c(0.2, 1), k = 2)$fractions, c(0.2, 1))
})

test_that("Loess seed validation rejects invalid and accepts boundary values", {
    for (seed in list(-1, 1.5, 2^53 + 2)) {
        expect_error(Loess(seed = seed), "non-negative whole number")
    }
    for (seed in list("bad", NA_real_, Inf)) {
        expect_error(Loess(seed = seed), "single numeric value")
    }
    expect_s3_class(Loess(seed = 0), "Loess")
    expect_s3_class(Loess(seed = 2^53), "Loess")
})
