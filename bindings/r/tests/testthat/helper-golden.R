# Shared helpers for the stored golden fixtures under tests/testthat/fixtures.
# The generator and test use the same cases so committed values cannot drift
# from the behavior being checked.

golden_seed <- function() {
    20240911L
}

golden_series <- function(n = 60L) {
    set.seed(golden_seed())
    x <- as.double(seq(-3, 3, length.out = n))
    y <- as.double(sin(x) * x + rnorm(n, 0, 0.25))
    list(x = x, y = y)
}

golden_write_csv <- function(df, file) {
    cols <- lapply(seq_along(df), function(j) {
        v <- df[[j]]
        if (is.double(v)) sprintf("%.17g", v) else as.character(v)
    })
    out <- do.call(data.frame, c(cols, list(stringsAsFactors = FALSE)))
    names(out) <- names(df)
    utils::write.csv(out, file, row.names = FALSE, quote = FALSE, na = "")
}

golden_read_csv <- function(file) {
    utils::read.csv(file, stringsAsFactors = FALSE)
}

golden_batch_default <- function(d) {
    model <- Loess(
        fraction = 0.67,
        iterations = 3L,
        boundary_policy = "extend",
        parallel = FALSE
    )
    res <- fit(model, d$x, d$y)
    data.frame(x = d$x, y = d$y, yhat = res$y)
}

golden_batch_intervals <- function(d) {
    model <- Loess(
        fraction = 0.3,
        iterations = 2L,
        boundary_policy = "extend",
        surface_mode = "direct",
        return_se = TRUE,
        return_gradient = TRUE,
        confidence_intervals = 0.9,
        prediction_intervals = 0.9,
        parallel = FALSE
    )
    res <- fit(model, d$x, d$y)
    data.frame(
        x = d$x,
        y = d$y,
        yhat = res$y,
        se = res$standard_errors,
        ci_lower = res$confidence_lower,
        ci_upper = res$confidence_upper,
        pi_lower = res$prediction_lower,
        pi_upper = res$prediction_upper,
        gradient = res$gradient
    )
}

golden_batch_robust <- function(d) {
    y <- d$y
    y[c(9L, 33L)] <- y[c(9L, 33L)] + 4
    model <- Loess(
        fraction = 0.4,
        iterations = 5L,
        boundary_policy = "extend",
        return_robustness_weights = TRUE,
        parallel = FALSE
    )
    res <- fit(model, d$x, y)
    data.frame(
        x = d$x,
        y = y,
        yhat = res$y,
        robustness_weight = res$robustness_weights
    )
}

golden_streaming_chunked <- function(d) {
    sl <- StreamingLoess(
        fraction = 0.3,
        chunk_size = 20L,
        overlap = 0L,
        iterations = 1L,
        parallel = FALSE
    )
    res <- process_chunk(sl, d$x, d$y)
    fin <- finalize(sl)
    data.frame(
        x = c(as.double(res$x), as.double(fin$x)),
        y = d$y,
        yhat = c(as.double(res$y), as.double(fin$y))
    )
}

golden_online_full <- function(d) {
    ol <- OnlineLoess(
        fraction = 0.3,
        window_capacity = 16L,
        min_points = 4L,
        update_mode = "full",
        iterations = 2L
    )
    out <- lapply(seq_along(d$x), function(i) add_point(ol, d$x[[i]], d$y[[i]]))
    yhat <- vapply(
        out,
        function(r) if (is.null(r)) NA_real_ else r$y,
        numeric(1)
    )
    data.frame(x = d$x, y = d$y, yhat = yhat)
}

golden_cases <- function() {
    d <- golden_series()
    list(
        batch_default = golden_batch_default(d),
        batch_intervals = golden_batch_intervals(d),
        batch_robust = golden_batch_robust(d),
        streaming_chunked = golden_streaming_chunked(d),
        online_full = golden_online_full(d)
    )
}
