#' rfastloess: High-performance LOESS Smoothing for R
#'
#' @description
#' A high-performance LOESS (Locally Weighted Scatterplot Smoothing)
#' implementation built on the Rust `fastLoess` crate.
#'
#' For comprehensive documentation, see:
#' <https://github.com/thisisamirv/loess-project/tree/main/docs>
#'
#' @docType package
#' @srrstats {G1.0} Package-level documentation for statistical software review.
#' @srrstats {G1.1} Thin R wrapper interface for core Rust algorithms.
#' @srrstats {G1.2} Package lifecycle documented in README and NEWS.md.
#' @srrstats {G1.4, G1.4a} All exported functions documented with roxygen2.
#' @srrstats {G1.5} Benchmarks against stats::loess in examples/r.

#' @section Main Classes:
#' \itemize{
#'   \item \code{\link{Loess}}: Primary interface for batch processing
#'   \item \code{\link{StreamingLoess}}: Chunked processing for large
#'     datasets
#'   \item \code{\link{OnlineLoess}}: Sliding window for real-time data
#' }
#'
#' @section Documentation:
#' For comprehensive documentation, tutorials, and API reference, see:
#' \url{https://loess.readthedocs.io/}
#'
#' @examples
#' # Basic smoothing
#' x <- seq(1, 10, length.out = 100)
#' y <- sin(x) + rnorm(100, sd = 0.2)
#' model <- Loess(fraction = 0.3)
#' result <- model$fit(x, y)
#' plot(x, y)
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @useDynLib rfastloess, .registration = TRUE
#' @importFrom stats smooth
#' @importFrom BiocGenerics normalize
"_PACKAGE"
