#' rfastloess: High-performance LOESS Smoothing for R
#'
#' @description
#' A high-performance LOESS (Locally Estimated Scatterplot Smoothing)
#' implementation built on the Rust `fastLoess` crate.
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{fastloess}}: Primary interface for batch processing
#'   \item \code{\link{fastloess_streaming}}: Chunked processing for large
#'     datasets
#'   \item \code{\link{fastloess_online}}: Sliding window for real-time data
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
#' result <- fastloess(x, y, fraction = 0.3)
#' plot(x, y)
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @useDynLib rfastloess, .registration = TRUE
#' @importFrom stats smooth
"_PACKAGE"
