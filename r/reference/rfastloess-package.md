# rfastloess: High-performance LOESS Smoothing for R

A high-performance LOESS (Locally Estimated Scatterplot Smoothing)
implementation built on the Rust `fastLoess` crate.

For comprehensive documentation, see:
<https://github.com/thisisamirv/loess-project/tree/main/docs>

## Main Classes

- [`Loess`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md):
  Primary interface for batch processing

- [`StreamingLoess`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md):
  Chunked processing for large datasets

- [`OnlineLoess`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md):
  Sliding window for real-time data

## Documentation

For comprehensive documentation, tutorials, and API reference, see:
<https://loess.readthedocs.io/>

## See also

Useful links:

- <https://github.com/thisisamirv/loess-project>

- Report bugs at <https://github.com/thisisamirv/loess-project/issues>

## Author

**Maintainer**: Amir Valizadeh <thisisamirv@gmail.com>
([ORCID](https://orcid.org/0000-0001-5983-8527)) \[funder\]

Authors:

- Amir Valizadeh <thisisamirv@gmail.com>
  ([ORCID](https://orcid.org/0000-0001-5983-8527)) \[funder\]

## Examples

``` r
# Basic smoothing
x <- seq(1, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)
model <- Loess(fraction = 0.3)
result <- fit(model, x, y)
plot(x, y)
lines(result$x, result$y, col = "red", lwd = 2)

```
