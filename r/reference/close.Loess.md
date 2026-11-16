# Release native resources held by a LOESS model

S3 methods for base [`close`](https://rdrr.io/r/base/connections.html).
For `Loess`, clears retained predict state early. For `StreamingLoess`
and `OnlineLoess`, releases the native adapter. Safe to call multiple
times.

## Usage

``` r
# S3 method for class 'Loess'
close(con, ...)

# S3 method for class 'StreamingLoess'
close(con, ...)

# S3 method for class 'OnlineLoess'
close(con, ...)
```

## Arguments

- con:

  A `Loess`, `StreamingLoess`, or `OnlineLoess` object.

- ...:

  Must be empty.

## Value

`NULL`, invisibly.

## Examples

``` r
model <- Loess(fraction = 0.3, retain_model = TRUE)
invisible(fit(model, 1:10, sin(1:10)))
close(model)
```
