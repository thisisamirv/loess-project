# Process a data chunk through a streaming LOESS model

Process a data chunk through a streaming LOESS model

## Usage

``` r
process_chunk(model, ...)

# S3 method for class 'StreamingLoess'
process_chunk(model, x, y, ...)
```

## Arguments

- model:

  A `StreamingLoess` object.

- ...:

  Must be empty.

- x:

  Numeric vector of x values.

- y:

  Numeric vector of y values.

## Value

A `LoessResult` for this chunk.

## Examples

``` r
x <- seq(0, 10, length.out = 100)
y <- sin(x) + rnorm(100, 0, 0.1)
model <- StreamingLoess(fraction = 0.2, chunk_size = 50L)
res <- process_chunk(model, x[1:50], y[1:50])
```
