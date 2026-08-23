# Print Loess Result

Print Loess Result

## Usage

``` r
# S3 method for class 'LoessResult'
print(x, ...)
```

## Arguments

- x:

  A LoessResult object.

- ...:

  Additional arguments (ignored).

## Value

The input object `x`, invisibly.

## Examples

``` r
x <- seq(0, 10, length.out = 50)
y <- sin(x) + rnorm(50, 0, 0.1)
model <- Loess(fraction = 0.3)
result <- fit(model, x, y)
print(result)
#> <LoessResult>
#>   Points:            50 
#>   Fraction Used:     0.3 
#>   Iterations Used:   3 
```
