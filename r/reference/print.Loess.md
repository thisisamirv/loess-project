# Print Loess Model

Print Loess Model

## Usage

``` r
# S3 method for class 'Loess'
print(x, ...)
```

## Arguments

- x:

  A Loess object.

- ...:

  Additional arguments (ignored).

## Value

The input object `x`, invisibly.

## Examples

``` r
model <- Loess(fraction = 0.3)
print(model)
#> <Loess Model>
#>   Fraction:          0.3 
#>   Iterations:        3 
#>   Weight Function:   tricube 
#>   Parallel:          TRUE 
```
