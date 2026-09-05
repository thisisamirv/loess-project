# Print OnlineLoess Model

Print OnlineLoess Model

## Usage

``` r
# S3 method for class 'OnlineLoess'
print(x, ...)
```

## Arguments

- x:

  An OnlineLoess object.

- ...:

  Additional arguments.

## Value

The input object `x`, invisibly.

## Examples

``` r
model <- OnlineLoess(fraction = 0.2, window_capacity = 20L)
print(model)
#> <OnlineLoess Model>
#>   Fraction:          0.2 
#>   Window Capacity:   20 
#>   Min Points:        3 
#>   Update Mode:       
```
