# Print StreamingLoess Model

Print StreamingLoess Model

## Usage

``` r
# S3 method for class 'StreamingLoess'
print(x, ...)
```

## Arguments

- x:

  A StreamingLoess object.

- ...:

  Additional arguments.

## Value

The input object `x`, invisibly.

## Examples

``` r
model <- StreamingLoess(fraction = 0.3, chunk_size = 50L)
print(model)
#> <StreamingLoess Model>
#>   Fraction:          0.3 
#>   Chunk Size:        50 
#>   Parallel:          TRUE 
```
