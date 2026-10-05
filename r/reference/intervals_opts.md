# Interval options for fitting and prediction

Interval options for fitting and prediction

## Usage

``` r
intervals_opts(confidence = NULL, prediction = NULL)
```

## Arguments

- confidence:

  Confidence coverage level in (0, 1), or `NULL`.

- prediction:

  Prediction coverage level in (0, 1), or `NULL`.

## Value

A named list for the `intervals` argument.

## Examples

``` r
model <- Loess(intervals = intervals_opts(confidence = 0.95))
```
