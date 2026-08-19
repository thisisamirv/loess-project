# StreamingLoess — R API Reference

See also: [fastLoess R API Reference](r.md)

## Class

### `StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```r
library(rfastloess)

stream <- StreamingLoess(fraction = 0.3, chunk_size = 50L, overlap = 10L)
print(stream)
#> <StreamingLoess Model>
#>   Fraction:          0.3
#>   Chunk Size:        50
#>   Parallel:          TRUE
```

**Methods:**

```r
library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

stream <- StreamingLoess(fraction = 0.3, chunk_size = 50L, overlap = 10L)
partial_result <- process_chunk(stream, x[seq_len(50)], y[seq_len(50)])
print(partial_result)
#> <LoessResult>
#>   Points:            40
#>   Fraction Used:     0.3
#>   Iterations Used:   3
```

* Processes a chunk of data. Returns partial results.

```r
library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

stream <- StreamingLoess(fraction = 0.3, chunk_size = 50L, overlap = 10L)
partial1 <- process_chunk(stream, x[seq_len(50)], y[seq_len(50)])
partial2 <- process_chunk(stream, x[51:100], y[51:100])
final_result <- finalize(stream)
print(final_result)
#> <LoessResult>
#>   Points:            10
#>   Fraction Used:     0.3
```

* Finalizes the smoothing process and returns any remaining buffered results.

## Result Structure

### `LoessResult`

Returned by `process_chunk()` and `finalize()`. An S3 list with class `"LoessResult"` containing:

**Supported S3 Methods:** `print(result)`, `plot(result)`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `numeric` | Sorted x values |
| `y` | `numeric` | Smoothed y values |
| `fraction_used` | `numeric` | Fraction used |
| `iterations_used` | `integer \| NULL` | Robustness iterations actually performed |
| `residuals` | `numeric \| NULL` | Residuals (if `return_residuals`) |
| `robustness_weights` | `numeric \| NULL` | Robustness weights (if `return_robustness_weights`) |
| `diagnostics` | `list \| NULL` | Fit metrics (if `return_diagnostics`) |
| `dimensions` | `integer` | Number of predictor dimensions |

See [r.md](r.md) for the full `LoessResult` field reference.

## Options Structure

### `StreamingOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `integer` | `5000L` | Data chunk size |
| `overlap` | `integer` | `500L` | Overlap between chunks |
| `merge_strategy` | `character` | `"weighted_average"` | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
