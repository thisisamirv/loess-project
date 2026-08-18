# StreamingLoess — R API Reference

See also: [fastLoess R API Reference](r.md)

## Class

### `StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```r
library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

library(rfastloess)
stream <- StreamingLoess(fraction = 0.3, chunk_size = 50, overlap = 10)
```

* `...`: Arguments corresponding to `LoessOptions` and `StreamingOptions` fields.

**Methods:**

```r
library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

stream <- StreamingLoess(fraction = 0.3, chunk_size = 50, overlap = 10)
partial_result <- process_chunk(stream, x[seq_len(50)], y[seq_len(50)])
```

* Processes a chunk of data. Returns partial results.

```r
library(rfastloess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

stream <- StreamingLoess(fraction = 0.3, chunk_size = 50, overlap = 10)
process_chunk(stream, x, y)
final_result <- finalize(stream)
```

* Finalizes the smoothing process and returns any remaining buffered results.

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
