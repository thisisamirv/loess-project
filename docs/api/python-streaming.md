# StreamingLoess — Python API Reference

See also: [fastLoess Python API Reference](python.md)

## Class

### `StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```python
import fastloess as fl

stream = fl.StreamingLoess(chunk_size=50, overlap=10)
```

**Methods:**

```python
import fastloess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

stream = fl.StreamingLoess(fraction=0.5, chunk_size=50, overlap=10)
partial_result = stream.process_chunk(x[:50], y[:50])
print(partial_result)
# LoessResult(n=40, fraction_used=0.5000)
```

* Processes a chunk of data. Returns partial results.

```python
import fastloess as fl
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 0.1

stream = fl.StreamingLoess(fraction=0.3, chunk_size=50, overlap=10)
stream.process_chunk(x[:50], y[:50])
stream.process_chunk(x[50:], y[50:])
final_result = stream.finalize()
print(final_result)
# LoessResult(n=10, fraction_used=0.3000)
```

* Finalizes the smoothing process and returns any remaining buffered results.

## Result Structure

### `LoessResult`

Returned by `process_chunk()` and `finalize()`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `ndarray` | Sorted x values |
| `y` | `ndarray` | Smoothed y values |
| `fraction_used` | `float` | Fraction used |
| `iterations_used` | `int \| None` | Robustness iterations actually performed |
| `residuals` | `ndarray \| None` | Residuals (if `return_residuals`) |
| `robustness_weights` | `ndarray \| None` | Robustness weights (if `return_robustness_weights`) |
| `diagnostics` | `Diagnostics \| None` | Fit metrics (if `return_diagnostics`) |
| `dimensions` | `int` | Number of predictor dimensions |

See [python.md](python.md) for the full `LoessResult` field reference.

## Options Structure

### `StreamingOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | `500` | Overlap between chunks |
| `merge_strategy` | `str` | `"weighted_average"` | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
