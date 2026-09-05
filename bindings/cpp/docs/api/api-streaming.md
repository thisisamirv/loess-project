\page api_streaming StreamingLoess API

# StreamingLoess API

See also: [fastLoess](api.md)

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Class

### fastloess::StreamingLoess

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }
    fastloess::StreamingOptions opts;
    opts.fraction = 0.5;
    opts.chunk_size = 50;
    opts.overlap = 10;
    fastloess::StreamingLoess model(opts);
    std::vector<double> x1(x.begin(), x.begin() + 50), y1(y.begin(), y.begin() + 50);
    auto result = model.process_chunk(x1, y1).value();
    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: 0.224537
```

- `options`: A `StreamingOptions` struct (inherits from `LoessOptions`) with additional `chunk_size` and `overlap` parameters.

#### `process_chunk(x, y)`

Processes a chunk of data. Returns partial results.

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastloess::StreamingOptions opts;
    opts.fraction = 0.5;
    opts.chunk_size = 50;
    opts.overlap = 10;
    fastloess::StreamingLoess model(opts);
    std::vector<double> x1(x.begin(), x.begin() + 50), y1(y.begin(), y.begin() + 50);
    auto partial = model.process_chunk(x1, y1).value();
    std::cout << partial.fraction_used() << std::endl;  // 0.5

    return 0;
}
```

```output
0.5
```

#### `finalize()`

Finalizes the smoothing process and returns any remaining buffered results.

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastloess::StreamingOptions opts;
    opts.fraction = 0.5;
    opts.chunk_size = 50;
    opts.overlap = 10;
    fastloess::StreamingLoess model(opts);
    std::vector<double> x1(x.begin(), x.begin() + 50), y1(y.begin(), y.begin() + 50);
    std::vector<double> x2(x.begin() + 50, x.end()), y2(y.begin() + 50, y.end());
    model.process_chunk(x1, y1);
    model.process_chunk(x2, y2);
    auto result = model.finalize().value();
    std::cout << result.fraction_used() << std::endl;  // 0.5

    return 0;
}
```

```output
0.5
```

## Options Structure

### StreamingOptions (inherits LoessOptions)

`StreamingOptions` inherits every field from `LoessOptions` (see [fastLoess](api.md#loessoptions)) and adds:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | `chunk_size / 10` | Overlap between chunks |
| `merge_strategy` | `std::string` | `"weighted_average"` | Strategy for blending overlap regions |

Confidence/prediction intervals, standard errors, and cross-validation are Batch-only; setting these inherited fields has no effect on `StreamingLoess` — see [fastLoess](api.md) for those.

## Options

### chunk_size

Number of points processed per call to `process_chunk()`. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying.

### overlap

Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `merge_strategy`. A good starting point is 10–20% of `chunk_size`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice.

- `-1` (the `StreamingOptions` default) — "use the library default", computing `chunk_size / 10` clamped to `[1, chunk_size - 10]`
- Any non-negative integer `< chunk_size`

### merge_strategy

*See: [Merge Strategies](../advanced/merge.md)*

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

![Merge Strategies](merge_comparison.svg)

## Result Structure

### fastloess::LoessResult

Returned (inside `Expected`) by `process_chunk()` and `finalize()`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | x values (same order as input) |
| `y_vector()` | `std::vector<double>` | Smoothed y values |
| `fraction_used()` | `double` | Fraction used |
| `iterations_used()` | `int` | Robustness iterations actually performed (-1 = N/A) |
| `standard_errors()` | `std::vector<double>` | Always empty (Batch only) |
| `confidence_lower()`, `confidence_upper()` | `std::vector<double>` | Always empty (Batch only) |
| `prediction_lower()`, `prediction_upper()` | `std::vector<double>` | Always empty (Batch only) |
| `residuals()` | `std::vector<double>` | Residuals (if `return_residuals`; empty if not) |
| `robustness_weights()` | `std::vector<double>` | Robustness weights (if `return_robustness_weights`; empty if not) |
| `cv_scores()` | `std::vector<double>` | Always empty (Batch only) |
| `diagnostics()` | `Diagnostics` | Fit metrics — check `has_value()` (if `return_diagnostics`) |
| `dimensions()` | `int` | Number of predictor dimensions |

See [cpp.md](api.md) for the full `LoessResult` field reference.

---

> **Always call finalize():** The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.
