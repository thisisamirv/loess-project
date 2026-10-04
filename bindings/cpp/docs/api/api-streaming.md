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

### StreamingOptions

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `weight_function` | `std::string` | `"tricube"` | Weight function name |
| `robustness_method` | `std::string` | `"bisquare"` | Robustness method name |
| `degree` | `std::string` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `int` | `1` | Number of predictor dimensions |
| `distance_metric` | `std::string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `std::vector<double>` | `{}` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode` | `std::string` | `"interpolation"` | Surface computation mode |
| `cell` | `double` | `NaN` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `0` | Number of interpolation vertices (0 for default) |
| `zero_weight_fallback` | `std::string` | `"use_local_mean"` | Zero-weight handling strategy |
| `boundary_policy` | `std::string` | `"extend"` | Boundary handling policy |
| `boundary_degree_fallback` | `int` | `-1` | Fall back to lower polynomial degree at boundaries (-1 = unset/library default, 0 = false, 1 = true) |
| `scaling_method` | `std::string` | `"mad"` | Residual scaling method |
| `auto_converge` | `double` | `NaN` | Auto-convergence tolerance (NaN to disable) |
| `missing` | `std::string` | `"error"` | Policy for non-finite (NaN/Inf) values in each chunk |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | `chunk_size / 10` | Overlap between chunks |
| `merge_strategy` | `std::string` | `"weighted_average"` | Strategy for blending overlap regions |
| `outputs` | `std::vector<std::string>` | `{}` | Optional fields: `diagnostics`, `residuals`, `weights`, `gradient`/`derivative`, `se` |
| `intervals` | `IntervalsOptions` | `disabled` | Grouped confidence and prediction coverage levels. |

Cross-validation and the `"sorted"` output are Batch-only; `StreamingLoess` ignores `"sorted"` — see [fastLoess](api.md) for those.

## Options

### fraction

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### iterations

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### weight_function

*See: [Weight Functions](../weighting/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](../weighting/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### degree

*See: [Polynomial Degree](../advanced/degree.md)*

- `"constant"` or `"0"` (degree 0)
- `"linear"` or `"1"` (default, degree 1)
- `"quadratic"` or `"2"` (degree 2)
- `"cubic"` or `"3"` (degree 3)
- `"quartic"` or `"4"` (degree 4)

### dimensions

*See: [Multivariate LOESS](../advanced/dimensions.md)*

Number of predictor dimensions. Set to match the number of columns in a multivariate `x` array.

- Any integer `>= 1`; `1` (default) is univariate

### distance_metric

*See: [Multivariate LOESS](../advanced/dimensions.md)*

- `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` (use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
- `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### weighted_metric_weights

*See: [Multivariate LOESS](../advanced/dimensions.md)*

Per-dimension weights, one per dimension declared in `dimensions`. Only used when `distance_metric = "weighted"`; setting `distance_metric = "weighted"` without providing this raises an error.

- `{}` (default, empty vector) — has no effect unless `distance_metric = "weighted"` is set
- A non-empty `std::vector<double>` of per-dimension weights, required when `distance_metric = "weighted"`

### surface_mode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### cell

Cell size for the interpolation grid, as a fraction of the data range. Smaller values place more vertices (denser grid), improving accuracy at the cost of speed. Only applies when `surface_mode = "interpolation"`.

- `NaN` (default) — uses the library default (`0.2`)
- Any value in `(0, 1]`

### interpolation_vertices

Caps the maximum number of interpolation vertices, overriding the count implied by `cell`. Only applies when `surface_mode = "interpolation"`.

- `0` (default) — uses the library default (no explicit cap)
- Any integer `>= 1`

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### boundary_degree_fallback

Whether to reduce the polynomial degree at boundary vertices when the requested `degree` can't be fit there (e.g., not enough neighbours). Only applies when `surface_mode = "interpolation"`.

- `-1` (default) — uses the library default (enabled)
- `1` — falls back to a lower degree at boundaries
- `0` — raises an error instead of silently falling back

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### missing

Policy for handling non-finite (NaN/Inf) values within each chunk:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error if any value in the chunk is non-finite |
| `"drop"` | Silently remove rows where any x dimension or y is non-finite before merging the chunk with the overlap buffer |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

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

### outputs

Select optional result fields by name. An empty vector (default) requests only fitted values.

| Name | Result |
| --- | --- |
| `"diagnostics"` | Fit metrics (RMSE, MAE, R², residual SD); AIC/AICc/effective degrees of freedom are unavailable in Streaming |
| `"residuals"` | Per-point residuals (`y - fitted`) |
| `"weights"` | Final per-point robustness weights |
| `"gradient"` or `"derivative"` | Per-point local fit gradient; requires `surface_mode = "direct"` |
| `"se"` | Standard errors, computed per chunk and merged across overlap boundaries via `merge_strategy` |

Confidence and prediction intervals remain controlled by their numeric level fields and include standard errors automatically.

### intervals.confidence

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`), computed per chunk and merged across overlap boundaries the same way `y` is, via `merge_strategy`. `NaN` (default) disables confidence intervals.

### intervals.prediction

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`); same per-chunk computation and overlap-merging as `intervals.confidence`. `NaN` (default) disables prediction intervals.

## Result Structure

### fastloess::LoessResult

Returned (inside `Expected`) by `process_chunk()` and `finalize()`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | x values (same order as input) |
| `y_vector()` | `std::vector<double>` | Smoothed y values |
| `fraction_used()` | `double` | Fraction used |
| `iterations_used()` | `int` | Robustness iterations actually performed (-1 = N/A) |
| `standard_errors()` | `std::vector<double>` | Standard errors, if `outputs` contains `"se"` or an interval level was set (empty otherwise) |
| `confidence_lower()`, `confidence_upper()` | `std::vector<double>` | Confidence interval bounds, if `intervals.confidence` was set (empty otherwise) |
| `prediction_lower()`, `prediction_upper()` | `std::vector<double>` | Prediction interval bounds, if `intervals.prediction` was set (empty otherwise) |
| `residuals()` | `std::vector<double>` | Residuals (if `outputs` contains `"residuals"`; empty if not) |
| `robustness_weights()` | `std::vector<double>` | Robustness weights (if `outputs` contains `"weights"`; empty if not) |
| `cv_scores()` | `std::vector<double>` | Always empty (Batch only) |
| `diagnostics()` | `Diagnostics` | Fit metrics — check `has_value()` (if `outputs` contains `"diagnostics"`) |
| `gradient()` | `std::vector<double>` | Per-point local fit gradient, flattened (if `outputs` contains `"gradient"` or `"derivative"`, `surface_mode = "direct"` only; empty if not computed) |
| `dimensions()` | `int` | Number of predictor dimensions |

### fastloess::Diagnostics

All accessors are const methods (not public fields):

| Method | Return Type | Description |
| --- | --- | --- |
| `rmse()` | `double` | Root Mean Squared Error |
| `mae()` | `double` | Mean Absolute Error |
| `r_squared()` | `double` | R-squared |
| `residual_sd()` | `double` | Residual standard deviation |
| `effective_df()` | `double` | Always `NaN` (requires standard errors, Batch only) |
| `aic()` | `double` | Always `NaN` (requires `effective_df`, Batch only) |
| `aicc()` | `double` | Always `NaN` (requires `effective_df`, Batch only) |

See [cpp.md](api.md) for the full `LoessResult` field reference.

---

> **Always call finalize():** The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.
