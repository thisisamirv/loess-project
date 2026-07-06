# fastLoess C++ API Reference

The C++ bindings provide a modern, object-oriented wrapper around the core Rust library, mirroring the Rust API structure.

## Classes

### `fastloess::Loess`

The `Loess` class allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```cpp
fastloess::LoessOptions opts;
opts.fraction = 0.5;
fastloess::Loess model(opts);
```

* `options`: A `LoessOptions` struct containing configuration parameters.

**Methods:**

```cpp
fastloess::Loess model;
auto result = model.fit(x, y).value();
```

* Fits the model to the provided `x` and `y` data vectors.
* Returns an `Expected<LoessResult>` — call `.has_value()` to check for errors, `.value()` to unwrap (throws `LoessError` on failure).

### `fastloess::StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```cpp
fastloess::StreamingOptions opts;
opts.chunk_size = 5;
fastloess::StreamingLoess model(opts);
```

* `options`: A `StreamingOptions` struct (inherits from `LoessOptions`) with additional `chunk_size` and `overlap` parameters.

**Methods:**

```cpp
fastloess::StreamingOptions opts;
opts.chunk_size = 10;
fastloess::StreamingLoess model(opts);
auto result = model.process_chunk(x, y).value();
```

* Processes a chunk of data. Returns partial results.

```cpp
fastloess::StreamingOptions opts;
opts.chunk_size = 10;
fastloess::StreamingLoess model(opts);
model.process_chunk(x, y);
auto result = model.finalize().value();
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `fastloess::OnlineLoess`

The `OnlineLoess` class updates the model incrementally with new data points.

**Constructor:**

```cpp
fastloess::OnlineOptions opts;
opts.window_capacity = 10;
fastloess::OnlineLoess model(opts);
```

* `options`: An `OnlineOptions` struct (inherits from `LoessOptions`) with `window_capacity`, `min_points`, and `update_mode`.

**Methods:**

```cpp
fastloess::OnlineOptions opts;
opts.window_capacity = 10;
fastloess::OnlineLoess model(opts);
auto result = model.add_points(x, y).value();
```

* Adds new points to the model and returns the smoothed values (retrospective or prospective depending on mode).

## Options Structures

### `LoessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `weight_function` | `std::string` | `"tricube"` | Kernel weight function |
| `robustness_method` | `std::string` | `"bisquare"` | Robustness method |
| `scaling_method` | `std::string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `std::string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `std::string` | `"use_local_mean"` | Zero-weight handling strategy |
| `auto_converge` | `double` | `NaN` | Auto-convergence tolerance (NaN to disable) |
| `custom_weights` | `std::vector<double>` | `{}` | Per-observation case weights (Batch only) |
| `confidence_intervals` | `double` | `NaN` | Confidence level (e.g., 0.95; NaN to disable) |
| `prediction_intervals` | `double` | `NaN` | Prediction level (e.g., 0.95; NaN to disable) |
| `return_diagnostics` | `bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `bool` | `false` | Include robustness weights in result |
| `return_se` | `bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `bool` | `false` | Enable parallel execution |
| `degree` | `std::string` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `int` | `1` | Number of predictor dimensions |
| `distance_metric` | `std::string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `surface_mode` | `std::string` | `"interpolation"` | Surface computation mode |
| `weighted_metric_weights` | `std::vector<double>` | `{}` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `cell` | `double` | `NaN` | Cell size for interpolation grid (NaN to use default; smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `0` | Number of interpolation vertices (0 for default) |
| `boundary_degree_fallback` | `int` | `-1` | Fall back to lower polynomial degree at boundaries (-1 = off, 0 = false, 1 = true) |
| `cv_seed` | `uint64_t` | `0` | Random seed for cross-validation shuffling (Batch only; 0 = random) |
| `cv_fractions` | `std::vector<double>` | `{}` | Fractions to test for cross-validation |
| `cv_method` | `std::string` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) |
| `cv_k` | `int` | `5` | Number of folds for k-fold CV |

### `StreamingOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | `-1` (auto) | Overlap between chunks (-1 for auto 10%) |
| `merge_strategy` | `std::string` | `"weighted_average"` | Strategy for blending overlap: see Merge Strategies |

### `OnlineOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `int` | `100` | Max points in sliding window |
| `min_points` | `int` | `2` | Min points before smoothing starts |
| `update_mode` | `std::string` | `"full"` | Update mode (`"full"` or `"incremental"`) |

## Result Structure

### `fastloess::LoessResult`

A RAII wrapper around the C result struct `fastloess_CppLoessResult`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | Sorted x values |
| `y_vector()` | `std::vector<double>` | Smoothed y values |
| `fraction_used()` | `double` | Fraction used (set or selected by CV) |
| `iterations_used()` | `int` | Robustness iterations actually performed |
| `standard_errors()` | `std::vector<double>` | Per-point SE (if `return_se`) |
| `confidence_lower()` | `std::vector<double>` | Lower confidence bounds |
| `confidence_upper()` | `std::vector<double>` | Upper confidence bounds |
| `prediction_lower()` | `std::vector<double>` | Lower prediction bounds |
| `prediction_upper()` | `std::vector<double>` | Upper prediction bounds |
| `residuals()` | `std::vector<double>` | Residuals (if `return_residuals`) |
| `robustness_weights()` | `std::vector<double>` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores()` | `std::vector<double>` | CV score per tested fraction |
| `diagnostics()` | `Diagnostics` | Fit metrics (if `return_diagnostics`) |
| `enp()` | `double` | Equivalent number of parameters (if `return_se`) |
| `trace_hat()` | `double` | Trace of hat matrix (if `return_se`) |
| `delta1()` | `double` | First delta statistic (if `return_se`) |
| `delta2()` | `double` | Second delta statistic (if `return_se`) |
| `residual_scale()` | `double` | Residual scale estimate (if `return_se`) |
| `leverage()` | `std::vector<double>` | Per-point hat-matrix diagonal (if `return_se`) |
| `dimensions()` | `int` | Number of predictor dimensions |

### `fastloess::Diagnostics`

All accessors are const methods (not public fields):

| Method | Return Type | Description |
| --- | --- | --- |
| `rmse()` | `double` | Root Mean Squared Error |
| `mae()` | `double` | Mean Absolute Error |
| `r_squared()` | `double` | R-squared |
| `residual_sd()` | `double` | Residual standard deviation |
| `effective_df()` | `double` | Effective degrees of freedom |
| `aic()` | `double` | AIC |
| `aicc()` | `double` | AICc |

## String Options

### Weight Functions

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"`
* `"biweight"`
* `"triangle"`
* `"cosine"`

### Robustness Methods

* `"bisquare"` (default)
* `"huber"`
* `"talwar"`

### Boundary Policies

* `"extend"` (default - linear extrapolation)
* `"reflect"`
* `"zero"`
* `"noboundary"`

### Scaling Methods

* `"mad"` (default - Median Absolute Deviation)
* `"mar"` (Median Absolute Residual)
* `"mean"` (Mean Absolute Residual)

### Zero Weight Fallback

* `"use_local_mean"` (default)
* `"return_original"`
* `"return_none"`

### Polynomial Degrees

* `"constant"` (degree 0)
* `"linear"` (default, degree 1)
* `"quadratic"` (degree 2)
* `"cubic"` (degree 3)
* `"quartic"` (degree 4)

### Distance Metrics

* `"normalized"` (default — scales each dimension by its range)
* `"euclidean"`
* `"manhattan"`
* `"chebyshev"`
* `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
* `"weighted"` (set `weighted_metric_weights` for per-dimension scaling)

### Surface Modes

* `"interpolation"` (default — faster, uses a spatial grid)
* `"direct"` (fits every point exactly; slower but more accurate)

### Merge Strategies (Streaming)

* `"weighted_average"` (default — weighted blend of overlapping regions)
* `"average"` (simple mean of overlapping regions)
* `"take_first"` (keep values from the earlier chunk)
* `"take_last"` (keep values from the later chunk)

### Update Modes (Online)

* `"full"` (default — re-smooth entire window each update)
* `"incremental"` (faster, O(1) incremental update)

## Example

```cpp
#include "fastloess.hpp"
#include <iostream>

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2.1, 4.0, 6.2, 8.0, 10.1};

    fastloess::LoessOptions opts;
    opts.fraction = 0.5;
    
    fastloess::Loess model(opts);
    auto expected = model.fit(x, y);

    if (expected.has_value()) {
        auto y_hat = expected.value().y_vector();
        for (double val : y_hat) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```
