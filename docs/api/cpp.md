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
// or with custom weights:
auto result = model.fit(x, y, weights).value();
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
| `custom_weights` | `std::vector<double>` | `{}` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |
| `confidence_intervals` | `double` | `NaN` | Confidence level (e.g., 0.95; NaN to disable) |
| `prediction_intervals` | `double` | `NaN` | Prediction level (e.g., 0.95; NaN to disable) |
| `return_diagnostics` | `bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `bool` | `false` | Include robustness weights in result |
| `return_se` | `bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `bool` | `true` | Enable parallel execution |
| `degree` | `std::string` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `int` | `1` | Number of predictor dimensions |
| `distance_metric` | `std::string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `surface_mode` | `std::string` | `"interpolation"` | Surface computation mode |
| `weighted_metric_weights` | `std::vector<double>` | `{}` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `cell` | `double` | `NaN` | Cell size for interpolation grid (NaN to use default; smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `0` | Number of interpolation vertices (0 for default) |
| `boundary_degree_fallback` | `int` | `-1` | Fall back to lower polynomial degree at boundaries (-1 = unset/library default, 0 = false, 1 = true) |
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
| `window_capacity` | `int` | `1000` | Max points in sliding window |
| `min_points` | `int` | `2` | Min points before smoothing starts |
| `update_mode` | `std::string` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `bool` | `false` | Enable parallel execution (off by default; online LOESS fits one point at a time) |

## Result Structure

### `fastloess::LoessResult`

A RAII wrapper around the C result struct `fastloess_CppLoessResult`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | Sorted x values |
| `y_vector()` | `std::vector<double>` | Smoothed y values |
| `fraction_used()` | `double` | Fraction used (set or selected by CV) |
| `iterations_used()` | `int` | Robustness iterations actually performed (-1 = N/A) |
| `standard_errors()` | `std::vector<double>` | Per-point SE (if `return_se`; empty if not computed) |
| `confidence_lower()` | `std::vector<double>` | Lower confidence bounds (empty if not computed) |
| `confidence_upper()` | `std::vector<double>` | Upper confidence bounds (empty if not computed) |
| `prediction_lower()` | `std::vector<double>` | Lower prediction bounds (empty if not computed) |
| `prediction_upper()` | `std::vector<double>` | Upper prediction bounds (empty if not computed) |
| `residuals()` | `std::vector<double>` | Residuals (if `return_residuals`; empty if not computed) |
| `robustness_weights()` | `std::vector<double>` | Robustness weights (if `return_robustness_weights`; empty if not computed) |
| `cv_scores()` | `std::vector<double>` | CV score per tested fraction (empty if CV not run) |
| `diagnostics()` | `Diagnostics` | Fit metrics — check `diagnostics().has_value()` before use (if `return_diagnostics`) |
| `enp()` | `double` | Equivalent number of parameters (NaN if not computed) |
| `trace_hat()` | `double` | Trace of hat matrix (NaN if not computed) |
| `delta1()` | `double` | First delta statistic (NaN if not computed) |
| `delta2()` | `double` | Second delta statistic (NaN if not computed) |
| `residual_scale()` | `double` | Residual scale estimate (NaN if not computed) |
| `leverage()` | `std::vector<double>` | Per-point hat-matrix diagonal (if `return_se`; empty if not computed) |
| `dimensions()` | `int` | Number of predictor dimensions |

### `fastloess::Diagnostics`

All accessors are const methods (not public fields):

| Method | Return Type | Description |
| --- | --- | --- |
| `rmse()` | `double` | Root Mean Squared Error |
| `mae()` | `double` | Mean Absolute Error |
| `r_squared()` | `double` | R-squared |
| `residual_sd()` | `double` | Residual standard deviation |
| `effective_df()` | `double` | Effective degrees of freedom (NaN if not computed) |
| `aic()` | `double` | AIC (NaN if not computed) |
| `aicc()` | `double` | AICc (NaN if not computed) |

## Options

### weight_function

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### degree

* `"constant"` or `"0"` (degree 0)
* `"linear"` or `"1"` (default, degree 1)
* `"quadratic"` or `"2"` (degree 2)
* `"cubic"` or `"3"` (degree 3)
* `"quartic"` or `"4"` (degree 4)

### distance_metric

* `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
* `"euclidean"` (alias: `"euclid"`)
* `"manhattan"` (alias: `"l1"`)
* `"chebyshev"` (alias: `"linf"`)
* `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
* `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### surface_mode

* `"interpolation"` (default — faster, uses a spatial grid)
* `"direct"` (fits every point exactly; slower but more accurate)

### merge_strategy

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)

### update_mode

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)

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
