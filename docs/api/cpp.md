# fastLoess C++ API Reference

The C++ bindings provide a modern, object-oriented wrapper around the core Rust library, mirroring the Rust API structure.

## Classes

### `fastloess::Loess`

The `Loess` class allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```cpp
explicit Loess(const LoessOptions &options = {})
```

* `options`: A `LoessOptions` struct containing configuration parameters.

**Methods:**

```cpp
LoessResult fit(const std::vector<double> &x, const std::vector<double> &y)
```

* Fits the model to the provided `x` and `y` data vectors.
* Returns a `LoessResult` object containing the smoothed values and optional diagnostics.

### `fastloess::StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```cpp
explicit StreamingLoess(const StreamingOptions &options = {})
```

* `options`: A `StreamingOptions` struct (inherits from `LoessOptions`) with additional `chunk_size` and `overlap` parameters.

**Methods:**

```cpp
LoessResult process_chunk(const std::vector<double> &x, const std::vector<double> &y)
```

* Processes a chunk of data. Returns partial results.

```cpp
LoessResult finalize()
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `fastloess::OnlineLoess`

The `OnlineLoess` class updates the model incrementally with new data points.

**Constructor:**

```cpp
explicit OnlineLoess(const OnlineOptions &options = {})
```

* `options`: An `OnlineOptions` struct (inherits from `LoessOptions`) with `window_capacity`, `min_points`, and `update_mode`.

**Methods:**

```cpp
LoessResult add_points(const std::vector<double> &x, const std::vector<double> &y)
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
| `xVector()` | `std::vector<double>` | Sorted x values |
| `yVector()` | `std::vector<double>` | Smoothed y values |
| `fractionUsed()` | `double` | Fraction used (set or selected by CV) |
| `iterationsUsed()` | `int` | Robustness iterations actually performed |
| `standardErrors()` | `std::vector<double>` | Per-point SE (if `return_se`) |
| `confidenceLower()` | `std::vector<double>` | Lower confidence bounds |
| `confidenceUpper()` | `std::vector<double>` | Upper confidence bounds |
| `predictionLower()` | `std::vector<double>` | Lower prediction bounds |
| `predictionUpper()` | `std::vector<double>` | Upper prediction bounds |
| `residuals()` | `std::vector<double>` | Residuals (if `return_residuals`) |
| `robustnessWeights()` | `std::vector<double>` | Robustness weights (if `return_robustness_weights`) |
| `cvScores()` | `std::vector<double>` | CV score per tested fraction |
| `diagnostics()` | `Diagnostics` | Fit metrics (if `return_diagnostics`) |
| `enp()` | `double` | Equivalent number of parameters (if `return_se`) |
| `traceHat()` | `double` | Trace of hat matrix (if `return_se`) |
| `delta1()` | `double` | First delta statistic (if `return_se`) |
| `delta2()` | `double` | Second delta statistic (if `return_se`) |
| `residualScale()` | `double` | Residual scale estimate (if `return_se`) |
| `leverage()` | `std::vector<double>` | Per-point hat-matrix diagonal (if `return_se`) |
| `dimensions()` | `int` | Number of predictor dimensions |

### `fastloess::Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `double` | Root Mean Squared Error |
| `mae` | `double` | Mean Absolute Error |
| `r_squared` | `double` | R-squared |
| `residual_sd` | `double` | Residual standard deviation |
| `effective_df` | `double` | Effective degrees of freedom |
| `aic` | `double` | AIC |
| `aicc` | `double` | AICc |

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
    auto result = model.fit(x, y);

    if (result.valid()) {
        auto y_hat = result.yVector();
        for (double val : y_hat) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```
