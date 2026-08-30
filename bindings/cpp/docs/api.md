# API

The C++ bindings provide a modern, object-oriented wrapper around the core Rust library, mirroring the Rust API structure.

> **StreamingLoess** and **OnlineLoess** are documented separately: [cpp-streaming.md](api-streaming.md), [cpp-online.md](api-online.md)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

![Gap Handling](gap_handling.svg)

## Classes

### fastloess::Loess

The `Loess` class allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

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
    fastloess::LoessOptions opts;
    opts.fraction = 0.5;
    fastloess::Loess model(opts);
    auto result = model.fit(x, y).value();
    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: 0.327376
```

- `options`: A `LoessOptions` struct containing configuration parameters.

**Methods:**

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

    fastloess::LoessOptions opts;
    opts.fraction = 0.5;
    fastloess::Loess model(opts);
    auto result = model.fit(x, y).value();
    std::cout << result.fraction_used() << std::endl;  // 0.5
    std::cout << result.iterations_used() << std::endl;  // 3
    // or with custom weights:
    std::vector<double> weights(x.size(), 1.0);
    auto resultW = model.fit(x, y, weights).value();

    return 0;
}
```

```output
0.5
3
```

- Fits the model to the provided `x` and `y` data vectors.
- Returns an `Expected<LoessResult>` — call `.has_value()` to check for errors, `.value()` to unwrap (throws `LoessError` on failure).

See [cpp-streaming.md](api-streaming.md) for the `StreamingLoess` class.

See [cpp-online.md](api-online.md) for the `OnlineLoess` class.

## Options Structures

### LoessOptions

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `int` | `3` | Number of robustifying iterations |
| `weight_function` | `std::string` | `"tricube"` | Kernel weight function |
| `robustness_method` | `std::string` | `"bisquare"` | Robustness method |
| `scaling_method` | `std::string` | `"mad"` | Residual scaling method |
| `boundary_policy` | `std::string` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `std::string` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge` | `double` | `NaN` | Auto-convergence tolerance (NaN to disable) |
| `custom_weights` | `std::vector<double>` | `{}` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only; see [Custom Weights](custom-weights.md)) |
| `confidence_intervals` | `double` | `NaN` | Confidence level (e.g., 0.95; NaN to disable) — see [Intervals](intervals.md) |
| `prediction_intervals` | `double` | `NaN` | Prediction level (e.g., 0.95; NaN to disable) — see [Intervals](intervals.md) |
| `return_diagnostics` | `bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `bool` | `false` | Include robustness weights in result |
| `return_se` | `bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `bool` | `true` | Enable parallel execution |
| `degree` | `std::string` | `"linear"` | Polynomial degree of local fit — see [Polynomial Degree](degree.md) |
| `dimensions` | `int` | `1` | Number of predictor dimensions — see [Multivariate LOESS](dimensions.md) |
| `distance_metric` | `std::string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `std::vector<double>` | `{}` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode` | `std::string` | `"interpolation"` | Surface computation mode |
| `cell` | `double` | `NaN` | Cell size for interpolation grid (NaN to use default; smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `0` | Number of interpolation vertices (0 for default) |
| `boundary_degree_fallback` | `int` | `-1` | Fall back to lower polynomial degree at boundaries (-1 = unset/library default, 0 = false, 1 = true) |
| `cv_method` | `std::string` | `"kfold"` | CV method (`"kfold"` fast or `"loocv"` slow, exhaustive) (Batch only) |
| `cv_k` | `int` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `std::vector<double>` | `{}` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `uint64_t` | `0` | Random seed for cross-validation shuffling (Batch only; 0 = random) |

`fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

`iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

See [cpp-streaming.md](api-streaming.md) for `StreamingOptions`.

See [cpp-online.md](api-online.md) for `OnlineOptions`.

## Result Structure

See [cpp-online.md](api-online.md) for `OnlineOutput`.

### fastloess::LoessResult

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

### fastloess::Diagnostics

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

*See: [Weight Functions](kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### boundary_policy

*See: [Boundary Handling](boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### degree

*See: [Polynomial Degree](degree.md)*

- `"constant"` or `"0"` (degree 0)
- `"linear"` or `"1"` (default, degree 1)
- `"quadratic"` or `"2"` (degree 2)
- `"cubic"` or `"3"` (degree 3)
- `"quartic"` or `"4"` (degree 4)

### distance_metric

*See: [Multivariate LOESS](dimensions.md)*

- `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
- `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### surface_mode

*See: [Polynomial Degree](degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### merge_strategy

See [cpp-streaming.md](api-streaming.md).

### update_mode

See [cpp-online.md](api-online.md).

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

```output
2.1 4 6.2 8 10.1
```
