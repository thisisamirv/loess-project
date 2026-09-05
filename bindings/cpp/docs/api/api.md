\page api API

# API

The C++ bindings provide a modern, object-oriented wrapper around the core Rust library, mirroring the Rust API structure.

> **StreamingLoess** and **OnlineLoess** are documented separately: \subpage api_streaming, \subpage api_online

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

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

#### `fit(x, y)`

Fits the model to the provided `x` and `y` data vectors. Returns an `Expected<LoessResult>` — call `.has_value()` to check for errors, `.value()` to unwrap (throws `LoessError` on failure).

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
| `confidence_intervals` | `double` | `NaN` | Confidence level (e.g., 0.95; NaN to disable) |
| `prediction_intervals` | `double` | `NaN` | Prediction level (e.g., 0.95; NaN to disable) |
| `return_diagnostics` | `bool` | `false` | Include diagnostics in result |
| `return_residuals` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `bool` | `false` | Include weights in result |
| `return_se` | `bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `bool` | `true` | Enable parallel execution |
| `degree` | `std::string` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `int` | `1` | Number of predictor dimensions |
| `distance_metric` | `std::string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `std::vector<double>` | `{}` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode` | `std::string` | `"interpolation"` | Surface computation mode |
| `cell` | `double` | `NaN` | Cell size for interpolation grid (NaN to use default; smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `int` | `0` | Number of interpolation vertices (0 for default) |
| `boundary_degree_fallback` | `int` | `-1` | Fall back to lower polynomial degree at boundaries (-1 = unset/library default, 0 = false, 1 = true) |
| `cv_method` | `std::string` | `"kfold"` | CV method (`"kfold"` fast or `"loocv"` slow, exhaustive) |
| `cv_k` | `int` | `5` | Number of folds for k-fold CV |
| `cv_fractions` | `std::vector<double>` | `{}` | Fractions to test for cross-validation |
| `cv_seed` | `uint64_t` | `0` | Random seed for cross-validation shuffling (0 = random) |
| `custom_weights` | `std::vector<double>` | `{}` | Per-observation case weights — passed to `fit()`, not the constructor |

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

### scaling_method

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### zero_weight_fallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### auto_converge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `NaN` (default) disables early stopping.

### confidence_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). `NaN` (default) disables confidence intervals.

### prediction_intervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). `NaN` (default) disables prediction intervals.

### return_diagnostics

*See: [`Diagnostics`](#fastloessdiagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R², AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/effective degrees of freedom additionally require `return_se = true` (or confidence/prediction intervals) to be populated, since they depend on hat-matrix statistics.

- `false` (default) — leaves `diagnostics()` empty
- `true` — populates `diagnostics()`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `false` (default) — leaves `residuals()` empty
- `true` — populates `residuals()`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `false` (default) — leaves `robustness_weights()` empty
- `true` — populates `robustness_weights()`

### return_se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

- `false` (default) — leaves `standard_errors()` and the hat-matrix accessors empty/NaN
- `true` — computes standard errors and hat-matrix statistics

### parallel

Enable multi-threaded execution via Rayon.

- `true` (default) — parallelizes the local regression fits across CPU cores
- `false` — forces single-threaded execution (useful for benchmarking or deterministic profiling)

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
- `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
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

### boundary_degree_fallback

Whether to reduce the polynomial degree at boundary vertices when the requested `degree` can't be fit there (e.g., not enough neighbours). Only applies when `surface_mode = "interpolation"`.

- `-1` (default) — uses the library default (enabled)
- `1` — falls back to a lower degree at boundaries
- `0` — raises an error instead of silently falling back

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

- `cv_method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `cv_k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `cv_k`: Number of folds for k-fold CV. Ignored when `cv_method = "loocv"`.
- `cv_fractions`: Candidate fractions to evaluate. Cross-validation is disabled unless this is set.
- `cv_seed`: Seed for reproducible k-fold shuffling. `0` (default) uses a random seed.

### custom_weights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `fit()` rather than the constructor.

## Result Structure

### fastloess::LoessResult

A RAII wrapper around the C result struct `fastloess_CppLoessResult`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | x values (same order as input) |
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
