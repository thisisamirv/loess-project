# fastLoess & loess-rs Rust API Reference

The Rust crates provide the core implementation and high-performance extensions.

## Structs & Usage

Both crates expose the same three entry types via their `prelude`: `Loess` for batch mode, `StreamingLoess` for chunked processing, and `OnlineLoess` for sliding-window updates.

> **StreamingLoess** and **OnlineLoess** are documented separately: [rust-streaming.md](rust-streaming.md), [rust-online.md](rust-online.md)

```text
use fastLoess::prelude::*;  // or: use loess_rs::prelude::*;
```

### `Loess` (Batch)

Standard in-memory smoothing.

**Constructor:**

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let builder = Loess::new(); // Batch is default

    Ok(())
}
```

**Methods:**

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new().fraction(0.5f64).build()?;
    let result = model.fit(&x, &y)?;
    println!("{}", result.fraction_used);  // 0.5
    println!("{:?}", result.iterations_used);  // Some(3)

    Ok(())
}
```

* Fits the model to the provided `x` and `y` arrays.
* Returns `Result<LoessResult<T>, LoessError>`.

See [rust-streaming.md](rust-streaming.md) for the `StreamingLoess` struct.

See [rust-online.md](rust-online.md) for the `OnlineLoess` struct.

## Builder Configuration

These chained methods configure the builder. They correspond to the "Options Structures" in other bindings.

### Loess Options

| Method | Argument Type | Default | Description |
| --- | --- | --- | --- |
| `fraction(T)` | `T: Float` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations(usize)` | `usize` | `3` | Number of robustifying iterations |
| `weight_function(...)` | `weight_function` | `"tricube"` | Kernel weight function |
| `robustness_method(...)` | `robustness_method` | `"bisquare"` | Robustness method |
| `scaling_method(...)` | `scaling_method` | `"mad"` | Residual scaling method |
| `boundary_policy(...)` | `boundary_policy` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback(...)` | `zero_weight_fallback` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge(T)` | `T: Float` | disabled | Auto-convergence tolerance |
| `custom_weights(Vec<T>)` | `Vec<T: Float>` | disabled | Per-observation case weights (Batch only) |
| `confidence_intervals(T)` | `T: Float` | disabled | Confidence level (e.g., 0.95) |
| `prediction_intervals(T)` | `T: Float` | disabled | Prediction level (e.g., 0.95) |
| `return_diagnostics()` | `bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals()` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights()` | `bool` | `false` | Include robustness weights in result |
| `return_se()` | `bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel(bool)` | `bool` | `true` | Enable parallel execution |
| `degree(...)` | `degree` | `"linear"` | Polynomial degree |
| `dimensions(usize)` | `usize` | `1` | Number of predictor dimensions |
| `distance_metric(...)` | `distance_metric` | `"normalized"` | Distance metric |
| `weighted_metric_weights(Vec<T>)` | `Vec<T: Float>` | disabled | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode(...)` | `surface_mode` | `"interpolation"` | Surface computation mode |
| `cell(T)` | `T: Float` | disabled | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices(usize)` | `usize` | disabled | Number of interpolation vertices |
| `boundary_degree_fallback(bool)` | `bool` | `true` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_method(...)` | `&str` | disabled | Cross-validation method |
| `cv_k(...)` | `usize` | disabled | Number of folds for K-fold cross-validation |
| `cv_fractions(...)` | `Vec<f64>` | disabled | Candidate fractions to evaluate during cross-validation |
| `cv_seed(...)` | `u64` | disabled | Random seed for reproducible fold assignments |

See [rust-streaming.md](rust-streaming.md) for Streaming Options.

See [rust-online.md](rust-online.md) for Online Options.

## Result Structure

### `LoessResult<T>`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vec<T>` | Sorted x values |
| `y` | `Vec<T>` | Smoothed y values |
| `fraction_used` | `T` | Fraction used (set or selected by CV) |
| `iterations_used` | `Option<usize>` | Robustness iterations actually performed |
| `standard_errors` | `Option<Vec<T>>` | Per-point SE (if `return_se()`) |
| `confidence_lower` | `Option<Vec<T>>` | Lower confidence bounds |
| `confidence_upper` | `Option<Vec<T>>` | Upper confidence bounds |
| `prediction_lower` | `Option<Vec<T>>` | Lower prediction bounds |
| `prediction_upper` | `Option<Vec<T>>` | Upper prediction bounds |
| `residuals` | `Option<Vec<T>>` | Residuals (if `return_residuals()`) |
| `robustness_weights` | `Option<Vec<T>>` | Robustness weights (if `return_robustness_weights()`) |
| `cv_scores` | `Option<Vec<T>>` | CV score per tested fraction |
| `diagnostics` | `Option<Diagnostics<T>>` | Fit metrics (if `return_diagnostics()`) |
| `enp` | `Option<T>` | Equivalent number of parameters (if `return_se()`) |
| `trace_hat` | `Option<T>` | Trace of hat matrix (if `return_se()`) |
| `delta1` | `Option<T>` | First delta statistic (if `return_se()`) |
| `delta2` | `Option<T>` | Second delta statistic (if `return_se()`) |
| `residual_scale` | `Option<T>` | Residual scale estimate (if `return_se()`) |
| `leverage` | `Option<Vec<T>>` | Per-point hat-matrix diagonal (if `return_se()`) |
| `dimensions` | `usize` | Number of predictor dimensions |
| `polynomial_degree` | `PolynomialDegree` (internal) | Polynomial degree used; implements `Display` (e.g. `"linear"`) |
| `distance_metric` | `DistanceMetric<T>` (internal) | Distance metric used; implements `Display` (e.g. `"normalized"`) |

### `Diagnostics<T>`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `T` | Root Mean Squared Error |
| `mae` | `T` | Mean Absolute Error |
| `r_squared` | `T` | R-squared |
| `residual_sd` | `T` | Residual standard deviation |
| `effective_df` | `Option<T>` | Effective degrees of freedom |
| `aic` | `Option<T>` | AIC |
| `aicc` | `Option<T>` | AICc |

## Options

### weight_function

*See: [Weight Functions](../user-guide/kernels.md)*

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"` (alias: `"boxcar"`)
* `"biweight"` (alias: `"bisquare"`)
* `"triangle"` (alias: `"triangular"`)
* `"cosine"`

### robustness_method

*See: [Robustness](../user-guide/robustness.md)*

* `"bisquare"` (default; alias: `"biweight"`)
* `"huber"`
* `"talwar"`

### boundary_policy

*See: [Boundary Handling](../user-guide/boundary.md)*

* `"extend"` (default; alias: `"pad"`)
* `"reflect"` (alias: `"mirror"`)
* `"zero"`
* `"noboundary"` (alias: `"none"`)

### scaling_method

*See: [Scaling Methods](../user-guide/scaling.md)*

* `"mad"` (default; alias: `"median_absolute_deviation"`)
* `"mar"` (alias: `"median_absolute_residual"`)
* `"mean"` (alias: `"mean_absolute_residual"`)

### zero_weight_fallback

*See: [Parameters](../user-guide/parameters.md)*

* `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`)
* `"return_original"` (alias: `"original"`)
* `"return_none"` (alias: `"none"`)

### distance_metric

*See: [Multivariate LOESS](../user-guide/dimensions.md)*

* `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
* `"euclidean"` (alias: `"euclid"`)
* `"manhattan"` (alias: `"l1"`)
* `"chebyshev"` (alias: `"linf"`)
* `"minkowski"` or `"minkowski:p"` for a custom exponent
* `"weighted"` plus `.weighted_metric_weights(vec![...])` (alias: `"weighted_euclidean"`)

### degree

*See: [Polynomial Degree](../user-guide/degree.md)*

* `"constant"` or `"0"` (degree 0)
* `"linear"` or `"1"` (default, degree 1)
* `"quadratic"` or `"2"` (degree 2)
* `"cubic"` or `"3"` (degree 3)
* `"quartic"` or `"4"` (degree 4)

### surface_mode

*See: [Parameters](../user-guide/parameters.md)*

* `"interpolation"` (default)
* `"direct"` (fits every point exactly; slower but more accurate)

### merge_strategy

See [rust-streaming.md](rust-streaming.md).

### update_mode

See [rust-online.md](rust-online.md).

## Example

```rust
use fastLoess::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new()
        .fraction(0.5)
        .iterations(3)
        .build()?;

    let result = model.fit(&x, &y)?;

    println!("Smoothed Y: {:?}", result.y);

    Ok(())
}
```
