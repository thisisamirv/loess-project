# API

The Rust crates provide the core implementation and high-performance extensions.

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Structs & Usage

> **StreamingLoess** and **OnlineLoess** are documented separately: [Streaming Adapter](crate::doc::api::streaming), [Online Adapter](crate::doc::api::online)

```text
use loess_rs::prelude::*;  // or: use fastLoess::prelude::*;
```

### `Loess` (Batch)

Standard in-memory smoothing.

**Constructor:**

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let builder = Loess::<f64>::new(); // Batch is default

    Ok(())
}
```

#### `fit(&x, &y)`

Fits the model to the provided `x` and `y` arrays. Returns `Result<LoessResult<T>, LoessError>`.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

    let model = Loess::new().fraction(0.5f64).build()?;
    let result = model.fit(&x, &y)?;
    println!("Fraction used: {}", result.fraction_used);
    println!("Iterations used: {:?}", result.iterations_used);

    Ok(())
}
```

```output
Fraction used: 0.5
Iterations used: Some(3)
```

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
| `missing(...)` | `missing` | `"error"` | Policy for non-finite (NaN/Inf) values in input data |
| `auto_converge(T)` | `T: Float` | disabled | Auto-convergence tolerance |
| `intervals(IntervalsBuilder<T>)` | `IntervalsBuilder<T>` | disabled | Group confidence, prediction, and bootstrap settings |
| `outputs([&str])` | iterable of names | `[]` | Select `"diagnostics"`, `"residuals"`, `"weights"`, `"gradient"`/`"derivative"`, `"se"`, `"sorted"` |
| `return_diagnostics()` | `bool` | `false` | Include diagnostics in result |
| `return_residuals()` | `bool` | `false` | Include residuals in result |
| `return_robustness_weights()` | `bool` | `false` | Include weights in result |
| `return_se()` | `bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `return_sorted()` | `bool` | `false` | Return results sorted ascending by `x` instead of in original input order |
| `degree(...)` | `degree` | `"linear"` | Polynomial degree |
| `dimensions(usize)` | `usize` | `1` | Number of predictor dimensions |
| `distance_metric(...)` | `distance_metric` | `"normalized"` | Distance metric |
| `weighted_metric_weights(Vec<T>)` | `Vec<T: Float>` | disabled | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode(...)` | `surface_mode` | `"interpolation"` | Surface computation mode |
| `cell(T)` | `T: Float` | disabled | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices(usize)` | `usize` | disabled | Number of interpolation vertices |
| `boundary_degree_fallback(bool)` | `bool` | `true` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv(CVOptions<T>)` | `CVOptions<T>` | disabled | Group method, folds, and candidate fractions via `CVBuilder` |
| `seed(...)` | `u64` | default algorithm seeds | Shared seed for CV and residual bootstrap |
| `custom_weights(Vec<T>)` | `Vec<T: Float>` | disabled | Per-observation case weights |
| `retain_model(bool)` | `bool` | `false` | Retain training data, enabling `predict()` on the result |
| `return_gradient()` | `bool` | `false` | Include the per-point local fit gradient in the result (`surface_mode = "direct"` only) |

`CVBuilder` and `IntervalsBuilder` are exported by `loess_rs::prelude`. Use `.cv(CVBuilder::new().method("kfold").k(5).fraction(vec![0.3, 0.5])).seed(42)` and `.intervals(IntervalsBuilder::new().confidence(0.90).prediction(0.95).bootstrap(200))`. CV options do not contain a seed; the outer `.seed(...)` controls both algorithms and does not enable either by itself. The old individual interval and `cv_*` setters were removed.

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

*See: [Weight Functions](crate::doc::weighting::kernels)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### robustness_method

*See: [Robustness](crate::doc::weighting::robustness)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### scaling_method

*See: [Scaling Methods](crate::doc::weighting::scaling)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### boundary_policy

*See: [Boundary Handling](crate::doc::advanced::boundary)*

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

### missing

Policy for handling non-finite (NaN/Inf) values in `x`/`y` (and, `custom_weights`):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error if any value is non-finite |
| `"drop"` | Silently remove observations (rows) where any x dimension or y is non-finite before fitting |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

### auto_converge

*See: [Robustness](crate::doc::weighting::robustness#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. Disabled by default.

### intervals: confidence

*See: [Intervals](crate::doc::guide::intervals)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). Disabled by default.

### intervals: prediction

*See: [Intervals](crate::doc::guide::intervals)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). Disabled by default.

### outputs

Select optional result components together using `.outputs([...])`. The existing
`return_*()` methods remain available and combine with grouped selections.

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let _model = Loess::<f64>::new()
        .surface_mode("direct")
        .outputs(["diagnostics", "residuals", "weights", "gradient", "se", "sorted"])
        .build()?;
    Ok(())
}
```

`"derivative"` is an alias for `"gradient"`, which requires the direct surface.
`"se"` includes hat-matrix statistics; diagnostics need it (or interval levels)
for AIC/AICc and effective degrees of freedom. Unknown names are collected and
reported together when `.build()` is called.

### return_diagnostics

Populates `LoessResult::diagnostics` with RMSE, MAE, R2, AIC/AICc, and effective degrees of freedom. `aic`/`aicc`/`effective_df` additionally require `.outputs(["se"])`, `.return_se()`, or confidence/prediction intervals to be populated, since they depend on hat-matrix statistics. `false` by default.

### return_residuals

Populates `LoessResult::residuals` (`y - fitted`). `false` by default.

### return_robustness_weights

Populates `LoessResult::robustness_weights` with the final per-point robustness weights. `false` by default.

### return_se

Computes hat-matrix statistics (`enp`, `trace_hat`, `delta1`, `delta2`, `residual_scale`, `leverage`) in addition to standard errors. `false` by default.

### return_sorted

Reorders every result field (residuals, intervals, etc.) ascending by `x`, instead of leaving them in original input order. To get both orderings, sort the default result client-side instead of calling `.fit()` twice. `false` by default.

### degree

*See: [Polynomial Degree](crate::doc::advanced::degree)*

- `"constant"` or `"0"` (degree 0)
- `"linear"` or `"1"` (default, degree 1)
- `"quadratic"` or `"2"` (degree 2)
- `"cubic"` or `"3"` (degree 3)
- `"quartic"` or `"4"` (degree 4)

### dimensions

*See: [Multivariate LOESS](crate::doc::advanced::dimensions)*

Number of predictor dimensions. `1` (default) is univariate; set to match the number of columns in a multivariate `x`.

### distance_metric

*See: [Multivariate LOESS](crate::doc::advanced::dimensions)*

- `"normalized"` (default — scales each dimension by its 10%-trimmed sample standard deviation; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` or `"minkowski:p"` for a custom exponent
- `"weighted"` plus `.weighted_metric_weights(vec![...])` (alias: `"weighted_euclidean"`)

### weighted_metric_weights

*See: [Multivariate LOESS](crate::doc::advanced::dimensions)*

Per-dimension weights, one per dimension. Only used when `distance_metric` is `"weighted"`; calling `.distance_metric("weighted")` without also calling this returns a `LoessError`.

### surface_mode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### cell

Cell size for the interpolation grid, as a fraction of the data range in `(0, 1]`. Disabled by default (uses the library default `0.2`). Only applies when `surface_mode` is `"interpolation"`.

### interpolation_vertices

Caps the maximum number of interpolation vertices, overriding the count implied by `cell`. Disabled by default (no explicit cap). Only applies when `surface_mode` is `"interpolation"`.

### boundary_degree_fallback

Whether to reduce the polynomial degree at boundary vertices when the requested `degree` can't be fit there. `true` by default. Only applies when `surface_mode` is `"interpolation"`.

### CV Options

*See: [Cross-Validation](crate::doc::guide::cross_validation)*

- `CVBuilder::new()` defaults to k-fold CV with five folds; `.method("loocv")` selects leave-one-out CV.
- `.k(...)` changes the fold count for k-fold CV and is ignored for LOOCV.
- `.fraction(vec![...])` supplies candidate fractions and produces the options passed to `.cv(...)`.
- The outer `.seed(...)` makes fold assignment and bootstrap sampling reproducible.

### custom_weights

*See: [Custom Weights](crate::doc::weighting::custom_weights)*

Per-observation case weights. Must have the same length as `y`; all values must be non-negative.

### retain_model

*See: [Predict](crate::doc::guide::predict)*

Retains the fitted model's training data, enabling `Predict::call(&result, new_x)` to evaluate the fit at out-of-sample query points not in the training set. Off by default (no extra memory/clone cost unless requested).

### return_gradient

Each local polynomial fit (degree >= linear) already computes per-dimension coefficients internally, but only the fitted value is normally kept; this exposes that per-point gradient (rate of change of the smoothed surface, `dimensions` values per point, flattened) in `LoessResult::gradient`, enabling sensitivity/rate-of-change analysis at effectively no extra computation cost. Only supported when `surface_mode` is `"direct"` — the default `"interpolation"` mode only stores value+gradient at a sparse grid of vertices, not enough to reconstruct an exact per-point gradient, so `gradient` stays `None` there. `false` by default.

## Result Structure

### `LoessResult<T>`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vec<T>` | x values (same order as input) |
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
| `gradient` | `Option<Vec<T>>` | Per-point local fit gradient, flattened (if `return_gradient()`, `surface_mode = "direct"` only) |
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

## Predict

*See: [Predict](crate::doc::guide::predict)*

### `Predict::call(&result, new_x) -> PredictOutput<T>`

Evaluates the fitted model at out-of-sample query points (flattened, `dimensions` values per point). Requires `.retain_model(true)` on the builder before `fit()`, otherwise returns `LoessError::PredictionUnavailable`.

## Example

```rust
use loess_rs::prelude::*;
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

    println!("Smoothed Y (first 5): {:?}", &result.y[..5]);

    Ok(())
}
```

```output
Smoothed Y (first 5): [0.3273755400709721, 0.3507395450049653, 0.3773907605267919, 0.40790905868327654, 0.44013801836005656]
```
