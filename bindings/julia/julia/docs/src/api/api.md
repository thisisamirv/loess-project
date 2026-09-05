# Batch Adapter

The Julia bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

> **StreamingLoess** and **OnlineLoess** are documented separately: [Streaming Adapter](api-streaming.md), [Online Adapter](api-online.md)

## When to Use Batch Adapter

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

## Classes

### `Loess`

The `Loess` type allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```@example batch
using FastLOESS

model = Loess(; fraction=0.5, iterations=3)
println(typeof(model))
```

- Keyword arguments configure the `Loess` model; see [Options Structures](#options-structures) below.

#### `fit(model, x, y)`

Fits the model to the provided `x` and `y` vectors. `custom_weights`: Optional `Vector{Float64}` of per-observation weights. All values must be ≥ 0 and length must match `x`. Returns a `LoessResult` containing the smoothed values and optional diagnostics.

```@example batch
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

result = fit(model, x, y)
println("First smoothed value: ", result.y[1])
```

## Options Structures

### `Loess` keyword arguments

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `weight_function` | `String` | `"tricube"` | Weight function name |
| `robustness_method` | `String` | `"bisquare"` | Robustness method name |
| `scaling_method` | `String` | `"mad"` | Residual scaling method |
| `boundary_policy` | `String` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling strategy |
| `missing` | `String` | `"error"` | Policy for non-finite (NaN/Inf) values in input data |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance |
| `confidence_intervals` | `Float64` | `NaN` | Confidence level (e.g., 0.95) |
| `prediction_intervals` | `Float64` | `NaN` | Prediction level (e.g., 0.95) |
| `return_diagnostics` | `Bool` | `false` | Include diagnostics in result |
| `return_residuals` | `Bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `Bool` | `false` | Include weights in result |
| `return_se` | `Bool` | `false` | Return standard errors |
| `return_sorted` | `Bool` | `false` | Return results sorted ascending by `x` instead of in original input order |
| `parallel` | `Bool` | `true` | Enable parallel execution |
| `degree` | `String` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `Int` | `1` | Number of predictor dimensions |
| `distance_metric` | `String` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `Union{Vector{Float64}, Nothing}` | `nothing` | Per-dimension weights (used when `distance_metric="weighted"`) |
| `surface_mode` | `String` | `"interpolation"` | Surface computation mode |
| `cell` | `Union{Float64, Nothing}` | `nothing` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `Union{Int, Nothing}` | `nothing` | Number of interpolation vertices |
| `boundary_degree_fallback` | `Union{Bool, Nothing}` | `nothing` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_method` | `String` | `"kfold"` | CV method (`"kfold"` fast or `"loocv"` slow, exhaustive) |
| `cv_k` | `Int` | `5` | Number of folds for k-fold CV |
| `cv_fractions` | `Vector{Float64}` | `Float64[]` | Fractions to test for cross-validation |
| `cv_seed` | `Union{Int, Nothing}` | `nothing` | Random seed for cross-validation shuffling |
| `custom_weights` | `Vector{Float64}` | `nothing` | Per-observation case weights — passed to `fit`, not the constructor |

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

### missing

Policy for handling non-finite (NaN/Inf) values in `x`/`y` (and `custom_weights`):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Raise an error if any value is non-finite |
| `"drop"` | Silently remove observations (rows) where any x dimension or y is non-finite before fitting |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

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

*See: [`Diagnostics`](#diagnostics)*

Include a `Diagnostics` object (RMSE, MAE, R2, AIC/AICc, effective degrees of freedom) in the result. AIC/AICc/`effective_df` additionally require `return_se=true` (or confidence/prediction intervals) to be populated, since they depend on hat-matrix statistics.

- `false` (default) — leaves `result.diagnostics` as `nothing`
- `true` — populates `result.diagnostics`

### return_residuals

Include per-point residuals (`y - fitted`) in the result.

- `false` (default) — leaves `result.residuals` as `nothing`
- `true` — populates `result.residuals`

### return_robustness_weights

Include the final per-point robustness weights (from the last robustness iteration) in the result.

- `false` (default) — leaves `result.robustness_weights` as `nothing`
- `true` — populates `result.robustness_weights`

### return_se

*See: [Intervals](../guide/intervals.md#standard-errors)*

Computes hat-matrix statistics (effective degrees of freedom, leverage, delta1/delta2) in addition to standard errors.

### return_sorted

When set to `true`, it reorders every result field (residuals, intervals, etc.) by `x` in an ascending manner, instead of in original input order.
To get both orderings, sort the default result client-side (e.g. `sortperm(result.x)`) instead of calling `fit` twice.

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
- `"minkowski"` (use `"minkowski:p"` string for custom exponent, e.g. `"minkowski:3"`)
- `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### weighted_metric_weights

*See: [Multivariate LOESS](../advanced/dimensions.md)*

Per-dimension weights, one per dimension declared in `dimensions`. Only used when `distance_metric="weighted"`; setting `distance_metric="weighted"` without providing this raises an error.

- `nothing` (default) — has no effect unless `distance_metric="weighted"` is set
- A `Vector{Float64}` of per-dimension weights, required when `distance_metric="weighted"`

### surface_mode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### cell

Cell size for the interpolation grid, as a fraction of the data range. Smaller values place more vertices (denser grid), improving accuracy at the cost of speed. Only applies when `surface_mode="interpolation"`.

- `nothing` (default) — uses the library default (`0.2`)
- Any `Float64` in `(0, 1]`

### interpolation_vertices

Caps the maximum number of interpolation vertices, overriding the count implied by `cell`. Only applies when `surface_mode="interpolation"`.

- `nothing` (default) — uses the library default (no explicit cap)
- Any integer `>= 1`

### boundary_degree_fallback

Whether to reduce the polynomial degree at boundary vertices when the requested `degree` can't be fit there (e.g., not enough neighbours). Only applies when `surface_mode="interpolation"`.

- `nothing` (default) — uses the library default (enabled)
- `true` — falls back to a lower degree at boundaries
- `false` — raises an error instead of silently falling back

### CV Options

*See: [Cross-Validation](../guide/cross-validation.md)*

- `cv_method`: `"kfold"` (default) — fast, evaluates each candidate fraction over `cv_k` folds; `"loocv"` — slow, exhaustive leave-one-out cross-validation
- `cv_k`: Number of folds for k-fold CV. Ignored when `cv_method="loocv"`.
- `cv_fractions`: Candidate fractions to evaluate. Cross-validation is disabled unless this is set.
- `cv_seed`: Seed for reproducible k-fold shuffling. `nothing` (default) uses a random seed.

### custom_weights

*See: [Custom Weights](../weighting/custom-weights.md)*

Per-observation weights, passed to `fit` rather than the constructor.

## Result Structure

### `LoessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vector{Float64}` | x values (same order as input) |
| `y` | `Vector{Float64}` | Smoothed y values |
| `fraction_used` | `Float64` | Fraction used (set or selected by CV) |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations actually performed |
| `standard_errors` | `Union{Vector{Float64}, Nothing}` | Per-point standard errors |
| `confidence_lower` | `Union{Vector{Float64}, Nothing}` | Lower confidence bounds |
| `confidence_upper` | `Union{Vector{Float64}, Nothing}` | Upper confidence bounds |
| `prediction_lower` | `Union{Vector{Float64}, Nothing}` | Lower prediction bounds |
| `prediction_upper` | `Union{Vector{Float64}, Nothing}` | Upper prediction bounds |
| `residuals` | `Union{Vector{Float64}, Nothing}` | Residuals (if `return_residuals`) |
| `robustness_weights` | `Union{Vector{Float64}, Nothing}` | Robustness weights (if `return_robustness_weights`) |
| `cv_scores` | `Union{Vector{Float64}, Nothing}` | CV score per tested fraction |
| `diagnostics` | `Union{Diagnostics, Nothing}` | Fit metrics (if `return_diagnostics`) |
| `enp` | `Union{Float64, Nothing}` | Equivalent number of parameters (if `return_se`) |
| `trace_hat` | `Union{Float64, Nothing}` | Trace of hat matrix (if `return_se`) |
| `delta1` | `Union{Float64, Nothing}` | First delta statistic (if `return_se`) |
| `delta2` | `Union{Float64, Nothing}` | Second delta statistic (if `return_se`) |
| `residual_scale` | `Union{Float64, Nothing}` | Residual scale estimate (if `return_se`) |
| `leverage` | `Union{Vector{Float64}, Nothing}` | Per-point hat-matrix diagonal (if `return_se`) |
| `dimensions` | `Int` | Number of predictor dimensions |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `Float64` | Root Mean Squared Error |
| `mae` | `Float64` | Mean Absolute Error |
| `r_squared` | `Float64` | R-squared |
| `residual_sd` | `Float64` | Residual standard deviation |
| `effective_df` | `Union{Float64, Nothing}` | Effective degrees of freedom |
| `aic` | `Union{Float64, Nothing}` | AIC |
| `aicc` | `Union{Float64, Nothing}` | AICc |

## Example

```@example batch
using FastLOESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

model = Loess(;
    fraction=0.5,
    iterations=3,
    confidence_intervals=0.95,
    prediction_intervals=0.95,
    return_diagnostics=true,
    parallel=true
)
result = fit(model, x, y)
println("First smoothed value: ", result.y[1])
```

---
