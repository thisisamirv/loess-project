# FastLOESS Julia API Reference

The Julia bindings provide a modern interface to the core Rust library, mirroring the Rust API structure.

## Classes

### `Loess`

The `Loess` struct allows configuring the LOESS parameters once and fitting multiple datasets using those parameters.

**Constructor:**

```julia
using FastLOESS

model = Loess(fraction=0.5)
```

**Methods:**

```julia
using FastLOESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

model = Loess(fraction=0.5)
result = fit(model, x, y)
println(result.fraction_used)
# 0.5
```

* Fits the model to the provided `x` and `y` data vectors.
* `custom_weights`: Optional per-observation weights. All values must be ≥ 0 and length must match `x`. Batch only.
* Returns a `LoessResult` struct containing the smoothed values and optional diagnostics.

### `StreamingLoess`

The `StreamingLoess` struct processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```julia
using FastLOESS

stream = StreamingLoess(chunk_size=50, overlap=10)
```

**Methods:**

```julia
using FastLOESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

stream = StreamingLoess(fraction=0.5, chunk_size=50, overlap=10)
partial_result = process_chunk(stream, x[1:50], y[1:50])
println(partial_result.fraction_used)
# 0.5
```

* Processes a chunk of data. Returns partial results.

```julia
using FastLOESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

stream = StreamingLoess(fraction=0.5, chunk_size=50, overlap=10)
process_chunk(stream, x[1:50], y[1:50])
process_chunk(stream, x[51:end], y[51:end])
final_result = finalize(stream)
println(final_result.fraction_used)
# 0.5
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLoess`

The `OnlineLoess` struct updates the model incrementally with new data points.

**Constructor:**

```julia
using FastLOESS

online = OnlineLoess(fraction=0.5, window_capacity=50)
```

**Methods:**

```julia
using FastLOESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

online = OnlineLoess(fraction=0.5, window_capacity=50)

# Returns nothing until min_points (3) are reached
result = add_point(online, x[1], y[1])  # nothing
result = add_point(online, x[2], y[2])  # nothing

# Returns OnlineOutput once enough points are available
result = add_point(online, x[3], y[3])
println(result.y)
# 0.22659245357374927
```

* Adds a single point to the sliding window. Returns `nothing` while the window is still filling (fewer than `min_points` seen), and an `OnlineOutput` once smoothing begins.

## Options Structures

### `LoessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `Float64` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `Int` | `3` | Number of robustifying iterations |
| `weight_function` | `String` | `"tricube"` | Kernel weight function |
| `robustness_method` | `String` | `"bisquare"` | Robustness method |
| `scaling_method` | `String` | `"mad"` | Residual scaling method |
| `boundary_policy` | `String` | `"extend"` | Boundary handling policy |
| `zero_weight_fallback` | `String` | `"use_local_mean"` | Zero-weight handling |
| `auto_converge` | `Float64` | `NaN` | Auto-convergence tolerance (NaN to disable) |
| `custom_weights` | `Union{Vector{Float64}, Nothing}` | `nothing` | Per-observation case weights — passed to `fit()`, not the constructor (Batch only) |
| `confidence_intervals` | `Float64` | `NaN` | Confidence level (e.g., 0.95; NaN to disable) |
| `prediction_intervals` | `Float64` | `NaN` | Prediction level (e.g., 0.95; NaN to disable) |
| `return_diagnostics` | `Bool` | `false` | Compute RMSE, MAE, R², AIC |
| `return_residuals` | `Bool` | `false` | Include residuals in result |
| `return_robustness_weights` | `Bool` | `false` | Include robustness weights in result |
| `return_se` | `Bool` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `Bool` | `true` | Enable parallel execution |
| `degree` | `String` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `Int` | `1` | Number of predictor dimensions |
| `distance_metric` | `String` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `weighted_metric_weights` | `Union{Vector{Float64}, Nothing}` | `nothing` | Per-dimension weights (used when `distance_metric = "weighted"`) |
| `surface_mode` | `String` | `"interpolation"` | Surface computation mode |
| `cell` | `Union{Float64, Nothing}` | `nothing` | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `interpolation_vertices` | `Union{Int, Nothing}` | `nothing` | Number of interpolation vertices |
| `boundary_degree_fallback` | `Union{Bool, Nothing}` | `nothing` | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `cv_method` | `String` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) (Batch only) |
| `cv_k` | `Int` | `5` | Number of folds for k-fold CV (Batch only) |
| `cv_fractions` | `Vector{Float64}` | `Float64[]` | Fractions to test for cross-validation (Batch only) |
| `cv_seed` | `Union{Int, Nothing}` | `nothing` | Random seed for cross-validation shuffling (Batch only) |

See [julia-streaming.md](julia-streaming.md) for `StreamingOptions`.

See [julia-online.md](julia-online.md) for `OnlineOptions`.

## Result Structure

See [julia-online.md](julia-online.md) for `OnlineOutput`.

### `LoessResult`

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Vector{Float64}` | Sorted x values |
| `y` | `Vector{Float64}` | Smoothed y values |
| `fraction_used` | `Float64` | Fraction used (set or selected by CV) |
| `iterations_used` | `Int` | Robustness iterations actually performed (-1 = N/A) |
| `standard_errors` | `Union{Vector{Float64}, Nothing}` | Per-point SE (if `return_se`) |
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
| `effective_df` | `Float64` | Effective degrees of freedom (NaN if not computed) |
| `aic` | `Float64` | AIC (NaN if not computed) |
| `aicc` | `Float64` | AICc (NaN if not computed) |

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

### degree

*See: [Polynomial Degree](../user-guide/degree.md)*

* `"constant"` or `"0"` (degree 0)
* `"linear"` or `"1"` (default, degree 1)
* `"quadratic"` or `"2"` (degree 2)
* `"cubic"` or `"3"` (degree 3)
* `"quartic"` or `"4"` (degree 4)

### distance_metric

*See: [Multivariate LOESS](../user-guide/dimensions.md)*

* `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
* `"euclidean"` (alias: `"euclid"`)
* `"manhattan"` (alias: `"l1"`)
* `"chebyshev"` (alias: `"linf"`)
* `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
* `"weighted"` plus `weighted_metric_weights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### surface_mode

*See: [Parameters](../user-guide/parameters.md)*

* `"interpolation"` (default — faster, uses a spatial grid)
* `"direct"` (fits every point exactly; slower but more accurate)

### merge_strategy

See [julia-streaming.md](julia-streaming.md).

### update_mode

See [julia-online.md](julia-online.md).

## Example

```julia
using FastLOESS

x = collect(range(0, 2π, length=100))
y = sin.(x) .+ 0.1

# Configure model
model = Loess(fraction=0.5, iterations=3)

# Fit data (throws on error)
result = fit(model, x, y)

println("Smoothed Y: ", result.y)
```
