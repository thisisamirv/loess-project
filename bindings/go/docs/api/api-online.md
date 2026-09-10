---
title: "OnlineLoess API"
weight: 34
---

For real-time data: processes one `(x, y)` point at a time and returns a smoothed value immediately once enough points have been seen.

See also: [API](api.md)

![Online Adapter](../assets/diagrams/online_comparison.svg)

## `fastloess.DefaultOnlineOptions() OnlineOptions`

```go
opts := fastloess.DefaultOnlineOptions()
opts.WindowCapacity = 200
opts.MinPoints = 10
```

`OnlineOptions` embeds [`Options`](api.md) (all the same fields apply, except `CVFractions`/`CVMethod`/`CVK`/`CVSeed`, and `Parallel`, which are batch-only). `ConfidenceIntervals`/`PredictionIntervals`/`ReturnSe` require `UpdateMode = "full"`. `AddPoint` only accepts a single x coordinate: online mode does not support multivariate predictors even if `Dimensions` was set on construction. Fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `Fraction` | `float64` | `0.67` | Smoothing fraction (bandwidth) |
| `Iterations` | `int` | `3` | Number of robustifying iterations |
| `WeightFunction` | `string` | `"tricube"` | Kernel weight function |
| `RobustnessMethod` | `string` | `"bisquare"` | Robustness method |
| `ScalingMethod` | `string` | `"mad"` | Residual scaling method |
| `BoundaryPolicy` | `string` | `"extend"` | Boundary handling policy |
| `ZeroWeightFallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `Missing` | `string` | `"error"` | Policy for non-finite (NaN/Inf) values in each point |
| `AutoConverge` | `*float64` | `nil` (disabled) | Auto-convergence tolerance |
| `ReturnRobustnessWeights` | `bool` | `false` | Include `RobustnessWeight` in result |
| `Degree` | `string` | `"linear"` | Polynomial degree |
| `Dimensions` | `int` | `1` | Number of predictor dimensions |
| `DistanceMetric` | `string` | `"normalized"` | Distance metric |
| `WeightedMetricWeights` | `[]float64` | `nil` | Per-dimension weights (used when `DistanceMetric = "weighted"`) |
| `SurfaceMode` | `string` | `"interpolation"` | Surface computation mode |
| `Cell` | `*float64` | `nil` (auto) | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `InterpolationVertices` | `*int` | `nil` (auto) | Number of interpolation vertices |
| `BoundaryDegreeFallback` | `*bool` | `nil` (auto) | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `ReturnGradient` | `bool` | `false` | Include the latest point's local fit gradient in the result (`SurfaceMode = "direct"` only) |
| `ConfidenceIntervals` | `*float64` | `nil` | Confidence level for confidence intervals; requires `UpdateMode = "full"` |
| `PredictionIntervals` | `*float64` | `nil` | Confidence level for prediction intervals; requires `UpdateMode = "full"` |
| `ReturnSe` | `bool` | `false` | Include standard error in result; requires `UpdateMode = "full"` |
| `WindowCapacity` | `int` | `1000` | Maximum number of recent points retained |
| `MinPoints` | `int` | `2` | Minimum points required before output starts |
| `UpdateMode` | `string` | `"incremental"` | How the window is updated as new points arrive |

Cross-validation, `ReturnSorted`, `ReturnDiagnostics`, and `ReturnResiduals` are Batch-only (or Batch/Streaming-only) and not available here; see [API](api.md) for those.

## `fastloess.NewOnlineLoess(opts OnlineOptions) (*OnlineLoess, error)`

## `(*OnlineLoess) AddPoint(x, y float64) (res PointResult, ok bool, err error)`

Adds a single observation. `ok` is `false` while the window is still filling (fewer than `MinPoints` seen so far); once `ok` is `true`, `res` holds the smoothed value for the most recently added point. Once the window reaches `WindowCapacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `UpdateMode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

## `(*OnlineLoess) Close() error`

Releases native resources. Safe to call multiple times.

## Options

### Fraction

`Fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

### Iterations

`Iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

### WeightFunction

*See: [Weight Functions](../weighting/kernels.md)*

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"` (alias: `"boxcar"`)
- `"biweight"` (alias: `"bisquare"`)
- `"triangle"` (alias: `"triangular"`)
- `"cosine"`

### RobustnessMethod

*See: [Robustness](../weighting/robustness.md)*

- `"bisquare"` (default; alias: `"biweight"`)
- `"huber"`
- `"talwar"`

### ScalingMethod

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### BoundaryPolicy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### ZeroWeightFallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### Missing

Policy for handling a non-finite (NaN/Inf) `x` or `y` value passed to `AddPoint` (overrides the row-dropping behavior described in [API](api.md) since Online processes one point at a time):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error |
| `"drop"` | Silently ignore the point — `AddPoint` returns `ok == false` instead of adding it to the window |

### AutoConverge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `nil` (default) disables early stopping.

### ReturnRobustnessWeights

Populate `PointResult.RobustnessWeight` with the robustness weight for the latest point (from the last robustness iteration).

- `false` (default) — leaves `RobustnessWeight` as `NaN`
- `true` — populates `RobustnessWeight`

### Degree

*See: [Polynomial Degree](../advanced/degree.md)*

- `"constant"` (degree 0)
- `"linear"` (default, degree 1)
- `"quadratic"` (degree 2)
- `"cubic"` (degree 3)
- `"quartic"` (degree 4)

### Dimensions

*See: [Multivariate LOESS](../advanced/dimensions.md)*

Number of predictor dimensions. Set to match the number of columns in a multivariate `x` array. `1` (default) is univariate.

### DistanceMetric

*See: [Multivariate LOESS](../advanced/dimensions.md)*

- `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
- `"weighted"` plus `WeightedMetricWeights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### WeightedMetricWeights

*See: [Multivariate LOESS](../advanced/dimensions.md)*

Per-dimension weights, one per dimension declared in `Dimensions`. Only used when `DistanceMetric = "weighted"`; setting `DistanceMetric = "weighted"` without providing this raises an error.

- `nil` (default) — has no effect unless `DistanceMetric = "weighted"` is set
- A non-empty `[]float64` of per-dimension weights, required when `DistanceMetric = "weighted"`

### SurfaceMode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

### Cell

Cell size for the interpolation grid, as a fraction of the data range. Smaller values place more vertices (denser grid), improving accuracy at the cost of speed. Only applies when `SurfaceMode` is `"interpolation"`.

- `nil` (default) — uses the library default (`0.2`)
- Any value in `(0, 1]`

### InterpolationVertices

Caps the maximum number of interpolation vertices, overriding the count implied by `Cell`. Only applies when `SurfaceMode` is `"interpolation"`.

- `nil` (default) — uses the library default (no explicit cap)
- Any integer `>= 1`

### BoundaryDegreeFallback

Whether to reduce the polynomial degree at boundary vertices when the requested `Degree` can't be fit there (e.g., not enough neighbours). Only applies when `SurfaceMode` is `"interpolation"`.

- `nil` (default) — uses the library default (enabled)
- `true` — falls back to a lower degree at boundaries
- `false` — raises an error instead of silently falling back

### ReturnGradient

Each local polynomial fit (degree >= linear) already computes per-dimension coefficients internally; this exposes the latest point's gradient (`Dimensions` values) in `PointResult.Gradient` at effectively no extra computation cost. Only supported when `SurfaceMode` is `"direct"` — returns an error instead of silently leaving `Gradient` as `nil` if requested under the default `"interpolation"` mode. `false` by default.

### ConfidenceIntervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`). Only computed under `UpdateMode = "full"` — returns an error at construction if set (or `ReturnSe`/`PredictionIntervals` is set) while `UpdateMode` is left at its default `"incremental"`, since incremental updates never compute standard errors. `nil` (default) disables confidence intervals.

### PredictionIntervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`). Same `UpdateMode = "full"` requirement as `ConfidenceIntervals`. `nil` (default) disables prediction intervals.

### ReturnSe

Include the standard error for the latest point in the result (`PointResult.StandardError`). Same `UpdateMode = "full"` requirement as `ConfidenceIntervals`.

- `false` (default) — leaves `StandardError` as `NaN`
- `true` — populates `StandardError`

### WindowCapacity

Maximum number of most recent points kept in the sliding window; older points are discarded as new ones arrive. Each `AddPoint` call costs O(`WindowCapacity`) rather than growing with total history.

### MinPoints

Minimum number of points required before `AddPoint` starts returning `ok == true`.

### UpdateMode

*See: [Execution Modes](../guide/adapter-choice.md)*

| Mode | Alias | Behavior | Speed |
| --- | --- | --- | --- |
| `"incremental"` (default) | `"single"` | Update only affected fits | Faster |
| `"full"` | `"resmooth"` | Recompute entire window | More accurate |

See [API](api.md) for the descriptions of all inherited fields not covered above.

## `PointResult` fields

| Field | Type | Notes |
| --- | --- | --- |
| `Y` | `float64` | Smoothed value. |
| `StandardError` | `float64` | Standard error, if `ReturnSe`/`ConfidenceIntervals`/`PredictionIntervals` was set and `UpdateMode = "full"` (`NaN` otherwise). |
| `Residual` | `float64` | Residual y − smoothed; always present (there is no `ReturnResiduals` option for Online). |
| `RobustnessWeight` | `float64` | Robustness weight, if `ReturnRobustnessWeights` was set (`NaN` otherwise). |
| `IterationsUsed` | `int` | Robustness iterations performed (`-1` if not applicable). |
| `ConfidenceLower` / `ConfidenceUpper` | `float64` | Confidence interval bounds, if `ConfidenceIntervals` was set and `UpdateMode = "full"` (`NaN` otherwise). |
| `PredictionLower` / `PredictionUpper` | `float64` | Prediction interval bounds, if `PredictionIntervals` was set and `UpdateMode = "full"` (`NaN` otherwise). |
| `Gradient` | `[]float64` | Latest point's local fit gradient (`Dimensions` values), if `ReturnGradient` was set (`SurfaceMode = "direct"` only). |

There is no `Diagnostics` type or `ReturnDiagnostics` option for `OnlineLoess`: `PointResult` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.

## Example

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/loess-project/bindings/go/fastloess/v2"
)

func main() {
 const n = 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastloess.DefaultOnlineOptions()
 opts.Fraction = 0.5
 opts.WindowCapacity = 50
 opts.MinPoints = 3

 model, err := fastloess.NewOnlineLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 _, ok1, err := model.AddPoint(x[0], y[0])
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println(ok1)

 _, ok2, err := model.AddPoint(x[1], y[1])
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println(ok2)

 res, ok3, err := model.AddPoint(x[2], y[2])
 if err != nil {
  log.Fatal(err)
 }
 if ok3 {
  fmt.Println(res.Y)
 }
}
```

```output
false
false
0.22659245357374927
```
