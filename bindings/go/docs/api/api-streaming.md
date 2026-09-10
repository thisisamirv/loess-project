---
title: "StreamingLoess API"
weight: 32
---

For datasets that don't fit in memory or arrive in chunks. Processes data incrementally, merging overlapping regions between chunks.

See also: [API](api.md)

## `fastloess.DefaultStreamingOptions() StreamingOptions`

```go
opts := fastloess.DefaultStreamingOptions()
opts.ChunkSize = 2000
opts.Overlap = 200
```

`StreamingOptions` embeds [`Options`](api.md) (all the same fields apply, except `CVFractions`/`CVMethod`/`CVK`/`CVSeed`, which are batch-only), plus:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `Fraction` | `float64` | `0.67` | Smoothing fraction (bandwidth) |
| `Iterations` | `int` | `3` | Number of robustifying iterations |
| `WeightFunction` | `string` | `"tricube"` | Kernel weight function |
| `RobustnessMethod` | `string` | `"bisquare"` | Robustness method |
| `ScalingMethod` | `string` | `"mad"` | Residual scaling method |
| `BoundaryPolicy` | `string` | `"extend"` | Boundary handling policy |
| `ZeroWeightFallback` | `string` | `"use_local_mean"` | Zero-weight handling |
| `Missing` | `string` | `"error"` | Policy for non-finite (NaN/Inf) values in each chunk |
| `AutoConverge` | `*float64` | `nil` (disabled) | Auto-convergence tolerance |
| `ReturnDiagnostics` | `bool` | `false` | Compute RMSE, MAE, R2 |
| `ReturnResiduals` | `bool` | `false` | Include residuals in result |
| `ReturnRobustnessWeights` | `bool` | `false` | Include weights in result |
| `Degree` | `string` | `"linear"` | Polynomial degree |
| `Dimensions` | `int` | `1` | Number of predictor dimensions |
| `DistanceMetric` | `string` | `"normalized"` | Distance metric |
| `WeightedMetricWeights` | `[]float64` | `nil` | Per-dimension weights (used when `DistanceMetric = "weighted"`) |
| `SurfaceMode` | `string` | `"interpolation"` | Surface computation mode |
| `Cell` | `*float64` | `nil` (auto) | Cell size for interpolation grid (smaller → more vertices, higher accuracy) |
| `InterpolationVertices` | `*int` | `nil` (auto) | Number of interpolation vertices |
| `BoundaryDegreeFallback` | `*bool` | `nil` (auto) | Fall back to lower polynomial degree at boundaries when higher degrees fail |
| `ReturnGradient` | `bool` | `false` | Include the per-point local fit gradient in the result (`SurfaceMode = "direct"` only) |
| `ConfidenceIntervals` | `*float64` | `nil` | Confidence level for confidence intervals, computed per chunk and merged across overlap boundaries via `MergeStrategy` |
| `PredictionIntervals` | `*float64` | `nil` | Confidence level for prediction intervals; same per-chunk computation and overlap-merging as `ConfidenceIntervals` |
| `ReturnSe` | `bool` | `false` | Include standard errors in result, computed per chunk and merged across overlap boundaries via `MergeStrategy` |
| `ChunkSize` | `int` | `5000` | Number of points processed per chunk. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying. |
| `Overlap` | `int` | `ChunkSize / 10` | Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `MergeStrategy`. A good starting point is 10–20% of `ChunkSize`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice. Negative (the `DefaultStreamingOptions()` value, `-1`) means "use the library default", clamped to `[1, ChunkSize - 10]`. |
| `MergeStrategy` | `string` | `"weighted_average"` | How overlapping chunk results are combined. |

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

*See also: [Merge Strategies](../advanced/merge.md)*

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

Confidence/prediction intervals and standard errors are computed per chunk and merged across overlap boundaries via `MergeStrategy`, same as `Y`. Cross-validation and `ReturnSorted` are Batch-only and not available here; see [API](api.md) for those.

## `fastloess.NewStreamingLoess(opts StreamingOptions) (*StreamingLoess, error)`

## `(*StreamingLoess) ProcessChunk(x, y []float64) (Result, error)`

Fits and returns the result for one chunk. Each chunk is fit together with the trailing `Overlap` points buffered from the previous call, then only the points that are fully resolved are returned — the tail of the chunk (the next `Overlap` points) is held back internally, since it will be refit once the following chunk arrives and its estimate reconciled via `MergeStrategy`. This is what lets the adapter process a dataset far larger than memory allows, one bounded-size chunk at a time, without ever materializing the whole dataset at once. For multivariate input (`Dimensions > 1`), `x` is flattened row-major. Call repeatedly as chunks arrive.

## `(*StreamingLoess) Finalize() (Result, error)`

Flushes the overlap points still buffered from the last `ProcessChunk` call. Because each call withholds its tail until the next chunk arrives to resolve it, the final chunk's tail would never be emitted otherwise — always call `Finalize` once after the last chunk to retrieve it.

## `(*StreamingLoess) Close() error`

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

Policy for handling non-finite (NaN/Inf) values within each chunk:

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error if any value in the chunk is non-finite |
| `"drop"` | Silently remove rows where any x dimension or y is non-finite before merging the chunk with the overlap buffer |

**Note:** A length mismatch between `x` and `y` always errors, even under `"drop"`.

### AutoConverge

*See: [Robustness](../weighting/robustness.md#auto-convergence)*

Convergence tolerance for early stopping of robustness iterations. `nil` (default) disables early stopping.

### ReturnDiagnostics

Populate `Result.Diagnostics` with RMSE, MAE, R², and residual_sd. `EffectiveDF`/`AIC`/`AICc` require standard errors, which are Batch-only, so they're always unset here.

- `false` (default) — leaves `Result.Diagnostics` as `nil`
- `true` — populates `Result.Diagnostics`

### ReturnResiduals

Populate `Result.Residuals` (`y - fitted`).

- `false` (default) — leaves `Result.Residuals` as `nil`
- `true` — populates `Result.Residuals`

### ReturnRobustnessWeights

Populate `Result.RobustnessWeights` with the final per-point robustness weights (from the last robustness iteration).

- `false` (default) — leaves `Result.RobustnessWeights` as `nil`
- `true` — populates `Result.RobustnessWeights`

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

Each local polynomial fit (degree >= linear) already computes per-dimension coefficients internally; this exposes the per-point gradient (`Dimensions` values per point, flattened) in `Result.Gradient` at effectively no extra computation cost. Only supported when `SurfaceMode` is `"direct"` — returns an error instead of silently leaving `Gradient` as `nil` if requested under the default `"interpolation"` mode. `false` by default. Gradient values in the overlap region are merged across chunk boundaries the same way `Y` is, via `MergeStrategy`.

### ConfidenceIntervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the confidence interval around the mean response (e.g. `0.95`), computed per chunk and merged across overlap boundaries the same way `Y` is, via `MergeStrategy`. `nil` (default) disables confidence intervals.

### PredictionIntervals

*See: [Intervals](../guide/intervals.md)*

Confidence level for the prediction interval for new observations (e.g. `0.95`); same per-chunk computation and overlap-merging as `ConfidenceIntervals`. `nil` (default) disables prediction intervals.

### ReturnSe

Include standard errors in the result (`Result.StandardErrors`), computed per chunk and merged across overlap boundaries via `MergeStrategy`.

- `false` (default) — leaves `StandardErrors` as `nil`
- `true` — populates `StandardErrors`

### ChunkSize

Number of points processed per chunk. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying.

### Overlap

Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `MergeStrategy`. A good starting point is 10–20% of `ChunkSize`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice.

- Negative (default, `-1`) — computes `ChunkSize / 10`, clamped to at least 1 and at most `ChunkSize - 10`
- Any `int >= 1` and `< ChunkSize`

### MergeStrategy

*See: [Merge Strategies](../advanced/merge.md)*

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

## Result

`ProcessChunk` and `Finalize` return the same [`Result`](api.md#result-fields) type as `Loess.Fit`. Fields tied to Batch-only options (`CVScores`) are always left at their zero value here.

## Example

```go
package main

import (
 "fmt"
 "log"

 "github.com/thisisamirv/loess-project/bindings/go/fastloess/v2"
)

func main() {
 const n = 20
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i)
  y[i] = float64(i) + 0.1
 }

 opts := fastloess.DefaultStreamingOptions()
 opts.ChunkSize = 10
 opts.Overlap = 2

 model, err := fastloess.NewStreamingLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 if _, err := model.ProcessChunk(x[:10], y[:10]); err != nil {
  log.Fatal(err)
 }
 if _, err := model.ProcessChunk(x[10:], y[10:]); err != nil {
  log.Fatal(err)
 }

 result, err := model.Finalize()
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("y[0]: %.4f\n", result.Y[0])
}
```

```output
y[0]: 18.1000
```
