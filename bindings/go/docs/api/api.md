---
title: "API"
weight: 30
---

Batch `Loess` reference. Best suited when the dataset fits in memory and you need intervals, cross-validation, multivariate predictors, or diagnostics.

> **StreamingLoess** and **OnlineLoess** are documented separately: [api-streaming.md](api-streaming.md), [api-online.md](api-online.md)

## `fastloess.DefaultOptions() Options`

Returns recommended defaults. Start from this and override only the fields you need:

```go
opts := fastloess.DefaultOptions()
opts.Fraction = 0.3
opts.ReturnDiagnostics = true
```

## `Options` fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `Fraction` | `float64` | `0.67` | Smoothing fraction, in (0, 1]. |
| `Iterations` | `int` | `3` | Robustness iterations, in [0, 1000]. |
| `WeightFunction` | `string` | `"tricube"` | Kernel: `tricube`, `gaussian`, `uniform`, `cosine`, `epanechnikov`, `biweight`, `triangle`. |
| `RobustnessMethod` | `string` | `"bisquare"` | Outlier downweighting: `bisquare`, `huber`, `talwar`. |
| `ScalingMethod` | `string` | `"mad"` | Residual scale estimator: `mad`, `mar`, `mean`. |
| `BoundaryPolicy` | `string` | `"extend"` | Boundary handling: `extend`, `reflect`, `zero`, `noboundary`. |
| `ZeroWeightFallback` | `string` | `"use_local_mean"` | Fallback when all robustness weights hit zero: `use_local_mean`, `return_original`, `return_none`. |
| `Degree` | `string` | `"linear"` | Polynomial degree of the local fit — see [Polynomial Degree](../advanced/degree.md). |
| `Dimensions` | `int` | `1` | Number of predictor dimensions — see [Multivariate LOESS](../advanced/dimensions.md). |
| `DistanceMetric` | `string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p. |
| `WeightedMetricWeights` | `[]float64` | `nil` | Per-dimension weights (used when `DistanceMetric = "weighted"`). |
| `SurfaceMode` | `string` | `"interpolation"` | Surface computation mode. |
| `ReturnSE` | `bool` | `false` | Populate `Result.StandardErrors`/`Result.HatMatrix` (hat-matrix statistics). |
| `Cell` | `*float64` | `nil` (auto) | Interpolation cell size tuning parameter, in (0, 1]. Only applies when `SurfaceMode` is `"interpolation"`. |
| `InterpolationVertices` | `*int` | `nil` (auto) | Caps the number of interpolation vertices. Only applies when `SurfaceMode` is `"interpolation"`. |
| `BoundaryDegreeFallback` | `*bool` | `nil` (auto) | Reduce polynomial degree near boundary vertices to avoid extrapolation artifacts. |
| `ConfidenceIntervals` | `*float64` | `nil` (disabled) | Confidence level in (0, 1), e.g. `0.95`. |
| `PredictionIntervals` | `*float64` | `nil` (disabled) | Confidence level in (0, 1), e.g. `0.95`. |
| `AutoConverge` | `*float64` | `nil` (disabled) | Convergence tolerance for early stopping. |
| `ReturnDiagnostics` | `bool` | `false` | Populate `Result.Diagnostics`. |
| `ReturnResiduals` | `bool` | `false` | Populate `Result.Residuals`. |
| `ReturnRobustnessWeights` | `bool` | `false` | Populate `Result.RobustnessWeights`. |
| `CVFractions` | `[]float64` | `nil` (disabled) | Candidate fractions for cross-validation. |
| `CVMethod` | `string` | `"kfold"` | `kfold` or `loocv`. |
| `CVK` | `int` | `5` | Number of folds for k-fold CV. |
| `CVSeed` | `*uint64` | `nil` (random) | RNG seed for reproducible k-fold splits. |
| `Parallel` | `bool` | `true` | Enable parallel processing. |

`Fraction` is the most important parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

`Iterations` controls robustness to outliers, at the cost of speed.

| Value | Effect | Performance |
| --- | --- | --- |
| 0 | No robustness | Fastest |
| 1-3 | Moderate | Recommended |
| 4-6 | Strong | Contaminated data |
| 7+ | Very strong | Heavy outliers |

## `fastloess.NewLoess(opts Options) (*Loess, error)`

Creates a new batch model. Returns an error if any option is out of range (some validation is eager at construction; some — e.g. `Cell`, `Fraction` — is deferred to `Fit`).

## `(*Loess) Fit(x, y []float64, customWeights ...[]float64) (Result, error)`

Smooths `y` as a function of `x`. For multivariate input (`Dimensions > 1`), `x` is flattened row-major (length `len(y)*Dimensions`). `x` and `y` must be non-empty. An optional `customWeights` slice (same length as `y`) applies per-observation case weights.

## `(*Loess) Close() error`

Releases native resources. Safe to call multiple times. A finalizer is registered as a safety net, but call `Close` explicitly (e.g. via `defer`) rather than relying on the garbage collector.

## `Result` fields

| Field | Type | Populated when |
| --- | --- | --- |
| `X`, `Y` | `[]float64` | Always. |
| `StandardErrors` | `[]float64` | `ReturnSE` |
| `ConfidenceLower`, `ConfidenceUpper` | `[]float64` | `ConfidenceIntervals` set |
| `PredictionLower`, `PredictionUpper` | `[]float64` | `PredictionIntervals` set |
| `Residuals` | `[]float64` | `ReturnResiduals` |
| `RobustnessWeights` | `[]float64` | `ReturnRobustnessWeights` |
| `CVScores` | `[]float64` | `CVFractions` set |
| `FractionUsed` | `float64` | Always. |
| `IterationsUsed` | `int` | Always (`-1` if not available). |
| `Dimensions` | `int` | Always. |
| `Diagnostics` | `*Diagnostics` | `ReturnDiagnostics` |
| `HatMatrix` | `*HatMatrixStats` | `ReturnSE` |

`Diagnostics` holds `RMSE`, `MAE`, `RSquared`, `AIC`, `AICc`, `EffectiveDF`, `ResidualSD`.

`HatMatrixStats` holds `ENP`, `TraceHat`, `Delta1`, `Delta2`, `ResidualScale`, `Leverage`.

## Options

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

### BoundaryPolicy

*See: [Boundary Handling](../advanced/boundary.md)*

- `"extend"` (default; alias: `"pad"`)
- `"reflect"` (alias: `"mirror"`)
- `"zero"`
- `"noboundary"` (alias: `"none"`)

### ScalingMethod

*See: [Scaling Methods](../weighting/scaling.md)*

- `"mad"` (default; alias: `"median_absolute_deviation"`)
- `"mar"` (alias: `"median_absolute_residual"`)
- `"mean"` (alias: `"mean_absolute_residual"`)

### ZeroWeightFallback

Behavior when all neighborhood weights are zero:

| Option | Behavior |
| --- | --- |
| `"use_local_mean"` (default; aliases: `"local_mean"`, `"mean"`) | Use the mean of the neighborhood |
| `"return_original"` (alias: `"original"`) | Return the original y value |
| `"return_none"` (alias: `"none"`) | Return `NaN` |

### Degree

*See: [Polynomial Degree](../advanced/degree.md)*

- `"constant"` (degree 0)
- `"linear"` (default, degree 1)
- `"quadratic"` (degree 2)
- `"cubic"` (degree 3)
- `"quartic"` (degree 4)

### DistanceMetric

*See: [Multivariate LOESS](../advanced/dimensions.md)*

- `"normalized"` (default — scales each dimension by its range; alias: `"norm"`)
- `"euclidean"` (alias: `"euclid"`)
- `"manhattan"` (alias: `"l1"`)
- `"chebyshev"` (alias: `"linf"`)
- `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)
- `"weighted"` plus `WeightedMetricWeights` for per-dimension scaling (alias: `"weighted_euclidean"`)

### SurfaceMode

*See: [Polynomial Degree](../advanced/degree.md#surface-mode)*

Controls whether the local polynomial is evaluated at every query point or at a sparser grid of anchor vertices with Hermite cubic interpolation in between.

| Mode | Behavior | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at vertices, interpolate between | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Slower | Full precision |

## Custom weights

```go
weights := make([]float64, len(x))
for i := range weights {
 weights[i] = 1.0
}
weights[0] = 5.0 // trust the first observation more

result, err := model.Fit(x, y, weights)
```

## Cross-validation

```go
opts := fastloess.DefaultOptions()
opts.CVFractions = []float64{0.1, 0.2, 0.3, 0.5}
opts.CVMethod = "kfold"
opts.CVK = 5
seed := uint64(42)
opts.CVSeed = &seed

model, _ := fastloess.NewLoess(opts)
defer model.Close()
result, _ := model.Fit(x, y)
fmt.Println(result.CVScores, result.FractionUsed) // FractionUsed = the CV-selected fraction
```

## Example

```go
package main

import (
 "fmt"

 "github.com/thisisamirv/loess-project/bindings/go/fastloess"
)

func main() {
 x := []float64{1, 2, 3, 4, 5}
 y := []float64{2.1, 4.0, 6.2, 8.0, 10.1}

 opts := fastloess.DefaultOptions()
 opts.Fraction = 0.5

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  panic(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  panic(err)
 }
 fmt.Println(result.Y)
}
```

```output
[2.1 4 6.2 8 10.1]
```
