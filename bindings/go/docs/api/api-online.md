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

`OnlineOptions` embeds [`Options`](api.md) (all the same fields apply, except `CVFractions`/`CVMethod`/`CVK`/`CVSeed`, and `Parallel`, which are batch-only). `AddPoint` only accepts a single x coordinate: online mode does not support multivariate predictors even if `Dimensions` was set on construction. Additional fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `WindowCapacity` | `int` | `1000` | Maximum number of recent points retained. |
| `MinPoints` | `int` | `2` | Minimum points required before output starts. |
| `UpdateMode` | `string` | `"incremental"` | How the window is updated as new points arrive. |

## `fastloess.NewOnlineLoess(opts OnlineOptions) (*OnlineLoess, error)`

## `(*OnlineLoess) AddPoint(x, y float64) (res PointResult, ok bool, err error)`

Adds a single observation. `ok` is `false` while the window is still filling (fewer than `MinPoints` seen so far); once `ok` is `true`, `res` holds the smoothed value for the most recently added point. Once the window reaches `WindowCapacity`, each new point evicts the oldest one, so memory stays bounded regardless of how much history has passed through. `UpdateMode` controls how much work each call does: `"incremental"` re-fits only the newest point, while `"full"` re-smooths the entire window for a more accurate but slower result.

## `(*OnlineLoess) Close() error`

Releases native resources. Safe to call multiple times.

## Options

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

### Missing

Policy for handling a non-finite (NaN/Inf) `x` or `y` value passed to `AddPoint` (overrides the row-dropping behavior described in [API](api.md) since Online processes one point at a time):

| Option | Behavior |
| --- | --- |
| `"error"` (default) | Return an error |
| `"drop"` | Silently ignore the point — `AddPoint` returns `ok == false` instead of adding it to the window |

See [API](api.md) for the descriptions of all inherited fields (`Fraction`, `WeightFunction`, `DistanceMetric`, `WeightedMetricWeights`, etc.).

## `PointResult` fields

| Field | Type | Notes |
| --- | --- | --- |
| `Y` | `float64` | Smoothed value. |
| `StandardError` | `float64` | Always `NaN` — standard errors require confidence intervals, which are Batch-only. |
| `Residual` | `float64` | Residual y − smoothed; always present (there is no `ReturnResiduals` option for Online). |
| `RobustnessWeight` | `float64` | Robustness weight, if `ReturnRobustnessWeights` was set (`NaN` otherwise). |
| `IterationsUsed` | `int` | Robustness iterations performed (`-1` if not applicable). |

There is no `Diagnostics` type or `ReturnDiagnostics` option for `OnlineLoess`: `PointResult` carries no diagnostics field, since diagnostics like RMSE/R² need more than one point's worth of history to be meaningful.

## Example

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/loess-project/bindings/go/fastloess"
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
