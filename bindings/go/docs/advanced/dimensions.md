---
title: "Multivariate LOESS"
weight: 60
---

<!-- markdownlint-disable MD046 -->

Smoothing over multiple predictor dimensions simultaneously.

## Overview

Standard LOESS operates on a single predictor $x$. Setting `Dimensions > 1` extends the neighbourhood search and local polynomial fit into an $n$-dimensional predictor space, enabling surface smoothing over spatial grids, time–altitude combinations, and similar multi-predictor datasets. `x` is passed as a flat, row-major slice of length `len(y)*Dimensions`.

![Multivariate LOESS](../assets/diagrams/multivariate_loess.svg)

| Dimensions | Use Case | Input Shape |
| --- | --- | --- |
| `1` | Time series, 1D signal (default) | `x`: length `n` |
| `2` | Spatial surface, 2-predictor model | `x`: length `n*2`, row-major |
| `3+` | High-dimensional regression | `x`: length `n*d`, row-major |

> **Computational cost:** Neighbourhood search scales with `d` dimensions. For `Dimensions >= 3` keep `Fraction` small.

---

## 1D — Standard (Default)

Single predictor. No configuration required.

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/loess-project/bindings/go/fastloess"
)

func main() {
 n := 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastloess.DefaultOptions()
 opts.Fraction = 0.3

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (1D LOESS):", result.Y[0])
}
```

```output
First smoothed value (1D LOESS): 0.2473103228645228
```

---

## 2D — Spatial Surface

Two predictors (e.g., latitude/longitude, time/altitude). Pass a flat, row-major slice of length `n*2` as `x`.

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/loess-project/bindings/go/fastloess"
)

func main() {
 n := 100
 x2d := make([]float64, n*2) // row-major: [lat0, lon0, lat1, lon1, ...]
 z := make([]float64, n)
 for i := 0; i < n; i++ {
  lat := float64(i) * 2 * math.Pi / float64(n-1)
  lon := float64(i) * 2 * math.Pi / float64(n-1)
  x2d[i*2] = lat
  x2d[i*2+1] = lon
  z[i] = math.Sin(lat) + math.Cos(lon)
 }

 opts := fastloess.DefaultOptions()
 opts.Dimensions = 2
 opts.Fraction = 0.3

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x2d, z)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (2D LOESS, lat/lon):", result.Y[0])
}
```

```output
First smoothed value (2D LOESS, lat/lon): 1.1662589769190939
```

---

## 3D and Higher

Three or more predictors. The neighbourhood radius grows in each additional dimension, so a larger `Fraction` (or smaller dataset) is typically needed.

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/loess-project/bindings/go/fastloess"
)

func main() {
 n := 100
 x3d := make([]float64, n*3) // row-major: [x1_0, x2_0, x3_0, x1_1, x2_1, x3_1, ...]
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x1 := float64(i) * 2 * math.Pi / float64(n-1)
  x2 := float64(i) / float64(n-1)
  x3 := 1.0 - float64(i)/float64(n-1)
  x3d[i*3] = x1
  x3d[i*3+1] = x2
  x3d[i*3+2] = x3
  y[i] = math.Sin(x1) + x2 - x3
 }

 opts := fastloess.DefaultOptions()
 opts.Dimensions = 3
 opts.Fraction = 0.5

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x3d, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (3D LOESS):", result.Y[0])
}
```

```output
First smoothed value (3D LOESS): -0.7006549793435668
```

---

## Distance Metrics for Multivariate Data

When `Dimensions > 1` you can also control how inter-point distances are computed.

| Metric | Description | When to Use |
| --- | --- | --- |
| `"normalized"` | Each dimension scaled to unit range (default) | Predictors on different scales |
| `"euclidean"` | Raw Euclidean distance | Predictors already on same scale |
| `"minkowski:p"` | Generalised Minkowski ($L_p$) norm | Custom distance geometry |
| `"weighted"` | Per-dimension weighted Euclidean | Domain-specific importance |

See [API Reference](../api/api.md) for the full list of options.
