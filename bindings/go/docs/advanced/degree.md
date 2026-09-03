---
title: "Polynomial Degree"
weight: 58
---

<!-- markdownlint-disable MD046 -->

Degree of the local polynomial fitted at each point.

## Overview

At each target point, LOESS fits a polynomial to the neighbouring data using weighted least squares. The `Degree` parameter controls the order of that polynomial.

![Degree Comparison](../assets/diagrams/degree_comparison.svg)

| Degree | Local Fit | Captures | Risk |
| --- | --- | --- | --- |
| `constant` | Constant | Level only | Over-smooth, biased at edges |
| `linear` | Linear | Trend (default) | Rarely overfits |
| `quadratic` | Quadratic | Curvature | Overfits with small `Fraction` |
| `cubic` | Cubic | Inflections | Requires larger `Fraction` |
| `quartic` | Quartic | Fine structure | High variance, rarely needed |

---

## Constant — Local Constant

$$\hat{y}(x_0) = \arg\min_a \sum_i w_i(x_0)\,(y_i - a)^2$$

The fit at each point is simply a weighted mean. Produces very smooth results but ignores local slope, introducing bias wherever the true function changes.

**Use when**: Maximum smoothness is more important than accuracy; computationally cheapest option.

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
 opts.Degree = "constant"
 opts.Fraction = 0.5

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (constant/Nadaraya-Watson):", result.Y[0])
}
```

```output
First smoothed value (constant/Nadaraya-Watson): 0.7051995803591182
```

---

## Linear — Local Linear (Default)

$$\hat{y}(x_0) = \arg\min_{a,b} \sum_i w_i(x_0)\,(y_i - a - b x_i)^2$$

Fits a weighted line through the neighbourhood. Removes first-order bias and handles boundary regions correctly. The right choice for the vast majority of applications.

**Use when**: Default; monotone or gently curved data; boundary accuracy matters.

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
 opts.Degree = "linear"
 opts.Fraction = 0.5

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (linear local regression):", result.Y[0])
}
```

```output
First smoothed value (linear local regression): 0.32737554007097214
```

---

## Quadratic — Local Quadratic

$$\hat{y}(x_0) = \arg\min_{a,b,c} \sum_i w_i(x_0)\,(y_i - a - b x_i - c x_i^2)^2$$

Fits a weighted parabola through the neighbourhood. Removes second-order bias and captures local curvature more faithfully, but requires more data per neighbourhood — pair with a larger `Fraction` (≥ 0.4) to avoid overfitting.

**Use when**: Data with pronounced peaks, valleys, or curvature; `Fraction` ≥ 0.4.

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
 opts.Degree = "quadratic"
 opts.Fraction = 0.5

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (quadratic local regression):", result.Y[0])
}
```

```output
First smoothed value (quadratic local regression): 0.06774573827604194
```

---

## Cubic — Local Cubic

$$\hat{y}(x_0) = \arg\min_{a,b,c,d} \sum_i w_i(x_0)\,(y_i - a - b x_i - c x_i^2 - d x_i^3)^2$$

Fits a weighted cubic polynomial. Captures inflection points and S-shaped local behaviour. Requires a substantially larger neighbourhood than quadratic — use `Fraction` ≥ 0.5 and verify visually for overfitting.

**Use when**: Data has clear S-shaped curves or multiple inflection points; `Fraction` ≥ 0.5.

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
 opts.Degree = "cubic"
 opts.Fraction = 0.6

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (cubic local regression):", result.Y[0])
}
```

```output
First smoothed value (cubic local regression): 0.24223407215677695
```

---

## Quartic — Local Quartic

$$\hat{y}(x_0) = \arg\min_{a,...,e} \sum_i w_i(x_0)\,(y_i - a - b x_i - \cdots - e x_i^4)^2$$

Fits a weighted quartic polynomial. Rarely needed in practice; only useful for capturing highly oscillatory local structure. Very prone to overfitting — require `Fraction` ≥ 0.6 and cross-validate.

**Use when**: Fine oscillatory structure is physically meaningful and the dataset is large; always cross-validate.

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
 opts.Degree = "quartic"
 opts.Fraction = 0.7

 model, err := fastloess.NewLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 result, err := model.Fit(x, y)
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println("First smoothed value (quartic local regression):", result.Y[0])
}
```

```output
First smoothed value (quartic local regression): 0.2385333930042555
```

---

## Choosing the Right Degree

| Situation | Recommended Degree |
| --- | --- |
| Monotone trend, general purpose | `linear` (default) |
| Maximum smoothness, speed | `constant` |
| Clear peaks / valleys / inflections | `quadratic` (with `Fraction` ≥ 0.4) |
| S-shaped curves, multiple inflections | `cubic` (with `Fraction` ≥ 0.5) |
| Fine oscillatory structure (rare) | `quartic` (with `Fraction` ≥ 0.6, cross-validate) |
| Boundary accuracy is critical | `linear` or `quadratic` (not `constant`) |
| Very small dataset (n < 50) | `linear` |

---

## Higher Degree Effects

![Higher Degree Comparison](../assets/diagrams/higher_degree_comparison.svg)

---

## Surface Mode

The `SurfaceMode` parameter controls whether LOESS evaluates the local polynomial at every query point or at a sparser grid of vertices with Hermite cubic interpolation in between.

| Mode | Behaviour | Speed | Accuracy |
| --- | --- | --- | --- |
| `"interpolation"` (default) | Evaluate at anchor vertices, blend via Hermite cubic | Faster | Slight approximation |
| `"direct"` | Evaluate at every query point | Exact | Full precision |

![Surface Mode Comparison](../assets/diagrams/surface_comparison.svg)

![Degree x Interpolation](../assets/diagrams/degree_interpolation_comparison.svg)
