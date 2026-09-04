\page dimensions Multivariate LOESS

# Multivariate LOESS

Smoothing over multiple predictor dimensions simultaneously.

## Overview

Standard LOESS operates on a single predictor \f$x\f$. Setting `dimensions > 1` extends the neighbourhood search and local polynomial fit into an \f$n\f$-dimensional predictor space, enabling surface smoothing over spatial grids, time–altitude combinations, and similar multi-predictor datasets.

![Multivariate LOESS](multivariate_loess.svg)

| Dimensions | Use Case | Input Shape |
| --- | --- | --- |
| `1` | Time series, 1D signal (default) | `x`: 1-D array |
| `2` | Spatial surface, 2-predictor model | `x`: n × 2 matrix |
| `3+` | High-dimensional regression | `x`: n × d matrix |

> **Computational cost:** Neighbourhood search scales with \f$d\f$ dimensions. For `dimensions ≥ 3` keep `fraction` small.

<hr>

## 1D — Standard (Default)

Single predictor. No configuration required.

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastloess::Loess model({ .fraction = 0.3 });
    auto result = model.fit(x, y).value();

    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: 0.24731
```

---

## 2D — Spatial Surface

Two predictors (e.g., latitude/longitude, time/altitude). Pass an \f$n \times 2\f$ matrix as `x`.

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> lat(n), lon(n), z(n), x2d(n * 2);
    for (int i = 0; i < n; ++i) {
        lat[i] = i * 2 * M_PI / (n - 1);
        lon[i] = i * 2 * M_PI / (n - 1);
        z[i] = std::sin(lat[i]) + std::cos(lon[i]) + 0.05;
        x2d[2 * i]     = lat[i];
        x2d[2 * i + 1] = lon[i];
    }

    // x is an (n × 2) row-major matrix
    fastloess::LoessOptions d2_opts;
    d2_opts.dimensions = 2;
    d2_opts.fraction = 0.3;
    fastloess::Loess model(d2_opts);
    auto result = model.fit(x2d, z).value();

    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: 1.19329
```

---

## 3D and Higher

Three or more predictors. The neighbourhood radius grows in each additional dimension, so a larger `fraction` (or smaller dataset) is typically needed.

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> y(n), x3d(n * 3);
    for (int i = 0; i < n; ++i) {
        double x1 = i * 2 * M_PI / (n - 1);
        double x2 = static_cast<double>(i) / (n - 1);
        double x3 = 1.0 - static_cast<double>(i) / (n - 1);
        y[i] = std::sin(x1) + x2 - x3 + 0.05;
        x3d[3 * i]     = x1;
        x3d[3 * i + 1] = x2;
        x3d[3 * i + 2] = x3;
    }

    fastloess::LoessOptions d3_opts;
    d3_opts.dimensions = 3;
    d3_opts.fraction = 0.5;
    fastloess::Loess model(d3_opts);
    auto result = model.fit(x3d, y).value();

    std::cout << "y[0]: " << result.y_vector()[0] << "\n";
    return 0;
}
```

```output
y[0]: -0.670791
```

---

## Distance Metrics for Multivariate Data

When `dimensions > 1` you can also control how inter-point distances are computed.

| Metric | Description | When to Use |
| --- | --- | --- |
| `"normalized"` | Each dimension scaled to unit range (default) | Predictors on different scales |
| `"euclidean"` | Raw Euclidean distance | Predictors already on same scale |
| `"minkowski:p"` | Generalised Minkowski (\f$L_p\f$) norm | Custom distance geometry |
| `"weighted"` | Per-dimension weighted Euclidean | Domain-specific importance |

See [API Reference](../api/api.md#distance_metric) for the full list of options per language.
