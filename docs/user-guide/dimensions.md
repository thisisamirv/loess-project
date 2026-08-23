<!-- markdownlint-disable MD024 MD033 -->
# Multivariate LOESS

Smoothing over multiple predictor dimensions simultaneously.

## Overview

Standard LOESS operates on a single predictor $x$. Setting `dimensions > 1` extends the neighbourhood search and local polynomial fit into an $n$-dimensional predictor space, enabling surface smoothing over spatial grids, time–altitude combinations, and similar multi-predictor datasets.

![Multivariate LOESS](../assets/diagrams/multivariate_loess.svg)

| Dimensions | Use Case | Input Shape |
| --- | --- | --- |
| `1` | Time series, 1D signal (default) | `x`: 1-D array |
| `2` | Spatial surface, 2-predictor model | `x`: n × 2 matrix |
| `3+` | High-dimensional regression | `x`: n × d matrix |

!!! warning "Computational cost"
    Neighbourhood search scales with $d$ dimensions. For `dimensions ≥ 3` keep `fraction` small and consider increasing `delta` to activate interpolation.

---

## 1D — Standard (Default)

Single predictor. No configuration required.

=== "Python"
    ```python
    import numpy as np
    import fastloess as fl

    x = np.linspace(0, 10, 200)
    y = np.sin(x) + np.random.normal(0, 0.2, 200)
    model = fl.Loess(fraction=0.3)
    result = model.fit(x, y)
    ```

=== "Node.js"
    ```javascript
    const { Loess } = require('fastloess');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Loess({ fraction: 0.3 });
    const result = model.fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Loess({ fraction: 0.3 });
    const result = model.fit(x, y);
    ```

=== "C++"
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

        return 0;
    }
    ```

---

## 2D — Spatial Surface

Two predictors (e.g., latitude/longitude, time/altitude). Pass an $n \times 2$ matrix as `x`.

=== "Python"
    ```python
    import numpy as np
    import fastloess as fl

    rng = np.random.default_rng(42)
    n = 100
    lat = np.linspace(0, 2 * np.pi, n)
    lon = np.linspace(0, 2 * np.pi, n)
    z = np.sin(lat) + np.cos(lon) + rng.normal(0, 0.1, n)

    # x is an (n, 2) array flattened to 1D (Python binding requires flat input)
    x2d = np.column_stack([lat, lon]).ravel()
    model = fl.Loess(dimensions=2, fraction=0.3)
    result = model.fit(x2d, z)
    ```

=== "Node.js"
    ```javascript
    const { Loess } = require('fastloess');

    const n = 100;
    const lat = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const lon = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const z = Float64Array.from({ length: n }, (_, i) => Math.sin(lat[i]) + Math.cos(lon[i]) + 0.05);
    // x is a flat Float64Array of length n*2, row-major
    const x2d = Float64Array.from({ length: n * 2 }, (_, k) => k % 2 === 0 ? lat[k >> 1] : lon[k >> 1]);

    const model = new Loess({ dimensions: 2, fraction: 0.3 });
    const result = model.fit(x2d, z);
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 100;
    const lat = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const lon = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const z = Float64Array.from({ length: n }, (_, i) => Math.sin(lat[i]) + Math.cos(lon[i]) + 0.05);
    const x2d = Float64Array.from({ length: n * 2 }, (_, k) => k % 2 === 0 ? lat[k >> 1] : lon[k >> 1]);

    const model = new Loess({ dimensions: 2, fraction: 0.3 });
    const result = model.fit(x2d, z);
    ```

=== "C++"
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

        return 0;
    }
    ```

---

## 3D and Higher

Three or more predictors. The neighbourhood radius grows in each additional dimension, so a larger `fraction` (or smaller dataset) is typically needed.

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = np.linspace(0, 1, n)
    x3 = np.linspace(1, 0, n)
    y = np.sin(x1) + x2 - x3 + rng.normal(0, 0.1, n)

    x3d = np.column_stack([x1, x2, x3]).ravel()   # (n*3,) flat
    model = fl.Loess(dimensions=3, fraction=0.5)
    result = model.fit(x3d, y)
    ```

=== "Node.js"
    ```javascript
    const { Loess } = require('fastloess');

    const n = 100;
    const x1 = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const x2 = Float64Array.from({ length: n }, (_, i) => i / (n - 1));
    const x3 = Float64Array.from({ length: n }, (_, i) => 1 - i / (n - 1));
    const y = Float64Array.from({ length: n }, (_, i) => Math.sin(x1[i]) + x2[i] - x3[i] + 0.05);
    const x3d = Float64Array.from({ length: n * 3 }, (_, k) => {
        const i = Math.floor(k / 3), d = k % 3;
        return d === 0 ? x1[i] : d === 1 ? x2[i] : x3[i];
    });

    const model = new Loess({ dimensions: 3, fraction: 0.5 });
    const result = model.fit(x3d, y);
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 100;
    const x1 = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const x2 = Float64Array.from({ length: n }, (_, i) => i / (n - 1));
    const x3 = Float64Array.from({ length: n }, (_, i) => 1 - i / (n - 1));
    const y = Float64Array.from({ length: n }, (_, i) => Math.sin(x1[i]) + x2[i] - x3[i] + 0.05);
    const x3d = Float64Array.from({ length: n * 3 }, (_, k) => {
        const i = Math.floor(k / 3), d = k % 3;
        return d === 0 ? x1[i] : d === 1 ? x2[i] : x3[i];
    });

    const model = new Loess({ dimensions: 3, fraction: 0.5 });
    const result = model.fit(x3d, y);
    ```

=== "C++"
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

        return 0;
    }
    ```

---

## Distance Metrics for Multivariate Data

When `dimensions > 1` you can also control how inter-point distances are computed.

| Metric | Description | When to Use |
| --- | --- | --- |
| `"normalized"` | Each dimension scaled to unit range (default) | Predictors on different scales |
| `"euclidean"` | Raw Euclidean distance | Predictors already on same scale |
| `"minkowski:p"` | Generalised Minkowski ($L_p$) norm | Custom distance geometry |
| `"weighted"` | Per-dimension weighted Euclidean | Domain-specific importance |

See [Parameters](parameters.md#distance_metric) for the full list of options per language.
