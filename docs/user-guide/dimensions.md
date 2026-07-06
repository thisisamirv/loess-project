<!-- markdownlint-disable MD024 -->
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

=== "R"
    ```r
    x <- seq(0, 10, length.out = 200)
    y <- sin(x) + rnorm(200, sd = 0.2)
    result <- Loess(fraction = 0.3)$fit(x, y)
    ```

=== "Python"
    ```python
    import numpy as np
    import fastloess as fl

    x = np.linspace(0, 10, 200)
    y = np.sin(x) + np.random.normal(0, 0.2, 200)
    result = fl.Loess(fraction=0.3).fit(x, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .fraction(0.3)
        .adapter(Batch)
        .build()?;
    let result = model.fit(&x, &y)?;
    ```

=== "Julia"
    ```julia
    result = fit(Loess(; fraction=0.3), x, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ fraction: 0.3 }).fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = new Loess({ fraction: 0.3 }).fit(x, y);
    ```

=== "C++"
    ```cpp
    fastloess::Loess model({ .fraction = 0.3 });
    auto result = model.fit(x, y).value();
    ```

---

## 2D — Spatial Surface

Two predictors (e.g., latitude/longitude, time/altitude). Pass an $n \times 2$ matrix as `x`.

=== "R"
    ```r
    # n × 2 predictor matrix
    coords <- cbind(lat, lon)
    result <- Loess(dimensions = 2L, fraction = 0.3)$fit(coords, z)
    ```

=== "Python"
    ```python
    import numpy as np
    import fastloess as fl

    # x is an (n, 2) array flattened to 1D (Python binding requires flat input)
    x2d = np.column_stack([lat, lon]).ravel()
    result = fl.Loess(dimensions=2, fraction=0.3).fit(x2d, z)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .dimensions(2)
        .fraction(0.3)
        .adapter(Batch)
        .build()?;
    let result = model.fit(&x2d, &z)?;
    ```

=== "Julia"
    ```julia
    # x is an (n, 2) matrix of predictors
    x2d = hcat(lat, lon)
    result = fit(Loess(; dimensions=2, fraction=0.3), x2d, z)
    ```

=== "Node.js"
    ```javascript
    // x is a flat Float64Array of length n*2, row-major
    const result = new Loess({ dimensions: 2, fraction: 0.3 }).fit(x2d, z);
    ```

=== "WebAssembly"
    ```javascript
    const result = new Loess({ dimensions: 2, fraction: 0.3 }).fit(x2d, z);
    ```

=== "C++"
    ```cpp
    // x is an (n × 2) row-major matrix
    fastloess::LoessOptions d2_opts;
    d2_opts.dimensions = 2;
    d2_opts.fraction = 0.3;
    fastloess::Loess model(d2_opts);
    auto result = model.fit(x2d, z).value();
    ```

---

## 3D and Higher

Three or more predictors. The neighbourhood radius grows in each additional dimension, so a larger `fraction` (or smaller dataset) is typically needed.

=== "R"
    ```r
    predictors <- cbind(x1, x2, x3)   # n × 3
    result <- Loess(dimensions = 3L, fraction = 0.5)$fit(predictors, y)
    ```

=== "Python"
    ```python
    x3d = np.column_stack([x1, x2, x3]).ravel()   # (n*3,) flat
    result = fl.Loess(dimensions=3, fraction=0.5).fit(x3d, y)
    ```

=== "Rust"
    ```rust
    let model = Loess::new()
        .dimensions(3)
        .fraction(0.5)
        .adapter(Batch)
        .build()?;
    let result = model.fit(&x3d, &y)?;
    ```

=== "Julia"
    ```julia
    # x is an (n, 3) matrix of predictors
    x3d = hcat(x1, x2, x3)
    result = fit(Loess(; dimensions=3, fraction=0.5), x3d, y)
    ```

=== "Node.js"
    ```javascript
    const result = new Loess({ dimensions: 3, fraction: 0.5 }).fit(x3d, y);
    ```

=== "WebAssembly"
    ```javascript
    const result = new Loess({ dimensions: 3, fraction: 0.5 }).fit(x3d, y);
    ```

=== "C++"
    ```cpp
    fastloess::LoessOptions d3_opts;
    d3_opts.dimensions = 3;
    d3_opts.fraction = 0.5;
    fastloess::Loess model(d3_opts);
    auto result = model.fit(x3d, y).value();
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
