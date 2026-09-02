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

:::{warning} Computational cost
Neighbourhood search scales with $d$ dimensions. For `dimensions ≥ 3` keep `fraction` small and consider increasing `delta` to activate interpolation.
:::

---

## 1D — Standard (Default)

Single predictor. No configuration required.

:::{jupyter-execute}
import numpy as np
import fastloess as fl

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 200)
y = np.sin(x) + rng.normal(0, 0.2, 200)
model = fl.Loess(fraction=0.3)
result = model.fit(x, y)
print(f"Smoothed y[0]: {result.y[0]:.4f}")
:::

---

## 2D — Spatial Surface

Two predictors (e.g., latitude/longitude, time/altitude). Pass an $n \times 2$ matrix as `x`.

:::{jupyter-execute}
import numpy as np
import fastloess as fl

rng = np.random.default_rng(42)
n = 100
lat = np.linspace(0, 2 *np.pi, n)
lon = np.linspace(0, 2* np.pi, n)
z = np.sin(lat) + np.cos(lon) + rng.normal(0, 0.1, n)

## x is an (n, 2) array flattened to 1D (Python binding requires flat input)

x2d = np.column_stack([lat, lon]).ravel()
model = fl.Loess(dimensions=2, fraction=0.3)
result = model.fit(x2d, z)
print(f"Smoothed y[0]: {result.y[0]:.4f}")
:::

---

## 3D and Higher

Three or more predictors. The neighbourhood radius grows in each additional dimension, so a larger `fraction` (or smaller dataset) is typically needed.

:::{jupyter-execute}
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
print(f"Smoothed y[0]: {result.y[0]:.4f}")
:::

---

## Distance Metrics for Multivariate Data

When `dimensions > 1` you can also control how inter-point distances are computed.

| Metric | Description | When to Use |
| --- | --- | --- |
| `"normalized"` | Each dimension scaled to unit range (default) | Predictors on different scales |
| `"euclidean"` | Raw Euclidean distance | Predictors already on same scale |
| `"minkowski:p"` | Generalised Minkowski ($L_p$) norm | Custom distance geometry |
| `"weighted"` | Per-dimension weighted Euclidean | Domain-specific importance |

See [API Reference](../api/api.md#distance_metric) for the full list of options per language.
