# Alternative Software

`fastloess` is a fast, configurable LOESS implementation for Python. This page compares it with [`skmisc.loess`](https://has2k1.github.io/scikit-misc/stable/loess.html), SciKit-Misc's implementation of LOESS, and shows which settings give closely matching Gaussian fits.

In short:

- With matching span, degree, Gaussian family, and direct surface fitting, `fastloess` reproduces `skmisc.loess` to floating-point precision (about `1e-14` on the examples below).
- Robust fits are not numerically identical, even with symmetric/bisquare weighting and the same iteration count; see [Comparing robust fits](#comparing-robust-fits).
- Defaults differ: `skmisc.loess` defaults to a quadratic Gaussian fit with span `0.75`, while `fastloess` defaults to a linear fit with fraction `0.67` and three robustness iterations.

---

## Reproducing `skmisc.loess`

`skmisc.loess` calls its smoothing parameter `span`; in `fastloess` the equivalent is `fraction`. For this direct Gaussian comparison, match the degree and span, disable robust iterations, and turn off `fastloess` boundary padding. The example compares both linear and quadratic fits.

:::{jupyter-execute}
import fastloess as fl
import numpy as np
from skmisc.loess import loess

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 60)
y = np.sin(x) + rng.normal(0, 0.2, 60)

for degree, fast_degree in ((1, "linear"), (2, "quadratic")):
    fast_model = fl.Loess(
        fraction=2 / 3,
        iterations=0,
        degree=fast_degree,
        boundary_policy="noboundary",
        surface_mode="direct",
        parallel=False,
    )
    fast_result = fast_model.fit(x, y)

    reference = loess(
        x,
        y,
        span=2 / 3,
        degree=degree,
        family="gaussian",
        iterations=0,
        surface="direct",
    )
    reference.fit()
    reference_y = np.asarray(reference.outputs.fitted_values)
    max_difference = np.max(np.abs(fast_result.y - reference_y))
    print(f"Degree {degree} max abs difference: {max_difference:.3g}")
    assert max_difference < 1e-12
:::

With sorted one-dimensional input and `surface_mode="direct"`, both return one fitted value per input observation, so the arrays can be compared directly. For other surface modes, the implementations use different interpolation strategies and should not be expected to agree at floating-point precision.

---

## Comparing robust fits

`skmisc.loess` enables robust reweighting with `family="symmetric"`; `fastloess` uses `iterations` together with its `robustness_method` and `scaling_method` options. Even after matching the iteration count, bisquare weighting, MAR scale, degree, span, and direct fitting, the robust fits differ because the implementations' robust reweighting calculations are not identical.

:::{jupyter-execute}
import fastloess as fl
import numpy as np
from skmisc.loess import loess

rng = np.random.default_rng(42)
x = np.linspace(0, 2 * np.pi, 60)
y = np.sin(x) + rng.normal(0, 0.2, 60)

fast_model = fl.Loess(
    fraction=2 / 3,
    iterations=3,
    degree="linear",
    boundary_policy="noboundary",
    scaling_method="mar",
    surface_mode="direct",
    parallel=False,
)
fast_result = fast_model.fit(x, y)

reference = loess(
    x,
    y,
    span=2 / 3,
    degree=1,
    family="symmetric",
    iterations=3,
    surface="direct",
)
reference.fit()
reference_y = np.asarray(reference.outputs.fitted_values)
print("Max abs difference:", np.max(np.abs(fast_result.y - reference_y)))
:::

---

## Why the defaults differ

| Option | `skmisc.loess` default | `fastloess` default |
| --- | --- | --- |
| `span` / `fraction` | `0.75` | `0.67` |
| Local polynomial degree | Quadratic (`2`) | Linear (`1`) |
| Robust fitting | Gaussian family (no robust reweighting) | 3 bisquare iterations |
| Surface calculation | Interpolated | Interpolated |
| Boundary handling | No padding | Extend |

The defaults are useful starting points, not a parity configuration. For a direct Gaussian comparison, explicitly set the options in the first example. For robustness comparisons, also choose an appropriate `scaling_method`; matching that option and the iteration count still does not make the two robust procedures identical.

---

## What this package adds

Both libraries support LOESS fitting, prediction, per-observation weights, and confidence intervals. `fastloess` additionally provides:

| Feature | `fastloess` | `skmisc.loess` |
| --- | :---: | :---: |
| Polynomial degree | 0–4 | 0–2 |
| Kernel functions | 7 options | Tricube |
| Robustness weighting | 3 methods | Symmetric / bisquare |
| Residual scale choices | MAD, MAR, mean | Fixed by implementation |
| Boundary padding | 4 policies | None |
| Prediction intervals | Yes | Not provided |
| Cross-validation for `fraction` | K-fold, LOOCV | No automatic selection |
| Streaming / online modes | Yes | No |
| Multivariate prediction | Yes | Not implemented |
| Parallel execution | Yes | No built-in parallel fitting |

See [Concepts](../introduction/concepts.md) for an overview of the LOESS options, or [Benchmarks](../benchmarks.md) for performance comparisons.
