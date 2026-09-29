<!-- markdownlint-disable MD036 MD046 -->
# Alternative Software

`FastLOESS.jl` is a faster, more configurable alternative to [`Loess.jl`](https://github.com/JuliaStats/Loess.jl), a widely used Julia implementation of LOESS. This page compares their methods, demonstrates how to align their settings, and summarizes the features each package provides.

In short:

- Both implement local-polynomial LOESS. `Loess.jl` supports linear and quadratic fits and uses a KD-tree approximation; `FastLOESS.jl` supports degrees 0–4 and offers direct and interpolated surface modes.
- The closest comparison uses linear fits, matching spans, and no robust iterations. The two independent implementations do not produce identical values; see [Comparing the two packages](#comparing-the-two-packages).
- `FastLOESS.jl` adds configurable robustness, boundary policies, cross-validation, and streaming and online adapters; see [What this package adds](#what-this-package-adds).

> **Note:** Both packages are LOESS implementations. `Loess.jl` currently does not support multivariate prediction blending.

---

## How the implementations differ

| Aspect | `FastLOESS.jl` | `Loess.jl` |
| --- | --- | --- |
| Method | LOESS (local-polynomial regression) | LOESS (KD-tree-based approximation) |
| Local fit degree | 0–4 | Linear or quadratic (default: quadratic) |
| Robustness reweighting | Configurable (`iterations`, 3 methods) | None |
| Surface computation | Direct or interpolated | KD-tree vertices with interpolation |
| Multivariate prediction | Supported | Not yet supported |
| Boundary handling | 4 explicit policies | No configurable padding |

Both packages use a smoothing parameter that controls the local neighborhood: `fraction` in `FastLOESS.jl` and `span` in `Loess.jl`.

---

## Comparing the two packages

This example disables robust reweighting, selects linear fits in both packages, and uses direct, unpadded fitting in `FastLOESS.jl`. `Loess.jl` uses its KD-tree interpolation, so matching the degree and span does not make the implementations numerically identical.

```@example alt-software-compare
using FastLOESS, Random
import Loess

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=60))
y = sin.(x) .+ randn(rng, 60) .* 0.2

model = FastLOESS.Loess(
    ; fraction=2 / 3,
    iterations=0,
    degree="linear",
    boundary_policy="noboundary",
    surface_mode="direct",
    parallel=false,
)
result = FastLOESS.fit(model, x, y)

baseline = Loess.loess(x, y; span=2 / 3, degree=1)
baseline_y = Loess.predict(baseline, x)

println("Max abs difference: ", maximum(abs.(result.y .- baseline_y)))
```

The difference reflects the independent neighborhood and surface-computation strategies. `FastLOESS.jl` also supports robust reweighting; `Loess.jl` does not, so leave `iterations=0` when comparing their non-robust fits.

---

## What this package adds

Both packages provide LOESS fitting and prediction. `FastLOESS.jl` additionally offers:

| Feature | `FastLOESS.jl` | `Loess.jl` |
| --- | :---: | :---: |
| Polynomial degree | 0–4 | 1–2 |
| Kernel functions | 7 options | Tricube |
| Robustness weighting | 3 methods | None |
| Residual scale estimation | MAD, MAR, mean | Not applicable |
| Boundary padding | 4 policies | None |
| Confidence intervals | Yes | Yes |
| Prediction intervals | Yes | Not implemented |
| Cross-validation for `fraction` | K-fold, LOOCV | No automatic selection |
| Streaming / online modes | Yes | No |
| Custom per-observation weights | Yes | No |
| Parallel execution | Yes | No built-in parallel fitting |
| Multivariate prediction | Yes | Not yet supported |

See [Concepts](../introduction/concepts.md) for an overview of the LOESS options, or [Benchmarks](../benchmarks.md) for performance comparisons.
