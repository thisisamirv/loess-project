# FastLOESS.jl

High-performance LOESS (Locally Estimated Scatterplot Smoothing) for Julia,
backed by a Rust library.

See the [main documentation](https://loess.readthedocs.io/) and the
[GitHub repository](https://github.com/thisisamirv/loess-project) for full details.

## Quick Start

```julia
using FastLOESS

x = collect(1.0:0.1:10.0)
y = sin.(x) .+ 0.1 .* randn(length(x))

result = fit(Loess(fraction = 0.3), x, y)
println(result.y)
```

## Installation

```julia
using Pkg
Pkg.add("FastLOESS")
```
