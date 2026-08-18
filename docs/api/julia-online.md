# OnlineLoess — Julia API Reference

See also: [FastLOESS Julia API Reference](julia.md)

## Struct

### `OnlineLoess`

The `OnlineLoess` struct updates the model incrementally with new data points.

**Constructor:**

```julia
using FastLOESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

online = OnlineLoess()
```

* `kwargs`: Keyword arguments corresponding to `OnlineOptions` fields.

**Methods:**

```julia
using FastLOESS
using Random, Statistics

rng = MersenneTwister(42)
x = collect(range(0, 2π, length=100))
y = sin.(x) .+ randn(rng, 100) .* 0.3

online = OnlineLoess()
result = add_point(online, x[1], y[1])  # returns OnlineOutput or nothing
```

* Adds a single point to the sliding window. Returns `nothing` while the window is still filling (fewer than `min_points` seen), and an `OnlineOutput` once smoothing begins.

## Options Structure

### `OnlineOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `window_capacity` | `Int` | `1000` | Max points in sliding window |
| `min_points` | `Int` | `3` | Min points before smoothing starts |
| `update_mode` | `String` | `"full"` | Update mode (`"full"` or `"incremental"`) |
| `parallel` | `Bool` | `false` | Enable parallel execution (off by default; online LOESS fits one point at a time) |

## Result Structure

### `OnlineOutput`

Returned by `add_point()` once the window has enough points (`nothing` until then).

| Field | Type | Description |
| --- | --- | --- |
| `smoothed` | `Float64` | Smoothed value for the latest point |
| `std_error` | `Union{Float64, Nothing}` | Standard error (if requested) |
| `residual` | `Union{Float64, Nothing}` | Residual y − smoothed (if requested) |
| `robustness_weight` | `Union{Float64, Nothing}` | Robustness weight (if requested) |
| `iterations_used` | `Union{Int, Nothing}` | Robustness iterations performed |

## Options

### update_mode

*See: [Execution Modes](../user-guide/adapters.md)*

* `"full"` (default; alias: `"resmooth"`)
* `"incremental"` (alias: `"single"`)
