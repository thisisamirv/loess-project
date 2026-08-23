=== "From crates.io"

```toml
# loess-rs (no_std compatible)
[dependencies]
loess-rs = "0.9"

# fastLoess (parallel + GPU)
[dependencies]
fastLoess = { version = "0.9", features = ["cpu"] }
```

=== "Feature Flags"

    | Crate | Feature | Description |
    | --- | --- | --- |
    | `loess-rs` | `std` | Enable standard library (default) |
    | `fastLoess` | `cpu` | Enable CPU parallelism via Rayon |
    | `fastLoess` | `gpu` | Enable GPU acceleration via wgpu (beta) |