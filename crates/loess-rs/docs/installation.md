=== "From crates.io"

```toml
# loess-rs (no_std compatible)
[dependencies]
loess-rs = "0.9"

# loess_rs (parallel + GPU)
[dependencies]
loess_rs = { version = "0.9", features = ["cpu"] }
```

=== "Feature Flags"

    | Crate | Feature | Description |
    | --- | --- | --- |
    | `loess-rs` | `std` | Enable standard library (default) |
    | `loess_rs` | `cpu` | Enable CPU parallelism via Rayon |
    | `loess_rs` | `gpu` | Enable GPU acceleration via wgpu (beta) |