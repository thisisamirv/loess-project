<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOESS library for your preferred language.

```toml
# loess-rs (no_std compatible)
[dependencies]
loess-rs = "0.9"

# loess_rs (parallel + GPU)
[dependencies]
loess_rs = { version = "0.9", features = ["cpu"] }
```

## Feature Flags

| Crate | Feature | Description |
| --- | --- | --- |
| `loess-rs` | `std` | Enable standard library (default) |
| `loess_rs` | `cpu` | Enable CPU parallelism via Rayon |
| `loess_rs` | `gpu` | Enable GPU acceleration via wgpu (beta) |

---

## Verify Installation

```rust
use loess_rs::prelude::*;

fn main() -> Result<(), LoessError> {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    
    let model = Loess::new().build()?;
    let result = model.fit(&x, &y)?;
    
    println!("Installed successfully!");
    Ok(())
}
```
