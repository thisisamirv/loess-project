<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOESS library for your preferred language.

```toml
# loess-rs (no_std compatible)
[dependencies]
loess-rs = "*"

# fastLoess (parallel + GPU)
[dependencies]
fastLoess = { version = "*", features = ["cpu"] }
```

## Feature Flags

| Crate | Feature | Description |
| --- | --- | --- |
| `loess-rs` | `std` | Enable standard library (default) |
| `fastLoess` | `cpu` | Enable CPU parallelism via Rayon |
| `fastLoess` | `gpu` | Enable GPU acceleration via wgpu (beta) |

---

## Verify Installation

```rust
use fastLoess::prelude::*;

fn main() -> Result<(), LoessError> {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    
    let model = Loess::new().build()?;
    let result = model.fit(&x, &y)?;
    
    println!("Installed successfully!");
    Ok(())
}
```

```output
Installed successfully!
```
