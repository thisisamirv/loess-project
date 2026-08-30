# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::loess` (higher is better):

| Category | R (stats) | Serial |
| --- | --- | --- |
| **Clustered** | 1× | 18× |
| **Constant Y** | 1× | 15× |
| **Extreme Outliers** | 1× | 7× |
| **Financial** (500–5K) | 1× | 5× |
| **Fraction** (0.05–0.67) | 1× | 22× |
| **Genomic** (1K–5K) | 1× | 6× |
| **Genomic** (100K) | 1× | 137× |
| **High Noise** | 1× | 22× |
| **Iterations** (1–10) | 1× | 13× |
| **Scale** (1K–10K) | 1× | 8× |
| **Scientific** (500–5K) | 1× | 4× |

*Averages across all sizes within each category. The `loess-rs` crate has no `parallel` or `gpu` feature — for CPU-parallel and GPU-accelerated numbers, see the `fastLoess` crate's benchmarks.*

## Reproducing Benchmarks

Use `std::time::Instant` to time fit calls:

```rust
use loess_rs::prelude::*;
use std::time::Instant;

fn main() -> Result<(), LoessError> {
    let n = 5000usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() + (((i * 7 + 3) % 17) as f64 / 17.0 - 0.5) * 0.6)
        .collect();

    let reps = 10u32;
    let run = || Loess::new().fraction(0.67).build().unwrap().fit(&x, &y).unwrap();
    run(); // warm-up
    let t0 = Instant::now();
    for _ in 0..reps {
        run();
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(reps);

    println!("Fit: {:.2} ms", elapsed_ms);

    Ok(())
}
```
