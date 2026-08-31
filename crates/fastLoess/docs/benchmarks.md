# Benchmarks

## CPU Benchmarks

Speedup relative to R's `stats::loess` (higher is better):

| Category | R (stats) | Serial | Parallel |
| --- | --- | --- | --- |
| **Clustered** | 1× | 18× | **21×** |
| **Constant Y** | 1× | 15× | **21×** |
| **Extreme Outliers** | 1× | 7× | **8×** |
| **Financial** (500–5K) | 1× | 5× | **5×** |
| **Fraction** (0.05–0.67) | 1× | **22×** | 18× |
| **Genomic** (1K–5K) | 1× | 6× | **7×** |
| **Genomic** (100K) | 1× | 137× | **201×** |
| **High Noise** | 1× | 22× | **25×** |
| **Iterations** (1–10) | 1× | 13× | **16×** |
| **Large** (Direct) | 1× | 0.9× | **3×** |
| **Large** (High Fraction) | 1× | 577× | **695×** |
| **Large** (High Iterations) | 1× | 6× | **9×** |
| **Large** (Interpolate) | 1× | 60× | **83×** |
| **Scale** (1K–10K) | 1× | **8×** | 8× |
| **Scientific** (500–5K) | 1× | 4× | **5×** |

*Averages across all sizes within each category.*

## Reproducing Benchmarks

Use `std::time::Instant` to time serial vs parallel runs:

```rust
use fastLoess::prelude::*;
use std::time::Instant;

fn main() -> Result<(), LoessError> {
    let n = 5000usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() + (((i * 7 + 3) % 17) as f64 / 17.0 - 0.5) * 0.6)
        .collect();

    let bench_ms = |parallel: bool, reps: u32| -> f64 {
        let run = || Loess::new().fraction(0.67).parallel(parallel).build().unwrap().fit(&x, &y).unwrap();
        run(); // warm-up
        let t0 = Instant::now();
        for _ in 0..reps {
            run();
        }
        t0.elapsed().as_secs_f64() * 1000.0 / f64::from(reps)
    };

    let serial_ms = bench_ms(false, 10);
    let parallel_ms = bench_ms(true, 10);

    println!("Serial:   {:.2} ms", serial_ms);
    println!("Parallel: {:.2} ms", parallel_ms);
    println!("Speedup:  {:.2}x", serial_ms / parallel_ms);

    Ok(())
}
```

```output
Serial:   48.03 ms
Parallel: 32.03 ms
Speedup:  1.50x
```
