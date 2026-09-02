<!-- markdownlint-disable MD024 MD033 MD046 -->
# Merge Strategies

How overlapping chunk boundaries are reconciled in Streaming mode.

## Overview

Streaming LOESS processes data in fixed-size chunks with a configurable overlap. Points inside the overlap zone are fitted twice — once by the left chunk and once by the right chunk. The `merge_strategy` decides how those two estimates are combined into a single output value.

```text
Chunk A:   [=========|=====]
Chunk B:            [=====|=========]
Overlap:            [=====]
                      ↑
                 merge_strategy
                 applied here
```

| Strategy | Method | Robustness | Speed |
| --- | --- | --- | --- |
| `"average"` | Simple mean of both estimates | Low | Fastest |
| `"take_first"` | Left-chunk estimate only | Low | Fastest |
| `"take_last"` | Right-chunk estimate only | Low | Fastest |
| `"weighted_average"` | Distance-weighted mean | High | Moderate |

![Merge Strategies](https://raw.githubusercontent.com/thisisamirv/loess-project/main/crates/loess-rs/assets/diagrams/merge_comparison.svg)

---

## Average

Takes the arithmetic mean of the left-chunk and right-chunk estimates in the overlap region. Fast and sufficient when both chunks have similar smoothing quality.

**Use when**: Chunks are large and the overlap region has uniform data density.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut model = StreamingLoess::new()
        .merge_strategy("average")
        .chunk_size(60usize)
        .overlap(20usize)
        .build()?;
    let _ = model.process_chunk(&x_chunk[..60], &y_chunk[..60])?;
    // The second chunk's overlap region (its first 20 points) is where
    // merge_strategy actually blends the two chunks' estimates.
    let result = model.process_chunk(&x_chunk[60..], &y_chunk[60..])?;
    println!("Merged value in overlap region (average): {}", result.y[5]);
    Ok(())
}
```

```output
Merged value in overlap region (average): 0.3676296872490885
```

---

## Take First

Keeps only the left-chunk estimate in the overlap zone and discards the right-chunk estimate. Produces a definitive, non-revised output as soon as the right boundary of each chunk is reached.

**Use when**: You need final output values immediately after each chunk (no look-ahead revision); left-chunk data quality is higher.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut model = StreamingLoess::new()
        .merge_strategy("take_first")
        .chunk_size(60usize)
        .overlap(20usize)
        .build()?;
    let _ = model.process_chunk(&x_chunk[..60], &y_chunk[..60])?;
    let result = model.process_chunk(&x_chunk[60..], &y_chunk[60..])?;
    println!("Merged value in overlap region (take_first): {}", result.y[5]);
    Ok(())
}
```

```output
Merged value in overlap region (take_first): 0.3576370673373696
```

---

## Take Last

Keeps only the right-chunk estimate in the overlap zone. The right chunk sees more of the surrounding data, so its fit can be more accurate near the left boundary of the new chunk.

**Use when**: Right-chunk context improves overlap quality; you are post-processing complete data rather than streaming live.

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut model = StreamingLoess::new()
        .merge_strategy("take_last")
        .chunk_size(60usize)
        .overlap(20usize)
        .build()?;
    let _ = model.process_chunk(&x_chunk[..60], &y_chunk[..60])?;
    let result = model.process_chunk(&x_chunk[60..], &y_chunk[60..])?;
    println!("Merged value in overlap region (take_last): {}", result.y[5]);
    Ok(())
}
```

```output
Merged value in overlap region (take_last): 0.3776223071608074
```

---

## Weighted Average

Assigns each overlap point a weight proportional to its proximity to the centre of its respective chunk: points near the left-chunk centre get higher left weight; points near the right-chunk centre get higher right weight. This produces the smoothest transition across chunk boundaries.

$$\hat{y} = \frac{w_L \hat{y}_L + w_R \hat{y}_R}{w_L + w_R}$$

where $w_L$ and $w_R$ are linear distance weights from the chunk centres.

**Use when**: Minimising boundary artefacts is more important than speed; moderate overlap (10–20 % of chunk size).

```rust
use loess_rs::prelude::*;
use std::f64::consts::TAU;

fn main() -> Result<(), LoessError> {
    let n = 100usize;
    let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
    let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

    let mut model = StreamingLoess::new()
        .merge_strategy("weighted_average")
        .chunk_size(60usize)
        .overlap(20usize)
        .build()?;
    let _ = model.process_chunk(&x_chunk[..60], &y_chunk[..60])?;
    let result = model.process_chunk(&x_chunk[60..], &y_chunk[60..])?;
    println!("Merged value in overlap region (weighted_average): {}", result.y[5]);
    Ok(())
}
```

```output
Merged value in overlap region (weighted_average): 0.36263337729322903
```

---

## Choosing a Strategy

| Situation | Recommended Strategy |
| --- | --- |
| General purpose | `"weighted_average"` |
| Maximum throughput | `"average"` |
| Immediate finalised output | `"take_first"` |
| Post-processing, right context better | `"take_last"` |
| Minimising boundary artefacts | `"weighted_average"` |

!!! tip "Overlap size matters"
    A larger overlap gives the merge strategy more room to blend, reducing boundary artefacts regardless of the strategy chosen. A good starting point is 10 % of `chunk_size`.
