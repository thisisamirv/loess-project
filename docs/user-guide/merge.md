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

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

---

## Average

Takes the arithmetic mean of the left-chunk and right-chunk estimates in the overlap region. Fast and sufficient when both chunks have similar smoothing quality.

**Use when**: Chunks are large and the overlap region has uniform data density.

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)
    set.seed(42)
    n <- 100
    x_chunk <- seq(0, 2 * pi, length.out = n)
    y_chunk <- sin(x_chunk) + rnorm(n, sd = 0.3)

    model <- StreamingLoess(
        merge_strategy = "average",
        chunk_size = 5000,
        overlap = 500
    )
    result <- model$process_chunk(x_chunk, y_chunk)
    ```

=== "Python"
    ```python
    from fastloess import StreamingLoess
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100
    x_chunk = np.linspace(0, 2 * np.pi, n)
    y_chunk = np.sin(x_chunk) + rng.normal(0, 0.3, n)

    model = StreamingLoess(merge_strategy="average", chunk_size=5000, overlap=500)
    result = model.process_chunk(x_chunk, y_chunk)
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

        let mut model = StreamingLoess::new()
            .merge_strategy("average")
            .chunk_size(5000)
            .overlap(500)
            .build()?;
        let result = model.process_chunk(&x_chunk, &y_chunk)?;;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3

    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    n = 100
    x_chunk = collect(range(0, 2π, length=n))
    y_chunk = sin.(x_chunk) .+ randn(rng, n) .* 0.3

    model = StreamingLoess(; merge_strategy="average", chunk_size=5000, overlap=500)
    result = process_chunk(model, x_chunk, y_chunk)
    ```

=== "Node.js"
    ```javascript
    const { StreamingLoess } = require('fastloess');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess(
        {},
        { merge_strategy: "average", chunk_size: 5000, overlap: 500 }
    );
    const result = processor.process_chunk(xChunk, yChunk);
    ```

=== "WebAssembly"
    ```javascript
    const { StreamingLoess } = require('fastloess-wasm');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess(
        {},
        { merge_strategy: "average", chunk_size: 5000, overlap: 500 }
    );
    const result = processor.process_chunk(xChunk, yChunk);
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = i * 2 * M_PI / (n - 1);
            y[i] = std::sin(x[i]) + 0.1;
        }

        fastloess::StreamingOptions opts;
        opts.merge_strategy = "average";
        opts.chunk_size = 5000;
        opts.overlap = 500;
        fastloess::StreamingLoess stream(opts);
        (void)stream.process_chunk(x, y);
        auto result = stream.finalize().value();

        return 0;
    }
    ```

---

## Take First

Keeps only the left-chunk estimate in the overlap zone and discards the right-chunk estimate. Produces a definitive, non-revised output as soon as the right boundary of each chunk is reached.

**Use when**: You need final output values immediately after each chunk (no look-ahead revision); left-chunk data quality is higher.

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)
    set.seed(42)
    n <- 100
    x_chunk <- seq(0, 2 * pi, length.out = n)
    y_chunk <- sin(x_chunk) + rnorm(n, sd = 0.3)

    model <- StreamingLoess(merge_strategy = "take_first")
    result <- model$process_chunk(x_chunk, y_chunk)
    final <- model$finalize()
    ```

=== "Python"
    ```python
    from fastloess import StreamingLoess
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100
    x_chunk = np.linspace(0, 2 * np.pi, n)
    y_chunk = np.sin(x_chunk) + rng.normal(0, 0.3, n)

    model = StreamingLoess(merge_strategy="take_first")
    model.process_chunk(x_chunk, y_chunk)
    result = model.finalize()
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

        let mut model = StreamingLoess::new()
            .merge_strategy("take_first")
            .build()?;
        let result = model.process_chunk(&x_chunk, &y_chunk)?;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3

    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    n = 100
    x_chunk = collect(range(0, 2π, length=n))
    y_chunk = sin.(x_chunk) .+ randn(rng, n) .* 0.3

    model = StreamingLoess(; merge_strategy="take_first")
    process_chunk(model, x_chunk, y_chunk)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const { StreamingLoess } = require('fastloess');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess({}, { merge_strategy: "take_first" });
    const result = processor.process_chunk(xChunk, yChunk);
    const final_ = processor.finalize();
    ```

=== "WebAssembly"
    ```javascript
    const { StreamingLoess } = require('fastloess-wasm');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess({}, { merge_strategy: "take_first" });
    const result = processor.process_chunk(xChunk, yChunk);
    const final_ = processor.finalize();
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = i * 2 * M_PI / (n - 1);
            y[i] = std::sin(x[i]) + 0.1;
        }

        fastloess::StreamingOptions s_opts;
        s_opts.merge_strategy = "take_first";
        fastloess::StreamingLoess stream(s_opts);
        (void)stream.process_chunk(x, y);
        auto result = stream.finalize().value();

        return 0;
    }
    ```

---

## Take Last

Keeps only the right-chunk estimate in the overlap zone. The right chunk sees more of the surrounding data, so its fit can be more accurate near the left boundary of the new chunk.

**Use when**: Right-chunk context improves overlap quality; you are post-processing complete data rather than streaming live.

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)
    set.seed(42)
    n <- 100
    x_chunk <- seq(0, 2 * pi, length.out = n)
    y_chunk <- sin(x_chunk) + rnorm(n, sd = 0.3)

    model <- StreamingLoess(merge_strategy = "take_last")
    result <- model$process_chunk(x_chunk, y_chunk)
    final <- model$finalize()
    ```

=== "Python"
    ```python
    from fastloess import StreamingLoess
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100
    x_chunk = np.linspace(0, 2 * np.pi, n)
    y_chunk = np.sin(x_chunk) + rng.normal(0, 0.3, n)

    model = StreamingLoess(merge_strategy="take_last")
    model.process_chunk(x_chunk, y_chunk)
    result = model.finalize()
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

        let mut model = StreamingLoess::new()
            .merge_strategy("take_last")
            .build()?;
        let result = model.process_chunk(&x_chunk, &y_chunk)?;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3

    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    n = 100
    x_chunk = collect(range(0, 2π, length=n))
    y_chunk = sin.(x_chunk) .+ randn(rng, n) .* 0.3

    model = StreamingLoess(; merge_strategy="take_last")
    process_chunk(model, x_chunk, y_chunk)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const { StreamingLoess } = require('fastloess');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess({}, { merge_strategy: "take_last" });
    const result = processor.process_chunk(xChunk, yChunk);
    const final_ = processor.finalize();
    ```

=== "WebAssembly"
    ```javascript
    const { StreamingLoess } = require('fastloess-wasm');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess({}, { merge_strategy: "take_last" });
    const result = processor.process_chunk(xChunk, yChunk);
    const final_ = processor.finalize();
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = i * 2 * M_PI / (n - 1);
            y[i] = std::sin(x[i]) + 0.1;
        }

        fastloess::StreamingOptions s_opts;
        s_opts.merge_strategy = "take_last";
        fastloess::StreamingLoess stream(s_opts);
        (void)stream.process_chunk(x, y);
        auto result = stream.finalize().value();

        return 0;
    }
    ```

---

## Weighted Average

Assigns each overlap point a weight proportional to its proximity to the centre of its respective chunk: points near the left-chunk centre get higher left weight; points near the right-chunk centre get higher right weight. This produces the smoothest transition across chunk boundaries.

$$\hat{y} = \frac{w_L \hat{y}_L + w_R \hat{y}_R}{w_L + w_R}$$

where $w_L$ and $w_R$ are linear distance weights from the chunk centres.

**Use when**: Minimising boundary artefacts is more important than speed; moderate overlap (10–20 % of chunk size).

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)
    set.seed(42)
    n <- 100
    x_chunk <- seq(0, 2 * pi, length.out = n)
    y_chunk <- sin(x_chunk) + rnorm(n, sd = 0.3)

    model <- StreamingLoess(
        merge_strategy = "weighted_average",
        chunk_size = 5000,
        overlap = 500
    )
    result <- model$process_chunk(x_chunk, y_chunk)
    final <- model$finalize()
    ```

=== "Python"
    ```python
    from fastloess import StreamingLoess
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100
    x_chunk = np.linspace(0, 2 * np.pi, n)
    y_chunk = np.sin(x_chunk) + rng.normal(0, 0.3, n)

    model = StreamingLoess(
        merge_strategy="weighted_average",
        chunk_size=5000,
        overlap=500
    )
    model.process_chunk(x_chunk, y_chunk)
    result = model.finalize()
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let x_chunk: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y_chunk: Vec<f64> = x_chunk.iter().map(|&xi| xi.sin() + 0.1).collect();

        let mut model = StreamingLoess::new()
            .merge_strategy("weighted_average")
            .chunk_size(5000)
            .overlap(500)
            .build()?;
        let result = model.process_chunk(&x_chunk, &y_chunk)?;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3

    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    n = 100
    x_chunk = collect(range(0, 2π, length=n))
    y_chunk = sin.(x_chunk) .+ randn(rng, n) .* 0.3

    model = StreamingLoess(;
        merge_strategy="weighted_average",
        chunk_size=5000,
        overlap=500
    )
    process_chunk(model, x_chunk, y_chunk)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const { StreamingLoess } = require('fastloess');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess(
        {},
        { merge_strategy: "weighted_average", chunk_size: 5000, overlap: 500 }
    );
    const result = processor.process_chunk(xChunk, yChunk);
    const final_ = processor.finalize();
    ```

=== "WebAssembly"
    ```javascript
    const { StreamingLoess } = require('fastloess-wasm');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess(
        {},
        { merge_strategy: "weighted_average", chunk_size: 5000, overlap: 500 }
    );
    const result = processor.process_chunk(xChunk, yChunk);
    const final_ = processor.finalize();
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = i * 2 * M_PI / (n - 1);
            y[i] = std::sin(x[i]) + 0.1;
        }

        fastloess::StreamingOptions s_opts;
        s_opts.merge_strategy = "weighted_average";
        fastloess::StreamingLoess stream(s_opts);
        (void)stream.process_chunk(x, y);
        auto result = stream.finalize().value();

        return 0;
    }
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
