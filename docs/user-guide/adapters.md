<!-- markdownlint-disable MD024 MD033 MD046 -->
# Execution Modes

Choose the right adapter for your use case.

## Overview

```mermaid
graph LR
    A[Data] --> B{Size?}
    B -->|Fits in memory| C{Real-time?}
    B -->|Too large| D[Streaming]
    C -->|No| E[Batch]
    C -->|Yes| F[Online]
```

| Mode | Use Case | Memory | Features |
| --- | --- | --- | --- |
| **Batch** | Complete datasets | Full | All features |
| **Streaming** | Large files (>100K) | Chunked | Residuals, robustness |
| **Online** | Real-time sensors | Fixed window | Incremental updates |

![Adapter Comparison](../assets/diagrams/adapter_comparison.svg)

---

## Batch Adapter

Standard mode for complete datasets. **Supports all features.**

### When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

![Gap Handling](../assets/diagrams/gap_handling.svg)

### Example

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    model <- Loess(
        fraction = 0.5,
        iterations = 3,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95,
        return_diagnostics = TRUE,
        parallel = TRUE
    )
    result <- model$fit(x, y)
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np

    rng = np.random.default_rng(42)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + rng.normal(0, 0.3, 100)

    model = fl.Loess(
        fraction=0.5,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95,
        return_diagnostics=True,
        parallel=True
    )
    result = model.fit(x, y)
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

        let model = Loess::new()
            .fraction(0.5)
            .iterations(3)
            .confidence_intervals(0.95)
            .prediction_intervals(0.95)
            .return_diagnostics()
            .parallel(true)
            .build()?;

        let result = model.fit(&x, &y)?;

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

    model = Loess(;
        fraction=0.5,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95,
        return_diagnostics=true,
        parallel=true
    )
    result = fit(model, x, y)
    ```

=== "Node.js"
    ```javascript
    const { Loess } = require('fastloess');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Loess({
        fraction: 0.5,
        iterations: 3,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
        return_diagnostics: true
    });
    const result = model.fit(x, y);
    ```

=== "WebAssembly"
    ```javascript
    import init, { Loess } from 'fastloess-wasm';
    await init();

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Loess({
        fraction: 0.5,
        iterations: 3,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
        return_diagnostics: true
    });
    const result = model.fit(x, y);
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

        fastloess::Loess model({
            .fraction = 0.5,
            .iterations = 3,
            .confidence_intervals = 0.95,
            .prediction_intervals = 0.95,
            .return_diagnostics = true,
            .parallel = true
        });
        auto result = model.fit(x, y).value();

        return 0;
    }
    ```

---

## Streaming Adapter

Process large datasets in chunks with configurable overlap.

### When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `chunk_size` | 5000 | Points per chunk |
| `overlap` | 500 | Overlap between chunks |
| `merge_strategy` | `"weighted_average"` | How to merge overlaps |

### Merge Strategies

| Strategy | Behavior |
| --- | --- |
| `"average"` | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend (default) |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

### Example

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    model <- StreamingLoess(
        fraction = 0.3,
        iterations = 2,
        chunk_size = 5000,
        overlap = 500,
        merge_strategy = "average"
    )
    result <- model$process_chunk(x, y)
    final <- model$finalize()
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np

    rng = np.random.default_rng(42)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + rng.normal(0, 0.3, 100)

    model = fl.StreamingLoess(
        fraction=0.3,
        iterations=2,
        chunk_size=5000,
        overlap=500,
        merge_strategy="average"
    )
    model.process_chunk(x, y)
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

        let mut processor = StreamingLoess::new()
            .fraction(0.3)
            .iterations(2)
            .chunk_size(50)
            .overlap(10)
            .merge_strategy("average")
            .build()?;

        let result = processor.process_chunk(&x_chunk, &y_chunk)?;
        println!("Chunk processed: {} points", result.y.len());

        let final_result = processor.finalize()?;
        println!("Final: {} points", final_result.y.len());

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

    model = StreamingLoess(;
        fraction=0.3,
        iterations=2,
        chunk_size=5000,
        overlap=500,
        merge_strategy="average"
    )
    process_chunk(model, x, y)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const { StreamingLoess } = require('fastloess');

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess(
        { fraction: 0.3, iterations: 2 },
        { chunk_size: 5000, overlap: 500 }
    );

    const result = processor.process_chunk(xChunk, yChunk);
    const finalResult = processor.finalize();
    ```

=== "WebAssembly"
    ```javascript
    import init, { StreamingLoess } from 'fastloess-wasm';
    await init();

    const n = 100;
    const xChunk = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const yChunk = Float64Array.from(xChunk, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new StreamingLoess(
        { fraction: 0.3, iterations: 2 },
        { chunk_size: 5000, overlap: 500 }
    );

    const result = processor.process_chunk(xChunk, yChunk);
    const finalResult = processor.finalize();
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
        opts.fraction = 0.3;
        opts.iterations = 2;
        opts.chunk_size = 5000;
        opts.overlap = 500;

        fastloess::StreamingLoess stream(opts);
        (void)stream.process_chunk(x, y);
        auto result = stream.finalize().value();

        return 0;
    }
    ```

---

!!! warning "Always call finalize()"
    In Rust, always call `processor.finalize()` after processing all chunks to retrieve buffered overlap data.

## Online Adapter

Incremental updates with a sliding window for real-time data.

### When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](../assets/diagrams/online_comparison.svg)

### Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `window_capacity` | 1000 | Max points in window |
| `min_points` | 2 | Points before output starts |
| `update_mode` | `"incremental"` | Update strategy |

### Update Modes

| Mode | Behavior | Speed |
| --- | --- | --- |
| `"incremental"` | Update only affected fits | Faster |
| `"full"` | Recompute entire window | More accurate |

### Example

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    model <- OnlineLoess(
        fraction = 0.2,
        iterations = 1,
        window_capacity = 100,
        min_points = 5,
        update_mode = "incremental"
    )
    for (i in seq_along(x)) {
        result <- model$add_point(x[i], y[i])
        if (!is.null(result)) cat(result$smoothed, "\n")
    }
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np

    rng = np.random.default_rng(42)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + rng.normal(0, 0.3, 100)

    model = fl.OnlineLoess(
        fraction=0.2,
        iterations=1,
        window_capacity=100,
        min_points=5,
        update_mode="incremental"
    )
    for xi, yi in zip(x, y):
        result = model.add_point(float(xi), float(yi))
        if result is not None:
            print(result.smoothed)
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

        let mut processor = OnlineLoess::new()
            .fraction(0.2)
            .iterations(1)
            .window_capacity(100)
            .min_points(5)
            .update_mode("incremental")
            .build()?;

        for i in 0..x.len() {
            if let Some(output) = processor.add_point(&[x[i]], y[i])? {
                println!("Smoothed: {:.2}", output.smoothed);
            }
        }

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

    model = OnlineLoess(;
        fraction=0.2,
        iterations=1,
        window_capacity=100,
        min_points=5,
        update_mode="incremental"
    )
    for i in eachindex(x)
        result = add_point(model, x[i], y[i])
        if result !== nothing
            println(result.smoothed)
        end
    end
    ```

=== "Node.js"
    ```javascript
    const { OnlineLoess } = require('fastloess');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new OnlineLoess(
        { fraction: 0.2, iterations: 1 },
        { window_capacity: 100, min_points: 5, update_mode: "incremental" }
    );

    for (let i = 0; i < n; i++) {
        const res = processor.add_point(x[i], y[i]);
        if (res !== null) {
            console.log(`Smoothed: ${res.smoothed.toFixed(2)}`);
        }
    }
    ```

=== "WebAssembly"
    ```javascript
    import init, { OnlineLoess } from 'fastloess-wasm';
    await init();

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new OnlineLoess(
        { fraction: 0.2, iterations: 1 },
        { window_capacity: 100, min_points: 5, update_mode: "incremental" }
    );

    for (let i = 0; i < n; i++) {
        const res = processor.add_point(x[i], y[i]);
        if (res !== undefined) {
            console.log(`Smoothed: ${res.smoothed.toFixed(2)}`);
        }
    }
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

        fastloess::OnlineOptions opts;
        opts.fraction = 0.2;
        opts.iterations = 1;
        opts.window_capacity = 100;
        opts.min_points = 5;
        opts.update_mode = "incremental";

        fastloess::OnlineLoess model(opts);
        for (size_t i = 0; i < x.size(); ++i) {
            auto out = model.add_point(x[i], y[i]).value();
            if (out.has_value())
                std::cout << out.smoothed() << std::endl;
        }

        return 0;
    }
    ```

---

## Feature Comparison

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Cross-validation | ✓ | ✗ | ✗ |
| Diagnostics | ✓ | ✓ | ✗ |
| Residuals | ✓ | ✓ | ✓ |
| Robustness weights | ✓ | ✓ | ✓ |
| Parallel execution | ✓ | ✓ | ✗ |

---

## Next Steps

- [Parameters](parameters.md) — All configuration options
- [Tutorials](../tutorials/real-time.md) — Real-time processing guide
