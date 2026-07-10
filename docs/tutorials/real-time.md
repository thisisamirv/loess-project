# Real-Time Processing

Streaming and online LOESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

### Sensor Data Example

=== "R"
    ```r
    library(rfastloess)

    set.seed(42)
    times <- 1:100
    temperatures <- 20 + 5 * sin(times / 10) + rnorm(100)

    model <- OnlineLoess(
        fraction = 0.3,
        window_capacity = 25,
        min_points = 5,
        update_mode = "incremental"
    )
    for (i in seq_along(times)) {
        result <- model$add_point(times[i], temperatures[i])
        if (!is.null(result))
            cat(sprintf("Time %d: %.2f\n", times[i], result$smoothed))
    }
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np

    # Simulate sensor readings arriving over time
    np.random.seed(42)
    n_readings = 100
    times = np.arange(n_readings, dtype=float)
    temperatures = 20 + 5 * np.sin(times / 10) + np.random.normal(0, 1, n_readings)

    # Process with online mode
    online = fl.OnlineLoess(
        fraction=0.3,
        window_capacity=25,    # Keep last 25 points
        min_points=5,          # Wait for 5 points before output
        update_mode="incremental"
    )
    for xi, yi in zip(times, temperatures):
        result = online.add_point(float(xi), float(yi))
        if result is not None:
            print(f"Time {xi:.0f}: smoothed = {result.smoothed:.2f}")
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;

    let mut processor = OnlineLoess::new()
        .fraction(0.3)
        .iterations(1)
        .window_capacity(25)
        .min_points(5)
        .update_mode("incremental")
        .build()?;

    for i in 0..100 {
        let xi = i as f64;
        let yi = 20.0 + 5.0 * (xi / 10.0).sin() + (xi * 1.7).sin() * 0.5;

        if let Some(output) = processor.add_point(&[xi], yi)? {
            println!("Time {}: smoothed = {:.2}", xi, output.smoothed);
        }
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS

    # Simulate sensor readings 
    times = collect(Float64, 1:100)
    temperatures = 20.0 .+ 5.0 .* sin.(times ./ 10.0) .+ randn(100)

    # Process with online mode
    model = OnlineLoess(;
        fraction=0.3,
        window_capacity=25,
        min_points=5,
        update_mode="incremental"
    )
    for i in eachindex(times)
        result = add_point(model, times[i], temperatures[i])
        if result !== nothing
            println("Time $(times[i]): smoothed = $(round(result.smoothed; digits=2))")
        end
    end
    ```

=== "Node.js"
    ```javascript
    const { OnlineLoess } = require('fastloess');

    const processor = new OnlineLoess(
        { fraction: 0.3, iterations: 1 },
        { window_capacity: 25, min_points: 5, update_mode: "incremental" }
    );

    // Simulate real-time data arrival
    for (let i = 0; i < 100; i++) {
        const x = i;
        const y = 20 + 5 * Math.sin(x / 10) + Math.random();
        
        const res = processor.add_point(x, y);
        if (res !== null) {
            console.log(`Time ${x}: smoothed = ${res.smoothed.toFixed(2)}`);
        }
    }
    ```

=== "WebAssembly"
    ```javascript
    import { OnlineLoess } from 'fastloess-wasm';

    const processor = new OnlineLoess(
        { fraction: 0.3, iterations: 1 },
        { window_capacity: 25, min_points: 5, update_mode: "incremental" }
    );

    // Simulate real-time data arrival
    for (let i = 0; i < readings.length; i++) {
        const res = processor.add_point(readings[i].x, readings[i].y);
        if (res !== undefined) {
            // Update dashboard UI with res.smoothed
        }
    }
    ```

=== "C++"
    ```cpp
    #include "fastloess.hpp"

    // Online mode processes points incrementally
    fastloess::OnlineOptions opts;
    opts.fraction = 0.3;
    opts.iterations = 1;
    opts.window_capacity = 25;
    opts.min_points = 5;
    opts.update_mode = "incremental";

    fastloess::OnlineLoess model(opts);
    for (size_t i = 0; i < times.size(); ++i) {
        auto res = model.add_point(times[i], temperatures[i]).value();
        if (res.has_value()) {
            std::cout << "Time " << times[i] << ": " << res.smoothed() << std::endl;
        }
    }
    ```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

### Log File Processing

=== "R"
    ```r
    x <- seq(0, 100000, by = 1)
    y <- sin(x / 1000) + rnorm(length(x), sd = 0.1)

    model <- StreamingLoess(
        fraction = 0.05,
        chunk_size = 10000,
        overlap = 1000,
        merge_strategy = "weighted_average"
    )
    result <- model$process_chunk(x, y)
    final <- model$finalize()
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np

    # Simulate large dataset arriving in chunks
    total_points = 100000
    chunk_size = 10000
    
    # All at once with streaming handles chunking internally
    x = np.arange(total_points, dtype=float)
    y = np.sin(x / 1000) + np.random.normal(0, 0.1, total_points)
    
    model = fl.StreamingLoess(
        fraction=0.05,
        chunk_size=10000,
        overlap=1000,
        merge_strategy="weighted_average"
    )
    model.process_chunk(x, y)
    result = model.finalize()
    
    print(f"Processed {len(result.y)} points")
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;

    let mut processor = StreamingLoess::new()
        .fraction(0.1)
        .iterations(2)
        .chunk_size(50)
        .overlap(10)
        .merge_strategy("weighted_average")
        .build()?;

    // Process chunks as they arrive
    processor.process_chunk(&chunk1_x, &chunk1_y)?;
    processor.process_chunk(&chunk2_x, &chunk2_y)?;

    // CRITICAL: Get buffered overlap data
    let final_result = processor.finalize()?;
    ```

=== "Julia"
    ```julia
    using FastLOESS

    # Large dataset
    x = collect(0.0:1.0:100000.0)
    y = sin.(x ./ 1000) .+ randn(length(x)) .* 0.1

    # Streaming mode handles everything internally
    model = StreamingLoess(;
        fraction=0.05,
        chunk_size=10000,
        overlap=1000,
        merge_strategy="weighted_average"
    )
    process_chunk(model, x, y)
    result = finalize(model)
    ```

=== "Node.js"
    ```javascript
    const { StreamingLoess } = require('fastloess');

    const processor = new StreamingLoess(
        { fraction: 0.1, iterations: 2 },
        { chunk_size: 5000, overlap: 500 }
    );

    // Process chunks
    const r1 = processor.processChunk(chunk1_x, chunk1_y);
    const r2 = processor.processChunk(chunk2_x, chunk2_y);

    // Always get buffered data
    const finalResult = processor.finalize();
    ```

=== "WebAssembly"
    ```javascript
    import { StreamingLoess } from 'fastloess-wasm';

    const processor = new StreamingLoess(
        { fraction: 0.1, iterations: 2 },
        { chunk_size: 5000, overlap: 500 }
    );

    // Process chunks as they arrive
    const result1 = processor.processChunk(x1, y1);
    const result2 = processor.processChunk(x2, y2);
    const finalResult = processor.finalize();
    ```

=== "C++"
    ```cpp
    #include "fastloess.hpp"

    fastloess::StreamingOptions opts;
    opts.fraction = 0.1;
    opts.iterations = 2;
    opts.chunk_size = 5000;
    opts.overlap = 500;

    fastloess::StreamingLoess stream(opts);
    (void)stream.process_chunk(x, y);
    auto result = stream.finalize().value();

    std::cout << "Processed " << result.y_vector().size() << " points" << std::endl;
    ```

!!! warning "Always call finalize()"
    The streaming adapter buffers overlap data. Always call `finalize()` to retrieve the last chunk.

---

## Real-Time Dashboard Example

=== "R"
    ```r
    library(rfastloess)

    # Simulated real-time dashboard
    window_capacity <- 50
    data_x <- numeric(0)
    data_y <- numeric(0)

    for (i in 1:200) {
        x <- i
        y <- 25.0 + 10 * sin(i / 20) + rnorm(1, sd = 2)
        
        data_x <- c(data_x, x)
        data_y <- c(data_y, y)
        
        if (length(data_x) > window_capacity) {
            data_x <- tail(data_x, window_capacity)
            data_y <- tail(data_y, window_capacity)
        }
        
        if (length(data_x) >= 5) {
            model <- Loess(fraction = 0.4)
            result <- model$fit(data_x, data_y)
            current_smoothed <- tail(result$y, 1)
        }
    }
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np

    # Simulated real-time dashboard sliding window
    window_capacity = 50
    data_x, data_y = [], []
    
    for i in range(200):
        x, y = i, 25.0 + 10 * np.sin(i / 20) + np.random.normal(0, 2)
        data_x.append(x)
        data_y.append(y)
        
        if len(data_x) > window_capacity:
            data_x = data_x[-window_capacity:]
            data_y = data_y[-window_capacity:]
        
        if len(data_x) >= 5:
            model = fl.Loess(fraction=0.4)
            result = model.fit(np.array(data_x, dtype=float), np.array(data_y, dtype=float))
            current_smoothed = result.y[-1]
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastloess');

    const window_capacity = 50;
    let dataX = [], dataY = [];

    for (let i = 0; i < 200; i++) {
        dataX.push(i);
        dataY.push(25.0 + 10 * Math.sin(i / 20) + Math.random() * 4 - 2);

        if (dataX.length > window_capacity) {
            dataX.shift();
            dataY.shift();
        }

        if (dataX.length >= 5) {
            const xArr = new Float64Array(dataX);
            const yArr = new Float64Array(dataY);
            const model = new fl.Loess({ fraction: 0.4 });
            const result = model.fit(xArr, yArr);
            const currentSmoothed = result.y[result.y.length - 1];
        }
    }
    ```

=== "WebAssembly"
    ```javascript
    import { Loess } from 'fastloess-wasm';

    // Sliding window logic
    for (const point of stream) {
        windowX.push(point.x);
        windowY.push(point.y);
        
        if (windowX.length > 50) {
            windowX.shift();
            windowY.shift();
        }

        const model = new Loess({ 
            fraction: 0.4 
        });
        const result = model.fit(new Float64Array(windowX), new Float64Array(windowY));
        const smoothed = result.y[result.y.length - 1];
    }
    ```

=== "C++"
    ```cpp
    // Sliding window over times/temperatures (skip until window has ≥2 points)
    for (size_t i = 0; i < times.size(); ++i) {
        windowX.push_back(times[i]);
        windowY.push_back(temperatures[i]);

        if (windowX.size() > 50) {
            windowX.erase(windowX.begin());
            windowY.erase(windowY.begin());
        }
        if (windowX.size() < 2) continue;

        fastloess::LoessOptions sw_opts;
        sw_opts.fraction = 0.4;
        fastloess::Loess model(sw_opts);
        auto result = model.fit(windowX, windowY).value();
        const auto smoothed = result.y_vector().back();
        (void)smoothed;
    }
    ```

---

## Choosing Parameters

### Online Mode

| Parameter | Guidance |
| --- | --- |
| `window_capacity` | Enough history for `fraction` to work |
| `min_points` | 2–5 typically; higher for stability |
| `update_mode` | `"incremental"` for speed, `"full"` for accuracy |

### Streaming Mode

| Parameter | Guidance |
| --- | --- |
| `chunk_size` | Balance memory vs. processing overhead |
| `overlap` | 10–20% of chunk_size for smooth transitions |
| `merge_strategy` | `"weighted_average"` for best quality, `"average"` for simplicity |

---

## Performance Considerations

| Mode | Memory | Latency | Use Case |
| --- | --- | --- | --- |
| **Online** | Fixed (window) | ~1ms/point | Sensors, dashboards |
| **Streaming** | ~chunk_size | ~100ms/chunk | Large files, ETL |
| **Batch** | Full dataset | N/A | Analysis, reports |

---

## See Also

- [Execution Modes](../user-guide/adapters.md) — Detailed mode comparison
- [Merge Strategies](../user-guide/merge.md) — Chunk reconciliation in depth
- [Scaling Methods](../user-guide/scaling.md) — Robustness scale estimation
- [Time Series](time-series.md) — General time series analysis
