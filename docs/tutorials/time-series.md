<!-- markdownlint-disable MD024 MD046 MD033 -->
# Time Series Analysis

LOESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)

    set.seed(42)
    t <- seq(0, 100, length.out = 500)
    trend <- 10 + 0.5 * t + 3 * sin(t / 10)
    noise <- rnorm(500, sd = 3)
    y <- trend + noise

    model <- Loess(fraction = 0.1, iterations = 3)
    result <- model$fit(t, y)

    plot(t, y, col = "gray", pch = ".",
         xlab = "Time", ylab = "Value", main = "Trend Extraction")
    lines(result$x, result$y, col = "blue", lwd = 2)
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    # Simulate noisy time series with trend
    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend = 10 + 0.5 * t + 3 * np.sin(t / 10)
    noise = np.random.normal(0, 3, len(t))
    y = trend + noise

    # Extract trend with LOESS
    model = fl.Loess(fraction=0.1, iterations=3)
    result = model.fit(t, y)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.5, label="Observed")
    plt.plot(t, result.y, "b-", linewidth=2, label="Trend (LOESS)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Trend Extraction")
    plt.show()
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;

    fn main() -> Result<(), LoessError> {
        let n = 500usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 100.0 / (n - 1) as f64).collect();
        let y: Vec<f64> = t.iter().enumerate()
            .map(|(i, &ti)| 10.0 + 0.5 * ti + 3.0 * (ti / 10.0).sin()
                          + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 3.0)
            .collect();

        let model = Loess::new()
            .fraction(0.1)
            .iterations(3)
            .build()?;

        let result = model.fit(&t, &y)?;
        // result.y contains the trend

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS

    t = collect(range(0, 100, length=500))
    trend_true = 10.0 .+ 0.5 .* t .+ 3.0 .* sin.(t ./ 10.0)
    y = trend_true .+ randn(500) .* 3.0

    # Extract trend
    model = Loess(; fraction=0.1, iterations=3)
    result = fit(model, t, y)

    println("Extracted trend points: ", length(result.y))
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastloess');

    const n = 500;
    const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
    const y = Float64Array.from(t, ti => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (Math.random()-0.5)*6);

    // t and y are your time series arrays (Float64Array)
    const model = new fl.Loess({ 
        fraction: 0.1, 
        iterations: 3 
    });
    const result = model.fit(t, y);

    console.log("Extracted trend:", result.y);
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Loess({ 
        fraction: 0.1, 
        iterations: 3 
    });
    const result = model.fit(x, y);

    // Trend values in result.y
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> t(n), y(n);
        for (int i = 0; i < n; ++i) {
            t[i] = i * 2.0 * M_PI / (n - 1);
            y[i] = std::sin(t[i]) + 0.1;
        }

        fastloess::LoessOptions trend_opts;
        trend_opts.fraction = 0.1;
        trend_opts.iterations = 3;
        fastloess::Loess basic_model(trend_opts);
        auto basic_result = basic_model.fit(t, y).value();

        // Trend in basic_result.y_vector()

        return 0;
    }
    ```

---

## Detrending

Remove trend to analyze residual patterns:

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)
    set.seed(42)
    t <- seq(0, 100, length.out = 500)
    trend_true <- 10 + 0.5 *t + 3* sin(t / 10)
    y <- trend_true + rnorm(500, sd = 3)

    model <- Loess(fraction = 0.3, iterations = 3, return_residuals = TRUE)
    result <- model$fit(t, y)

    trend <- result$y
    detrended <- result$residuals

    par(mfrow = c(1, 2))
    plot(t, trend, type = "l", main = "Trend")
    plot(t, detrended, type = "l", main = "Detrended")
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend_true = 10 + 0.5 * t + 3 * np.sin(t / 10)
    y = trend_true + np.random.normal(0, 3, len(t))

    # Smooth to get trend
    model = fl.Loess(fraction=0.3, iterations=3, return_residuals=True)
    result = model.fit(t, y)

    trend = result.y
    detrended = result.residuals

    # Analyze residuals for seasonality, etc.
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, trend)
    plt.title("Extracted Trend")

    plt.subplot(1, 2, 2)
    plt.plot(t, detrended)
    plt.title("Detrended (Residuals)")
    plt.tight_layout()
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = t.iter().map(|&ti| ti.sin() + 0.1).collect();

        let model = Loess::new()
            .fraction(0.3)
            .iterations(3)
            .return_residuals()
            .build()?;

        let result = model.fit(&t, &y)?;
        let trend = &result.y;
        let detrended = result.residuals.as_ref().unwrap();

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    t = collect(range(0, 100, length=500))
    y = 10.0 .+ 0.5 .* t .+ 3.0 .* sin.(t ./ 10.0) .+ randn(rng, 500) .* 3.0

    # Smooth to get trend and residuals
    model = Loess(; fraction=0.3, iterations=3, return_residuals=true)
    result = fit(model, t, y)

    trend = result.y
    detrended = result.residuals

    println("Detrended variance: ", var(detrended))
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastloess');

    const n = 500;
    const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
    const y = Float64Array.from(t, ti => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (Math.random()-0.5)*6);

    const model = new fl.Loess({
        fraction: 0.3,
        iterations: 3,
        return_residuals: true
    });
    const result = model.fit(t, y);

    const trend = result.y;
    const detrended = result.residuals;
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Loess({ 
        fraction: 0.3, 
        iterations: 3, 
        return_residuals: true 
    });
    const result = model.fit(x, y);

    // Access result.y (trend) and result.residuals (detrended)
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> t(n), y(n);
        for (int i = 0; i < n; ++i) {
            t[i] = i * 2.0 * M_PI / (n - 1);
            y[i] = std::sin(t[i]) + 0.1;
        }

        fastloess::Loess model({
            .fraction = 0.3,
            .iterations = 3,
            .return_residuals = true
        });
        auto result = model.fit(t, y).value();

        auto trend = result.y_vector();
        auto detrended = result.residuals();

        return 0;
    }
    ```

---

## Forecasting with Prediction Intervals

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)
    set.seed(42)
    t <- seq(0, 100, length.out = 500)
    trend_true <- 10 + 0.5 *t + 3* sin(t / 10)
    y <- trend_true + rnorm(500, sd = 3)

    model <- Loess(
        fraction = 0.2,
        iterations = 3,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95
    )
    result <- model$fit(t, y)

    plot(t, y, col = "gray", pch = 16)
    lines(result$x, result$y, col = "blue", lwd = 2)
    lines(result$x, result$prediction_lower, col = "blue", lty = 2)
    lines(result$x, result$prediction_upper, col = "blue", lty = 2)
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend_true = 10 + 0.5 * t + 3 * np.sin(t / 10)
    y = trend_true + np.random.normal(0, 3, len(t))

    model = fl.Loess(
        fraction=0.2,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95
    )
    result = model.fit(t, y)

    # Plot with uncertainty bands
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.3)
    plt.plot(t, result.y, "b-", linewidth=2, label="Trend")
    plt.fill_between(
        t,
        result.prediction_lower,
        result.prediction_upper,
        alpha=0.2, color="blue", label="95% Prediction"
    )
    plt.legend()
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = t.iter().map(|&ti| ti.sin() + 0.1).collect();

        let model = Loess::new()
            .fraction(0.2)
            .iterations(3)
            .confidence_intervals(0.95)
            .prediction_intervals(0.95)
            .build()?;

        let result = model.fit(&t, &y)?;
        // Access result.prediction_lower and result.prediction_upper

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    t = collect(range(0, 100, length=500))
    y = 10.0 .+ 0.5 .* t .+ 3.0 .* sin.(t ./ 10.0) .+ randn(rng, 500) .* 3.0

    model = Loess(;
        fraction=0.2,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95
    )
    result = fit(model, t, y)

    # Intervals are available in result.prediction_lower/upper
    println("First point 95% PI: [$(result.prediction_lower[1]), $(result.prediction_upper[1])]")
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastloess');

    const n = 500;
    const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
    const y = Float64Array.from(t, ti => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (Math.random()-0.5)*6);

    const model = new fl.Loess({
        fraction: 0.2,
        iterations: 3,
        prediction_intervals: 0.95
    });
    const result = model.fit(t, y);

    console.log(`95% PI: [${result.prediction_lower[0]}, ${result.prediction_upper[0]}]`);
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Loess({
        fraction: 0.2,
        iterations: 3,
        prediction_intervals: 0.95
    });
    const result = model.fit(x, y);

    // Access result.prediction_lower and result.prediction_upper
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> t(n), y(n);
        for (int i = 0; i < n; ++i) {
            t[i] = i * 2.0 * M_PI / (n - 1);
            y[i] = std::sin(t[i]) + 0.1;
        }

        fastloess::Loess forecast_model({
            .fraction = 0.2,
            .iterations = 3,
            .confidence_intervals = 0.95,
            .prediction_intervals = 0.95
        });
        auto result = forecast_model.fit(t, y).value();

        // Access result.prediction_lower() and result.prediction_upper()

        return 0;
    }
    ```

---

## Handling Missing Data

LOESS naturally handles irregular time sampling:

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)

    t_irregular <- sort(runif(200, 0, 100))
    y_irregular <- 10 + 0.3 * t_irregular + rnorm(200, sd = 2)

    model <- Loess(fraction = 0.2)
    result <- model$fit(t_irregular, y_irregular)
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np

    rng = np.random.default_rng(42)

    # Irregular time points (gaps in data)
    t_irregular = np.sort(np.random.uniform(0, 100, 200))
    y_irregular = 10 + t_irregular * 0.3 + np.random.normal(0, 2, 200)

    # LOESS handles this seamlessly
    model = fl.Loess(fraction=0.2)
    result = model.fit(t_irregular, y_irregular)
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;

    fn main() -> Result<(), LoessError> {
        let t_irregular: Vec<f64> = (0..100).map(|i| i as f64 * 1.0 + (i * 31 % 10) as f64 * 0.1).collect();
        let y_irregular: Vec<f64> = t_irregular.iter().map(|&t| 10.0 + t * 0.3 + 2.0 * (t * 0.1).sin()).collect();

        // Irregular sampling - no special handling needed
        let model = Loess::new()
            .fraction(0.2)
            .build()?;

        let result = model.fit(&t_irregular, &y_irregular)?;

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

    # Irregular time points (gaps in data)
    t_irregular = sort(rand(200) .*100.0)
    y_irregular = 10.0 .+ t_irregular .* 0.3 .+ randn(200) .* 2.0

    # LOESS handles this seamlessly
    model = Loess(; fraction=0.2)
    result = fit(model, t_irregular, y_irregular)
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastloess');

    const n = 500;
    const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
    const y = Float64Array.from(t, ti => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (Math.random()-0.5)*6);

    const tIrregular = Float64Array.from({ length: 200 }, () => Math.random() * 100).sort((a,b)=>a-b);
    const yIrregular = Float64Array.from(tIrregular, t => 10 + 0.3 * t + Math.random() * 2);

    // No special handling needed for irregular spacing
    const model = new fl.Loess({ fraction: 0.2 });
    const result = model.fit(tIrregular, yIrregular);
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 100;
    const tIrregular = Float64Array.from({ length: n }, (_, i) => i * 1.0 + (i * 31 % 10) * 0.1).sort((a, b) => a - b);
    const yIrregular = Float64Array.from(tIrregular, t => 10 + 0.3 * t + 2.0 * Math.sin(t * 0.1));
    const model = new Loess({ fraction: 0.2 });
    const result = model.fit(tIrregular, yIrregular);
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> tIrregular(n), yIrregular(n);
        for (int i = 0; i < n; ++i) {
            tIrregular[i] = i * 1.0 + (i * 31 % 10) * 0.1;
            yIrregular[i] = 10.0 + 0.3 * tIrregular[i] + 2.0 * std::sin(tIrregular[i] * 0.1);
        }

        fastloess::Loess missing_model({ .fraction = 0.2 });
        auto result = missing_model.fit(tIrregular, yIrregular).value();

        return 0;
    }
    ```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)
    set.seed(42)
    t <- seq(0, 100, length.out = 500)
    trend_true <- 10 + 0.5 *t + 3* sin(t / 10)
    y <- trend_true + rnorm(500, sd = 3)

    fractions <- c(0.05, 0.2, 0.5)

    plot(t, y, col = "gray", pch = ".", main = "Multi-Scale LOESS")
    colors <- c("red", "blue", "green")
    for (i in seq_along(fractions)) {
        model <- Loess(fraction = fractions[i])
        result <- model$fit(t, y)
        lines(result$x, result$y, col = colors[i], lwd = 2)
    }
    legend("topleft", legend = paste("f =", fractions), col = colors, lwd = 2)
    ```

=== "Python"
    ```python
    import fastloess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend_true = 10 + 0.5 * t + 3 * np.sin(t / 10)
    y = trend_true + np.random.normal(0, 3, len(t))

    # Multiple smoothing scales
    fractions = [0.05, 0.2, 0.5]

    plt.figure(figsize=(12, 5))
    plt.plot(t, y, "gray", alpha=0.3, label="Data")
    
    for f in fractions:
        model = fl.Loess(fraction=f)
        result = model.fit(t, y)
        plt.plot(t, result.y, label=f"fraction={f}")
    
    plt.legend()
    plt.title("Multi-Scale LOESS")
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LoessError> {
        let n = 100usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = t.iter().map(|&ti| ti.sin() + 0.1).collect();

        let fractions = [0.05, 0.2, 0.5];

        for f in fractions {
            let model = Loess::new()
                .fraction(f)
                .build()?;
            let result = model.fit(&t, &y)?;
            // Store or plot result.y for each scale
        }

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOESS
    using Random, Statistics

    rng = MersenneTwister(42)
    t = collect(range(0, 100, length=500))
    y = 10.0 .+ 0.5 .* t .+ 3.0 .* sin.(t ./ 10.0) .+ randn(rng, 500) .* 3.0

    fractions = [0.05, 0.2, 0.5]

    results = [fit(Loess(; fraction=f), t, y) for f in fractions]
    # results[i].y contains smoothed values for each fraction
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastloess');

    const n = 500;
    const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
    const y = Float64Array.from(t, ti => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (Math.random()-0.5)*6);

    const scales = [0.05, 0.2, 0.5];
    const trends = scales.map(f => {
        return new fl.Loess({ fraction: f }).fit(t, y).y;
    });
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const trends = [0.05, 0.2, 0.5].map(f => {
        const model = new Loess({ fraction: f });
        const result = model.fit(x, y);
        return result.y;
    });
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 100;
        std::vector<double> t(n), y(n);
        for (int i = 0; i < n; ++i) {
            t[i] = i * 2.0 * M_PI / (n - 1);
            y[i] = std::sin(t[i]) + 0.1;
        }

        std::vector<double> scales = {0.05, 0.2, 0.5};
        std::vector<std::vector<double>> trends;
        for (auto f : scales) {
            fastloess::Loess scale_model({ .fraction = f });
            auto result = scale_model.fit(t, y).value();
            trends.push_back(result.y_vector());
        }

        return 0;
    }
    ```

---

## Gene Expression Time Course

Biological application:

=== "R"
    ```r
    library(rfastloess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastloess)

    # Gene expression over 24 hours
    hours <- seq(0, 24, by = 0.5)

    # Circadian pattern with measurement noise
    expression <- 100 * (1 + 0.5 * sin(hours * pi / 12)) + rnorm(49, sd = 10)

    model <- Loess(
        fraction = 0.3,
        iterations = 3,
        confidence_intervals = 0.95,
        return_diagnostics = TRUE
    )
    result <- model$fit(hours, expression)

    # Plot
    plot(hours, expression, pch = 16, col = "gray",
         xlab = "Time (hours)", ylab = "Expression Level",
         main = "Gene Expression Time Course")
    lines(result$x, result$y, col = "red", lwd = 2)
    lines(result$x, result$confidence_lower, col = "red", lty = 2)
    lines(result$x, result$confidence_upper, col = "red", lty = 2)

    cat("R²:", result$diagnostics$r_squared, "\n")
    ```

=== "Python"
    ```python
    import numpy as np
    import fastloess as fl

    # Gene expression over 24 hours
    hours = np.arange(0, 24.5, 0.5)
    expression = 100 * (1 + 0.5 * np.sin(hours * np.pi / 12)) + np.random.normal(0, 10, len(hours))

    model = fl.Loess(
        fraction=0.3,
        iterations=3,
        confidence_intervals=0.95,
        return_diagnostics=True
    )
    result = model.fit(hours, expression)

    print(f"R²: {result.diagnostics.r_squared:.3f}")
    ```

=== "Rust"
    ```rust
    use fastLoess::prelude::*;
    use std::f64::consts::PI;

    fn main() -> Result<(), LoessError> {
        let hours: Vec<f64> = (0..49).map(|i| i as f64 * 0.5).collect(); // 0.0..24.0 step 0.5
        let expression: Vec<f64> = hours.iter().enumerate()
            .map(|(i, &h)| 100.0 * (1.0 + 0.5 * (h * PI / 12.0).sin())
                          + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 10.0)
            .collect();

        let model = Loess::new()
            .fraction(0.3)
            .iterations(3)
            .confidence_intervals(0.95)
            .return_diagnostics()
            .build()?;

        let result = model.fit(&hours, &expression)?;
        if let Some(diag) = &result.diagnostics {
            println!("R²: {:.3}", diag.r_squared);
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

    using FastLOESS

    hours = collect(range(0, 24, step=0.5))
    expression = 100 .*(1.0 .+ 0.5 .* sin.(hours .*pi ./ 12.0)) .+ randn(length(hours)) .* 10.0

    model = Loess(;
        fraction=0.3,
        iterations=3,
        confidence_intervals=0.95,
        return_diagnostics=true
    )
    result = fit(model, hours, expression)

    println("R²: ", result.diagnostics.r_squared)
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastloess');

    const hours = Float64Array.from({ length: 49 }, (_, i) => i * 0.5);
    const expression = Float64Array.from(hours, h => 100*(1+0.5*Math.sin(h*Math.PI/12))+(Math.random()-0.5)*20);

    const model = new fl.Loess({
        fraction: 0.3,
        iterations: 3,
        return_diagnostics: true
    });
    const result = model.fit(hours, expression);

    console.log(`R²: ${result.diagnostics.r_squared.toFixed(3)}`);
    ```

=== "WebAssembly"
    ```javascript
    const { Loess } = require('fastloess-wasm');

    const n = 24;
    const hours = Float64Array.from({ length: n }, (_, i) => i);
    const expression = Float64Array.from(hours, h => 5 + 3 * Math.sin(h * Math.PI / 12) + (h % 3) * 0.2);
    const model = new Loess({ fraction: 0.3, iterations: 3, return_diagnostics: true });
    const result = model.fit(hours, expression);

    console.log("R²:", result.diagnostics?.r_squared);
    ```

=== "C++"
    ```cpp
    #include <fastloess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 49;
        std::vector<double> hours(n), expression(n);
        for (int i = 0; i < n; ++i) {
            hours[i] = i * 0.5;
            expression[i] = 100.0 * (1.0 + 0.5 * std::sin(hours[i] * M_PI / 12.0));
        }

        fastloess::Loess gene_model({
            .fraction = 0.3,
            .iterations = 3,
            .return_diagnostics = true
        });
        auto result = gene_model.fit(hours, expression).value();

        std::cout << "R²: " << result.diagnostics().r_squared() << std::endl;

        return 0;
    }
    ```

---

## Choosing Fraction for Time Series

| Data Type | Recommended Fraction | Rationale |
| --- | --- | --- |
| Daily data (years) | 0.3–0.5 | Capture annual trends |
| Hourly data (days) | 0.1–0.2 | Capture daily patterns |
| Sensor data (minutes) | 0.05–0.1 | Preserve short-term features |
| Noisy data | Higher | Reduce noise impact |
| Clean data | Lower | Preserve detail |

---

## See Also

- [Real-Time Processing](real-time.md) — For streaming time series
- [Cross-Validation](../user-guide/cross-validation.md) — Optimal fraction selection
- [Polynomial Degree](../user-guide/degree.md) — Degree 2 for curved trends
- [Boundary Handling](../user-guide/boundary.md) — Edge bias in trend extraction
- [Parameters](../user-guide/parameters.md) — Full parameter reference
