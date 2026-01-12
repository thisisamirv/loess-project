<!-- markdownlint-disable MD024 -->
# C++ API Reference

C and C++ bindings for fastLoess.

## Installation

### From Source

```bash
git clone https://github.com/thisisamirv/loess-project
cd loess-project/bindings/cpp

# Build the library
cargo build --release

# Headers are at: include/fastloess.h (C) and include/fastloess.hpp (C++)
# Library is at: target/release/libfastloess_cpp.so (Linux)
#                target/release/libfastloess_cpp.dylib (macOS)
#                target/release/fastloess_cpp.dll (Windows)
```

---

## Quick Start

```cpp
#include <vector>
#include <iostream>
#include "fastloess.hpp"

int main() {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 3.9, 6.2, 8.0, 10.1};

    // Smooth with default options
    auto result = fastloess::smooth(x, y);

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result.x(i) << " -> " << result.y(i) << std::endl;
    }
    return 0;
}
```

---

## API

### `fastloess::smooth()`

Batch LOESS smoothing.

```cpp
LoessResult smooth(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const LoessOptions& options = {}
);
```

### `fastloess::streaming()`

Streaming LOESS for large datasets.

```cpp
LoessResult streaming(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const StreamingOptions& options = {}
);
```

### `fastloess::online()`

Online LOESS with sliding window.

```cpp
LoessResult online(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const OnlineOptions& options = {}
);
```

---

## Options

### `LoessOptions`

| Field                  | Type          | Default      | Description                       |
|------------------------|---------------|--------------|-----------------------------------|
| `fraction`             | `double`      | `0.67`       | Smoothing fraction                |
| `iterations`           | `int`         | `3`          | Robustness iterations             |
| `weight_function`      | `std::string` | `"tricube"`  | Weight function                   |
| `robustness_method`    | `std::string` | `"bisquare"` | Robustness method                 |
| `confidence_intervals` | `double`      | `NAN`        | Confidence level (NaN = disabled) |
| `return_diagnostics`   | `bool`        | `false`      | Return fit diagnostics            |
| `parallel`             | `bool`        | `false`      | Enable parallel processing        |

### `StreamingOptions`

Extends `LoessOptions` with:

| Field        | Type  | Default | Description                        |
|--------------|-------|---------|------------------------------------|
| `chunk_size` | `int` | `5000`  | Points per chunk                   |
| `overlap`    | `int` | `-1`    | Overlap between chunks (-1 = auto) |

### `OnlineOptions`

Extends `LoessOptions` with:

| Field             | Type          | Default  | Description                          |
|-------------------|---------------|----------|--------------------------------------|
| `window_capacity` | `int`         | `1000`   | Sliding window size                  |
| `min_points`      | `int`         | `2`      | Minimum points for smoothing         |
| `update_mode`     | `std::string` | `"full"` | Update mode: "full" or "incremental" |

---

## LoessResult

RAII wrapper with automatic memory management.

```cpp
class LoessResult {
public:
    size_t size() const;              // Number of points
    bool valid() const;               // Check if result is valid
    
    double x(size_t i) const;         // Access x value
    double y(size_t i) const;         // Access smoothed y value
    
    std::vector<double> x_vector() const;
    std::vector<double> y_vector() const;
    std::vector<double> residuals() const;
    std::vector<double> confidence_lower() const;
    std::vector<double> confidence_upper() const;
    
    double fraction_used() const;
    int iterations_used() const;
    Diagnostics diagnostics() const;
};
```

---

## Error Handling

Errors throw `fastloess::LoessError`:

```cpp
try {
    auto result = fastloess::smooth(x, y);
} catch (const fastloess::LoessError& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```
