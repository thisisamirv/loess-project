<!-- markdownlint-disable MD024 MD046 -->
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
| **Scale** (1K–10K) | 1× | **8×** | 8× |
| **Scientific** (500–5K) | 1× | 4× | **5×** |

*Averages across all sizes within each category.*

---

## Reproducing Benchmarks

Use `std::chrono` to time serial vs parallel runs:

```cpp
#include <fastloess.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

template <typename Fn>
static double bench_ms(Fn fn, int reps = 10) {
    // Warm-up
    fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main() {
    const int n = 5000;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1) * 10.0;
        y[i] = std::sin(x[i]) + ((i * 7 + 3) % 17 / 17.0 - 0.5) * 0.6;
    }

    fastloess::LoessOptions opts;
    opts.fraction = 0.67;

    auto serial_ms = bench_ms([&] {
        opts.parallel = false;
        fastloess::Loess(opts).fit(x, y);
    });

    auto parallel_ms = bench_ms([&] {
        opts.parallel = true;
        fastloess::Loess(opts).fit(x, y);
    });

    std::cout << "Serial:   " << serial_ms   << " ms\n";
    std::cout << "Parallel: " << parallel_ms << " ms\n";
    std::cout << "Speedup:  " << serial_ms / parallel_ms << "×\n";
    return 0;
}
```
