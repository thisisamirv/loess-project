\page merge Merge Strategies

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

![Merge Strategies](merge_comparison.svg)

---

## Average

Takes the arithmetic mean of the left-chunk and right-chunk estimates in the overlap region. Fast and sufficient when both chunks have similar smoothing quality.

**Use when**: Chunks are large and the overlap region has uniform data density.

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
    opts.chunk_size = 60;
    opts.overlap = 20;
    fastloess::StreamingLoess stream(opts);
    std::vector<double> x1(x.begin(), x.begin() + 60), y1(y.begin(), y.begin() + 60);
    std::vector<double> x2(x.begin() + 60, x.end()), y2(y.begin() + 60, y.end());
    (void)stream.process_chunk(x1, y1).value();
    // The second chunk's overlap region (its first 20 points) is where
    // merge_strategy actually blends the two chunks' estimates.
    auto result = stream.process_chunk(x2, y2).value();

    std::cout << "Merged value in overlap region (average): " << result.y_vector()[5] << "\n";
    return 0;
}
```

```output
Merged value in overlap region (average): 0.324852
```

---

## Take First

Keeps only the left-chunk estimate in the overlap zone and discards the right-chunk estimate. Produces a definitive, non-revised output as soon as the right boundary of each chunk is reached.

**Use when**: You need final output values immediately after each chunk (no look-ahead revision); left-chunk data quality is higher.

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
    s_opts.chunk_size = 60;
    s_opts.overlap = 20;
    fastloess::StreamingLoess stream(s_opts);
    std::vector<double> x1(x.begin(), x.begin() + 60), y1(y.begin(), y.begin() + 60);
    std::vector<double> x2(x.begin() + 60, x.end()), y2(y.begin() + 60, y.end());
    (void)stream.process_chunk(x1, y1).value();
    auto result = stream.process_chunk(x2, y2).value();

    std::cout << "Merged value in overlap region (take_first): " << result.y_vector()[5] << "\n";
    return 0;
}
```

```output
Merged value in overlap region (take_first): 0.353763
```

---

## Take Last

Keeps only the right-chunk estimate in the overlap zone. The right chunk sees more of the surrounding data, so its fit can be more accurate near the left boundary of the new chunk.

**Use when**: Right-chunk context improves overlap quality; you are post-processing complete data rather than streaming live.

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
    s_opts.chunk_size = 60;
    s_opts.overlap = 20;
    fastloess::StreamingLoess stream(s_opts);
    std::vector<double> x1(x.begin(), x.begin() + 60), y1(y.begin(), y.begin() + 60);
    std::vector<double> x2(x.begin() + 60, x.end()), y2(y.begin() + 60, y.end());
    (void)stream.process_chunk(x1, y1).value();
    auto result = stream.process_chunk(x2, y2).value();

    std::cout << "Merged value in overlap region (take_last): " << result.y_vector()[5] << "\n";
    return 0;
}
```

```output
Merged value in overlap region (take_last): 0.295941
```

---

## Weighted Average

Assigns each overlap point a weight proportional to its proximity to the centre of its respective chunk: points near the left-chunk centre get higher left weight; points near the right-chunk centre get higher right weight. This produces the smoothest transition across chunk boundaries.

\f[\hat{y} = \frac{w_L \hat{y}_L + w_R \hat{y}_R}{w_L + w_R}\f]

where \f$w_L\f$ and \f$w_R\f$ are linear distance weights from the chunk centres.

**Use when**: Minimising boundary artefacts is more important than speed; moderate overlap (10–20 % of chunk size).

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
    s_opts.chunk_size = 60;
    s_opts.overlap = 20;
    fastloess::StreamingLoess stream(s_opts);
    std::vector<double> x1(x.begin(), x.begin() + 60), y1(y.begin(), y.begin() + 60);
    std::vector<double> x2(x.begin() + 60, x.end()), y2(y.begin() + 60, y.end());
    (void)stream.process_chunk(x1, y1).value();
    auto result = stream.process_chunk(x2, y2).value();

    std::cout << "Merged value in overlap region (weighted_average): " << result.y_vector()[5] << "\n";
    return 0;
}
```

```output
Merged value in overlap region (weighted_average): 0.339308
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

> **Overlap size matters:** A larger overlap gives the merge strategy more room to blend, reducing boundary artefacts regardless of the strategy chosen. A good starting point is 10 % of `chunk_size`.
