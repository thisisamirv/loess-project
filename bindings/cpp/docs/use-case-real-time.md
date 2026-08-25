# Real-Time Processing

Streaming and online LOESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

`window_capacity = 25` limits the internal buffer to the 25 most recent observations; each `add_point` call costs O(window) rather than growing with total history. `min_points = 5` suppresses output until the window holds enough points for a stable fit — calls made before that threshold return `null`/`None`/`nothing`. `update_mode = "incremental"` re-fits only the most recent point rather than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> times(n), temperatures(n);
    for (int i = 0; i < n; ++i) {
        times[i] = i * 0.1;
        temperatures[i] = 20.0 + std::sin(times[i]);
    }

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
            std::cout << "Time " << times[i] << ": " << res.y() << std::endl;
        }
    }

    return 0;
}
```

```output
Time 0.4: 20.3894
Time 0.5: 20.4794
Time 0.6: 20.5646
Time 0.7: 20.6442
Time 0.8: 20.7174
Time 0.9: 20.7833
Time 1: 20.8415
Time 1.1: 20.8912
Time 1.2: 20.932
Time 1.3: 20.9545
Time 1.4: 20.9792
Time 1.5: 20.994
Time 1.6: 20.999
Time 1.7: 20.9939
Time 1.8: 20.9789
Time 1.9: 20.9579
Time 2: 20.9252
Time 2.1: 20.8833
Time 2.2: 20.8326
Time 2.3: 20.7736
Time 2.4: 20.7068
Time 2.5: 20.6329
Time 2.6: 20.5528
Time 2.7: 20.4671
Time 2.8: 20.3767
Time 2.9: 20.2826
Time 3: 20.1857
Time 3.1: 20.0869
Time 3.2: 19.9872
Time 3.3: 19.8877
Time 3.4: 19.7893
Time 3.5: 19.6929
Time 3.6: 19.5997
Time 3.7: 19.5105
Time 3.8: 19.4261
Time 3.9: 19.3475
Time 4: 19.2754
Time 4.1: 19.2105
Time 4.2: 19.1536
Time 4.3: 19.1051
Time 4.4: 19.0655
Time 4.5: 19.0353
Time 4.6: 19.0147
Time 4.7: 19.0039
Time 4.8: 19.0031
Time 4.9: 19.0123
Time 5: 19.0313
Time 5.1: 19.06
Time 5.2: 19.0981
Time 5.3: 19.1452
Time 5.4: 19.2009
Time 5.5: 19.2645
Time 5.6: 19.3355
Time 5.7: 19.4132
Time 5.8: 19.4967
Time 5.9: 19.5852
Time 6: 19.6778
Time 6.1: 19.7737
Time 6.2: 19.8719
Time 6.3: 19.9713
Time 6.4: 20.071
Time 6.5: 20.17
Time 6.6: 20.2673
Time 6.7: 20.3619
Time 6.8: 20.453
Time 6.9: 20.5395
Time 7: 20.6206
Time 7.1: 20.6955
Time 7.2: 20.7634
Time 7.3: 20.8237
Time 7.4: 20.8758
Time 7.5: 20.9192
Time 7.6: 20.9533
Time 7.7: 20.978
Time 7.8: 20.9928
Time 7.9: 20.9978
Time 8: 20.9927
Time 8.1: 20.9778
Time 8.2: 20.9531
Time 8.3: 20.9188
Time 8.4: 20.8754
Time 8.5: 20.8232
Time 8.6: 20.7629
Time 8.7: 20.6948
Time 8.8: 20.6199
Time 8.9: 20.5387
Time 9: 20.4522
Time 9.1: 20.3611
Time 9.2: 20.2665
Time 9.3: 20.1692
Time 9.4: 20.0702
Time 9.5: 19.9704
Time 9.6: 19.871
Time 9.7: 19.7729
Time 9.8: 19.677
Time 9.9: 19.5844
```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

`chunk_size` controls how many data points are processed in one pass; matching it to your file-read buffer or message-batch size avoids unnecessary copying. `overlap` retains that many points from the previous chunk as context so the neighbourhood at chunk boundaries is not artificially truncated. `merge_strategy = "weighted_average"` blends the overlapping region smoothly; use `"last"` if chunk boundaries are guaranteed to be well separated and no blending is needed.

!!! warning "Always call finalize()"
    The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.

### Log File Processing

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
    opts.fraction = 0.1;
    opts.iterations = 2;
    opts.chunk_size = 5000;
    opts.overlap = 500;

    fastloess::StreamingLoess stream(opts);
    (void)stream.process_chunk(x, y);
    auto result = stream.finalize().value();

    std::cout << "Processed " << result.y_vector().size() << " points" << std::endl;

    return 0;
}
```

```output
Processed 100 points
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOESS fit on a manually managed sliding window rather than `OnlineLoess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window²) refit on every tick; for high-frequency streams prefer `OnlineLoess` with `update_mode = "incremental"` to bound per-frame cost.

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> times(n), temperatures(n);
    for (int i = 0; i < n; ++i) {
        times[i] = i * 0.1;
        temperatures[i] = 25.0 + 10.0 * std::sin(times[i] / 2.0) + 2.0;
    }

    std::vector<double> windowX, windowY;
    std::vector<double> last_y;

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
        last_y = result.y_vector();
        const auto smoothed = result.y_vector().back();
        (void)smoothed;
    }

    std::cout << "y[0]: " << last_y[0] << "\n";
    return 0;
}
```

```output
y[0]: 32.262
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

- [Execution Modes](adapter-choice.md) — Detailed mode comparison
- [Merge Strategies](merge.md) — Chunk reconciliation in depth
- [Scaling Methods](scaling.md) — Robustness scale estimation
- [Time Series](use-case-time-series.md) — General time series analysis
