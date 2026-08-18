# StreamingLoess — C++ API Reference

See also: [fastLoess C++ API Reference](cpp.md)

## Class

### `fastloess::StreamingLoess`

The `StreamingLoess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```cpp
#include <fastloess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    fastloess::StreamingOptions opts;
    opts.chunk_size = 5;
    fastloess::StreamingLoess model(opts);

    return 0;
}
```

* `options`: A `StreamingOptions` struct (inherits from `LoessOptions`) with additional `chunk_size` and `overlap` parameters.

**Methods:**

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
    opts.chunk_size = 10;
    opts.overlap = 0;
    fastloess::StreamingLoess model(opts);
    (void)model.process_chunk(x, y);

    return 0;
}
```

* Processes a chunk of data. Returns partial results.

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
    opts.chunk_size = 10;
    opts.overlap = 0;
    fastloess::StreamingLoess model(opts);
    model.process_chunk(x, y);
    auto result = model.finalize().value();

    return 0;
}
```

* Finalizes the smoothing process and returns any remaining buffered results.

## Options Structure

### `StreamingOptions` (inherits `LoessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | `5000` | Data chunk size |
| `overlap` | `int` | `500` | Overlap between chunks |
| `merge_strategy` | `std::string` | `"weighted_average"` | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
