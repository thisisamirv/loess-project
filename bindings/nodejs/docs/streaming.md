# Streaming Adapter

Process large datasets in chunks with configurable overlap.

## When to Use

- Dataset >100,000 points
- Memory-constrained environments
- Batch processing pipelines

## Parameters

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

## Example

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

---

!!! warning "Always call finalize()"
    In Rust, always call `processor.finalize()` after processing all chunks to retrieve buffered overlap data.
