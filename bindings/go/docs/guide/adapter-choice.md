---
title: "Execution Modes"
weight: 25
---
<!-- markdownlint-disable MD024 -->
Choose the right adapter for your use case.

## Overview

Choose the first row below whose condition applies:

| Condition | Adapter |
| --- | --- |
| Data too large to fit in memory | `StreamingLoess` |
| Fits in memory, need real-time/incremental updates | `OnlineLoess` |
| Fits in memory, no real-time requirement | `Loess` |

| Mode | Use Case | Memory | Features |
| --- | --- | --- | --- |
| **Batch** (`Loess`) | Complete datasets | Full | All features |
| **Streaming** (`StreamingLoess`) | Large files (>100K) | Chunked | Residuals, robustness |
| **Online** (`OnlineLoess`) | Real-time sensors | Fixed window | Incremental updates |

![Adapter Comparison](../assets/diagrams/adapter_comparison.svg)

---

## Batch Adapter

Standard mode for complete datasets. **Supports all features.**

### When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

### Example

```go
opts := fastloess.DefaultOptions()
opts.Fraction = 0.5
opts.Iterations = 3
opts.ConfidenceIntervals = ptr(0.95)
opts.PredictionIntervals = ptr(0.95)
opts.ReturnDiagnostics = true
opts.Parallel = true

model, err := fastloess.NewLoess(opts)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

result, err := model.Fit(x, y)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("95%% CI at midpoint: [%v, %v]\n", result.ConfidenceLower[50], result.ConfidenceUpper[50])
fmt.Println("R2:", result.Diagnostics.RSquared)
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
| `ChunkSize` | 5000 | Points per chunk |
| `Overlap` | 500 | Overlap between chunks |
| `MergeStrategy` | `"weighted_average"` | How to merge overlaps |

### Merge Strategies

| Strategy | Behavior |
| --- | --- |
| `"average"` | Average overlapping values |
| `"weighted_average"` | Distance-weighted blend (default) |
| `"take_first"` | Keep left chunk values |
| `"take_last"` | Keep right chunk values |

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

### Example

```go
opts := fastloess.DefaultStreamingOptions()
opts.Fraction = 0.3
opts.Iterations = 2
opts.ChunkSize = 5000
opts.Overlap = 500
opts.MergeStrategy = "average"

model, err := fastloess.NewStreamingLoess(opts)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

if _, err := model.ProcessChunk(x, y); err != nil {
    log.Fatal(err)
}
result, err := model.Finalize()
if err != nil {
    log.Fatal(err)
}
fmt.Println("Smoothed y[0]:", result.Y[0])
```

---

> **Always call finalize():** The streaming adapter buffers overlap data. Call `Finalize()` after processing all chunks to retrieve the buffered tail.

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
| `WindowCapacity` | 1000 | Max points in window |
| `MinPoints` | 2 | Points before output starts |
| `UpdateMode` | `"incremental"` | Update strategy |

### Update Modes

| Mode | Behavior | Speed |
| --- | --- | --- |
| `"incremental"` | Update only affected fits | Faster |
| `"full"` | Recompute entire window | More accurate |

### Example

```go
opts := fastloess.DefaultOnlineOptions()
opts.Fraction = 0.2
opts.Iterations = 1
opts.WindowCapacity = 100
opts.MinPoints = 5
opts.UpdateMode = "incremental"

model, err := fastloess.NewOnlineLoess(opts)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

shown := 0
for i := range x {
    res, ok, err := model.AddPoint(x[i], y[i])
    if err != nil {
        log.Fatal(err)
    }
    if ok && shown < 5 {
        fmt.Println(res.Y)
        shown++
    }
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

- [API Reference](../api/api.md) — All configuration options
- [Streaming API](../api/api-streaming.md) · [Online API](../api/api-online.md)
