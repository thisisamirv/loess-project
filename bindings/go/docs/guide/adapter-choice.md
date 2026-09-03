---
title: "Execution Modes"
weight: 25
---

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
