---
title: "StreamingLoess API"
weight: 32
---

For datasets that don't fit in memory or arrive in chunks. Processes data incrementally, merging overlapping regions between chunks.

See also: [API](api.md)

## `fastloess.DefaultStreamingOptions() StreamingOptions`

```go
opts := fastloess.DefaultStreamingOptions()
opts.ChunkSize = 2000
opts.Overlap = 200
```

`StreamingOptions` embeds [`Options`](api.md) (all the same fields apply, except `CVFractions`/`CVMethod`/`CVK`/`CVSeed`, which are batch-only), plus:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `ChunkSize` | `int` | `5000` | Number of points processed per chunk. Larger chunks reduce per-chunk overhead and give each local fit more surrounding context, at the cost of higher peak memory; smaller chunks bound memory tightly but increase the fraction of points that fall in overlap regions. A good starting point is balancing available memory against how much processing overhead per chunk is acceptable — match it to your file-read buffer or message-batch size to avoid unnecessary copying. |
| `Overlap` | `int` | `ChunkSize / 10` | Number of points retained from the previous chunk as context, so the neighbourhood at chunk boundaries isn't artificially truncated. Points inside the overlap zone are fitted twice (once by each chunk) and reconciled via `MergeStrategy`. A good starting point is 10–20% of `ChunkSize`: too little overlap causes visible boundary artefacts, while too much wastes computation refitting the same points twice. Negative (the `DefaultStreamingOptions()` value, `-1`) means "use the library default", clamped to `[1, ChunkSize - 10]`. |
| `MergeStrategy` | `string` | `"weighted_average"` | How overlapping chunk results are combined. |

| Strategy | Alias | Behavior |
| --- | --- | --- |
| `"weighted_average"` (default) | `"weighted"` | Distance-weighted blend |
| `"average"` | `"mean"` | Average overlapping values |
| `"take_first"` | `"first"` | Keep left chunk values |
| `"take_last"` | `"last"` | Keep right chunk values |

*See also: [Merge Strategies](../advanced/merge.md)*

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

## `fastloess.NewStreamingLoess(opts StreamingOptions) (*StreamingLoess, error)`

## `(*StreamingLoess) ProcessChunk(x, y []float64) (Result, error)`

Fits and returns the result for one chunk. Each chunk is fit together with the trailing `Overlap` points buffered from the previous call, then only the points that are fully resolved are returned — the tail of the chunk (the next `Overlap` points) is held back internally, since it will be refit once the following chunk arrives and its estimate reconciled via `MergeStrategy`. This is what lets the adapter process a dataset far larger than memory allows, one bounded-size chunk at a time, without ever materializing the whole dataset at once. For multivariate input (`Dimensions > 1`), `x` is flattened row-major. Call repeatedly as chunks arrive.

## `(*StreamingLoess) Finalize() (Result, error)`

Flushes the overlap points still buffered from the last `ProcessChunk` call. Because each call withholds its tail until the next chunk arrives to resolve it, the final chunk's tail would never be emitted otherwise — always call `Finalize` once after the last chunk to retrieve it.

## `(*StreamingLoess) Close() error`

Releases native resources. Safe to call multiple times.

## Result

`ProcessChunk` and `Finalize` return the same [`Result`](api.md#result-fields) type as `Loess.Fit`. Fields tied to Batch-only options (`StandardErrors`, `ConfidenceLower`/`ConfidenceUpper`, `PredictionLower`/`PredictionUpper`, `CVScores`) are always left at their zero value here.

## Example

```go
package main

import (
 "fmt"
 "log"

 "github.com/thisisamirv/loess-project/bindings/go/fastloess"
)

func main() {
 const n = 20
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i)
  y[i] = float64(i) + 0.1
 }

 opts := fastloess.DefaultStreamingOptions()
 opts.ChunkSize = 10
 opts.Overlap = 2

 model, err := fastloess.NewStreamingLoess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 if _, err := model.ProcessChunk(x[:10], y[:10]); err != nil {
  log.Fatal(err)
 }
 if _, err := model.ProcessChunk(x[10:], y[10:]); err != nil {
  log.Fatal(err)
 }

 result, err := model.Finalize()
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("y[0]: %.4f\n", result.Y[0])
}
```

```output
y[0]: 18.1000
```
