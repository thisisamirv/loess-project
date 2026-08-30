# Package index

## Core LOESS Interface

Main S3 classes for LOESS smoothing.

- [`Loess()`](https://thisisamirv.github.io/loess-project/r/reference/Loess.md)
  : LOESS Batch Smoothing
- [`StreamingLoess()`](https://thisisamirv.github.io/loess-project/r/reference/StreamingLoess.md)
  : LOESS Streaming Smoothing
- [`OnlineLoess()`](https://thisisamirv.github.io/loess-project/r/reference/OnlineLoess.md)
  : LOESS Online Smoothing

## Results and Utilities

Objects returned by fit methods and helper functions.

- [`fit()`](https://thisisamirv.github.io/loess-project/r/reference/fit.md)
  : Fit a LOESS model to data
- [`process_chunk()`](https://thisisamirv.github.io/loess-project/r/reference/process_chunk.md)
  : Process a data chunk through a streaming LOESS model
- [`finalize()`](https://thisisamirv.github.io/loess-project/r/reference/finalize.md)
  : Finalize a streaming LOESS model
- [`add_point()`](https://thisisamirv.github.io/loess-project/r/reference/add_point.md)
  : Add a single point to an online LOESS model
- [`plot(`*`<LoessResult>`*`)`](https://thisisamirv.github.io/loess-project/r/reference/plot.LoessResult.md)
  : Plot Loess Result
- [`print(`*`<Loess>`*`)`](https://thisisamirv.github.io/loess-project/r/reference/print.Loess.md)
  : Print Loess Model
- [`print(`*`<LoessResult>`*`)`](https://thisisamirv.github.io/loess-project/r/reference/print.LoessResult.md)
  : Print Loess Result
- [`print(`*`<OnlineLoess>`*`)`](https://thisisamirv.github.io/loess-project/r/reference/print.OnlineLoess.md)
  : Print OnlineLoess Model
- [`print(`*`<StreamingLoess>`*`)`](https://thisisamirv.github.io/loess-project/r/reference/print.StreamingLoess.md)
  : Print StreamingLoess Model
