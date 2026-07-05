# fastLoess WebAssembly API Reference

The WebAssembly bindings provide a high-performance interface to the core Rust library, mirroring the Rust API structure.

## Classes and Functions

### `smooth`

The `smooth` function is the main entry point for batch smoothing.

**Signature:**

```javascript
const result = smooth(x, y, options);
```

* `x`: `Float64Array` of input x values.
* `y`: `Float64Array` of input y values.
* `options`: An object containing `LoessOptions` fields.
* Returns: A `LoessResult` object.

### `StreamingLoessWasm`

The `StreamingLoessWasm` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```javascript
const stream = new StreamingLoessWasm(options, streamingOptions);
```

* `options`: An object containing `LoessOptions` fields.
* `streamingOptions`: An object containing `StreamingOptions` fields.

**Methods:**

```javascript
const partialResult = stream.processChunk(x, y);
```

* Processes a chunk of data. Returns partial results.

```javascript
const finalResult = stream.finalize();
```

* Finalizes the smoothing process and returns any remaining buffered results.

### `OnlineLoessWasm`

The `OnlineLoessWasm` class updates the model incrementally with new data points.

**Constructor:**

```javascript
const online = new OnlineLoessWasm(options, onlineOptions);
```

* `options`: An object containing `LoessOptions` fields.
* `onlineOptions`: An object containing `OnlineOptions` fields.

**Methods:**

```javascript
const smoothed = online.update(x, y);
```

* Adds a **single** point `(x, y)` to the sliding window.
* Returns `number | null` — the smoothed value for the current point, or `null` if the window has fewer than `minPoints`.

> **Note:** For batch addition of multiple points, call `update()` in a loop. There is no `addPoints()` equivalent in the WASM binding.

## Options Structures

### `LoessOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `number` | `0.67` | Smoothing fraction (bandwidth) |
| `iterations` | `number` | `3` | Number of robustifying iterations |
| `weightFunction` | `string` | `"tricube"` | Kernel weight function |
| `robustnessMethod` | `string` | `"bisquare"` | Robustness method |
| `scalingMethod` | `string` | `"mad"` | Residual scaling method |
| `boundaryPolicy` | `string` | `"extend"` | Boundary handling policy |
| `zeroWeightFallback` | `string` | `"use_local_mean"` | Zero-weight handling strategy |
| `autoConverge` | `number` | `null` | Auto-convergence tolerance |
| `customWeights` | `number[]` | `null` | Per-observation case weights (Batch only) |
| `confidenceIntervals` | `number` | `null` | Confidence level (e.g., 0.95) |
| `predictionIntervals` | `number` | `null` | Prediction level (e.g., 0.95) |
| `returnDiagnostics` | `boolean` | `false` | Compute RMSE, MAE, R², AIC |
| `returnResiduals` | `boolean` | `false` | Include residuals in result |
| `returnRobustnessWeights` | `boolean` | `false` | Include robustness weights in result |
| `returnSe` | `boolean` | `false` | Compute hat-matrix statistics (enp, leverage …) |
| `parallel` | `boolean` | `true` | Enable parallel execution |
| `degree` | `string` | `"linear"` | Polynomial degree of local fit |
| `dimensions` | `number` | `1` | Number of predictor dimensions |
| `distanceMetric` | `string` | `"normalized"` | Distance metric; use `"minkowski:p"` for custom p |
| `surfaceMode` | `string` | `"interpolation"` | Surface computation mode |
| `cvFractions` | `number[]` | `null` | Fractions to test for cross-validation |
| `cvMethod` | `string` | `"kfold"` | CV method (`"kfold"` or `"loocv"`) |
| `cvK` | `number` | `5` | Number of folds for k-fold CV |

### `StreamingOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunkSize` | `number` | `5000` | Data chunk size |
| `overlap` | `number` | auto (10% of chunk) | Overlap between chunks |
| `mergeStrategy` | `string` | `"weighted_average"` | Strategy for blending overlap: see Merge Strategies |

### `OnlineOptions`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `windowCapacity` | `number` | `100` | Max points in sliding window |
| `minPoints` | `number` | `2` | Min points before smoothing starts |
| `updateMode` | `string` | `"full"` | Update mode (`"full"` or `"incremental"`) |

## Result Structure

### `LoessResult`

> **Note:** `LoessResultWasm` exposes `x`, `y`, `fractionUsed`, `iterationsUsed`, `standardErrors`, `confidenceLower/Upper`, `predictionLower/Upper`, `residuals`, `robustnessWeights`, `cvScores`, and `diagnostics`.
> Hat-matrix statistics (`enp`, `traceHat`, `delta1`, `delta2`, `residualScale`, `leverage`) are not yet exposed by the WASM binding; use the Node.js or Rust binding for those.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `Float64Array` | Sorted x values |
| `y` | `Float64Array` | Smoothed y values |
| `fractionUsed` | `number` | Fraction used (set or selected by CV) |
| `iterationsUsed` | number \| null | Robustness iterations actually performed |
| `standardErrors` | Float64Array \| null | Per-point SE (if `returnSe`) |
| `confidenceLower` | Float64Array \| null | Lower confidence bounds |
| `confidenceUpper` | Float64Array \| null | Upper confidence bounds |
| `predictionLower` | Float64Array \| null | Lower prediction bounds |
| `predictionUpper` | Float64Array \| null | Upper prediction bounds |
| `residuals` | Float64Array \| null | Residuals (if `returnResiduals`) |
| `robustnessWeights` | Float64Array \| null | Robustness weights (if `returnRobustnessWeights`) |
| `cvScores` | Float64Array \| null | CV score per tested fraction |
| `diagnostics` | Diagnostics \| null | Fit metrics (if `returnDiagnostics`) |

### `Diagnostics`

| Field | Type | Description |
| --- | --- | --- |
| `rmse` | `number` | Root Mean Squared Error |
| `mae` | `number` | Mean Absolute Error |
| `rSquared` | `number` | R-squared |
| `residualSd` | `number` | Residual standard deviation |
| `effectiveDf` | `number` | Effective degrees of freedom |
| `aic` | `number` | AIC |
| `aicc` | `number` | AICc |

## String Options

### Weight Functions

* `"tricube"` (default)
* `"epanechnikov"`
* `"gaussian"`
* `"uniform"`
* `"biweight"`
* `"triangle"`
* `"cosine"`

### Robustness Methods

* `"bisquare"` (default)
* `"huber"`
* `"talwar"`

### Boundary Policies

* `"extend"` (default - linear extrapolation)
* `"reflect"`
* `"zero"`
* `"noboundary"`

### Scaling Methods

* `"mad"` (default - Median Absolute Deviation)
* `"mar"` (Median Absolute Residual)
* `"mean"` (Mean Absolute Residual)

### Zero Weight Fallback

* `"use_local_mean"` (default)
* `"return_original"`
* `"return_none"`

### Polynomial Degrees

* `"constant"` (degree 0)
* `"linear"` (default, degree 1)
* `"quadratic"` (degree 2)
* `"cubic"` (degree 3)
* `"quartic"` (degree 4)

### Distance Metrics

* `"normalized"` (default — scales each dimension by its range)
* `"euclidean"`
* `"manhattan"`
* `"chebyshev"`
* `"minkowski"` (Euclidean when no suffix; use `"minkowski:p"` for custom p, e.g. `"minkowski:3"`)

### Surface Modes

* `"interpolation"` (default — faster, uses a spatial grid)
* `"direct"` (fits every point exactly; slower but more accurate)

### Merge Strategies (Streaming)

* `"weighted_average"` (default — weighted blend of overlapping regions)
* `"average"` (simple mean of overlapping regions)
* `"take_first"` (keep values from the earlier chunk)
* `"take_last"` (keep values from the later chunk)

### Update Modes (Online)

* `"full"` (default — re-smooth entire window each update)
* `"incremental"` (faster, O(1) incremental update)

## Example

```javascript
import init, { smooth } from 'fastloess-wasm';

await init();

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 4.0, 6.2, 8.0, 10.1]);

// Fit data
const result = smooth(x, y, { fraction: 0.5 });

console.log("Smoothed Y:", result.y);
```
