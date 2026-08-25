---
title: Time Series Analysis
---
<!-- markdownlint-disable MD024 MD046 MD033 -->
LOESS for trend extraction and temporal smoothing.

## Overview

Time series data often contains noise, seasonality, and trends. LOESS provides flexible trend extraction without parametric assumptions.

---

## Basic Trend Extraction

`fraction = 0.1` sizes the neighbourhood as 10% of the data at each evaluation point — narrow enough to follow a slowly varying trend without smearing periodic variation. Three robustness `iterations` down-weight noise spikes so they cannot bias the fitted curve; this is especially important when the signal-to-noise ratio is low or when occasional outliers are expected.

```javascript
const fl = require('fastloess');

const n = 500;
const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
const y = Float64Array.from(t, (ti, i) => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (((i*7+3)%17)/17-0.5)*6);

// t and y are your time series arrays (Float64Array)
const model = new fl.Loess({ 
    fraction: 0.1, 
    iterations: 3 
});
const result = model.fit(t, y);

console.log("Extracted trend (first 5):", [...result.y.slice(0, 5)].map(v => v.toFixed(4)));
```

```output
Extracted trend (first 5): [ '9.4350', '9.5812', '9.7368', '9.8902', '10.0435' ]
```

---

## Detrending

Remove trend to analyze residual patterns.

Setting `return_residuals = True` stores `observed − smoothed` alongside the smooth. A slightly wider `fraction = 0.3` produces a smoother baseline trend, so short-duration oscillations end up in the residuals rather than being absorbed into the trend component. The residual series is then ready for spectral analysis, seasonality detection, or change-point methods.

```javascript
const fl = require('fastloess');

const n = 500;
const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
const y = Float64Array.from(t, (ti, i) => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (((i*7+3)%17)/17-0.5)*6);

const model = new fl.Loess({
    fraction: 0.3,
    iterations: 3,
    return_residuals: true
});
const result = model.fit(t, y);

const trend = result.y;
const detrended = result.residuals;
console.log("Trend y[0]:", trend[0].toFixed(4), " residual:", detrended[0].toFixed(4));
```

```output
Trend y[0]: 10.8694  residual: -2.8106
```

---

## Forecasting with Prediction Intervals

Prediction intervals widen the uncertainty band to include both the uncertainty in the fitted curve (confidence interval) and the expected scatter of new observations around it. `fraction = 0.2` offers a balance between local detail and stable interval width — too small a fraction produces jagged interval edges; too large a fraction underestimates local variance near turning points.

```javascript
const fl = require('fastloess');

const n = 500;
const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
const y = Float64Array.from(t, (ti, i) => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (((i*7+3)%17)/17-0.5)*6);

const model = new fl.Loess({
    fraction: 0.2,
    iterations: 3,
    prediction_intervals: 0.95
});
const result = model.fit(t, y);

console.log(`95% PI: [${result.prediction_lower[0]}, ${result.prediction_upper[0]}]`);
```

```output
95% PI: [5.864903127502757, 14.544377281917292]
```

---

## Handling Missing Data

LOESS naturally handles irregular time sampling:

```javascript
const fl = require('fastloess');

const n = 500;
const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
const y = Float64Array.from(t, (ti, i) => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (((i*7+3)%17)/17-0.5)*6);

const tIrregular = Float64Array.from({ length: 200 }, (_, i) => i * 100 / 199 + ((i*7+3)%17)/17*0.5 - 0.25).sort((a, b) => a - b);
const yIrregular = Float64Array.from(tIrregular, (t, i) => 10 + 0.3 * t + ((i*7+3)%17)/17*2);

// No special handling needed for irregular spacing
const model = new fl.Loess({ fraction: 0.2 });
const result = model.fit(tIrregular, yIrregular);
console.log("Irregular fit y[0]:", result.y[0].toFixed(4));
```

```output
Irregular fit y[0]: 11.1532
```

---

## Multi-Scale Analysis

Use different fractions to extract features at different scales:

```javascript
const fl = require('fastloess');

const n = 500;
const t = Float64Array.from({ length: n }, (_, i) => i * 100 / (n - 1));
const y = Float64Array.from(t, (ti, i) => 10 + 0.5 * ti + 3 * Math.sin(ti / 10) + (((i*7+3)%17)/17-0.5)*6);

const scales = [0.05, 0.2, 0.5];
const trends = scales.map(f => {
    return new fl.Loess({ fraction: f }).fit(t, y).y;
});
console.log("Trend y[0] (fraction=0.05):", trends[0][0].toFixed(4), " (0.2):", trends[1][0].toFixed(4));
```

```output
Trend y[0] (fraction=0.05): 9.0056  (0.2): 10.2046
```

---

## Gene Expression Time Course

Biological application:

```javascript
const fl = require('fastloess');

const hours = Float64Array.from({ length: 49 }, (_, i) => i * 0.5);
const expression = Float64Array.from(hours, (h, i) => 100*(1+0.5*Math.sin(h*Math.PI/12))+(((i*7+3)%17)/17-0.5)*20);

const model = new fl.Loess({
    fraction: 0.3,
    iterations: 3,
    return_diagnostics: true
});
const result = model.fit(hours, expression);

console.log(`R²: ${result.diagnostics.r_squared.toFixed(3)}`);
```

```output
R²: 0.963
```

---

## Choosing Fraction for Time Series

| Data Type | Recommended Fraction | Rationale |
| --- | --- | --- |
| Daily data (years) | 0.3–0.5 | Capture annual trends |
| Hourly data (days) | 0.1–0.2 | Capture daily patterns |
| Sensor data (minutes) | 0.05–0.1 | Preserve short-term features |
| Noisy data | Higher | Reduce noise impact |
| Clean data | Lower | Preserve detail |

---

## See Also

- [Real-Time Processing](use-case-real-time.md) — For streaming time series
- [Cross-Validation](cross-validation.md) — Optimal fraction selection
- [Polynomial Degree](degree.md) — Degree 2 for curved trends
- [Boundary Handling](boundary.md) — Edge bias in trend extraction
- [Parameters](parameters.md) — Full parameter reference
