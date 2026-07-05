# Node.js Examples

Complete Node.js examples demonstrating fastloess with native N-API bindings.

## Batch Smoothing

Process complete datasets with confidence intervals and diagnostics.

```javascript
--8<-- "examples/nodejs/batch_smoothing.js"
```

[:material-download: Download batch_smoothing.js](https://github.com/thisisamirv/loess-project/blob/main/examples/nodejs/batch_smoothing.js)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks.

```javascript
--8<-- "examples/nodejs/streaming_smoothing.js"
```

[:material-download: Download streaming_smoothing.js](https://github.com/thisisamirv/loess-project/blob/main/examples/nodejs/streaming_smoothing.js)

---

## Online Smoothing

Real-time smoothing with sliding window for streaming data.

```javascript
--8<-- "examples/nodejs/online_smoothing.js"
```

[:material-download: Download online_smoothing.js](https://github.com/thisisamirv/loess-project/blob/main/examples/nodejs/online_smoothing.js)

---

## Running the Examples

```bash
# Install the package
cd bindings/nodejs
npm install
npm run build

# Run examples
node examples/nodejs/batch_smoothing.js
node examples/nodejs/streaming_smoothing.js
node examples/nodejs/online_smoothing.js
```

## Quick Start

```javascript
const { Loess } = require('fastloess');

// Generate sample data
const x = Array.from({ length: 100 }, (_, i) => i * 0.1);
const y = x.map(xi => Math.sin(xi) + Math.random() * 0.2);

// Basic smoothing
const result = new Loess({ fraction: 0.3 }).fit(x, y);
console.log('Smoothed values:', result.y);

// With options
const resultWithOptions = new Loess({
    fraction: 0.3,
    iterations: 3,
    confidenceIntervals: 0.95,
    returnDiagnostics: true
}).fit(x, y);

console.log('R²:', resultWithOptions.diagnostics.rSquared);
```

## TypeScript Support

The package includes TypeScript definitions:

```typescript
import { Loess, LoessResult } from 'fastloess';

const options = {
    fraction: 0.3,
    iterations: 3,
    confidenceIntervals: 0.95
};

const result: LoessResult = new Loess(options).fit(x, y);
```
