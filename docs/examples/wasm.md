# WebAssembly Examples

Complete WebAssembly examples demonstrating fastloess-wasm for browser-based smoothing.

## Batch Smoothing

Process complete datasets in the browser.

```html
--8<-- "examples/wasm/batch_smoothing.html"
```

[:material-download: Download batch_smoothing.html](https://github.com/thisisamirv/loess-project/blob/main/examples/wasm/batch_smoothing.html)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks in the browser.

```html
--8<-- "examples/wasm/streaming_smoothing.html"
```

[:material-download: Download streaming_smoothing.html](https://github.com/thisisamirv/loess-project/blob/main/examples/wasm/streaming_smoothing.html)

---

## Online Smoothing

Real-time smoothing with sliding window for browser applications.

```html
--8<-- "examples/wasm/online_smoothing.html"
```

[:material-download: Download online_smoothing.html](https://github.com/thisisamirv/loess-project/blob/main/examples/wasm/online_smoothing.html)

---

## Installation

### NPM

```bash
npm install fastloess-wasm
```

### CDN

```html
<script type="module">
  import init, { Loess } from 'https://unpkg.com/fastloess-wasm@latest';
  
  await init();
  // Ready to use
</script>
```

## Quick Start

### Browser (ES Modules)

```javascript
import init, { Loess } from 'fastloess-wasm';

async function main() {
    // Initialize WASM module
    await init();

    // Generate sample data
    const x = Float64Array.from({ length: 100 }, (_, i) => i * 0.1);
    const y = Float64Array.from(x, xi => Math.sin(xi) + Math.random() * 0.2);

    // Basic smoothing
    const result = new Loess({ fraction: 0.3 }).fit(x, y);
    console.log('Smoothed values:', result.y);

    // With options
    const resultWithOptions = new Loess({
        fraction: 0.3,
        iterations: 3,
        confidence_intervals: 0.95,
        return_diagnostics: true
    }).fit(x, y);

    console.log('R²:', resultWithOptions.diagnostics?.rSquared);
}

main();
```

### Node.js

```javascript
const { Loess } = require('fastloess-wasm');

// Same API as browser
const result = new Loess({ fraction: 0.3 }).fit(x, y);
```

## Features

The WebAssembly bindings provide:

- **Zero dependencies** - Pure WASM, no runtime requirements
- **TypedArray support** - Works with `Float64Array` for efficiency
- **Same API as Node.js** - Consistent interface across platforms
- **Small bundle size** - Optimized with `wasm-opt`
