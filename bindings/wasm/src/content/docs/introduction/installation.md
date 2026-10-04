---
title: Installation
---
<!-- markdownlint-disable MD024 MD046 -->
Install the LOESS library for your preferred language.

## From NPM (recommended)

```bash
npm install fastloess-wasm
```

## From CDN

```html
<script type="module">
  import init, { Loess } from "https://cdn.jsdelivr.net/npm/fastloess-wasm@0.9/fastloess_wasm.js";
  await init();
</script>
```

## From Source

```bash
# Install Rust first: https://rustup.rs/
# Install wasm-pack: https://rustwasm.github.io/wasm-pack/installer/
git clone https://github.com/thisisamirv/loess-project
cd loess-project/bindings/wasm
# For bundlers (Webpack, Vite, etc.)
wasm-pack build --target bundler
# For Node.js
wasm-pack build --target nodejs
# For browser (no bundler)
wasm-pack build --target web
```

## Check the Package Version

For a Node.js build (`wasm-pack build --target nodejs`):

```javascript
const { version } = require('fastloess-wasm');
console.log(version());
```

```output
2.1.0
```

For a browser build (`--target web`), initialize WebAssembly first:

```javascript
import init, { version } from 'fastloess-wasm';

await init();
console.log(version());
```

This reports the WASM binding package version.

---

## Verify Installation

```javascript
const { Loess } = require('fastloess-wasm');

const x = new Float64Array([1.0, 2.0, 3.0]);
const y = new Float64Array([2.0, 4.0, 6.0]);
const model = new Loess({});
const result = model.fit(x, y);
console.log("Installed successfully!");
```

```output
Installed successfully!
```
