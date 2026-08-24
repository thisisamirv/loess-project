---
title: Installation
---
<!-- markdownlint-disable MD024 MD046 -->
Install the LOESS library for your preferred language.

## From NPM (recommended)

```bash
npm install fastloess
```

## From Source

```bash
git clone https://github.com/thisisamirv/loess-project
cd loess-project/bindings/nodejs
npm install
npm run build
```

---

## Verify Installation

```javascript
const fl = require('fastloess');

const x = new Float64Array([1.0, 2.0, 3.0]);
const y = new Float64Array([2.0, 4.0, 6.0]);

const model = new fl.Loess({});
const result = model.fit(x, y);
console.log("Installed successfully!");
```

```output
Installed successfully!
```
