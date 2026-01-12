<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOESS library for your preferred language.

## R

**From R-universe (recommended):**

Pre-built binaries, no Rust toolchain required:

```r
install.packages("rfastloess", repos = "https://thisisamirv.r-universe.dev")
```

**From Source:**

Requires Rust toolchain:

```r
# Install Rust first: https://rustup.rs/
devtools::install_github("thisisamirv/loess-project", subdir = "bindings/r")
```

---

## Python

**From PyPI (recommended):**

```bash
pip install fastloess
```

**From conda-forge:**

```bash
conda install -c conda-forge fastloess
```

**From Source:**

```bash
git clone https://github.com/thisisamirv/loess-project
cd loess-project/bindings/python
pip install maturin
maturin develop --release
```

---

## Rust

**From crates.io:**

=== "loess (no_std compatible)"

    ```toml
    [dependencies]
    loess = "0.99"
    ```

=== "fastLoess (parallel + GPU)"

    ```toml
    [dependencies]
    fastLoess = { version = "0.99", features = ["cpu"] }
    ```

---

## Julia

**From General Registry (recommended):**

```julia
using Pkg
Pkg.add("fastloess")
```

**From Source:**

```julia
using Pkg
Pkg.develop(url="https://github.com/thisisamirv/loess-project", subdir="bindings/julia/julia")
```

---

## Node.js

**From NPM (recommended):**

```bash
npm install fastloess
```

**From Source:**

```bash
git clone https://github.com/thisisamirv/loess-project
cd loess-project/bindings/nodejs
npm install
npm run build
```

---

## WebAssembly

**From NPM (recommended):**

```bash
npm install fastloess-wasm
```

**From Source:**

Requires Rust toolchain and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/).

```bash
git clone https://github.com/thisisamirv/loess-project
cd loess-project/bindings/wasm
# For bundlers (Webpack, Vite, etc.)
wasm-pack build --target bundler
# For Node.js
wasm-pack build --target nodejs
# For browser (no bundler)
wasm-pack build --target web
```

---

## C++

**From Source:**

Requires Rust toolchain.

```bash
git clone https://github.com/thisisamirv/loess-project
cd loess-project/bindings/cpp

# Build the library
cargo build --release

# Headers are at: include/fastloess.h (C) and include/fastloess.hpp (C++)
# Library is at: target/release/libfastloess_cpp.so (Linux)
```

---

### Feature Flags

| Crate        | Feature | Description                             |
|--------------|---------|-----------------------------------------|
| `loess`     | `std`   | Enable standard library (default)       |
| `fastLoess` | `cpu`   | Enable CPU parallelism via Rayon        |
| `fastLoess` | `gpu`   | Enable GPU acceleration via wgpu (beta) |

### Minimum Supported Rust Version (MSRV)

Both crates require **Rust 1.85.0** or later.

---

## Verify Installation

=== "R"

    ```r
    library(rfastloess)
    
    x <- c(1, 2, 3)
    y <- c(2, 4, 6)
    
    result <- fastloess(x, y)
    print("Installed successfully!")
    ```

=== "Python"

    ```python
    import fastloess as fl
    import numpy as np
    
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    
    result = fl.smooth(x, y)
    print("Installed successfully!")
    ```

=== "Rust"

    ```rust
    use loess::prelude::*;
    
    fn main() -> Result<(), LoessError> {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        
        let model = Loess::new().adapter(Batch).build()?;
        let result = model.fit(&x, &y)?;
        
        println!("Installed successfully!");
        Ok(())
    }
    ```

=== "Julia"

    ```julia
    using fastloess
    
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    
    result = smooth(x, y)
    println("Installed successfully!")
    ```

=== "Node.js"

    ```javascript
    const fl = require('fastloess');
    
    const x = new Float64Array([1.0, 2.0, 3.0]);
    const y = new Float64Array([2.0, 4.0, 6.0]);
    
    const result = fl.smooth(x, y);
    console.log("Installed successfully!");
    ```

    See [Node.js API](../api/nodejs.md) for full reference.

=== "WebAssembly"

    ```javascript
    import init, { smooth } from 'fastloess-wasm';

    async function verify() {
        await init();
        const x = new Float64Array([1.0, 2.0, 3.0]);
        const y = new Float64Array([2.0, 4.0, 6.0]);
        const result = smooth(x, y);
        console.log("Installed successfully!");
    }
    verify();
    ```

    See [WebAssembly API](../api/wasm.md) for full reference.
