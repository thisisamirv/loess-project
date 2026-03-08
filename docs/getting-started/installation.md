<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOESS library for your preferred language.

=== "R"

    === "From R-universe (recommended)"

    ```r
    install.packages("rfastloess", repos = "https://thisisamirv.r-universe.dev")
    ```

    === "From conda-forge"

    ```r
    conda install -c conda-forge r-rfastloess
    ```

    === "From Source"

    ```r
    # Install Rust first: https://rustup.rs/
    devtools::install_github("thisisamirv/loess-project", subdir = "bindings/r")
    ```

=== "Python"

    === "From PyPI (recommended)"

    ```bash
    pip install fastloess
    ```

    === "From conda-forge"

    ```bash
    conda install -c conda-forge fastloess
    ```

    === "From Source"

    ```bash
    # Install Rust first: https://rustup.rs/
    git clone https://github.com/thisisamirv/loess-project
    cd loess-project/bindings/python
    pip install maturin
    maturin develop --release
    ```

=== "Rust"

    === "From crates.io"

    ```toml
    # loess (no_std compatible)
    [dependencies]
    loess = "1.1"

    # fastLoess (parallel + GPU)
    [dependencies]
    fastLoess = { version = "1.1", features = ["cpu"] }
    ```

    === "Feature Flags"

    | Crate        | Feature | Description                             |
    |--------------|---------|-----------------------------------------|
    | `loess`     | `std`   | Enable standard library (default)       |
    | `fastLoess` | `cpu`   | Enable CPU parallelism via Rayon        |
    | `fastLoess` | `gpu`   | Enable GPU acceleration via wgpu (beta) |

=== "Julia"

    === "From General Registry (recommended)"

    ```julia
    Pkg.add("FastLOESS")
    ```

    === "From Source"

    ```julia
    using Pkg
    Pkg.develop(url="https://github.com/thisisamirv/loess-project", subdir="bindings/julia/julia")
    ```

=== "Node.js"

    === "From NPM (recommended)"

    ```bash
    npm install fastloess
    ```

    === "From Source"

    ```bash
    git clone https://github.com/thisisamirv/loess-project
    cd loess-project/bindings/nodejs
    npm install
    npm run build
    ```

=== "WebAssembly"

    === "From NPM (recommended)"

    ```bash
    npm install fastloess-wasm
    ```

    === "From CDN"

    ```html
    <script type="module">
      import { smooth } from "https://cdn.jsdelivr.net/npm/fastloess-wasm@0.99/index.js";
    </script>
    ```

    === "From Source"

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

=== "C++"

    === "Pre-built Binaries (Linux (x64))"

    ```bash
    wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-x64.so
    wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
    g++ -o myapp myapp.cpp -L. -lfastloess-linux-x64
    ```

    === "Pre-built Binaries (macOS (x64))"

    ```bash
    curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-macos-x64.dylib
    curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
    clang++ -o myapp myapp.cpp -L. -lfastloess-macos-x64
    ```

    === "Pre-built Binaries (Windows (x64))"

    ```powershell
    wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess-win32-x64.dll
    wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
    cl myapp.cpp /link fastloess-win32-x64.lib
    ```

    === "From Source"

    ```bash
    # Install Rust first: https://rustup.rs/
    git clone https://github.com/thisisamirv/loess-project
    cd loess-project/bindings/cpp

    # Build the library
    cargo build --release

    # Headers are at: include/fastloess.hpp (C++)
    # Library is at: target/release/libfastloess_cpp.so (Linux)
    #                target/release/libfastloess_cpp.dylib (macOS)
    #                target/release/fastloess_cpp.dll (Windows)
    ```

    === "From conda-forge"

    ```bash
    conda install -c conda-forge libfastloess
    ```

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
    using FastLOESS
    
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

=== "C++"

    ```cpp
    #include <fastloess.hpp>
    #include <iostream>
    #include <vector>

    int main() {
        std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> y = {2.0, 4.1, 5.9, 8.2, 9.8};

        fastloess::Loess model;
        auto result = model.fit(x, y);

        std::cout << "Installed successfully!" << std::endl;
        return 0;
    }
    ```
