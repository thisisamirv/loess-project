<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOESS library for your preferred language.

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
      import init, { Loess } from "https://cdn.jsdelivr.net/npm/fastloess-wasm@0.9/fastloess_wasm.js";
      await init();
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

=== "Python"

    ```python
    import fastloess as fl
    import numpy as np
    
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    
    model = fl.Loess()
    result = model.fit(x, y)
    print("Installed successfully!")
    ```

=== "Node.js"

    ```javascript
    const fl = require('fastloess');
    
    const x = new Float64Array([1.0, 2.0, 3.0]);
    const y = new Float64Array([2.0, 4.0, 6.0]);
    
    const model = new fl.Loess({});
    const result = model.fit(x, y);
    console.log("Installed successfully!");
    ```

=== "WebAssembly"

    ```javascript
    const { Loess } = require('fastloess-wasm');

    const x = new Float64Array([1.0, 2.0, 3.0]);
    const y = new Float64Array([2.0, 4.0, 6.0]);
    const model = new Loess({});
    const result = model.fit(x, y);
    console.log("Installed successfully!");
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
        model.fit(x, y).value();

        std::cout << "Installed successfully!" << std::endl;
        return 0;
    }
    ```
