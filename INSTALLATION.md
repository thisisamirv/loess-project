<!-- markdownlint-disable MD025 MD041 -->

> [!NOTE]
>
> Installation instructions are available for:
>
> - [R](#r)
> - [Python](#python)
> - [Rust (loess)](#rust-loess-no_std-compatible)
> - [Rust (fastLoess)](#rust-fastloess-parallel--gpu)
> - [Julia](#julia)
> - [Node.js](#nodejs)
> - [WebAssembly](#webassembly)
> - [C++](#c)

---

# R

**From R-universe:**

```r
install.packages("rfastloess", repos = "https://thisisamirv.r-universe.dev")
```

**Or from conda-forge:**

```r
conda install -c conda-forge r-rfastloess
```

# Python

**From PyPI:**

```bash
pip install fastloess
```

**Or from conda-forge:**

```bash
conda install -c conda-forge fastloess
```

# Rust (loess, no_std compatible)

**From crates.io:**

```toml
[dependencies]
loess = "1.1"
```

# Rust (fastLoess, parallel + GPU)

**From crates.io:**

```toml
[dependencies]
fastLoess = { version = "1.1", features = ["cpu"] }
```

# Julia

**From General Registry:**

```julia
using Pkg
Pkg.add("FastLOESS")
```

# Node.js

**From npm:**

```bash
npm install fastloess
```

# WebAssembly

**From npm:**

```bash
npm install fastloess-wasm
```

**Or via CDN:**

```html
<script type="module">
  import init, { smooth } from 'https://unpkg.com/fastloess-wasm@latest';
  await init();
</script>
```

# C++

**From source:**

```bash
make cpp
# Links against libfastloess_cpp.so
```

**Or from conda-forge:**

```bash
conda install -c conda-forge libfastloess
```
