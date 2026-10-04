<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOESS library for your preferred language.

## From General Registry (recommended)

```julia
Pkg.add("FastLOESS")
```

## From Source

```julia
using Pkg
Pkg.develop(url="https://github.com/thisisamirv/loess-project", subdir="bindings/julia/julia")
```

---

## Verify Installation

```@example installation
using FastLOESS

x = [1.0, 2.0, 3.0]
y = [2.0, 4.0, 6.0]

model = Loess()
result = fit(model, x, y)
println("Installed successfully!")
```

## Check the Package Version

Read the Julia binding version from its installed package metadata:

```@example package_version
using FastLOESS
version()
```

This is the Julia package version, not the version of the native library.
