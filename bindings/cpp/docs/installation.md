# Installation

Install the LOESS library for your preferred language.

## Pre-built Binaries (Linux (x64))

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-x64.so
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
g++ -o myapp myapp.cpp -L. -lfastloess-linux-x64
```

## Pre-built Binaries (macOS (x64))

```bash
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-macos-x64.dylib
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
clang++ -o myapp myapp.cpp -L. -lfastloess-macos-x64
```

## Pre-built Binaries (Windows (x64))

```powershell
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess-win32-x64.dll
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
cl myapp.cpp /link fastloess-win32-x64.lib
```

## From Source

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

## From conda-forge

```bash
conda install -c conda-forge libfastloess
```

---

## Verify Installation

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

```output
Installed successfully!
```
