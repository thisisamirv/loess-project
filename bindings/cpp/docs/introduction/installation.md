\page installation Installation

# Installation

Install the LOESS library for your preferred language.

## Pre-built Binaries (Linux (x64))

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-x64.so
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
g++ -o myapp myapp.cpp -L. -lfastloess-linux-x64
```

## Pre-built Binaries (Linux (ARM64))

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-arm64.so
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
g++ -o myapp myapp.cpp -L. -lfastloess-linux-arm64
```

## Pre-built Binaries (Linux (x86), 32-bit)

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-x86.so
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
g++ -m32 -std=c++17 -I. -o myapp myapp.cpp -L. -lfastloess-linux-x86
```

## Pre-built Binaries (Linux (ARMv7), hard-float)

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-armv7.so
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
arm-linux-gnueabihf-g++ -std=c++17 -I. -o myapp myapp.cpp -L. -lfastloess-linux-armv7
```

## Pre-built Binaries (Linux (x64), musl/Alpine)

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-x64-musl.so
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
g++ -o myapp myapp.cpp -L. -lfastloess-linux-x64-musl
```

## Pre-built Binaries (Linux (ARM64), musl/Alpine)

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-arm64-musl.so
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
g++ -o myapp myapp.cpp -L. -lfastloess-linux-arm64-musl
```

## Pre-built Binaries (macOS (x64))

```bash
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-macos-x64.dylib
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
clang++ -o myapp myapp.cpp -L. -lfastloess-macos-x64
```

## Pre-built Binaries (macOS (ARM64))

```bash
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-macos-arm64.dylib
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
clang++ -o myapp myapp.cpp -L. -lfastloess-macos-arm64
```

## Pre-built Binaries (Android)

Choose the shared library matching the Android ABI used by your application:

| ABI | Release asset |
| --- | --- |
| `arm64-v8a` | `libfastloess-android-arm64-v8a.so` |
| `armeabi-v7a` | `libfastloess-android-armeabi-v7a.so` |
| `x86` | `libfastloess-android-x86.so` |
| `x86_64` | `libfastloess-android-x86_64.so` |

Download the C++ and C headers (`fastloess.hpp`, `fastloess.h`, and `fastloess_version.h`) from the same release. Bundle and link the `.so` for the ABI being built by the Android NDK.

## Pre-built Binaries (iOS)

The release provides static archives for physical devices and simulators. Use only the archive matching the active Xcode destination:

| Destination | Rust target | Release asset |
| --- | --- | --- |
| iOS device (arm64) | `aarch64-apple-ios` | `libfastloess-ios-arm64.a` |
| iOS simulator (Apple silicon) | `aarch64-apple-ios-sim` | `libfastloess-ios-simulator-arm64.a` |
| iOS simulator (Intel) | `x86_64-apple-ios` | `libfastloess-ios-simulator-x86_64.a` |

Download the C++ and C headers (`fastloess.hpp`, `fastloess.h`, and `fastloess_version.h`) from the same release and link the matching static archive into your app or framework.

## Pre-built Binaries (Windows (x64))

```powershell
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess-win32-x64.dll
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess-win32-x64.lib
cl /std:c++17 myapp.cpp /link fastloess-win32-x64.lib
```

## Pre-built Binaries (Windows (ARM64))

```powershell
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess-win32-arm64.dll
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess-win32-arm64.lib
cl /std:c++17 myapp.cpp /link fastloess-win32-arm64.lib
```

## Pre-built Binaries (Windows (x64), MinGW-w64)

```powershell
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess-win32-x64-gnu.dll
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-win32-x64-gnu.dll.a
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.hpp
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess.h
wget https://github.com/thisisamirv/loess-project/releases/latest/download/fastloess_version.h
g++ -std=c++17 -I. -o myapp myapp.cpp -L. -lfastloess-win32-x64-gnu
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

## From Spack

```bash
spack install fastloess-cpp
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

## Check the Header and Library Versions

Cargo and CMake generate `fastloess_version.h` from package metadata. Download it and `fastloess.h` alongside `fastloess.hpp` when using prebuilt binaries. The version header can be included on its own for compile-time checks, without linking the native library:

```cpp
#include <fastloess_version.h>

static_assert(FASTLOESS_CPP_VERSION_MAJOR >= 2,
 "This application requires fastloess-cpp 2 or later");

int main() {}
```

The macros `FASTLOESS_CPP_VERSION_MAJOR`, `FASTLOESS_CPP_VERSION_MINOR`, `FASTLOESS_CPP_VERSION_PATCH`, and `FASTLOESS_CPP_VERSION_STRING` describe the headers used to compile your application. `fastloess.hpp` includes this header automatically.

Use `cpp_version()` to identify the native library loaded at runtime:

```cpp
#include <fastloess.hpp>
#include <iostream>

int main() {
 std::cout << "Header version: " << FASTLOESS_CPP_VERSION_STRING << '\n';
 std::cout << "Loaded library version: " << cpp_version() << '\n';
}
```

```output
Header version: 2.1.0
Loaded library version: 2.1.0
```
