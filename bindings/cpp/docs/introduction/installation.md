\page installation Installation

# Installation

Install the LOESS library for your preferred language.

Each prebuilt platform archive contains that platform's library and the matching C++ and C headers. Download and extract the archive for your target; its files are placed in the current directory.

Future C++ releases build with the committed workspace `Cargo.lock` and include `THIRD_PARTY_LICENSES.html`, generated from the locked runtime dependency graph. The existing v2.0.0 binaries did not publish their build lockfile or dependency notices; a report generated later cannot establish the dependencies embedded in those binaries.

## Pre-built Binaries (Linux (x64))

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-x64.tar
tar -xf libfastloess-linux-x64.tar
g++ -o myapp myapp.cpp -L. -lfastloess-linux-x64
```

## Pre-built Binaries (Linux (ARM64))

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-arm64.tar
tar -xf libfastloess-linux-arm64.tar
g++ -o myapp myapp.cpp -L. -lfastloess-linux-arm64
```

## Pre-built Binaries (Linux (x86), 32-bit)

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-x86.tar
tar -xf libfastloess-linux-x86.tar
g++ -m32 -std=c++17 -I. -o myapp myapp.cpp -L. -lfastloess-linux-x86
```

## Pre-built Binaries (Linux (ARMv7), hard-float)

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-armv7.tar
tar -xf libfastloess-linux-armv7.tar
arm-linux-gnueabihf-g++ -std=c++17 -I. -o myapp myapp.cpp -L. -lfastloess-linux-armv7
```

## Pre-built Binaries (Linux (x64), musl/Alpine)

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-x64-musl.tar
tar -xf libfastloess-linux-x64-musl.tar
g++ -o myapp myapp.cpp -L. -lfastloess-linux-x64-musl
```

## Pre-built Binaries (Linux (ARM64), musl/Alpine)

```bash
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-linux-arm64-musl.tar
tar -xf libfastloess-linux-arm64-musl.tar
g++ -o myapp myapp.cpp -L. -lfastloess-linux-arm64-musl
```

## Pre-built Binaries (macOS (x64))

```bash
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-macos-x64.tar
tar -xf libfastloess-macos-x64.tar
clang++ -o myapp myapp.cpp -L. -lfastloess-macos-x64
```

## Pre-built Binaries (macOS (ARM64))

```bash
curl -LO https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-macos-arm64.tar
tar -xf libfastloess-macos-arm64.tar
clang++ -o myapp myapp.cpp -L. -lfastloess-macos-arm64
```

## Pre-built Binaries (Android)

Choose the shared library matching the Android ABI used by your application:

| ABI | Release archive |
| --- | --- |
| `arm64-v8a` | `libfastloess-android-arm64-v8a.tar` |
| `armeabi-v7a` | `libfastloess-android-armeabi-v7a.tar` |
| `x86` | `libfastloess-android-x86.tar` |
| `x86_64` | `libfastloess-android-x86_64.tar` |

Download and extract the archive for the ABI being built by the Android NDK. It contains the `.so` and all three headers.

## Pre-built Binaries (iOS)

The release provides static archives for physical devices and simulators. Use only the archive matching the active Xcode destination:

| Destination | Rust target | Release archive |
| --- | --- | --- |
| iOS device (arm64) | `aarch64-apple-ios` | `libfastloess-ios-arm64.tar` |
| iOS simulator (Apple silicon) | `aarch64-apple-ios-sim` | `libfastloess-ios-simulator-arm64.tar` |
| iOS simulator (Intel) | `x86_64-apple-ios` | `libfastloess-ios-simulator-x86_64.tar` |

Download and extract the archive matching the active Xcode destination. It contains the static library and all three headers.

## Pre-built Binaries (Windows (x64))

```powershell
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-windows-x64-msvc.tar
tar -xf libfastloess-windows-x64-msvc.tar
cl /std:c++17 myapp.cpp /link fastloess-win32-x64.lib
```

## Pre-built Binaries (Windows (ARM64))

```powershell
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-windows-arm64.tar
tar -xf libfastloess-windows-arm64.tar
cl /std:c++17 myapp.cpp /link fastloess-win32-arm64.lib
```

## Pre-built Binaries (Windows (x64), MinGW-w64)

```powershell
wget https://github.com/thisisamirv/loess-project/releases/latest/download/libfastloess-windows-x64-gnu.tar
tar -xf libfastloess-windows-x64-gnu.tar
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

The recipe links its homepage to the C++ documentation and provides checks for the installed headers and library directory. Recipes with the standalone smoke test can compile and run a small linear fit against the installed library:

```bash
spack test run --alias fastloess-cpp-smoke fastloess-cpp
spack test results -l fastloess-cpp-smoke
```

## From vcpkg (Repository Overlay)

The repository provides a `fastloess` overlay port for the published v2.0.0 CPU shared library. It is not yet part of vcpkg's curated registry. Rust and Cargo are not required for installation.

From the repository root, with `VCPKG_ROOT` pointing to an existing vcpkg checkout:

```powershell
& "$env:VCPKG_ROOT/vcpkg.exe" install fastloess:x64-windows --overlay-ports=bindings/cpp/vcpkg
```

Windows x64/ARM64 MSVC, glibc Linux x64/ARM64, and macOS x64/ARM64 dynamic targets are supported. Use dynamic triplets such as `x64-linux-dynamic` or `arm64-osx-dynamic` on Unix; default static triplets and musl are unsupported by this port. Debug and Release consumers use the same prebuilt Release library; Windows requires the dynamic CRT.

Configure your application with vcpkg's CMake toolchain and the same triplet, then link the installed target:

```cmake
find_package(fastloess CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE fastloess::fastloess)
```

The overlay README under `bindings/cpp/vcpkg` contains installation, consumer validation, and clangd configuration commands. The v2.0.0 port discloses its dependency-license provenance limitation; future releases must install the matching generated notices before that limitation can be resolved.

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
Header version: 3.0.0
Loaded library version: 3.0.0
```
