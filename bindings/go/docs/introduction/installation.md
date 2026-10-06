---
title: "Installation"
weight: 15
---

## Requirements

- Go 1.23+
- `CGO_ENABLED=1` and a C compiler:
  - Linux/macOS: GCC or Clang (usually already present)
  - Windows: a MinGW-w64 toolchain (e.g. via [MSYS2](https://www.msys2.org/) or [WinLibs](https://winlibs.com/)), since Go's `cgo` invokes `gcc` on Windows, not MSVC's `cl.exe`
- A supported target: Linux, macOS, or Windows on amd64 or arm64

## Within the `loess-project` monorepo

If you're working inside a checkout of [`loess-project`](https://github.com/thisisamirv/loess-project), the root `Makefile` handles building the native library for you:

```sh
make go        # build the Rust FFI crate, then `go build ./...`
make go-dev    # full dev checks: fmt, lint, tests, doc snippets
```

The Make targets select the `external_native` build tag and link the source-built library in `target/release-c` (Linux/macOS), `target/x86_64-pc-windows-gnu/release-c` (Windows amd64), or `target/aarch64-pc-windows-gnullvm/release-c` (Windows arm64).

For manual Go commands in a source checkout, pass `-tags=external_native`. Configure gopls `buildFlags` with `-tags=external_native` when editing source-checkout files; release bundles are generated in CI rather than stored on the development branch.

## As a standalone module

Starting with the next bundled Go release, the module includes the matching native header, CPU static libraries, dependency notices, and checksums. No Rust installation, separate native download, or `CGO_CFLAGS`/`CGO_LDFLAGS` configuration is needed on supported targets.

From your application's Go module:

```sh
go get github.com/thisisamirv/loess-project/bindings/go/fastloess/v2@latest
CGO_ENABLED=1 go build ./...
```

The existing v2.0.0 Go tag predates the required `/v2` module-path correction and contains no bundled native files. A new release is required; published tags are not moved or replaced. Until that release is published, use the source-build instructions below.

Linux defaults to the glibc archive. On Alpine or another musl system, select the musl archive explicitly:

```sh
CGO_ENABLED=1 go build -tags=musl ./...
```

Windows arm64 requires LLVM-MinGW with `CC=aarch64-w64-mingw32-clang`. Other Windows consumers need a compatible MinGW-w64 C compiler. Cross-compilation also requires a C compiler targeting the destination architecture and libc.

## Custom or source-built native libraries

To use your own library instead of the bundled CPU archives, select `external_native`. Put `fastloess_go.h` in the native installation's `include` directory and the matching archive under `lib/libfastloess_go.a`. Platform-suffixed archives downloaded separately from GitHub Releases must be renamed to this basename. Keep the Go module, header, and native library at the same release.

```sh
export CGO_ENABLED=1
export CGO_CFLAGS="-I/path/to/fastloess_go/include"
export CGO_LDFLAGS="-L/path/to/fastloess_go/lib -lfastloess_go -lm -ldl -lpthread"  # Linux
go build -tags=external_native ./...
```

On macOS, drop `-ldl -lpthread` (not needed). On Windows, use `-lws2_32 -luserenv -lbcrypt -lntdll -lpthread` instead, and ensure a MinGW-w64 `gcc.exe` is on `PATH`.

New releases also publish `libfastloess_go-<platform>.a` assets for `linux-x64`, `linux-x64-musl` (Alpine), `linux-arm64`, `linux-arm64-musl` (Alpine), `macos-x64`, `macos-arm64`, `win32-x64`, and `win32-arm64`. These separate assets are optional when consuming a bundled module.

Alternatively, build the native library yourself from the [`loess-project`](https://github.com/thisisamirv/loess-project) source:

```sh
git clone https://github.com/thisisamirv/loess-project
cd loess-project
cargo build --locked -p fastloess-go --profile release-c
```
