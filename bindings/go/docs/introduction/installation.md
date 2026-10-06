---
title: "Installation"
weight: 15
---

## Requirements

- Go 1.23+
- `CGO_ENABLED=1` and a C compiler:
  - Linux/macOS: GCC or Clang (usually already present)
  - Windows: a MinGW-w64 toolchain (e.g. via [MSYS2](https://www.msys2.org/) or [WinLibs](https://winlibs.com/)), since Go's `cgo` invokes `gcc` on Windows, not MSVC's `cl.exe`
- A prebuilt copy of the native `fastloess_go` static library and its header (`fastloess_go.h`)

## Within the `loess-project` monorepo

If you're working inside a checkout of [`loess-project`](https://github.com/thisisamirv/loess-project), the root `Makefile` handles building the native library for you:

```sh
make go        # build the Rust FFI crate, then `go build ./...`
make go-dev    # full dev checks: fmt, lint, tests, doc snippets
```

`bindings/go/fastloess/ffi.go`'s `#cgo` directives point at `../../../target/release-c` (Linux/macOS) or `../../../target/x86_64-pc-windows-gnu/release-c` (Windows), which is where `cargo build -p fastloess-go --profile release-c` places the static library within this repo's layout.

## As a standalone module

Outside the monorepo, download the prebuilt static library and header attached to a [GitHub release](https://github.com/thisisamirv/loess-project/releases), then point `cgo` at them:

Put `fastloess_go.h` in your native installation's `include` directory and rename the selected platform archive to `libfastloess_go.a` in its `lib` directory. The `-lfastloess_go` linker flag does not find the platform-suffixed release filename automatically. Use the Go module and native assets from the same release; `go get` does not install the native library or header.

The existing v2.0.0 Go tag predates the required `/v2` module-path correction. A new release containing the corrected module path is needed for public Go proxy resolution; published tags must not be moved or replaced.

```sh
export CGO_ENABLED=1
export CGO_CFLAGS="-I/path/to/fastloess_go/include"
export CGO_LDFLAGS="-L/path/to/fastloess_go/lib -lfastloess_go -lm -ldl -lpthread"  # Linux
go build ./...
```

On macOS, drop `-ldl -lpthread` (not needed). On Windows, use `-lws2_32 -luserenv -lbcrypt -lntdll -lpthread` instead, and ensure a MinGW-w64 `gcc.exe` is on `PATH`.

Each release publishes `libfastloess_go-<platform>.a` assets for `linux-x64`, `linux-x64-musl` (Alpine), `linux-arm64`, `linux-arm64-musl` (Alpine), `macos-x64`, `macos-arm64`, `win32-x64`, and `win32-arm64` — download the one matching your target `GOOS`/`GOARCH` and libc.

Alternatively, build the native library yourself from the [`loess-project`](https://github.com/thisisamirv/loess-project) source:

```sh
git clone https://github.com/thisisamirv/loess-project
cd loess-project
cargo build --locked -p fastloess-go --profile release-c
```
