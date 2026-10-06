// Package fastloess provides Go bindings for fastLoess, a high-performance
// implementation of LOESS (Locally Estimated Scatterplot Smoothing) written
// in Rust, exposed to Go via cgo.
//
// Three model types are available, matching the same three offered by every
// other fastLoess binding:
//
//   - Loess: batch smoothing. Processes an entire dataset at once and
//     supports every feature (confidence/prediction intervals,
//     cross-validation, multivariate predictors). Best when the dataset
//     fits in memory.
//   - StreamingLoess: chunked processing, for datasets that don't fit in
//     memory or arrive in chunks.
//   - OnlineLoess: point-by-point processing, for real-time data.
//
// # Quickstart
//
//	opts := fastloess.DefaultOptions()
//	opts.Fraction = 0.2
//	model, err := fastloess.NewLoess(opts)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer model.Close()
//
//	result, err := model.Fit(x, y)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Println(result.Y)
//
// # Building
//
// This package uses cgo to link against the native fastloess_go library
// (built from the sibling Rust crate in this same directory). Within this
// monorepo, `make go` selects external_native and builds the Rust library.
// Bundled Go releases include native headers and CPU libraries for Linux,
// macOS, and Windows on amd64/arm64. Select the musl build tag on Alpine.
// Custom libraries require external_native and CGO_CFLAGS/CGO_LDFLAGS.
//
// # Resource management
//
// Loess, StreamingLoess, and OnlineLoess all hold native (non-Go-GC-
// visible) memory and must be released with Close when no longer needed.
// A finalizer is registered as a safety net, but relying on the garbage
// collector delays releasing native memory - call Close explicitly (e.g.
// via defer) instead.
package fastloess
