package fastloess

// version is the released version of this Go binding (bindings/go), tracked
// independently of the underlying fastLoess Rust core's crate version.
const version = "1.1.0"

// Version returns the version of this Go binding package.
func Version() string {
	return version
}
