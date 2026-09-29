package fastloess

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo linux LDFLAGS: -L${SRCDIR}/../../../target/release-c -lfastloess_go -lm -ldl -lpthread
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../target/release-c -lfastloess_go
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/../../../target/x86_64-pc-windows-gnu/release-c -lfastloess_go -lws2_32 -luserenv -lbcrypt -lntdll -lpthread
// -static avoids depending on llvm-mingw's runtime DLLs (libunwind/libc++/
// libwinpthread) being discoverable on PATH at execution time, since they
// live in a target-specific sysroot subdirectory rather than next to the
// cross-compiler driver itself.
#cgo windows,arm64 LDFLAGS: -static -L${SRCDIR}/../../../target/aarch64-pc-windows-gnullvm/release-c -lfastloess_go -lws2_32 -luserenv -lbcrypt -lntdll -lpthread
#include <stdlib.h>
#include "fastloess_go.h"
*/
import "C"

import (
	"errors"
	"math"
	"runtime"
	"unsafe"
)

// lastError reads the thread-local error message set by the most recent
// failed constructor call. Callers MUST invoke this on the same OS thread as
// the call that may have set it - see withLockedThread.
func lastError() string {
	cmsg := C.go_last_error_message()
	if cmsg == nil {
		return "unknown error"
	}
	return C.GoString(cmsg)
}

// withLockedThread pins the calling goroutine to its current OS thread for
// the duration of f. This is required whenever we make a cgo call that may
// set fastLoess's thread-local last-error slot, followed by a second cgo
// call to read it back (go_last_error_message) - without this, Go's
// scheduler could migrate the goroutine to a different OS thread in
// between, and we'd read an unrelated thread's (empty) error slot.
func withLockedThread(f func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	f()
}

func cStringOrNil(s string) *C.char {
	if s == "" {
		return nil
	}
	return C.CString(s)
}

func freeCString(s *C.char) {
	if s != nil {
		C.free(unsafe.Pointer(s))
	}
}

func boolToCInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func optFloat(v float64, set bool) C.double {
	if !set {
		return C.double(math.NaN())
	}
	return C.double(v)
}

// cDoubles returns a pointer to the first element of xs (or nil if empty)
// and its length, suitable for passing to a `const double *, unsigned long`
// FFI parameter pair. The backing array of a []float64 contains no Go
// pointers, so passing its address across cgo is safe per the cgo pointer
// passing rules.
func cDoubles(xs []float64) (*C.double, C.ulong) {
	if len(xs) == 0 {
		return nil, 0
	}
	return (*C.double)(unsafe.Pointer(&xs[0])), C.ulong(len(xs))
}

// cDoubleSliceToGo copies n float64s out of a Rust-allocated buffer. Returns
// nil if ptr is nil (meaning the field was not computed).
func cDoubleSliceToGo(ptr *C.double, n int) []float64 {
	if ptr == nil || n == 0 {
		return nil
	}
	src := unsafe.Slice((*float64)(unsafe.Pointer(ptr)), n)
	out := make([]float64, n)
	copy(out, src)
	return out
}

func cDoubleOptional(value C.double) *float64 {
	f := float64(value)
	if math.IsNaN(f) {
		return nil
	}
	return &f
}

// Diagnostics holds goodness-of-fit metrics, populated when ReturnDiagnostics
// is enabled.
type Diagnostics struct {
	RMSE        *float64
	MAE         *float64
	RSquared    *float64
	AIC         *float64
	AICc        *float64
	EffectiveDF *float64
	ResidualSD  *float64
}

// HatMatrixStats holds hat-matrix statistics, populated when ReturnSE is
// enabled. Batch model only.
type HatMatrixStats struct {
	ENP           float64
	TraceHat      float64
	Delta1        float64
	Delta2        float64
	ResidualScale float64
	// Leverage is the per-point hat-matrix diagonal (length N).
	Leverage []float64
}

// Result is the outcome of a batch fit, streaming chunk/finalize, or is
// embedded conceptually (as PointResult) for the online model.
type Result struct {
	// X is the sorted input x values (length N, or N*Dimensions for
	// multivariate input).
	X []float64
	// Y is the smoothed y values (length N).
	Y []float64

	// StandardErrors is nil unless ReturnSE was requested.
	StandardErrors []float64
	// ConfidenceLower/ConfidenceUpper are nil unless ConfidenceIntervals was set.
	ConfidenceLower []float64
	ConfidenceUpper []float64
	// PredictionLower/PredictionUpper are nil unless PredictionIntervals was set.
	PredictionLower []float64
	PredictionUpper []float64
	// Residuals is nil unless ReturnResiduals was requested.
	Residuals []float64
	// RobustnessWeights is nil unless ReturnRobustnessWeights was requested.
	RobustnessWeights []float64
	// Gradient is nil unless ReturnGradient was requested (flattened, Dimensions
	// values per point; only populated when SurfaceMode is "direct").
	Gradient []float64
	// CVScores is nil unless cross-validation was configured.
	CVScores []float64

	// FractionUsed is the smoothing fraction actually applied.
	FractionUsed float64
	// IterationsUsed is the number of robustness iterations performed, or -1
	// if not available (e.g. for streaming intermediate chunks).
	IterationsUsed int
	// Dimensions is the number of predictor dimensions used.
	Dimensions int

	// Diagnostics is nil unless ReturnDiagnostics was requested.
	Diagnostics *Diagnostics
	// HatMatrix is nil unless ReturnSE was requested. Batch model only.
	HatMatrix *HatMatrixStats

	// PredictModel is non-nil only if Options.RetainModel was set to true. Enables
	// out-of-sample Predict() calls against this fitted model. Call Close (or let the
	// garbage collector finalize it) when no longer needed.
	PredictModel *PredictModel
}

func resultFromC(cres C.fastloess_GoLoessResult) (Result, error) {
	if cres.error != nil {
		msg := C.GoString(cres.error)
		C.go_loess_free_result(&cres)
		return Result{}, errors.New(msg)
	}

	n := int(cres.n)
	cvN := int(cres.cv_scores_len)

	r := Result{
		X:                 cDoubleSliceToGo(cres.x, n),
		Y:                 cDoubleSliceToGo(cres.y, n),
		StandardErrors:    cDoubleSliceToGo(cres.standard_errors, n),
		ConfidenceLower:   cDoubleSliceToGo(cres.confidence_lower, n),
		ConfidenceUpper:   cDoubleSliceToGo(cres.confidence_upper, n),
		PredictionLower:   cDoubleSliceToGo(cres.prediction_lower, n),
		PredictionUpper:   cDoubleSliceToGo(cres.prediction_upper, n),
		Residuals:         cDoubleSliceToGo(cres.residuals, n),
		RobustnessWeights: cDoubleSliceToGo(cres.robustness_weights, n),
		CVScores:          cDoubleSliceToGo(cres.cv_scores, cvN),
		FractionUsed:      float64(cres.fraction_used),
		IterationsUsed:    int(cres.iterations_used),
		Dimensions:        int(cres.dimensions),
		PredictModel:      predictModelFromC(cres.predict_handle),
	}

	if !math.IsNaN(float64(cres.rmse)) {
		r.Diagnostics = &Diagnostics{
			RMSE:        cDoubleOptional(cres.rmse),
			MAE:         cDoubleOptional(cres.mae),
			RSquared:    cDoubleOptional(cres.r_squared),
			AIC:         cDoubleOptional(cres.aic),
			AICc:        cDoubleOptional(cres.aicc),
			EffectiveDF: cDoubleOptional(cres.effective_df),
			ResidualSD:  cDoubleOptional(cres.residual_sd),
		}
	}

	if !math.IsNaN(float64(cres.enp)) {
		r.HatMatrix = &HatMatrixStats{
			ENP:           float64(cres.enp),
			TraceHat:      float64(cres.trace_hat),
			Delta1:        float64(cres.delta1),
			Delta2:        float64(cres.delta2),
			ResidualScale: float64(cres.residual_scale),
			Leverage:      cDoubleSliceToGo(cres.leverage, n),
		}
	}

	r.Gradient = cDoubleSliceToGo(cres.gradient, n*r.Dimensions)

	C.go_loess_free_result(&cres)
	return r, nil
}

// PredictModel is the retained fitted-model state enabling out-of-sample Predict()
// calls, returned via Result.PredictModel when Options.RetainModel was set to true.
//
// PredictModel is not safe for concurrent use; each goroutine should use its own
// instance, or callers must serialize access.
type PredictModel struct {
	ptr *C.fastloess_GoPredictHandle
}

func predictModelFromC(handle *C.fastloess_GoPredictHandle) *PredictModel {
	if handle == nil {
		return nil
	}
	pm := &PredictModel{ptr: handle}
	runtime.SetFinalizer(pm, finalizePredictModel)
	return pm
}

func finalizePredictModel(pm *PredictModel) {
	_ = pm.Close()
}

// Close releases the native resources held by this model. Safe to call multiple
// times. Relying on the garbage collector's finalizer instead delays releasing
// native memory - call Close explicitly (e.g. via defer) when possible.
func (pm *PredictModel) Close() error {
	if pm != nil && pm.ptr != nil {
		C.go_predict_handle_free(pm.ptr)
		pm.ptr = nil
		runtime.SetFinalizer(pm, nil)
	}
	return nil
}

// PredictOptions configures a PredictModel.Predict call.
type PredictOptions struct {
	// Outputs selects optional prediction components: "se", "derivative", or "gradient".
	Outputs []string
	// ReturnSE requests standard errors in the output.
	ReturnSE bool
	// ConfidenceLevel is the confidence interval coverage level (e.g. 0.95). Nil disables it.
	ConfidenceLevel *float64
	// PredictionLevel is the prediction interval coverage level (e.g. 0.95). Nil disables it.
	PredictionLevel *float64
	// ReturnDerivative requests the local fit's gradient at each query point.
	ReturnDerivative bool
	// Extrapolation is the behavior for query points outside the training range:
	// "clamp" (default), "linear", or "error".
	Extrapolation string
	// MaxExtrapolationDistance caps how far "linear" extrapolation may extend beyond
	// the training boundary before Predict errors instead of returning an unbounded
	// value. Nil disables the cap.
	MaxExtrapolationDistance *float64
	// MaxNeighborDistance caps the distance to the farthest point in a query's
	// neighbor window before Predict errors, catching in-range-but-sparse query
	// points. Nil disables the cap.
	MaxNeighborDistance *float64
}

// PredictResult is the outcome of PredictModel.Predict.
type PredictResult struct {
	// Y is the predicted value for each query point.
	Y []float64
	// StandardErrors is nil unless ReturnSE/ConfidenceLevel/PredictionLevel was set.
	StandardErrors []float64
	// ConfidenceLower/ConfidenceUpper are nil unless ConfidenceLevel was set.
	ConfidenceLower []float64
	ConfidenceUpper []float64
	// PredictionLower/PredictionUpper are nil unless PredictionLevel was set.
	PredictionLower []float64
	PredictionUpper []float64
	// Derivative is nil unless ReturnDerivative was requested (length
	// len(newX)*Dimensions, flattened like newX).
	Derivative []float64
}

// Predict evaluates the fitted model at out-of-sample query points not in the
// training set (flattened, Dimensions values per point).
func (pm *PredictModel) Predict(newX []float64, opts PredictOptions) (PredictResult, error) {
	if pm == nil || pm.ptr == nil {
		return PredictResult{}, errors.New("fastloess: Predict called on a nil/closed PredictModel (was RetainModel set?)")
	}
	if len(newX) == 0 {
		return PredictResult{}, errors.New("fastloess: newX must be non-empty")
	}

	extrap := cStringOrNil(opts.Extrapolation)
	defer freeCString(extrap)

	cl, clSet := optPtr(opts.ConfidenceLevel)
	pl, plSet := optPtr(opts.PredictionLevel)
	maxExtrap, maxExtrapSet := optPtr(opts.MaxExtrapolationDistance)
	maxNeighbor, maxNeighborSet := optPtr(opts.MaxNeighborDistance)
	newXPtr, newXLen := cDoubles(newX)

	cres := C.go_predict(
		pm.ptr,
		newXPtr, newXLen,
		boolToCInt(opts.ReturnSE || hasOutput(opts.Outputs, "se")),
		optFloat(cl, clSet),
		optFloat(pl, plSet),
		boolToCInt(opts.ReturnDerivative || hasOutput(opts.Outputs, "derivative") || hasOutput(opts.Outputs, "gradient")),
		extrap,
		optFloat(maxExtrap, maxExtrapSet),
		optFloat(maxNeighbor, maxNeighborSet),
	)
	if cres.error != nil {
		msg := C.GoString(cres.error)
		C.go_predict_free_result(&cres)
		return PredictResult{}, errors.New(msg)
	}

	n := int(cres.n)
	dims := int(cres.dimensions)
	r := PredictResult{
		Y:               cDoubleSliceToGo(cres.y, n),
		StandardErrors:  cDoubleSliceToGo(cres.standard_errors, n),
		ConfidenceLower: cDoubleSliceToGo(cres.confidence_lower, n),
		ConfidenceUpper: cDoubleSliceToGo(cres.confidence_upper, n),
		PredictionLower: cDoubleSliceToGo(cres.prediction_lower, n),
		PredictionUpper: cDoubleSliceToGo(cres.prediction_upper, n),
		Derivative:      cDoubleSliceToGo(cres.derivative, n*dims),
	}
	C.go_predict_free_result(&cres)
	return r, nil
}

// PointResult is the outcome of OnlineLoess.AddPoint once the window has
// enough points to produce a smoothed value.
type PointResult struct {
	Y                float64
	StandardError    float64 // NaN if not computed
	Residual         float64 // NaN if not computed
	RobustnessWeight float64 // NaN if not computed
	IterationsUsed   int     // -1 if not applicable
	// ConfidenceLower/ConfidenceUpper are NaN unless ConfidenceIntervals was
	// set and UpdateMode = "full".
	ConfidenceLower float64
	ConfidenceUpper float64
	// PredictionLower/PredictionUpper are NaN unless PredictionIntervals was
	// set and UpdateMode = "full".
	PredictionLower float64
	PredictionUpper float64
	// Gradient is nil unless ReturnGradient was requested (Dimensions values
	// for the latest point; only populated when SurfaceMode is "direct").
	Gradient []float64
}
