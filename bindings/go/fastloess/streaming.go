package fastloess

/*
#include "fastloess_go.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// StreamingOptions configures a StreamingLoess model. Confidence/prediction
// intervals, standard errors, and cross-validation are Batch-only and have
// no effect here.
type StreamingOptions struct {
	// Fraction is the smoothing fraction, in (0, 1]. Default: 0.67.
	Fraction float64
	// Iterations is the number of robustness iterations, in [0, 1000]. Default: 3.
	Iterations int

	// WeightFunction is the kernel weight function: "tricube" (default),
	// "gaussian", "uniform" (alias "boxcar"), "cosine", "epanechnikov",
	// "biweight" (alias "bisquare"), or "triangle" (alias "triangular").
	WeightFunction string
	// RobustnessMethod is the outlier downweighting method: "bisquare"
	// (default, alias "biweight"), "huber", or "talwar".
	RobustnessMethod string
	// ScalingMethod is the residual scale estimator for robustness weights:
	// "mad" (default, alias "median_absolute_deviation"), "mar" (alias
	// "median_absolute_residual"), or "mean" (alias "mean_absolute_residual").
	ScalingMethod string
	// BoundaryPolicy is the boundary handling strategy: "extend" (default,
	// alias "pad"), "reflect" (alias "mirror"), "zero", or "noboundary"
	// (alias "none").
	BoundaryPolicy string
	// ZeroWeightFallback is the fallback policy when all robustness weights
	// drop to zero: "use_local_mean" (default, aliases "local_mean", "mean"),
	// "return_original" (alias "original"), or "return_none" (alias "none").
	ZeroWeightFallback string

	// Degree is the local polynomial degree: "constant", "linear" (default),
	// "quadratic", "cubic", or "quartic".
	Degree string
	// Dimensions is the number of predictor dimensions. Default: 1.
	Dimensions int
	// DistanceMetric is the distance metric used for neighborhood search:
	// "normalized" (default), "euclidean", "manhattan", "chebyshev",
	// "minkowski:p" (e.g. "minkowski:3"), or "weighted" (requires
	// WeightedMetricWeights).
	DistanceMetric string
	// WeightedMetricWeights are per-dimension weights, used when
	// DistanceMetric is "weighted" (or omitted but weights are provided).
	WeightedMetricWeights []float64
	// SurfaceMode controls the fitting surface: "interpolation" (default,
	// fast, uses a k-d tree of vertices) or "direct" (exact, fits every
	// point directly).
	SurfaceMode string

	// Cell is the interpolation cell size tuning parameter, in (0, 1].
	// Nil uses the library default. Only applies when SurfaceMode is
	// "interpolation".
	Cell *float64
	// InterpolationVertices caps the number of interpolation vertices. Nil
	// uses the library default. Only applies when SurfaceMode is
	// "interpolation".
	InterpolationVertices *int
	// BoundaryDegreeFallback controls whether the polynomial degree is
	// reduced near boundary vertices to avoid extrapolation artifacts. Nil
	// uses the library default.
	BoundaryDegreeFallback *bool

	// AutoConverge is the convergence tolerance for early stopping of
	// robustness iterations. Nil disables early stopping.
	AutoConverge *float64

	// ReturnDiagnostics requests fit-quality metrics (RMSE, MAE, R-squared, AIC, etc.).
	ReturnDiagnostics bool
	// ReturnResiduals requests residuals in the result.
	ReturnResiduals bool
	// ReturnRobustnessWeights requests per-point robustness weights in the result.
	ReturnRobustnessWeights bool
	// Parallel enables parallel processing. Default: true.
	Parallel bool

	// ChunkSize is the number of points processed per chunk. Default: 5000.
	ChunkSize int
	// Overlap is the number of points shared between consecutive chunks.
	// Negative means "use the library default".
	Overlap int
	// MergeStrategy controls how overlapping chunk results are combined,
	// e.g. "weighted_average" (default).
	MergeStrategy string
}

// DefaultStreamingOptions returns recommended defaults for streaming use.
func DefaultStreamingOptions() StreamingOptions {
	return StreamingOptions{
		Fraction:           0.67,
		Iterations:         3,
		WeightFunction:     "tricube",
		RobustnessMethod:   "bisquare",
		ScalingMethod:      "mad",
		BoundaryPolicy:     "extend",
		ZeroWeightFallback: "use_local_mean",
		Degree:             "linear",
		Dimensions:         1,
		DistanceMetric:     "normalized",
		SurfaceMode:        "interpolation",
		Parallel:           true,
		ChunkSize:          5000,
		Overlap:            -1,
		MergeStrategy:      "weighted_average",
	}
}

// StreamingLoess processes data in chunks, useful for datasets that don't
// fit in memory or arrive incrementally.
//
// StreamingLoess is not safe for concurrent use.
type StreamingLoess struct {
	ptr *C.fastloess_GoStreamingLoess
}

// NewStreamingLoess creates a new streaming model with the given options.
func NewStreamingLoess(opts StreamingOptions) (*StreamingLoess, error) {
	wf := cStringOrNil(opts.WeightFunction)
	defer freeCString(wf)
	rm := cStringOrNil(opts.RobustnessMethod)
	defer freeCString(rm)
	sm := cStringOrNil(opts.ScalingMethod)
	defer freeCString(sm)
	bp := cStringOrNil(opts.BoundaryPolicy)
	defer freeCString(bp)
	zwf := cStringOrNil(opts.ZeroWeightFallback)
	defer freeCString(zwf)
	ms := cStringOrNil(opts.MergeStrategy)
	defer freeCString(ms)
	degree := cStringOrNil(opts.Degree)
	defer freeCString(degree)
	distanceMetric := cStringOrNil(opts.DistanceMetric)
	defer freeCString(distanceMetric)
	surfaceMode := cStringOrNil(opts.SurfaceMode)
	defer freeCString(surfaceMode)

	autoConverge, autoConvergeSet := optPtr(opts.AutoConverge)
	wmwPtr, wmwLen := cDoubles(opts.WeightedMetricWeights)

	cell, cellSet := 0.0, false
	if opts.Cell != nil {
		cell, cellSet = *opts.Cell, true
	}
	interpolationVertices := C.int(-1)
	if opts.InterpolationVertices != nil {
		interpolationVertices = C.int(*opts.InterpolationVertices)
	}
	boundaryDegreeFallback := C.int(-1)
	if opts.BoundaryDegreeFallback != nil {
		boundaryDegreeFallback = boolToCInt(*opts.BoundaryDegreeFallback)
	}

	var ptr *C.fastloess_GoStreamingLoess
	var errMsg string
	withLockedThread(func() {
		ptr = C.go_streaming_new(
			C.double(opts.Fraction),
			C.int(opts.Iterations),
			wf, rm, sm, bp,
			boolToCInt(opts.ReturnDiagnostics),
			boolToCInt(opts.ReturnResiduals),
			boolToCInt(opts.ReturnRobustnessWeights),
			zwf,
			optFloat(autoConverge, autoConvergeSet),
			boolToCInt(opts.Parallel),
			C.int(opts.ChunkSize),
			C.int(opts.Overlap),
			ms,
			degree,
			C.int(opts.Dimensions),
			distanceMetric,
			surfaceMode,
			optFloat(cell, cellSet),
			interpolationVertices,
			boundaryDegreeFallback,
			wmwPtr, wmwLen,
		)
		if ptr == nil {
			errMsg = lastError()
		}
	})
	if ptr == nil {
		return nil, errors.New(errMsg)
	}

	s := &StreamingLoess{ptr: ptr}
	runtime.SetFinalizer(s, finalizeStreaming)
	return s, nil
}

func finalizeStreaming(s *StreamingLoess) {
	_ = s.Close()
}

// ProcessChunk fits and returns the result for one chunk of data. For
// multivariate input (Dimensions > 1), x is flattened row-major.
func (s *StreamingLoess) ProcessChunk(x, y []float64) (Result, error) {
	if s == nil || s.ptr == nil {
		return Result{}, errors.New("fastloess: ProcessChunk called on a closed StreamingLoess model")
	}
	if len(x) == 0 || len(y) == 0 {
		return Result{}, errors.New("fastloess: x and y must be non-empty")
	}
	xPtr, xLen := cDoubles(x)
	yPtr, yLen := cDoubles(y)
	cres := C.go_streaming_process(s.ptr, xPtr, xLen, yPtr, yLen)
	return resultFromC(cres)
}

// Finalize flushes any buffered data and returns the final merged result.
func (s *StreamingLoess) Finalize() (Result, error) {
	if s == nil || s.ptr == nil {
		return Result{}, errors.New("fastloess: Finalize called on a closed StreamingLoess model")
	}
	cres := C.go_streaming_finalize(s.ptr)
	return resultFromC(cres)
}

// Close releases the native resources held by this model. Safe to call
// multiple times.
func (s *StreamingLoess) Close() error {
	if s != nil && s.ptr != nil {
		C.go_streaming_free(s.ptr)
		s.ptr = nil
		runtime.SetFinalizer(s, nil)
	}
	return nil
}
