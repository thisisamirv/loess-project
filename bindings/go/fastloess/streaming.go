package fastloess

/*
#include "fastloess_go.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// StreamingOptions configures a StreamingLoess model.
type StreamingOptions struct {
	Options
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
		Options:       DefaultOptions(),
		ChunkSize:     5000,
		Overlap:       -1,
		MergeStrategy: "weighted_average",
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

	ci, ciSet := optPtr(opts.ConfidenceIntervals)
	pi, piSet := optPtr(opts.PredictionIntervals)
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
			boolToCInt(opts.ReturnSE),
			optFloat(ci, ciSet),
			optFloat(pi, piSet),
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
