package fastloess

/*
#include "fastloess_go.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// OnlineOptions configures an OnlineLoess model. Confidence/prediction
// intervals, standard errors, cross-validation, and diagnostics/residuals
// are Batch-only (or Batch/Streaming-only) and have no effect here.
type OnlineOptions struct {
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

	// ReturnRobustnessWeights requests per-point robustness weights in the result.
	ReturnRobustnessWeights bool
	// Parallel enables parallel processing (used for internal KD-tree/
	// interval-pass dispatch). Default: false (online updates are
	// latency-sensitive).
	Parallel bool

	// WindowCapacity is the maximum number of recent points retained.
	// Default: 1000.
	WindowCapacity int
	// MinPoints is the minimum number of points required before the model
	// starts producing output. Default: 2.
	MinPoints int
	// UpdateMode controls how the window is updated as new points arrive,
	// e.g. "incremental" (default).
	UpdateMode string
}

// DefaultOnlineOptions returns recommended defaults for online use. Note
// Parallel defaults to false, since per-point updates rarely benefit from
// parallelism.
func DefaultOnlineOptions() OnlineOptions {
	return OnlineOptions{
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
		Parallel:           false,
		WindowCapacity:     1000,
		MinPoints:          2,
		UpdateMode:         "incremental",
	}
}

// OnlineLoess processes one (x, y) point at a time, useful for real-time
// streaming data where results are needed immediately as points arrive.
// AddPoint only accepts a single x coordinate: online mode does not support
// multivariate predictors even if Dimensions was set on construction.
//
// OnlineLoess is not safe for concurrent use.
type OnlineLoess struct {
	ptr *C.fastloess_GoOnlineLoess
}

// NewOnlineLoess creates a new online model with the given options.
func NewOnlineLoess(opts OnlineOptions) (*OnlineLoess, error) {
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
	um := cStringOrNil(opts.UpdateMode)
	defer freeCString(um)
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

	var ptr *C.fastloess_GoOnlineLoess
	var errMsg string
	withLockedThread(func() {
		ptr = C.go_online_new(
			C.double(opts.Fraction),
			C.int(opts.Iterations),
			wf, rm, sm, bp,
			boolToCInt(opts.ReturnRobustnessWeights),
			zwf,
			optFloat(autoConverge, autoConvergeSet),
			boolToCInt(opts.Parallel),
			C.int(opts.WindowCapacity),
			C.int(opts.MinPoints),
			um,
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

	o := &OnlineLoess{ptr: ptr}
	runtime.SetFinalizer(o, finalizeOnline)
	return o, nil
}

func finalizeOnline(o *OnlineLoess) {
	_ = o.Close()
}

// AddPoint adds a single (x, y) observation. ok is false while the window is
// still filling (fewer than MinPoints seen so far); once ok is true, res
// holds the smoothed value for the most recently added point.
func (o *OnlineLoess) AddPoint(x, y float64) (res PointResult, ok bool, err error) {
	if o == nil || o.ptr == nil {
		return PointResult{}, false, errors.New("fastloess: AddPoint called on a closed OnlineLoess model")
	}

	cout := C.go_online_add_point(o.ptr, C.double(x), C.double(y))
	if cout.error != nil {
		msg := C.GoString(cout.error)
		C.go_online_free_output(&cout)
		return PointResult{}, false, errors.New(msg)
	}
	if cout.has_value == 0 {
		return PointResult{}, false, nil
	}

	res = PointResult{
		Y:                float64(cout.y),
		StandardError:    float64(cout.standard_error),
		Residual:         float64(cout.residual),
		RobustnessWeight: float64(cout.robustness_weight),
		IterationsUsed:   int(cout.iterations_used),
	}
	return res, true, nil
}

// Close releases the native resources held by this model. Safe to call
// multiple times.
func (o *OnlineLoess) Close() error {
	if o != nil && o.ptr != nil {
		C.go_online_free(o.ptr)
		o.ptr = nil
		runtime.SetFinalizer(o, nil)
	}
	return nil
}
