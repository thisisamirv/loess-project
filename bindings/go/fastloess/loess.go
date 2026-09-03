package fastloess

/*
#include "fastloess_go.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// Options configures a Loess, StreamingLoess, or OnlineLoess model.
// Use DefaultOptions and override only the fields you need.
type Options struct {
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
	// ReturnSE requests hat-matrix statistics (effective degrees of
	// freedom, leverage, standard errors). Batch model only.
	ReturnSE bool

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

	// ConfidenceIntervals is the confidence level for confidence intervals,
	// in (0, 1) (e.g. 0.95). Nil disables confidence intervals.
	ConfidenceIntervals *float64
	// PredictionIntervals is the confidence level for prediction intervals,
	// in (0, 1) (e.g. 0.95). Nil disables prediction intervals.
	PredictionIntervals *float64
	// AutoConverge is the convergence tolerance for early stopping of
	// robustness iterations. Nil disables early stopping.
	AutoConverge *float64

	// ReturnDiagnostics requests fit-quality metrics (RMSE, MAE, R-squared, AIC, etc.).
	ReturnDiagnostics bool
	// ReturnResiduals requests residuals in the result.
	ReturnResiduals bool
	// ReturnRobustnessWeights requests per-point robustness weights in the result.
	ReturnRobustnessWeights bool

	// CVFractions is a set of candidate fractions for cross-validation.
	// Empty disables CV. Batch model only.
	CVFractions []float64
	// CVMethod is the cross-validation method: "kfold" (default) or "loocv".
	CVMethod string
	// CVK is the number of folds for k-fold CV. Default: 5.
	CVK int
	// CVSeed is the RNG seed for reproducible k-fold splits. Nil uses a
	// random seed.
	CVSeed *uint64

	// Parallel enables parallel processing. Default: true.
	Parallel bool
}

// DefaultOptions returns the library's recommended defaults. Start from this
// and override only the fields you need.
func DefaultOptions() Options {
	return Options{
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
		CVMethod:           "kfold",
		CVK:                5,
		Parallel:           true,
	}
}

func optPtr(p *float64) (float64, bool) {
	if p == nil {
		return 0, false
	}
	return *p, true
}

// Loess is a stateful batch LOESS smoothing model. It processes an entire
// dataset at once and supports every feature (multivariate predictors,
// confidence/prediction intervals, cross-validation).
//
// Loess is not safe for concurrent use; each goroutine should use its own
// instance, or callers must serialize access.
type Loess struct {
	ptr *C.fastloess_GoLoess
}

// NewLoess creates a new batch Loess model with the given options.
func NewLoess(opts Options) (*Loess, error) {
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
	cvMethod := cStringOrNil(opts.CVMethod)
	defer freeCString(cvMethod)
	degree := cStringOrNil(opts.Degree)
	defer freeCString(degree)
	distanceMetric := cStringOrNil(opts.DistanceMetric)
	defer freeCString(distanceMetric)
	surfaceMode := cStringOrNil(opts.SurfaceMode)
	defer freeCString(surfaceMode)

	ci, ciSet := optPtr(opts.ConfidenceIntervals)
	pi, piSet := optPtr(opts.PredictionIntervals)
	autoConverge, autoConvergeSet := optPtr(opts.AutoConverge)
	cvFracPtr, cvFracLen := cDoubles(opts.CVFractions)
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

	var ptr *C.fastloess_GoLoess
	var errMsg string
	withLockedThread(func() {
		ptr = C.go_loess_new(
			C.double(opts.Fraction),
			C.int(opts.Iterations),
			wf, rm, sm, bp,
			optFloat(ci, ciSet),
			optFloat(pi, piSet),
			boolToCInt(opts.ReturnDiagnostics),
			boolToCInt(opts.ReturnResiduals),
			boolToCInt(opts.ReturnRobustnessWeights),
			zwf,
			optFloat(autoConverge, autoConvergeSet),
			cvFracPtr, cvFracLen,
			cvMethod,
			C.int(opts.CVK),
			boolToCInt(opts.Parallel),
			degree,
			C.int(opts.Dimensions),
			distanceMetric,
			surfaceMode,
			boolToCInt(opts.ReturnSE),
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

	if opts.CVSeed != nil {
		C.go_loess_set_cv_seed(ptr, C.ulong(*opts.CVSeed))
	}

	l := &Loess{ptr: ptr}
	runtime.SetFinalizer(l, finalizeLoess)
	return l, nil
}

func finalizeLoess(l *Loess) {
	_ = l.Close()
}

// Fit smooths y as a function of x. For multivariate input (Dimensions > 1),
// x is flattened row-major (length len(y)*Dimensions). An optional
// customWeights slice (same length as y) applies per-observation case weights.
func (l *Loess) Fit(x, y []float64, customWeights ...[]float64) (Result, error) {
	if l == nil || l.ptr == nil {
		return Result{}, errors.New("fastloess: Fit called on a closed Loess model")
	}
	if len(x) == 0 || len(y) == 0 {
		return Result{}, errors.New("fastloess: x and y must be non-empty")
	}
	var cw []float64
	if len(customWeights) > 0 {
		cw = customWeights[0]
	}

	xPtr, xLen := cDoubles(x)
	yPtr, yLen := cDoubles(y)
	cwPtr, cwLen := cDoubles(cw)

	cres := C.go_loess_fit(l.ptr, xPtr, xLen, yPtr, yLen, cwPtr, cwLen)
	return resultFromC(cres)
}

// Close releases the native resources held by this model. It is safe to
// call Close multiple times, and Close is called automatically by the
// garbage collector if not called explicitly, but relying on that delays
// releasing native memory - call Close explicitly (e.g. via defer) instead.
func (l *Loess) Close() error {
	if l != nil && l.ptr != nil {
		C.go_loess_free(l.ptr)
		l.ptr = nil
		runtime.SetFinalizer(l, nil)
	}
	return nil
}
