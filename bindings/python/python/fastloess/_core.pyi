"""Type stubs for fastloess._core native extension module."""

# pylint: disable=unnecessary-ellipsis,unused-argument

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

class Diagnostics:
    """Diagnostic statistics for LOESS fit quality."""

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        ...

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        ...

    @property
    def r_squared(self) -> float:
        """R-squared (coefficient of determination)."""
        ...

    @property
    def aic(self) -> float | None:
        """Akaike Information Criterion."""
        ...

    @property
    def aicc(self) -> float | None:
        """Corrected AIC."""
        ...

    @property
    def effective_df(self) -> float | None:
        """Effective degrees of freedom."""
        ...

    @property
    def residual_sd(self) -> float:
        """Residual standard deviation."""
        ...

class OnlineOutput:
    """Result from a single add_point() call."""

    @property
    def smoothed(self) -> float:
        """Smoothed value for the latest point."""
        ...

    @property
    def std_error(self) -> float | None:
        """Standard error (None if not computed)."""
        ...

    @property
    def residual(self) -> float | None:
        """Residual y − smoothed (None if not computed)."""
        ...

    @property
    def robustness_weight(self) -> float | None:
        """Robustness weight (None if not computed)."""
        ...

    @property
    def iterations_used(self) -> int | None:
        """Number of robustness iterations performed."""
        ...

class LoessResult:
    """Result from LOESS smoothing."""

    @property
    def x(self) -> NDArray[np.float64]:
        """Sorted x values."""
        ...

    @property
    def y(self) -> NDArray[np.float64]:
        """Smoothed y values."""
        ...

    @property
    def standard_errors(self) -> NDArray[np.float64] | None:
        """Standard errors (if computed)."""
        ...

    @property
    def confidence_lower(self) -> NDArray[np.float64] | None:
        """Lower confidence interval bounds."""
        ...

    @property
    def confidence_upper(self) -> NDArray[np.float64] | None:
        """Upper confidence interval bounds."""
        ...

    @property
    def prediction_lower(self) -> NDArray[np.float64] | None:
        """Lower prediction interval bounds."""
        ...

    @property
    def prediction_upper(self) -> NDArray[np.float64] | None:
        """Upper prediction interval bounds."""
        ...

    @property
    def residuals(self) -> NDArray[np.float64] | None:
        """Residuals (original y - smoothed y)."""
        ...

    @property
    def robustness_weights(self) -> NDArray[np.float64] | None:
        """Robustness weights from final iteration."""
        ...

    @property
    def diagnostics(self) -> Diagnostics | None:
        """Diagnostic metrics."""
        ...

    @property
    def iterations_used(self) -> int | None:
        """Number of iterations performed."""
        ...

    @property
    def fraction_used(self) -> float:
        """Fraction used for smoothing."""
        ...

    @property
    def cv_scores(self) -> NDArray[np.float64] | None:
        """CV scores for tested fractions."""
        ...

    @property
    def enp(self) -> float | None:
        """Equivalent number of parameters (hat-matrix stat, if return_se was set)."""
        ...

    @property
    def trace_hat(self) -> float | None:
        """Trace of hat matrix (if return_se was set)."""
        ...

    @property
    def delta1(self) -> float | None:
        """First delta statistic (if return_se was set)."""
        ...

    @property
    def delta2(self) -> float | None:
        """Second delta statistic (if return_se was set)."""
        ...

    @property
    def residual_scale(self) -> float | None:
        """Residual scale estimate (if return_se was set)."""
        ...

    @property
    def leverage(self) -> NDArray[np.float64] | None:
        """Per-point leverage / hat-matrix diagonal (if return_se was set)."""
        ...

    @property
    def dimensions(self) -> int:
        """Number of predictor dimensions."""
        ...

class Loess:
    """Batch LOESS processor with configurable parameters."""

    def __init__(
        self,
        fraction: float = 0.67,
        iterations: int = 3,
        weight_function: str = "tricube",
        robustness_method: str = "bisquare",
        scaling_method: str = "mad",
        boundary_policy: str = "extend",
        confidence_intervals: float | None = None,
        prediction_intervals: float | None = None,
        return_diagnostics: bool = False,
        return_residuals: bool = False,
        return_robustness_weights: bool = False,
        zero_weight_fallback: str = "use_local_mean",
        auto_converge: float | None = None,
        cv_fractions: Sequence[float] | None = None,
        cv_method: str = "kfold",
        cv_k: int = 5,
        parallel: bool = True,
        degree: str = "linear",
        dimensions: int = 1,
        distance_metric: str = "normalized",
        surface_mode: str = "interpolation",
        return_se: bool = False,
        weighted_metric_weights: Sequence[float] | None = None,
        cell: float | None = None,
        interpolation_vertices: int | None = None,
        boundary_degree_fallback: bool | None = None,
        cv_seed: int | None = None,
    ) -> None:
        """Initialize the batch LOESS processor.""""
        ...

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        custom_weights: ArrayLike | None = None,
    ) -> LoessResult:
        """Fit LOESS model to data.

        Parameters
        ----------
        x : array_like
            Predictor values.
        y : array_like
            Response values.
        custom_weights : array_like, optional
            Per-observation weights (same length as y). Each weight multiplies the
            local kernel weight: w_ij = custom_weights[j] * K(d_ij/h) * rob_j.
            Analogous to the ``weights`` argument in R's ``stats::loess``.
        """
        ...

class StreamingLoess:
    """Streaming LOESS processor for incremental chunk-based smoothing."""

    def __init__(
        self,
        fraction: float = 0.67,
        chunk_size: int = 5000,
        overlap: int | None = None,
        iterations: int = 3,
        weight_function: str = "tricube",
        robustness_method: str = "bisquare",
        scaling_method: str = "mad",
        boundary_policy: str = "extend",
        auto_converge: float | None = None,
        return_diagnostics: bool = False,
        return_residuals: bool = False,
        return_robustness_weights: bool = False,
        zero_weight_fallback: str = "use_local_mean",
        parallel: bool = True,
        degree: str = "linear",
        dimensions: int = 1,
        distance_metric: str = "normalized",
        surface_mode: str = "interpolation",
        return_se: bool = False,
        merge_strategy: str = "weighted_average",
        weighted_metric_weights: Sequence[float] | None = None,
        cell: float | None = None,
        interpolation_vertices: int | None = None,
        boundary_degree_fallback: bool | None = None,
        confidence_intervals: float | None = None,
        prediction_intervals: float | None = None,
    ) -> None:
        """Initialize the streaming processor.""""
        ...

    def process_chunk(self, x: ArrayLike, y: ArrayLike) -> LoessResult:
        """Process a chunk of data and return smoothed values."""
        ...

    def finalize(self) -> LoessResult:
        """Finalize smoothing and return remaining buffered data."""
        ...

class OnlineLoess:
    """Online LOESS processor for real-time data streams."""

    def __init__(
        self,
        fraction: float = 0.67,
        window_capacity: int = 1000,
        min_points: int = 3,
        iterations: int = 3,
        weight_function: str = "tricube",
        robustness_method: str = "bisquare",
        scaling_method: str = "mad",
        boundary_policy: str = "extend",
        update_mode: str = "full",
        auto_converge: float | None = None,
        return_robustness_weights: bool = False,
        zero_weight_fallback: str = "use_local_mean",
        parallel: bool = False,
        degree: str = "linear",
        dimensions: int = 1,
        distance_metric: str = "normalized",
        surface_mode: str = "interpolation",
        return_se: bool = False,
        weighted_metric_weights: Sequence[float] | None = None,
        return_diagnostics: bool = False,
        return_residuals: bool = False,
        cell: float | None = None,
        interpolation_vertices: int | None = None,
        boundary_degree_fallback: bool | None = None,
        confidence_intervals: float | None = None,
        prediction_intervals: float | None = None,
    ) -> None:
        """Initialize the online processor."""
        ...

    def add_point(self, x: float, y: float) -> OnlineOutput | None:
        """Add a single point and return its smoothed value, or None if the window is still filling."""
        ...
