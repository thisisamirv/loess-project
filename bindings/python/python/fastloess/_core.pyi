"""Type stubs for fastloess._core native extension module."""

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

def smooth(
    x: ArrayLike,
    y: ArrayLike,
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
) -> LoessResult:
    """LOESS smoothing with the batch adapter.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    fraction : float, optional
        Smoothing fraction (default: 0.67).
    iterations : int, optional
        Number of robustness iterations (default: 3).
    weight_function : str, optional
        Kernel function: "tricube", "epanechnikov", "gaussian", "uniform",
        "biweight", "triangle", "cosine".
    robustness_method : str, optional
        Robustness method: "bisquare", "huber", "talwar".
    scaling_method : str, optional
        Scaling method: "mad", "mar", "mean" (default: "mad").
    boundary_policy : str, optional
        Boundary policy: "extend", "reflect", "zero", "noboundary".
    confidence_intervals : float, optional
        Confidence level for confidence intervals (e.g., 0.95).
    prediction_intervals : float, optional
        Confidence level for prediction intervals (e.g., 0.95).
    return_diagnostics : bool, optional
        Whether to compute diagnostics (default: False).
    return_residuals : bool, optional
        Whether to include residuals (default: False).
    return_robustness_weights : bool, optional
        Whether to include robustness weights (default: False).
    zero_weight_fallback : str, optional
        Fallback when all weights are zero.
    auto_converge : float, optional
        Tolerance for auto-convergence.
    cv_fractions : sequence of float, optional
        Fractions to test for cross-validation.
    cv_method : str, optional
        CV method: "loocv" or "kfold" (default: "kfold").
    cv_k : int, optional
        Number of folds for k-fold CV (default: 5).
    parallel : bool, optional
        Enable parallel execution (default: True).
    degree : str, optional
        Polynomial degree: "constant", "linear", "quadratic", etc. (default: "linear").
    dimensions : int, optional
        Number of predictor dimensions (default: 1).
    distance_metric : str, optional
        Distance metric: "normalized", "euclidean", etc. (default: "normalized").
    surface_mode : str, optional
        Surface mode: "interpolation" or "direct" (default: "interpolation").
    return_se : bool, optional
        Compute hat-matrix statistics (enp, trace_hat, etc.) (default: False).

    Returns
    -------
    LoessResult
        Result object with smoothed values and optional outputs.
    """
    ...

def smooth_streaming(
    x: ArrayLike,
    y: ArrayLike,
    fraction: float = 0.3,
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
) -> LoessResult:
    """Streaming LOESS for large datasets.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    fraction : float, optional
        Smoothing fraction (default: 0.3).
    chunk_size : int, optional
        Size of each processing chunk (default: 5000).
    overlap : int, optional
        Overlap between chunks (default: 10% of chunk_size).
    iterations : int, optional
        Number of robustness iterations (default: 3).
    weight_function : str, optional
        Kernel function.
    robustness_method : str, optional
        Robustness method.
    scaling_method : str, optional
        Scaling method.
    boundary_policy : str, optional
        Boundary policy.
    auto_converge : float, optional
        Tolerance for auto-convergence.
    return_diagnostics : bool, optional
        Whether to compute diagnostics.
    return_residuals : bool, optional
        Whether to include residuals.
    return_robustness_weights : bool, optional
        Whether to include robustness weights.
    zero_weight_fallback : str, optional
        Fallback when all weights are zero.
    parallel : bool, optional
        Enable parallel execution (default: True).
    degree : str, optional
        Polynomial degree (default: "linear").
    dimensions : int, optional
        Number of predictor dimensions (default: 1).
    distance_metric : str, optional
        Distance metric (default: "normalized").
    surface_mode : str, optional
        Surface mode (default: "interpolation").
    return_se : bool, optional
        Compute hat-matrix statistics (default: False).

    Returns
    -------
    LoessResult
        Result object with smoothed values.
    """
    ...

def smooth_online(
    x: ArrayLike,
    y: ArrayLike,
    fraction: float = 0.2,
    window_capacity: int = 100,
    min_points: int = 2,
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
) -> LoessResult:
    """Online LOESS with sliding window.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    fraction : float, optional
        Smoothing fraction (default: 0.2).
    window_capacity : int, optional
        Maximum points to retain in window (default: 100).
    min_points : int, optional
        Minimum points before smoothing starts (default: 2).
    iterations : int, optional
        Number of robustness iterations (default: 3).
    weight_function : str, optional
        Kernel function.
    robustness_method : str, optional
        Robustness method.
    scaling_method : str, optional
        Scaling method.
    boundary_policy : str, optional
        Boundary policy.
    update_mode : str, optional
        Update strategy: "full" or "incremental" (default: "full").
    auto_converge : float, optional
        Tolerance for auto-convergence.
    return_robustness_weights : bool, optional
        Whether to include robustness weights.
    zero_weight_fallback : str, optional
        Fallback when all weights are zero.
    parallel : bool, optional
        Enable parallel execution (default: False).
    degree : str, optional
        Polynomial degree (default: "linear").
    dimensions : int, optional
        Number of predictor dimensions (default: 1).
    distance_metric : str, optional
        Distance metric (default: "normalized").
    surface_mode : str, optional
        Surface mode (default: "interpolation").
    return_se : bool, optional
        Compute hat-matrix statistics (default: False).

    Returns
    -------
    LoessResult
        Result object with smoothed values.
    """
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
    ) -> None:
        """Initialize the batch LOESS processor."""
        ...

    def fit(self, x: ArrayLike, y: ArrayLike) -> LoessResult:
        """Fit LOESS model to data."""
        ...

class StreamingLoess:
    """Streaming LOESS processor for incremental chunk-based smoothing."""

    def __init__(
        self,
        fraction: float = 0.3,
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
    ) -> None:
        """Initialize the streaming processor."""
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
        fraction: float = 0.2,
        window_capacity: int = 100,
        min_points: int = 2,
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
    ) -> None:
        """Initialize the online processor."""
        ...

    def update(self, x: float, y: float) -> float | None:
        """Add a single point and return smoothed value if available."""
        ...

    def add_points(self, x: ArrayLike, y: ArrayLike) -> LoessResult:
        """Add multiple points and return smoothed results."""
        ...
