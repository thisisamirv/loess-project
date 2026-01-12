"""Type stubs for fastloess."""

import numpy as np
from typing import Optional

class Diagnostics:
    """Diagnostic statistics for assessing fit quality."""

    rmse: float
    mae: float
    r_squared: float
    residual_sd: float
    aic: Optional[float]
    aicc: Optional[float]
    effective_df: Optional[float]

class LoessResult:
    """Result object containing smoothed values and optional outputs."""

    x: np.ndarray
    y: np.ndarray
    dimensions: int
    distance_metric: str
    polynomial_degree: str
    fraction_used: float
    iterations_used: Optional[int]
    standard_errors: Optional[np.ndarray]
    confidence_lower: Optional[np.ndarray]
    confidence_upper: Optional[np.ndarray]
    prediction_lower: Optional[np.ndarray]
    prediction_upper: Optional[np.ndarray]
    residuals: Optional[np.ndarray]
    robustness_weights: Optional[np.ndarray]
    diagnostics: Optional[Diagnostics]
    cv_scores: Optional[np.ndarray]

def smooth(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.67,
    iterations: int = 3,
    weight_function: str = "tricube",
    robustness_method: str = "bisquare",
    scaling_method: str = "mad",
    boundary_policy: str = "extend",
    polynomial_degree: str = "linear",
    surface_mode: str = "interpolation",
    boundary_degree_fallback: bool = True,
    dimensions: int = 1,
    distance_metric: str = "normalized",
    cell: float = 0.2,
    interpolation_vertices: Optional[int] = None,
    return_se: bool = False,
    confidence_intervals: Optional[float] = None,
    prediction_intervals: Optional[float] = None,
    return_diagnostics: bool = False,
    return_residuals: bool = False,
    return_robustness_weights: bool = False,
    zero_weight_fallback: str = "use_local_mean",
    auto_converge: Optional[float] = None,
    cv_fractions: Optional[list[float]] = None,
    cv_method: str = "kfold",
    cv_k: int = 5,
    parallel: bool = True,
) -> LoessResult: ...
def smooth_streaming(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.3,
    chunk_size: int = 5000,
    overlap: int = 500,
    iterations: int = 3,
    weight_function: str = "tricube",
    robustness_method: str = "bisquare",
    scaling_method: str = "mad",
    boundary_policy: str = "extend",
    polynomial_degree: str = "linear",
    surface_mode: str = "interpolation",
    boundary_degree_fallback: bool = True,
    dimensions: int = 1,
    distance_metric: str = "normalized",
    cell: float = 0.2,
    interpolation_vertices: Optional[int] = None,
    merge_strategy: str = "average",
    auto_converge: Optional[float] = None,
    return_diagnostics: bool = False,
    return_residuals: bool = False,
    return_robustness_weights: bool = False,
    zero_weight_fallback: str = "use_local_mean",
    parallel: bool = True,
) -> LoessResult: ...
def smooth_online(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.2,
    window_capacity: int = 1000,
    min_points: int = 3,
    iterations: int = 3,
    weight_function: str = "tricube",
    robustness_method: str = "bisquare",
    scaling_method: str = "mad",
    boundary_policy: str = "extend",
    polynomial_degree: str = "linear",
    surface_mode: str = "interpolation",
    boundary_degree_fallback: bool = True,
    dimensions: int = 1,
    distance_metric: str = "normalized",
    cell: float = 0.2,
    interpolation_vertices: Optional[int] = None,
    update_mode: str = "incremental",
    auto_converge: Optional[float] = None,
    return_robustness_weights: bool = False,
    zero_weight_fallback: str = "use_local_mean",
    parallel: bool = False,
) -> LoessResult: ...

__version__: str
