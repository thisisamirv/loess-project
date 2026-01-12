//! Python bindings for fastLoess.
//!
//! Provides Python access to the fastLoess Rust library via PyO3.

#![allow(non_snake_case)]
#![deny(missing_docs)]

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt::Display;

use ::fastLoess::internals::api::{
    BoundaryPolicy, DistanceMetric, MergeStrategy, PolynomialDegree, RobustnessMethod,
    ScalingMethod, SurfaceMode, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use ::fastLoess::prelude::{
    Batch, KFold, LOOCV, Loess as LoessBuilder, LoessResult, MAD, MAR, Online, Streaming,
};
// ============================================================================
// Helper Functions
// ============================================================================

/// Convert a LoessError to a PyErr
fn to_py_error(e: impl Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Parse weight function from string
fn parse_weight_function(name: &str) -> PyResult<WeightFunction> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(PyValueError::new_err(format!(
            "Unknown weight function: {}. Valid options: tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            name
        ))),
    }
}

/// Parse robustness method from string
fn parse_robustness_method(name: &str) -> PyResult<RobustnessMethod> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(PyValueError::new_err(format!(
            "Unknown robustness method: {}. Valid options: bisquare, huber, talwar",
            name
        ))),
    }
}

/// Parse zero weight fallback from string
fn parse_zero_weight_fallback(name: &str) -> PyResult<ZeroWeightFallback> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(PyValueError::new_err(format!(
            "Unknown zero weight fallback: {}. Valid options: use_local_mean, return_original, return_none",
            name
        ))),
    }
}

/// Parse boundary policy from string
fn parse_boundary_policy(name: &str) -> PyResult<BoundaryPolicy> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(PyValueError::new_err(format!(
            "Unknown boundary policy: {}. Valid options: extend, reflect, zero, noboundary",
            name
        ))),
    }
}

/// Parse scaling method from string
fn parse_scaling_method(name: &str) -> PyResult<ScalingMethod> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        _ => Err(PyValueError::new_err(format!(
            "Unknown scaling method: {}. Valid options: mad, mar",
            name
        ))),
    }
}

/// Parse update mode from string
fn parse_update_mode(name: &str) -> PyResult<UpdateMode> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(PyValueError::new_err(format!(
            "Unknown update mode: {}. Valid options: full, incremental",
            name
        ))),
    }
}

/// Parse polynomial degree from string
fn parse_polynomial_degree(name: &str) -> PyResult<PolynomialDegree> {
    match name.to_lowercase().as_str() {
        "constant" | "0" => Ok(PolynomialDegree::Constant),
        "linear" | "1" => Ok(PolynomialDegree::Linear),
        "quadratic" | "2" => Ok(PolynomialDegree::Quadratic),
        "cubic" | "3" => Ok(PolynomialDegree::Cubic),
        "quartic" | "4" => Ok(PolynomialDegree::Quartic),
        _ => Err(PyValueError::new_err(format!(
            "Unknown polynomial degree: {}. Valid options: constant, linear, quadratic, cubic, quartic",
            name
        ))),
    }
}

/// Parse surface mode from string
fn parse_surface_mode(name: &str) -> PyResult<SurfaceMode> {
    match name.to_lowercase().as_str() {
        "interpolation" | "interp" => Ok(SurfaceMode::Interpolation),
        "direct" => Ok(SurfaceMode::Direct),
        _ => Err(PyValueError::new_err(format!(
            "Unknown surface mode: {}. Valid options: interpolation, direct",
            name
        ))),
    }
}

/// Parse distance metric from string
fn parse_distance_metric(name: &str) -> PyResult<DistanceMetric<f64>> {
    match name.to_lowercase().as_str() {
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "normalized" => Ok(DistanceMetric::Normalized),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        "chebyshev" | "linf" => Ok(DistanceMetric::Chebyshev),
        _ => Err(PyValueError::new_err(format!(
            "Unknown distance metric: {}. Valid options: euclidean, normalized, manhattan, chebyshev",
            name
        ))),
    }
}

/// Parse merge strategy from string (for streaming)
fn parse_merge_strategy(name: &str) -> PyResult<MergeStrategy> {
    match name.to_lowercase().as_str() {
        "average" => Ok(MergeStrategy::Average),
        "weighted_average" | "weighted" => Ok(MergeStrategy::WeightedAverage),
        "take_first" | "first" => Ok(MergeStrategy::TakeFirst),
        "take_last" | "last" => Ok(MergeStrategy::TakeLast),
        _ => Err(PyValueError::new_err(format!(
            "Unknown merge strategy: {}. Valid options: average, weighted_average, take_first, take_last",
            name
        ))),
    }
}

// ============================================================================
// Python Classes
// ============================================================================

/// Diagnostic statistics for LOESS fit quality.
#[pyclass(name = "Diagnostics")]
#[derive(Clone)]
pub struct PyDiagnostics {
    /// Root Mean Squared Error
    #[pyo3(get)]
    pub rmse: f64,

    /// Mean Absolute Error
    #[pyo3(get)]
    pub mae: f64,

    /// R-squared (coefficient of determination)
    #[pyo3(get)]
    pub r_squared: f64,

    /// Akaike Information Criterion
    #[pyo3(get)]
    pub aic: Option<f64>,

    /// Corrected AIC
    #[pyo3(get)]
    pub aicc: Option<f64>,

    /// Effective degrees of freedom
    #[pyo3(get)]
    pub effective_df: Option<f64>,

    /// Residual standard deviation
    #[pyo3(get)]
    pub residual_sd: f64,
}

#[pymethods]
impl PyDiagnostics {
    fn __repr__(&self) -> String {
        format!(
            "Diagnostics(rmse={:.6}, mae={:.6}, r_squared={:.6})",
            self.rmse, self.mae, self.r_squared
        )
    }
}

/// Result from LOESS smoothing.
#[pyclass(name = "LoessResult")]
pub struct PyLoessResult {
    inner: LoessResult<f64>,
}

#[pymethods]
impl PyLoessResult {
    /// Sorted x values
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.x.clone())
    }

    /// Smoothed y values
    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.y.clone())
    }

    /// Standard errors (if computed)
    #[getter]
    fn standard_errors<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Lower confidence interval bounds
    #[getter]
    fn confidence_lower<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Upper confidence interval bounds
    #[getter]
    fn confidence_upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Lower prediction interval bounds
    #[getter]
    fn prediction_lower<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Upper prediction interval bounds
    #[getter]
    fn prediction_upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Residuals (original y - smoothed y)
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .residuals
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Robustness weights from final iteration
    #[getter]
    fn robustness_weights<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Diagnostic metrics
    #[getter]
    fn diagnostics(&self) -> Option<PyDiagnostics> {
        self.inner.diagnostics.as_ref().map(|d| PyDiagnostics {
            rmse: d.rmse,
            mae: d.mae,
            r_squared: d.r_squared,
            aic: d.aic,
            aicc: d.aicc,
            effective_df: d.effective_df,
            residual_sd: d.residual_sd,
        })
    }

    /// Number of iterations performed
    #[getter]
    fn iterations_used(&self) -> Option<usize> {
        self.inner.iterations_used
    }

    /// Fraction used for smoothing
    #[getter]
    fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    /// CV scores for tested fractions
    #[getter]
    fn cv_scores<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    fn __repr__(&self) -> String {
        format!(
            "LoessResult(n={}, fraction_used={:.4})",
            self.inner.y.len(),
            self.inner.fraction_used
        )
    }
}

// ============================================================================
// Python Functions
// ============================================================================

/// LOESS smoothing with the batch adapter.
///
/// This is the primary interface for LOESS smoothing. Processes the entire
/// dataset in memory with optional parallel execution.
///
/// Parameters
/// ----------
/// x : array_like
///     Independent variable values.
/// y : array_like
///     Dependent variable values.
/// fraction : float, optional
///     Smoothing fraction (default: 0.67).
/// iterations : int, optional
///     Number of robustness iterations (default: 3).
/// weight_function : str, optional
///     Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// robustness_method : str, optional
///     Robustness method: "bisquare", "huber", "talwar".
/// scaling_method : str, optional
///     Scaling method for robustness: "mad", "mar" (default: "mad").
/// boundary_policy : str, optional
///     Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// polynomial_degree : str, optional
///     Polynomial degree: "constant", "linear", "quadratic", "cubic", "quartic" (default: "linear").
/// surface_mode : str, optional
///     Surface evaluation mode: "interpolation" (faster), "direct" (more accurate) (default: "interpolation").
/// boundary_degree_fallback : bool, optional
///     Reduce polynomial degree at boundaries (default: True).
/// dimensions : int, optional
///     Number of predictor dimensions for multivariate LOESS (default: 1).
/// distance_metric : str, optional
///     Distance metric for nD data: "euclidean", "normalized", "manhattan", "chebyshev" (default: "normalized").
/// cell : float, optional
///     Cell size for interpolation subdivision (default: 0.2).
/// interpolation_vertices : int, optional
///     Maximum vertices for interpolation surface (default: None = auto).
/// return_se : bool, optional
///     Compute and return standard errors (default: False).
/// confidence_intervals : float, optional
///     Confidence level for intervals (e.g., 0.95).
/// prediction_intervals : float, optional
///     Prediction level for intervals (e.g., 0.95).
/// return_diagnostics : bool, optional
///     Whether to include diagnostic statistics in output.
/// return_residuals : bool, optional
///     Whether to include residuals in output.
/// return_robustness_weights : bool, optional
///     Whether to include robustness weights in output.
/// zero_weight_fallback : str, optional
///     Fallback when all weights are zero: "use_local_mean", "return_original", "return_none".
/// auto_converge : float, optional
///     Tolerance for auto-convergence (disabled by default).
/// cv_fractions : list of float, optional
///     Fractions to test for cross-validation (disabled by default).
///     When provided, enables cross-validation to select optimal fraction.
/// cv_method : str, optional
///     CV method: "loocv" (leave-one-out) or "kfold". Default: "kfold".
/// cv_k : int, optional
///     Number of folds for k-fold CV (default: 5).
/// parallel : bool, optional
///     Enable parallel execution (default: True).
///
/// Returns
/// -------
/// LoessResult
///     Result object with smoothed values and optional outputs.
#[pyfunction]
#[pyo3(signature = (
    x, y,
    fraction=0.67,
    iterations=3,
    weight_function="tricube",
    robustness_method="bisquare",
    scaling_method="mad",
    boundary_policy="extend",
    polynomial_degree="linear",
    surface_mode="interpolation",
    boundary_degree_fallback=true,
    dimensions=1,
    distance_metric="normalized",
    cell=0.2,
    interpolation_vertices=None,
    return_se=false,
    confidence_intervals=None,
    prediction_intervals=None,
    return_diagnostics=false,
    return_residuals=false,
    return_robustness_weights=false,
    zero_weight_fallback="use_local_mean",
    auto_converge=None,
    cv_fractions=None,
    cv_method="kfold",
    cv_k=5,
    parallel=true
))]
#[allow(clippy::too_many_arguments)]
fn smooth<'py>(
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    fraction: f64,
    iterations: usize,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    polynomial_degree: &str,
    surface_mode: &str,
    boundary_degree_fallback: bool,
    dimensions: usize,
    distance_metric: &str,
    cell: f64,
    interpolation_vertices: Option<usize>,
    return_se: bool,
    confidence_intervals: Option<f64>,
    prediction_intervals: Option<f64>,
    return_diagnostics: bool,
    return_residuals: bool,
    return_robustness_weights: bool,
    zero_weight_fallback: &str,
    auto_converge: Option<f64>,
    cv_fractions: Option<Vec<f64>>,
    cv_method: &str,
    cv_k: usize,
    parallel: bool,
) -> PyResult<PyLoessResult> {
    let x_slice = x.as_slice().map_err(to_py_error)?;
    let y_slice = y.as_slice().map_err(to_py_error)?;

    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
    let bp = parse_boundary_policy(boundary_policy)?;
    let pd = parse_polynomial_degree(polynomial_degree)?;
    let surf_mode = parse_surface_mode(surface_mode)?;
    let dm = parse_distance_metric(distance_metric)?;

    let mut builder = LoessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.degree(pd); // Use .degree() method from loess-rs
    builder = builder.surface_mode(surf_mode);
    builder = builder.boundary_degree_fallback(boundary_degree_fallback);
    builder = builder.dimensions(dimensions);
    builder = builder.distance_metric(dm);
    builder = builder.cell(cell);
    if let Some(iv) = interpolation_vertices {
        builder = builder.interpolation_vertices(iv);
    }
    if return_se {
        builder = builder.return_se();
    }
    builder = builder.parallel(parallel);

    if let Some(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }

    // Cross-validation if fractions are provided
    if let Some(fractions) = cv_fractions {
        let cv_config = match cv_method.to_lowercase().as_str() {
            "simple" | "loo" | "loocv" | "leave_one_out" => LOOCV(&fractions),
            "kfold" | "k_fold" | "k-fold" => KFold(cv_k, &fractions),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown CV method: {}. Valid options: loocv, kfold",
                    cv_method
                )));
            }
        };

        builder = builder.cross_validate(cv_config);
    }

    // Convert to adapter and set adapter-specific parameters
    let mut builder = builder.adapter(Batch);

    if let Some(cl) = confidence_intervals {
        builder = builder.confidence_intervals(cl);
    }

    if let Some(pl) = prediction_intervals {
        builder = builder.prediction_intervals(pl);
    }

    if return_diagnostics {
        builder = builder.return_diagnostics(true);
    }

    if return_residuals {
        builder = builder.compute_residuals(true);
    }

    if return_robustness_weights {
        builder = builder.return_robustness_weights(true);
    }

    let result = builder
        .build()
        .map_err(to_py_error)?
        .fit(x_slice, y_slice)
        .map_err(to_py_error)?;

    Ok(PyLoessResult { inner: result })
}

/// Streaming LOESS for large datasets.
///
/// Processes data in chunks to maintain constant memory usage.
/// Calls process_chunk() followed by finalize() internally.
///
/// Parameters
/// ----------
/// x : array_like
///     Independent variable values.
/// y : array_like
///     Dependent variable values.
/// fraction : float, optional
///     Smoothing fraction (default: 0.3).
/// chunk_size : int, optional
///     Size of each processing chunk (default: 5000).
/// overlap : int, optional
///     Overlap between chunks (default: 10% of chunk_size).
/// iterations : int, optional
///     Number of robustness iterations (default: 3).
/// weight_function : str, optional
///     Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// robustness_method : str, optional
///     Robustness method: "bisquare", "huber", "talwar".
/// scaling_method : str, optional
///     Scaling method for robustness: "mad", "mar" (default: "mad").
/// boundary_policy : str, optional
///     Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// polynomial_degree : str, optional
///     Polynomial degree: "constant", "linear", "quadratic", "cubic", "quartic" (default: "linear").
/// surface_mode : str, optional
///     Surface evaluation mode: "interpolation", "direct" (default: "interpolation").
/// boundary_degree_fallback : bool, optional
///     Reduce polynomial degree at boundaries (default: True).
/// dimensions : int, optional
///     Number of predictor dimensions (default: 1).
/// distance_metric : str, optional
///     Distance metric: "euclidean", "normalized", "manhattan", "chebyshev" (default: "normalized").
/// cell : float, optional
///     Cell size for interpolation (default: 0.2).
/// interpolation_vertices : int, optional
///     Maximum interpolation vertices (default: None).
/// merge_strategy : str, optional
///     Strategy for merging overlapping chunks: "average", "weighted_average", "take_first", "take_last" (default: "weighted_average").
/// auto_converge : float, optional
///     Tolerance for auto-convergence (disabled by default).
/// return_diagnostics : bool, optional
///     Whether to compute cumulative diagnostics across chunks.
/// return_residuals : bool, optional
///     Whether to include residuals.
/// return_robustness_weights : bool, optional
///     Whether to include robustness weights.
/// zero_weight_fallback : str, optional
///     Fallback when all weights are zero.
/// parallel : bool, optional
///     Enable parallel execution (default: True).
///
/// Returns
/// -------
/// LoessResult
///     Result object with smoothed values.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x, y,
    fraction=0.3,
    chunk_size=5000,
    overlap=500,
    iterations=3,
    weight_function="tricube",
    robustness_method="bisquare",
    scaling_method="mad",
    boundary_policy="extend",
    polynomial_degree="linear",
    surface_mode="interpolation",
    boundary_degree_fallback=true,
    dimensions=1,
    distance_metric="normalized",
    cell=0.2,
    interpolation_vertices=None,
    merge_strategy="average",
    auto_converge=None,
    return_diagnostics=false,
    return_residuals=false,
    return_robustness_weights=false,
    zero_weight_fallback="use_local_mean",
    parallel=true
))]
fn smooth_streaming<'py>(
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    fraction: f64,
    chunk_size: usize,
    overlap: usize,
    iterations: usize,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    polynomial_degree: &str,
    surface_mode: &str,
    boundary_degree_fallback: bool,
    dimensions: usize,
    distance_metric: &str,
    cell: f64,
    interpolation_vertices: Option<usize>,
    merge_strategy: &str,
    auto_converge: Option<f64>,
    return_diagnostics: bool,
    return_residuals: bool,
    return_robustness_weights: bool,
    zero_weight_fallback: &str,
    parallel: bool,
) -> PyResult<PyLoessResult> {
    let x_slice = x.as_slice().map_err(to_py_error)?;
    let y_slice = y.as_slice().map_err(to_py_error)?;

    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
    let bp = parse_boundary_policy(boundary_policy)?;
    let pd = parse_polynomial_degree(polynomial_degree)?;
    let surf_mode = parse_surface_mode(surface_mode)?;
    let dm = parse_distance_metric(distance_metric)?;
    let ms = parse_merge_strategy(merge_strategy)?;

    let mut builder = LoessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.degree(pd);
    builder = builder.surface_mode(surf_mode);
    builder = builder.boundary_degree_fallback(boundary_degree_fallback);
    builder = builder.dimensions(dimensions);
    builder = builder.distance_metric(dm);
    builder = builder.cell(cell);
    if let Some(iv) = interpolation_vertices {
        builder = builder.interpolation_vertices(iv);
    }

    if let Some(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }

    let mut builder = builder.adapter(Streaming);
    builder = builder.chunk_size(chunk_size);
    builder = builder.overlap(overlap);
    builder = builder.merge_strategy(ms);
    builder = builder.parallel(parallel);

    if return_diagnostics {
        builder = builder.return_diagnostics(true);
    }
    if return_residuals {
        builder = builder.compute_residuals(true);
    }
    if return_robustness_weights {
        builder = builder.return_robustness_weights(true);
    }

    let mut processor = builder.build().map_err(to_py_error)?;

    // Process the data as a single chunk
    let chunk_result = processor
        .process_chunk(x_slice, y_slice)
        .map_err(to_py_error)?;

    // Finalize to get remaining buffered overlap data
    let final_result = processor.finalize().map_err(to_py_error)?;

    // Combine results from process_chunk and finalize
    let mut combined_x = chunk_result.x;
    let mut combined_y = chunk_result.y;
    let mut combined_se = chunk_result.standard_errors;
    let mut combined_cl = chunk_result.confidence_lower;
    let mut combined_cu = chunk_result.confidence_upper;
    let mut combined_pl = chunk_result.prediction_lower;
    let mut combined_pu = chunk_result.prediction_upper;
    let mut combined_res = chunk_result.residuals;
    let mut combined_rw = chunk_result.robustness_weights;

    combined_x.extend(final_result.x);
    combined_y.extend(final_result.y);

    if let (Some(mut s), Some(f)) = (combined_se.take(), final_result.standard_errors) {
        s.extend(f);
        combined_se = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_cl.take(), final_result.confidence_lower) {
        s.extend(f);
        combined_cl = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_cu.take(), final_result.confidence_upper) {
        s.extend(f);
        combined_cu = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_pl.take(), final_result.prediction_lower) {
        s.extend(f);
        combined_pl = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_pu.take(), final_result.prediction_upper) {
        s.extend(f);
        combined_pu = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_res.take(), final_result.residuals) {
        s.extend(f);
        combined_res = Some(s);
    }
    if let (Some(mut s), Some(f)) = (combined_rw.take(), final_result.robustness_weights) {
        s.extend(f);
        combined_rw = Some(s);
    }

    // Create combined result
    let result = LoessResult {
        x: combined_x,
        dimensions: 1,
        distance_metric: DistanceMetric::Normalized,
        polynomial_degree: pd,
        y: combined_y,
        standard_errors: combined_se,
        confidence_lower: combined_cl,
        confidence_upper: combined_cu,
        prediction_lower: combined_pl,
        prediction_upper: combined_pu,
        residuals: combined_res,
        robustness_weights: combined_rw,
        diagnostics: final_result.diagnostics, // diagnostics are cumulative in final
        iterations_used: chunk_result.iterations_used,
        fraction_used: chunk_result.fraction_used,
        cv_scores: None,
        enp: None,
        trace_hat: None,
        delta1: None,
        delta2: None,
        residual_scale: None,
        leverage: None,
    };

    Ok(PyLoessResult { inner: result })
}

/// Online LOESS with sliding window.
///
/// Maintains a sliding window for incremental updates.
///
/// Parameters
/// ----------
/// x : array_like
///     Independent variable values.
/// y : array_like
///     Dependent variable values.
/// fraction : float, optional
///     Smoothing fraction (default: 0.2).
/// window_capacity : int, optional
///     Maximum points to retain in window (default: 100).
/// min_points : int, optional
///     Minimum points before smoothing starts (default: 3).
/// iterations : int, optional
///     Number of robustness iterations (default: 3).
/// weight_function : str, optional
///     Kernel function: "tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle".
/// robustness_method : str, optional
///     Robustness method: "bisquare", "huber", "talwar".
/// scaling_method : str, optional
///     Scaling method for robustness: "mad", "mar" (default: "mad").
/// boundary_policy : str, optional
///     Handling of edge effects: "extend", "reflect", "zero", "noboundary" (default: "extend").
/// polynomial_degree : str, optional
///     Polynomial degree: "constant", "linear", "quadratic", "cubic", "quartic" (default: "linear").
/// surface_mode : str, optional
///     Surface evaluation mode: "interpolation", "direct" (default: "interpolation").
/// boundary_degree_fallback : bool, optional
///     Reduce polynomial degree at boundaries (default: True).
/// dimensions : int, optional
///     Number of predictor dimensions (default: 1).
/// distance_metric : str, optional
///     Distance metric: "euclidean", "normalized", "manhattan", "chebyshev" (default: "normalized").
/// cell : float, optional
///     Cell size for interpolation (default: 0.2).
/// interpolation_vertices : int, optional
///     Maximum interpolation vertices (default: None).
/// update_mode : str, optional
///     Update mode: "full" (resmooth all), "incremental" (single point) (default: "incremental").
/// auto_converge : float, optional
///     Tolerance for auto-convergence (disabled by default).
/// return_robustness_weights : bool, optional
///     Whether to include robustness weights.
/// zero_weight_fallback : str, optional
///     Fallback when all weights are zero.
/// parallel : bool, optional
///     Enable parallel execution (default: False for online).
///
/// Returns
/// -------
/// LoessResult
///     Result object with smoothed values.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x, y,
    fraction=0.2,
    window_capacity=1000,
    min_points=3,
    iterations=3,
    weight_function="tricube",
    robustness_method="bisquare",
    scaling_method="mad",
    boundary_policy="extend",
    polynomial_degree="linear",
    surface_mode="interpolation",
    boundary_degree_fallback=true,
    dimensions=1,
    distance_metric="normalized",
    cell=0.2,
    interpolation_vertices=None,
    update_mode="incremental",
    auto_converge=None,
    return_robustness_weights=false,
    zero_weight_fallback="use_local_mean",
    parallel=false
))]
fn smooth_online<'py>(
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    fraction: f64,
    window_capacity: usize,
    min_points: usize,
    iterations: usize,
    weight_function: &str,
    robustness_method: &str,
    scaling_method: &str,
    boundary_policy: &str,
    polynomial_degree: &str,
    surface_mode: &str,
    boundary_degree_fallback: bool,
    dimensions: usize,
    distance_metric: &str,
    cell: f64,
    interpolation_vertices: Option<usize>,
    update_mode: &str,
    auto_converge: Option<f64>,
    return_robustness_weights: bool,
    zero_weight_fallback: &str,
    parallel: bool,
) -> PyResult<PyLoessResult> {
    let x_slice = x.as_slice().map_err(to_py_error)?;
    let y_slice = y.as_slice().map_err(to_py_error)?;

    let wf = parse_weight_function(weight_function)?;
    let rm = parse_robustness_method(robustness_method)?;
    let sm = parse_scaling_method(scaling_method)?;
    let bp = parse_boundary_policy(boundary_policy)?;
    let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
    let um = parse_update_mode(update_mode)?;
    let pd = parse_polynomial_degree(polynomial_degree)?;
    let surf_mode = parse_surface_mode(surface_mode)?;
    let dm = parse_distance_metric(distance_metric)?;

    let mut builder = LoessBuilder::<f64>::new();
    builder = builder.fraction(fraction);
    builder = builder.iterations(iterations);
    builder = builder.weight_function(wf);
    builder = builder.robustness_method(rm);
    builder = builder.scaling_method(sm);
    builder = builder.zero_weight_fallback(zwf);
    builder = builder.boundary_policy(bp);
    builder = builder.degree(pd);
    builder = builder.surface_mode(surf_mode);
    builder = builder.boundary_degree_fallback(boundary_degree_fallback);
    builder = builder.dimensions(dimensions);
    builder = builder.distance_metric(dm);
    builder = builder.cell(cell);
    if let Some(iv) = interpolation_vertices {
        builder = builder.interpolation_vertices(iv);
    }

    if let Some(tol) = auto_converge {
        builder = builder.auto_converge(tol);
    }

    let mut builder = builder.adapter(Online);
    builder = builder.window_capacity(window_capacity);
    builder = builder.min_points(min_points);
    builder = builder.update_mode(um);
    builder = builder.parallel(parallel);

    if return_robustness_weights {
        builder = builder.return_robustness_weights(true);
    }

    let mut processor = builder.build().map_err(to_py_error)?;

    // Process each point individually
    let mut smoothed = Vec::with_capacity(x_slice.len());
    for (&xi, &yi) in x_slice.iter().zip(y_slice.iter()) {
        let output = processor.add_point(&[xi], yi).map_err(to_py_error)?;
        // Use smoothed value if available, otherwise use original
        smoothed.push(output.map_or(yi, |o| o.smoothed));
    }

    // Create result
    let result = LoessResult {
        x: x_slice.to_vec(),
        dimensions: 1,
        distance_metric: DistanceMetric::Normalized,
        polynomial_degree: pd,
        y: smoothed,
        standard_errors: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
        residuals: None,
        robustness_weights: None,
        diagnostics: None,
        iterations_used: Some(iterations),
        fraction_used: fraction,
        cv_scores: None,
        enp: None,
        trace_hat: None,
        delta1: None,
        delta2: None,
        residual_scale: None,
        leverage: None,
    };

    Ok(PyLoessResult { inner: result })
}

// ============================================================================
// Module Registration
// ============================================================================

/// fastloess: High-performance LOESS smoothing for Python.
#[pymodule]
fn fastloess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLoessResult>()?;
    m.add_class::<PyDiagnostics>()?;
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_streaming, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_online, m)?)?;
    Ok(())
}
