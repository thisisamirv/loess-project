//! Python bindings for fastLoess.

#![allow(non_snake_case)]
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt::Display;
use std::sync::Mutex;

use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::api::{Batch, LoessBuilder, Online, Streaming};
use ::fastLoess::internals::binding_support as shared_parse;
use ::fastLoess::internals::binding_support::{
    BoundaryPolicy, DistanceMetric, PolynomialDegree, RobustnessMethod, ScalingMethod, SurfaceMode,
    WeightFunction, ZeroWeightFallback,
};
use ::fastLoess::prelude::LoessResult;

// Helper Functions

fn to_py_error(err: shared_parse::BindingError) -> PyErr {
    match err.category {
        shared_parse::BindingErrorCategory::InvalidArg => PyValueError::new_err(err.message),
        shared_parse::BindingErrorCategory::Runtime => PyRuntimeError::new_err(err.message),
    }
}

fn map_invalid_arg<T, E: Display>(result: Result<T, E>) -> PyResult<T> {
    shared_parse::map_invalid_arg(result).map_err(to_py_error)
}

fn map_runtime<T, E: Display>(result: Result<T, E>) -> PyResult<T> {
    shared_parse::map_runtime(result).map_err(to_py_error)
}

fn to_py_invalid_arg_error(e: impl Display) -> PyErr {
    to_py_error(shared_parse::BindingError::invalid_arg(e.to_string()))
}

// Python Classes

// Diagnostic statistics for LOESS fit quality.
#[pyclass(name = "Diagnostics", from_py_object)]
#[derive(Clone)]
pub struct PyDiagnostics {
    // Root Mean Squared Error
    #[pyo3(get)]
    pub rmse: f64,

    // Mean Absolute Error
    #[pyo3(get)]
    pub mae: f64,

    // R-squared (coefficient of determination)
    #[pyo3(get)]
    pub r_squared: f64,

    // Akaike Information Criterion
    #[pyo3(get)]
    pub aic: Option<f64>,

    // Corrected AIC
    #[pyo3(get)]
    pub aicc: Option<f64>,

    // Effective degrees of freedom
    #[pyo3(get)]
    pub effective_df: Option<f64>,

    // Residual standard deviation
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

// Result from LOESS smoothing.
#[pyclass(name = "LoessResult")]
pub struct PyLoessResult {
    inner: LoessResult<f64>,
}

#[pymethods]
impl PyLoessResult {
    // Sorted x values
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.x.clone())
    }

    // Smoothed y values
    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.y.clone())
    }

    // Standard errors (if computed)
    #[getter]
    fn standard_errors<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Lower confidence interval bounds
    #[getter]
    fn confidence_lower<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Upper confidence interval bounds
    #[getter]
    fn confidence_upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Lower prediction interval bounds
    #[getter]
    fn prediction_lower<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Upper prediction interval bounds
    #[getter]
    fn prediction_upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Residuals (original y - smoothed y)
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .residuals
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Robustness weights from final iteration
    #[getter]
    fn robustness_weights<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .robustness_weights
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Diagnostic metrics
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

    // Number of iterations performed
    #[getter]
    fn iterations_used(&self) -> Option<usize> {
        self.inner.iterations_used
    }

    // Fraction used for smoothing
    #[getter]
    fn fraction_used(&self) -> f64 {
        self.inner.fraction_used
    }

    // CV scores for tested fractions
    #[getter]
    fn cv_scores<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .cv_scores
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Equivalent number of parameters (hat-matrix stat, if return_se was set)
    #[getter]
    fn enp(&self) -> Option<f64> {
        self.inner.enp
    }

    // Trace of hat matrix (if return_se was set)
    #[getter]
    fn trace_hat(&self) -> Option<f64> {
        self.inner.trace_hat
    }

    // First delta statistic (if return_se was set)
    #[getter]
    fn delta1(&self) -> Option<f64> {
        self.inner.delta1
    }

    // Second delta statistic (if return_se was set)
    #[getter]
    fn delta2(&self) -> Option<f64> {
        self.inner.delta2
    }

    // Residual scale estimate (if return_se was set)
    #[getter]
    fn residual_scale(&self) -> Option<f64> {
        self.inner.residual_scale
    }

    // Per-point leverage / hat-matrix diagonal (if return_se was set)
    #[getter]
    fn leverage<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .leverage
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    // Number of predictor dimensions
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions
    }

    fn __repr__(&self) -> String {
        format!(
            "LoessResult(n={}, fraction_used={:.4})",
            self.inner.y.len(),
            self.inner.fraction_used
        )
    }
}

// Python Classes - Stateful Adapters

// LOESS processor for incremental chunk-based smoothing.
#[pyclass(name = "StreamingLoess")]
pub struct PyStreamingLoess {
    inner: Mutex<ParallelStreamingLoess<f64>>,
}

#[pymethods]
impl PyStreamingLoess {
    #[new]
    #[pyo3(signature = (
        fraction=0.67,
        chunk_size=5000,
        overlap=None,
        iterations=3,
        weight_function="tricube",
        robustness_method="bisquare",
        scaling_method="mad",
        boundary_policy="extend",
        auto_converge=None,
        return_diagnostics=false,
        return_residuals=false,
        return_robustness_weights=false,
        zero_weight_fallback="use_local_mean",
        merge_strategy="weighted_average",
        parallel=true,
        degree="linear",
        dimensions=1usize,
        distance_metric="normalized",
        weighted_metric_weights=None,
        surface_mode="interpolation",
        return_se=false,
        cell=None,
        interpolation_vertices=None,
        boundary_degree_fallback=None,
        confidence_intervals=None,
        prediction_intervals=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        chunk_size: usize,
        overlap: Option<usize>,
        iterations: usize,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
        auto_converge: Option<f64>,
        return_diagnostics: bool,
        return_residuals: bool,
        return_robustness_weights: bool,
        zero_weight_fallback: &str,
        merge_strategy: &str,
        parallel: bool,
        degree: &str,
        dimensions: usize,
        distance_metric: &str,
        weighted_metric_weights: Option<Vec<f64>>,
        surface_mode: &str,
        return_se: bool,
        cell: Option<f64>,
        interpolation_vertices: Option<usize>,
        boundary_degree_fallback: Option<bool>,
        confidence_intervals: Option<f64>,
        prediction_intervals: Option<f64>,
    ) -> PyResult<Self> {
        let ms = map_invalid_arg(shared_parse::parse_merge_strategy(merge_strategy))?;
        let (builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
            LoessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                zero_weight_fallback: Some(zero_weight_fallback),
                boundary_policy: Some(boundary_policy),
                scaling_method: Some(scaling_method),
                auto_converge,
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                confidence_intervals,
                prediction_intervals,
                parallel: Some(parallel),
                degree: Some(degree),
                dimensions: Some(dimensions),
                distance_metric: Some(distance_metric),
                weighted_metric_weights: weighted_metric_weights.as_deref(),
                surface_mode: Some(surface_mode),
                return_se,
                cell,
                interpolation_vertices,
                boundary_degree_fallback,
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        ))?;

        let overlap_size = overlap.unwrap_or_else(|| {
            let default = chunk_size / 10;
            default.min(chunk_size.saturating_sub(10)).max(1)
        });

        let mut streaming_builder = builder.adapter(Streaming);
        streaming_builder = streaming_builder.chunk_size(chunk_size);
        streaming_builder = streaming_builder.overlap(overlap_size);
        streaming_builder = streaming_builder.parallel(parallel);
        streaming_builder = streaming_builder.merge_strategy(ms);

        if let Some(tol) = auto_converge {
            streaming_builder = streaming_builder.auto_converge(tol);
        }

        let processor = map_runtime(streaming_builder.build())?;
        Ok(PyStreamingLoess {
            inner: Mutex::new(processor),
        })
    }

    // Process a chunk of data.
    fn process_chunk<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyLoessResult> {
        let x_vec = x.as_slice().map_err(to_py_invalid_arg_error)?.to_vec();
        let y_vec = y.as_slice().map_err(to_py_invalid_arg_error)?.to_vec();

        let result = py.detach(move || {
            self.inner
                .lock()
                .map_err(|e| {
                    to_py_error(shared_parse::BindingError::runtime(
                        shared_parse::mutex_poisoned_message(&e.to_string()),
                    ))
                })?
                .process_chunk(&x_vec, &y_vec)
                .map_err(|e| to_py_error(shared_parse::BindingError::runtime(e.to_string())))
        })?;

        Ok(PyLoessResult { inner: result })
    }

    // Finalize smoothing and return remaining buffered data.
    fn finalize(&self, py: Python<'_>) -> PyResult<PyLoessResult> {
        let result = py.detach(move || {
            self.inner
                .lock()
                .map_err(|e| {
                    to_py_error(shared_parse::BindingError::runtime(
                        shared_parse::mutex_poisoned_message(&e.to_string()),
                    ))
                })?
                .finalize()
                .map_err(|e| to_py_error(shared_parse::BindingError::runtime(e.to_string())))
        })?;

        Ok(PyLoessResult { inner: result })
    }
}

// Result from a single online update step.
#[pyclass(name = "OnlineOutput", from_py_object)]
#[derive(Clone)]
pub struct PyOnlineOutput {
    #[pyo3(get)]
    pub smoothed: f64,
    #[pyo3(get)]
    pub std_error: Option<f64>,
    #[pyo3(get)]
    pub residual: Option<f64>,
    #[pyo3(get)]
    pub robustness_weight: Option<f64>,
    #[pyo3(get)]
    pub iterations_used: Option<usize>,
}

#[pymethods]
impl PyOnlineOutput {
    fn __repr__(&self) -> String {
        format!("OnlineOutput(smoothed={:.4})", self.smoothed)
    }
}

// LOESS processor for real-time data streams.
#[pyclass(name = "OnlineLoess")]
pub struct PyOnlineLoess {
    inner: Mutex<ParallelOnlineLoess<f64>>,
}

#[pymethods]
impl PyOnlineLoess {
    #[new]
    #[pyo3(signature = (
        fraction=0.67,
        window_capacity=1000,
        min_points=3,
        iterations=3,
        weight_function="tricube",
        robustness_method="bisquare",
        scaling_method="mad",
        boundary_policy="extend",
        update_mode="full",
        auto_converge=None,
        return_robustness_weights=false,
        zero_weight_fallback="use_local_mean",
        parallel=false,
        degree="linear",
        dimensions=1usize,
        distance_metric="normalized",
        weighted_metric_weights=None,
        surface_mode="interpolation",
        return_se=false,
        return_diagnostics=false,
        return_residuals=false,
        cell=None,
        interpolation_vertices=None,
        boundary_degree_fallback=None,
        confidence_intervals=None,
        prediction_intervals=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        window_capacity: usize,
        min_points: usize,
        iterations: usize,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
        update_mode: &str,
        auto_converge: Option<f64>,
        return_robustness_weights: bool,
        zero_weight_fallback: &str,
        parallel: bool,
        degree: &str,
        dimensions: usize,
        distance_metric: &str,
        weighted_metric_weights: Option<Vec<f64>>,
        surface_mode: &str,
        return_se: bool,
        return_diagnostics: bool,
        return_residuals: bool,
        cell: Option<f64>,
        interpolation_vertices: Option<usize>,
        boundary_degree_fallback: Option<bool>,
        confidence_intervals: Option<f64>,
        prediction_intervals: Option<f64>,
    ) -> PyResult<Self> {
        let um = map_invalid_arg(shared_parse::parse_update_mode(update_mode))?;
        let (builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
            LoessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                zero_weight_fallback: Some(zero_weight_fallback),
                boundary_policy: Some(boundary_policy),
                scaling_method: Some(scaling_method),
                auto_converge,
                return_residuals,
                return_robustness_weights: false,
                return_diagnostics,
                confidence_intervals,
                prediction_intervals,
                parallel: Some(parallel),
                degree: Some(degree),
                dimensions: Some(dimensions),
                distance_metric: Some(distance_metric),
                weighted_metric_weights: weighted_metric_weights.as_deref(),
                surface_mode: Some(surface_mode),
                return_se,
                cell,
                interpolation_vertices,
                boundary_degree_fallback,
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        ))?;

        let mut online_builder = builder.adapter(Online);
        online_builder = online_builder.window_capacity(window_capacity);
        online_builder = online_builder.min_points(min_points);
        online_builder = online_builder.update_mode(um);
        online_builder = online_builder.parallel(parallel);

        if let Some(tol) = auto_converge {
            online_builder = online_builder.auto_converge(tol);
        }
        if return_robustness_weights {
            online_builder = online_builder.return_robustness_weights(true);
        }

        let processor = map_runtime(online_builder.build())?;
        Ok(PyOnlineLoess {
            inner: Mutex::new(processor),
        })
    }

    // Add a single point and return its smoothed value, or None if the window
    // is not yet full enough to produce a result.
    fn add_point(&self, x: f64, y: f64) -> PyResult<Option<PyOnlineOutput>> {
        let mut inner = self.inner.lock().map_err(|e| {
            to_py_error(shared_parse::BindingError::runtime(
                shared_parse::mutex_poisoned_message(&e.to_string()),
            ))
        })?;
        let output = inner
            .add_point(&[x], y)
            .map_err(|e| to_py_error(shared_parse::BindingError::invalid_arg(e.to_string())))?;
        Ok(output.map(|o| PyOnlineOutput {
            smoothed: o.smoothed,
            std_error: o.std_error,
            residual: o.residual,
            robustness_weight: o.robustness_weight,
            iterations_used: o.iterations_used,
        }))
    }
}

// LOESS processor with configurable parameters.
//
// This class allows you to configure LOESS parameters once and then
// call `fit()` multiple times with different datasets.
#[pyclass(name = "Loess", from_py_object)]
#[derive(Clone)]
pub struct PyLoess {
    fraction: f64,
    iterations: usize,
    weight_function: WeightFunction,
    robustness_method: RobustnessMethod,
    scaling_method: ScalingMethod,
    zero_weight_fallback: ZeroWeightFallback,
    boundary_policy: BoundaryPolicy,
    auto_converge: Option<f64>,
    confidence_intervals: Option<f64>,
    prediction_intervals: Option<f64>,
    return_diagnostics: bool,
    return_residuals: bool,
    return_robustness_weights: bool,
    cv_fractions: Option<Vec<f64>>,
    cv_method: String,
    cv_k: usize,
    parallel: bool,
    degree: PolynomialDegree,
    dimensions: usize,
    distance_metric: DistanceMetric<f64>,
    surface_mode: SurfaceMode,
    return_se: bool,
    cell: Option<f64>,
    interpolation_vertices: Option<usize>,
    boundary_degree_fallback: Option<bool>,
    cv_seed: Option<u64>,
}

#[pymethods]
impl PyLoess {
    #[new]
    #[pyo3(signature = (
        fraction=0.67,
        iterations=3,
        weight_function="tricube",
        robustness_method="bisquare",
        scaling_method="mad",
        boundary_policy="extend",
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
        parallel=true,
        degree="linear",
        dimensions=1usize,
        distance_metric="normalized",
        weighted_metric_weights=None,
        surface_mode="interpolation",
        return_se=false,
        cell=None,
        interpolation_vertices=None,
        boundary_degree_fallback=None,
        cv_seed=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        iterations: usize,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
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
        degree: &str,
        dimensions: usize,
        distance_metric: &str,
        weighted_metric_weights: Option<Vec<f64>>,
        surface_mode: &str,
        return_se: bool,
        cell: Option<f64>,
        interpolation_vertices: Option<usize>,
        boundary_degree_fallback: Option<bool>,
        cv_seed: Option<u64>,
    ) -> PyResult<Self> {
        let wf = map_invalid_arg(shared_parse::parse_weight_function(weight_function))?;
        let rm = map_invalid_arg(shared_parse::parse_robustness_method(robustness_method))?;
        let sm = map_invalid_arg(shared_parse::parse_scaling_method(scaling_method))?;
        let zwf = map_invalid_arg(shared_parse::parse_zero_weight_fallback(
            zero_weight_fallback,
        ))?;
        let bp = map_invalid_arg(shared_parse::parse_boundary_policy(boundary_policy))?;
        let deg = map_invalid_arg(shared_parse::parse_polynomial_degree(degree))?;
        let (_, dm) = map_invalid_arg(shared_parse::apply_distance_metric(
            LoessBuilder::<f64>::new(),
            distance_metric,
            weighted_metric_weights.as_deref(),
        ))?;
        let surf = map_invalid_arg(shared_parse::parse_surface_mode(surface_mode))?;

        Ok(PyLoess {
            fraction,
            iterations,
            weight_function: wf,
            robustness_method: rm,
            scaling_method: sm,
            zero_weight_fallback: zwf,
            boundary_policy: bp,
            auto_converge,
            confidence_intervals,
            prediction_intervals,
            return_diagnostics,
            return_residuals,
            return_robustness_weights,
            cv_fractions,
            cv_method: cv_method.to_string(),
            cv_k,
            parallel,
            degree: deg,
            dimensions,
            distance_metric: dm,
            surface_mode: surf,
            return_se,
            cell,
            interpolation_vertices,
            boundary_degree_fallback,
            cv_seed,
        })
    }

    // Fit LOESS model to data.
    //
    // Parameters
    // ----------
    // x : array_like
    //     Independent variable values.
    // y : array_like
    //     Dependent variable values.
    //
    // Returns
    // -------
    // LoessResult
    //     Smoothed values and optional diagnostics.
    #[pyo3(signature = (x, y, custom_weights=None))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        custom_weights: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<PyLoessResult> {
        // 1. Copy data (Must be done with GIL)
        let x_vec = x.as_slice().map_err(to_py_invalid_arg_error)?.to_vec();
        let y_vec = y.as_slice().map_err(to_py_invalid_arg_error)?.to_vec();
        let uw_vec: Option<Vec<f64>> = custom_weights
            .map(|uw| {
                uw.as_slice()
                    .map(|s| s.to_vec())
                    .map_err(to_py_invalid_arg_error)
            })
            .transpose()?;

        // Used for builder configuration
        let params = self.clone();

        // 2. Release GIL
        let result = py.detach(move || {
            let (mut builder, _) =
                shared_parse::map_invalid_arg(shared_parse::apply_typed_builder_options(
                    LoessBuilder::<f64>::new(),
                    shared_parse::TypedBuilderOptionSet {
                        fraction: Some(params.fraction),
                        iterations: Some(params.iterations),
                        weight_function: Some(params.weight_function),
                        robustness_method: Some(params.robustness_method),
                        zero_weight_fallback: Some(params.zero_weight_fallback),
                        boundary_policy: Some(params.boundary_policy),
                        scaling_method: Some(params.scaling_method),
                        auto_converge: params.auto_converge,
                        return_residuals: params.return_residuals,
                        return_robustness_weights: params.return_robustness_weights,
                        return_diagnostics: params.return_diagnostics,
                        confidence_intervals: params.confidence_intervals,
                        prediction_intervals: params.prediction_intervals,
                        parallel: Some(params.parallel),
                        degree: Some(params.degree),
                        dimensions: Some(params.dimensions),
                        distance_metric: Some(params.distance_metric),
                        surface_mode: Some(params.surface_mode),
                        return_se: params.return_se,
                        cell: params.cell,
                        interpolation_vertices: params.interpolation_vertices,
                        boundary_degree_fallback: params.boundary_degree_fallback,
                        cv_fractions: params.cv_fractions,
                        cv_method: Some(params.cv_method),
                        cv_k: Some(params.cv_k),
                        cv_seed: params.cv_seed,
                    },
                ))?;

            if let Some(uw) = uw_vec {
                builder = builder.custom_weights(uw);
            }

            let model = shared_parse::map_invalid_arg(builder.adapter(Batch).build())?;
            shared_parse::map_invalid_arg(model.fit(&x_vec, &y_vec))
        });

        // 3. Handle result (Back with GIL)
        match result {
            Ok(inner) => Ok(PyLoessResult { inner }),
            Err(e) => Err(to_py_error(e)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Loess(fraction={:.4}, iterations={}, parallel={})",
            self.fraction, self.iterations, self.parallel
        )
    }
}

// Module Registration

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLoessResult>()?;
    m.add_class::<PyDiagnostics>()?;
    m.add_class::<PyOnlineOutput>()?;
    m.add_class::<PyLoess>()?;
    m.add_class::<PyStreamingLoess>()?;
    m.add_class::<PyOnlineLoess>()?;
    Ok(())
}
