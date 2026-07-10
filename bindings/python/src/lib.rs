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
use ::fastLoess::internals::binding_support as shared_parse;

use ::fastLoess::prelude::LoessResult;
use fastLoess::internals::api::LoessBuilder;

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
                ..Default::default()
            },
        ))?;

        let processor =
            shared_parse::build_streaming(builder, Some(chunk_size), overlap, Some(merge_strategy))
                .map_err(to_py_error)?;
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
                ..Default::default()
            },
        ))?;

        let processor = shared_parse::build_online(
            builder,
            Some(window_capacity),
            Some(min_points),
            Some(update_mode),
        )
        .map_err(to_py_error)?;
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
    builder: LoessBuilder<f64>,
    // Kept only for __repr__
    fraction: f64,
    iterations: usize,
    parallel: bool,
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
                cv_fractions: cv_fractions.as_deref(),
                cv_method: Some(cv_method),
                cv_k: Some(cv_k),
                cv_seed,
            },
        ))?;

        Ok(PyLoess {
            builder,
            fraction,
            iterations,
            parallel,
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

        // Clone the pre-built builder for this fit call
        let builder = self.builder.clone();

        // 2. Release GIL
        let result = py.detach(move || {
            let model = shared_parse::build_batch(builder, uw_vec)?;
            shared_parse::map_loess_result(model.fit(&x_vec, &y_vec))
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
