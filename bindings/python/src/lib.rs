//! Python bindings for fastLoess.

#![allow(non_snake_case)]
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::fmt::Display;
use std::sync::Mutex;

use ::fastLoess::internals::adapters::online::ParallelOnlineLoess;
use ::fastLoess::internals::adapters::streaming::ParallelStreamingLoess;
use ::fastLoess::internals::binding_support as shared_parse;

use ::fastLoess::prelude::{IntervalsBuilder, LoessResult};
use fastLoess::internals::api::LoessBuilder;
use fastLoess::prelude::Predict;

// ============================================================================
// Helper Functions
// ============================================================================

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

type ParsedCvOptions = (Option<Vec<f64>>, String, usize);

#[derive(Default)]
struct ParsedIntervals {
    confidence: Option<f64>,
    prediction: Option<f64>,
}

impl ParsedIntervals {
    fn builder(self) -> IntervalsBuilder<f64> {
        let mut options = IntervalsBuilder::new();
        if let Some(level) = self.confidence {
            options = options.confidence(level);
        }
        if let Some(level) = self.prediction {
            options = options.prediction(level);
        }
        options
    }
}

fn check_keys(group: &Bound<'_, PyDict>, allowed: &[&str], name: &str) -> PyResult<()> {
    for key in group.keys() {
        let key: String = key.extract().map_err(to_py_invalid_arg_error)?;
        if !allowed.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown {name} option '{key}'; expected one of: {}",
                allowed.join(", ")
            )));
        }
    }
    Ok(())
}

fn parse_intervals(intervals: Option<&Bound<'_, PyDict>>) -> PyResult<ParsedIntervals> {
    let mut options = ParsedIntervals::default();
    if let Some(group) = intervals {
        check_keys(group, &["confidence", "prediction"], "intervals")?;
        if let Some(value) = group
            .get_item("confidence")?
            .filter(|value| !value.is_none())
        {
            options.confidence = Some(value.extract().map_err(to_py_invalid_arg_error)?);
        }
        if let Some(value) = group
            .get_item("prediction")?
            .filter(|value| !value.is_none())
        {
            options.prediction = Some(value.extract().map_err(to_py_invalid_arg_error)?);
        }
    }
    Ok(options)
}

fn parse_cv_options(cv: Option<&Bound<'_, PyDict>>) -> PyResult<ParsedCvOptions> {
    let Some(cv) = cv else {
        return Ok((None, "kfold".to_owned(), 5));
    };
    check_keys(cv, &["fractions", "method", "k"], "cv")?;

    let fractions = match cv.get_item("fractions")? {
        Some(value) => Some(
            value
                .extract::<Vec<f64>>()
                .map_err(to_py_invalid_arg_error)?,
        ),
        None => None,
    };
    if fractions.is_none() {
        return Err(PyValueError::new_err("cv requires a 'fractions' sequence"));
    }
    let method = match cv.get_item("method")? {
        Some(value) => value.extract::<String>().map_err(to_py_invalid_arg_error)?,
        None => "kfold".to_owned(),
    };
    let k = match cv.get_item("k")? {
        Some(value) => value.extract::<usize>().map_err(to_py_invalid_arg_error)?,
        None => 5,
    };
    Ok((fractions, method, k))
}

// ============================================================================
// Python Classes
// ============================================================================

/// Diagnostic statistics for LOESS fit quality.
#[pyclass(name = "Diagnostics", from_py_object)]
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

/// Result from `LoessResult.predict()`.
#[pyclass(name = "PredictOutput")]
pub struct PyPredictOutput {
    inner: shared_parse::PredictOutput<f64>,
}

#[pymethods]
impl PyPredictOutput {
    /// Predicted y values, one per query point
    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.y.clone())
    }

    /// Standard errors (if requested)
    #[getter]
    fn standard_errors<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .standard_errors
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Lower confidence interval bounds (if requested)
    #[getter]
    fn confidence_lower<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .confidence_lower
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Upper confidence interval bounds (if requested)
    #[getter]
    fn confidence_upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .confidence_upper
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Lower prediction interval bounds (if requested)
    #[getter]
    fn prediction_lower<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .prediction_lower
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Upper prediction interval bounds (if requested)
    #[getter]
    fn prediction_upper<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .prediction_upper
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Local fit's gradient at each query point (if requested), `dimensions` values per
    /// query point, flattened
    #[getter]
    fn derivative<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .derivative
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    fn __repr__(&self) -> String {
        format!("PredictOutput(n={})", self.inner.y.len())
    }
}

#[pymethods]
impl PyLoessResult {
    /// x values, in the same order as the input
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

    /// Local fit's gradient at each point (if requested), `dimensions` values per point,
    /// flattened
    #[getter]
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .gradient
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

    /// Equivalent number of parameters (hat-matrix stat, if return_se was set)
    #[getter]
    fn enp(&self) -> Option<f64> {
        self.inner.enp
    }

    /// Trace of hat matrix (if return_se was set)
    #[getter]
    fn trace_hat(&self) -> Option<f64> {
        self.inner.trace_hat
    }

    /// First delta statistic (if return_se was set)
    #[getter]
    fn delta1(&self) -> Option<f64> {
        self.inner.delta1
    }

    /// Second delta statistic (if return_se was set)
    #[getter]
    fn delta2(&self) -> Option<f64> {
        self.inner.delta2
    }

    /// Residual scale estimate (if return_se was set)
    #[getter]
    fn residual_scale(&self) -> Option<f64> {
        self.inner.residual_scale
    }

    /// Per-point leverage / hat-matrix diagonal (if return_se was set)
    #[getter]
    fn leverage<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .leverage
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    /// Number of predictor dimensions
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions
    }

    /// Evaluate the fitted model at out-of-sample query points not in the training set.
    ///
    /// Requires `retain_model=True` to have been set on the builder before `fit()`.
    ///
    /// Parameters
    /// ----------
    /// new_x : array_like
    ///     Query points (flattened, `dimensions` values per point).
    /// outputs : sequence[str], optional
    ///     Select "se", "gradient", and/or "derivative".
    /// intervals : dict, optional
    ///     Grouped confidence and prediction coverage levels.
    /// extrapolation : str, optional
    ///     One of "clamp" (default), "linear", "error".
    /// max_extrapolation_distance : float, optional
    /// max_neighbor_distance : float, optional
    ///
    /// Returns
    /// -------
    /// PredictOutput
    #[pyo3(signature = (
        new_x,
        *,
        outputs=None,
        intervals=None,
        extrapolation="clamp",
        max_extrapolation_distance=None,
        max_neighbor_distance=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        new_x: &Bound<'py, PyAny>,
        outputs: Option<Vec<String>>,
        intervals: Option<&Bound<'_, PyDict>>,
        extrapolation: &str,
        max_extrapolation_distance: Option<f64>,
        max_neighbor_distance: Option<f64>,
    ) -> PyResult<PyPredictOutput> {
        validate_outputs(outputs.as_ref(), &["se", "gradient", "derivative"])?;
        let intervals = parse_intervals(intervals)?;
        let new_x_vec = array_like_to_vec(py, new_x)?;
        let output = py
            .detach(move || {
                shared_parse::run_predict(
                    &self.inner,
                    &new_x_vec,
                    shared_parse::PredictOptionSet {
                        return_se: has_output(outputs.as_ref(), "se"),
                        confidence_level: intervals.confidence,
                        prediction_level: intervals.prediction,
                        return_derivative: has_output(outputs.as_ref(), "gradient")
                            || has_output(outputs.as_ref(), "derivative"),
                        extrapolation: Some(extrapolation),
                        max_extrapolation_distance,
                        max_neighbor_distance,
                    },
                )
            })
            .map_err(to_py_error)?;
        Ok(PyPredictOutput { inner: output })
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
// Python Classes - Stateful Adapters
// ============================================================================

fn has_output(outputs: Option<&Vec<String>>, name: &str) -> bool {
    outputs.is_some_and(|values| values.iter().any(|value| value == name))
}

fn validate_outputs(outputs: Option<&Vec<String>>, allowed: &[&str]) -> PyResult<()> {
    if let Some(output) = outputs
        .into_iter()
        .flatten()
        .find(|output| !allowed.contains(&output.as_str()))
    {
        return Err(PyValueError::new_err(format!(
            "unknown output '{output}'. Valid outputs: {}",
            allowed.join(", ")
        )));
    }
    Ok(())
}

fn array_like_to_vec<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", numpy.getattr("float64")?)?;
    let array = numpy.call_method("ascontiguousarray", (value,), Some(&kwargs))?;
    let array: PyReadonlyArray1<'py, f64> = array.extract()?;
    array
        .as_slice()
        .map(|values| values.to_vec())
        .map_err(to_py_invalid_arg_error)
}

fn online_coordinate_to_vec<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(vec![scalar]);
    }
    array_like_to_vec(py, value)
}

/// Streaming LOESS processor for incremental chunk-based smoothing.
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
        *,
        overlap=None,
        iterations=3,
        weight_function="tricube",
        robustness_method="bisquare",
        scaling_method="mad",
        boundary_policy="extend",
        auto_converge=None,
        outputs=None,
        intervals=None,
        seed=None,
        zero_weight_fallback="use_local_mean",
        merge_strategy="weighted_average",
        parallel=true,
        degree="linear",
        dimensions=1usize,
        distance_metric="normalized",
        weighted_metric_weights=None,
        surface_mode="interpolation",
        cell=None,
        interpolation_vertices=None,
        boundary_degree_fallback=None,
        missing="error"
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
        outputs: Option<Vec<String>>,
        intervals: Option<&Bound<'_, PyDict>>,
        seed: Option<u64>,
        zero_weight_fallback: &str,
        merge_strategy: &str,
        parallel: bool,
        degree: &str,
        dimensions: usize,
        distance_metric: &str,
        weighted_metric_weights: Option<Vec<f64>>,
        surface_mode: &str,
        cell: Option<f64>,
        interpolation_vertices: Option<usize>,
        boundary_degree_fallback: Option<bool>,
        missing: &str,
    ) -> PyResult<Self> {
        validate_outputs(
            outputs.as_ref(),
            &[
                "diagnostics",
                "residuals",
                "weights",
                "gradient",
                "derivative",
                "se",
            ],
        )?;
        let (mut builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
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
                return_residuals: has_output(outputs.as_ref(), "residuals"),
                return_robustness_weights: has_output(outputs.as_ref(), "weights"),
                return_diagnostics: has_output(outputs.as_ref(), "diagnostics"),
                parallel: Some(parallel),
                degree: Some(degree),
                dimensions: Some(dimensions),
                distance_metric: Some(distance_metric),
                weighted_metric_weights: weighted_metric_weights.as_deref(),
                surface_mode: Some(surface_mode),
                return_se: has_output(outputs.as_ref(), "se"),
                cell,
                interpolation_vertices,
                boundary_degree_fallback,
                missing: Some(missing),
                ..Default::default()
            },
        ))?;
        if has_output(outputs.as_ref(), "gradient") || has_output(outputs.as_ref(), "derivative") {
            builder = builder.return_gradient();
        }

        builder = builder.intervals(parse_intervals(intervals)?.builder());
        if let Some(seed) = seed {
            builder = builder.seed(seed);
        }
        let processor =
            shared_parse::build_streaming(builder, Some(chunk_size), overlap, Some(merge_strategy))
                .map_err(to_py_error)?;
        Ok(PyStreamingLoess {
            inner: Mutex::new(processor),
        })
    }

    /// Process a chunk of data.
    #[pyo3(signature = (x, y, custom_weights=None))]
    fn process_chunk<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        custom_weights: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyLoessResult> {
        let x_vec = array_like_to_vec(py, x)?;
        let y_vec = array_like_to_vec(py, y)?;
        let weights_vec = custom_weights
            .map(|weights| array_like_to_vec(py, weights))
            .transpose()?;

        let result = py.detach(move || {
            let mut inner = self.inner.lock().map_err(|e| {
                to_py_error(shared_parse::BindingError::runtime(
                    shared_parse::mutex_poisoned_message(&e.to_string()),
                ))
            })?;
            let result = if let Some(weights) = weights_vec.as_deref() {
                inner.process_chunk_weighted(&x_vec, &y_vec, weights)
            } else {
                inner.process_chunk(&x_vec, &y_vec)
            };
            result.map_err(|e| to_py_error(shared_parse::BindingError::runtime(e.to_string())))
        })?;

        Ok(PyLoessResult { inner: result })
    }

    /// Finalize smoothing and return remaining buffered data.
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

/// Result from a single online update step.
#[pyclass(name = "OnlineOutput", from_py_object)]
#[derive(Clone)]
pub struct PyOnlineOutput {
    /// Smoothed value for the latest point
    #[pyo3(get)]
    pub y: f64,
    /// Standard error (if computed)
    #[pyo3(get)]
    pub standard_error: Option<f64>,
    /// Residual (raw input y minus this output's y) (if computed)
    #[pyo3(get)]
    pub residual: Option<f64>,
    /// Robustness weight for the latest point (if computed)
    #[pyo3(get)]
    pub robustness_weight: Option<f64>,
    /// Number of robustness iterations performed (if tracked)
    #[pyo3(get)]
    pub iterations_used: Option<usize>,
    /// Confidence interval lower bound (`update_mode="full"` only, if requested)
    #[pyo3(get)]
    pub confidence_lower: Option<f64>,
    /// Confidence interval upper bound (`update_mode="full"` only, if requested)
    #[pyo3(get)]
    pub confidence_upper: Option<f64>,
    /// Prediction interval lower bound (`update_mode="full"` only, if requested)
    #[pyo3(get)]
    pub prediction_lower: Option<f64>,
    /// Prediction interval upper bound (`update_mode="full"` only, if requested)
    #[pyo3(get)]
    pub prediction_upper: Option<f64>,
    /// Local fit gradient for the latest point (if requested), `dimensions` values
    gradient: Option<Vec<f64>>,
}

#[pymethods]
impl PyOnlineOutput {
    /// Local fit gradient for the latest point (if requested)
    #[getter]
    fn gradient<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.gradient
            .as_ref()
            .map(|v| PyArray1::from_vec(py, v.clone()))
    }

    fn __repr__(&self) -> String {
        format!("OnlineOutput(y={:.4})", self.y)
    }
}

/// Online LOESS processor for real-time data streams.
#[pyclass(name = "OnlineLoess")]
pub struct PyOnlineLoess {
    inner: Mutex<ParallelOnlineLoess<f64>>,
    dimensions: usize,
}

#[pymethods]
impl PyOnlineLoess {
    #[new]
    #[pyo3(signature = (
        fraction=0.67,
        window_capacity=1000,
        min_points=2,
        *,
        iterations=0,
        weight_function="tricube",
        robustness_method="bisquare",
        scaling_method="mad",
        boundary_policy="extend",
        update_mode="incremental",
        auto_converge=None,
        outputs=None,
        intervals=None,
        seed=None,
        zero_weight_fallback="use_local_mean",
        degree="linear",
        dimensions=1usize,
        distance_metric="normalized",
        weighted_metric_weights=None,
        surface_mode="interpolation",
        cell=None,
        interpolation_vertices=None,
        boundary_degree_fallback=None,
        missing="error"
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
        outputs: Option<Vec<String>>,
        intervals: Option<&Bound<'_, PyDict>>,
        seed: Option<u64>,
        zero_weight_fallback: &str,
        degree: &str,
        dimensions: usize,
        distance_metric: &str,
        weighted_metric_weights: Option<Vec<f64>>,
        surface_mode: &str,
        cell: Option<f64>,
        interpolation_vertices: Option<usize>,
        boundary_degree_fallback: Option<bool>,
        missing: &str,
    ) -> PyResult<Self> {
        validate_outputs(
            outputs.as_ref(),
            &["weights", "gradient", "derivative", "se"],
        )?;
        let (mut builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
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
                return_residuals: false,
                return_robustness_weights: has_output(outputs.as_ref(), "weights"),
                return_diagnostics: false,
                parallel: None,
                degree: Some(degree),
                dimensions: Some(dimensions),
                distance_metric: Some(distance_metric),
                weighted_metric_weights: weighted_metric_weights.as_deref(),
                surface_mode: Some(surface_mode),
                return_se: has_output(outputs.as_ref(), "se"),
                cell,
                interpolation_vertices,
                boundary_degree_fallback,
                missing: Some(missing),
                ..Default::default()
            },
        ))?;
        if has_output(outputs.as_ref(), "gradient") || has_output(outputs.as_ref(), "derivative") {
            builder = builder.return_gradient();
        }

        builder = builder.intervals(parse_intervals(intervals)?.builder());
        if let Some(seed) = seed {
            builder = builder.seed(seed);
        }
        let processor = shared_parse::build_online(
            builder,
            Some(window_capacity),
            Some(min_points),
            Some(update_mode),
        )
        .map_err(to_py_error)?;
        Ok(PyOnlineLoess {
            inner: Mutex::new(processor),
            dimensions,
        })
    }

    /// Add a point using a scalar x for 1D or a coordinate vector for multivariate input.
    #[pyo3(signature = (x, y, weight=1.0))]
    fn add_point<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: f64,
        weight: f64,
    ) -> PyResult<Option<PyOnlineOutput>> {
        let x_vec = online_coordinate_to_vec(py, x)?;
        if x_vec.len() != self.dimensions {
            return Err(PyValueError::new_err(format!(
                "x must have exactly {} values for dimensions={}, got {}",
                self.dimensions,
                self.dimensions,
                x_vec.len(),
            )));
        }
        let output = py.detach(move || {
            let mut inner = self.inner.lock().map_err(|e| {
                to_py_error(shared_parse::BindingError::runtime(
                    shared_parse::mutex_poisoned_message(&e.to_string()),
                ))
            })?;
            inner
                .add_point_weighted(&x_vec, y, weight)
                .map_err(|e| to_py_error(shared_parse::BindingError::invalid_arg(e.to_string())))
        })?;
        Ok(output.map(|o| PyOnlineOutput {
            y: o.y,
            standard_error: o.standard_error,
            residual: o.residual,
            robustness_weight: o.robustness_weight,
            iterations_used: o.iterations_used,
            confidence_lower: o.confidence_lower,
            confidence_upper: o.confidence_upper,
            prediction_lower: o.prediction_lower,
            prediction_upper: o.prediction_upper,
            gradient: o.gradient,
        }))
    }

    /// Compute diagnostics for the current sliding window on demand.
    fn window_diagnostics(&self, py: Python<'_>) -> PyResult<Option<PyDiagnostics>> {
        let diagnostics = py.detach(|| {
            self.inner
                .lock()
                .map_err(|e| {
                    to_py_error(shared_parse::BindingError::runtime(
                        shared_parse::mutex_poisoned_message(&e.to_string()),
                    ))
                })?
                .window_diagnostics()
                .map_err(|e| to_py_error(shared_parse::BindingError::runtime(e.to_string())))
        })?;
        Ok(diagnostics.map(|d| PyDiagnostics {
            rmse: d.rmse,
            mae: d.mae,
            r_squared: d.r_squared,
            aic: d.aic,
            aicc: d.aicc,
            effective_df: d.effective_df,
            residual_sd: d.residual_sd,
        }))
    }

    /// Predict query points using a fitted model of the current sliding window.
    #[pyo3(signature = (
        new_x,
        *,
        outputs=None,
        intervals=None,
        extrapolation="clamp",
        max_extrapolation_distance=None,
        max_neighbor_distance=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn predict_window<'py>(
        &self,
        py: Python<'py>,
        new_x: &Bound<'py, PyAny>,
        outputs: Option<Vec<String>>,
        intervals: Option<&Bound<'_, PyDict>>,
        extrapolation: &str,
        max_extrapolation_distance: Option<f64>,
        max_neighbor_distance: Option<f64>,
    ) -> PyResult<PyPredictOutput> {
        validate_outputs(outputs.as_ref(), &["se", "gradient", "derivative"])?;
        let intervals = parse_intervals(intervals)?;
        let new_x_vec = array_like_to_vec(py, new_x)?;
        let mut builder = Predict::new()
            .intervals(intervals.builder())
            .extrapolation(extrapolation);
        if let Some(names) = outputs.as_ref() {
            builder = builder.outputs(names.iter().map(String::as_str));
        }
        if let Some(distance) = max_extrapolation_distance {
            builder = builder.max_extrapolation_distance(distance);
        }
        if let Some(distance) = max_neighbor_distance {
            builder = builder.max_neighbor_distance(distance);
        }
        let options = builder.build().map_err(to_py_invalid_arg_error)?;
        let output = py.detach(move || {
            self.inner
                .lock()
                .map_err(|e| {
                    to_py_error(shared_parse::BindingError::runtime(
                        shared_parse::mutex_poisoned_message(&e.to_string()),
                    ))
                })?
                .predict_window(&new_x_vec, &options)
                .map_err(|e| to_py_error(shared_parse::BindingError::invalid_arg(e.to_string())))
        })?;
        Ok(PyPredictOutput { inner: output })
    }
}

/// Batch LOESS processor with configurable parameters.
///
/// This class allows you to configure LOESS parameters once and then
/// call `fit()` multiple times with different datasets.
#[pyclass(name = "Loess", from_py_object)]
#[derive(Clone)]
pub struct PyLoess {
    builder: LoessBuilder<f64>,
    /// Kept only for __repr__
    fraction: f64,
    iterations: usize,
    parallel: bool,
}

#[pymethods]
impl PyLoess {
    #[new]
    #[pyo3(signature = (
        fraction=0.67,
        *,
        iterations=3,
        weight_function="tricube",
        robustness_method="bisquare",
        scaling_method="mad",
        boundary_policy="extend",
        outputs=None,
        intervals=None,
        cv=None,
        seed=None,
        zero_weight_fallback="use_local_mean",
        auto_converge=None,
        parallel=true,
        degree="linear",
        dimensions=1usize,
        distance_metric="normalized",
        weighted_metric_weights=None,
        surface_mode="interpolation",
        cell=None,
        interpolation_vertices=None,
        boundary_degree_fallback=None,
        missing="error",
        retain_model=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        iterations: usize,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
        outputs: Option<Vec<String>>,
        intervals: Option<&Bound<'_, PyDict>>,
        cv: Option<Bound<'_, PyDict>>,
        seed: Option<u64>,
        zero_weight_fallback: &str,
        auto_converge: Option<f64>,
        parallel: bool,
        degree: &str,
        dimensions: usize,
        distance_metric: &str,
        weighted_metric_weights: Option<Vec<f64>>,
        surface_mode: &str,
        cell: Option<f64>,
        interpolation_vertices: Option<usize>,
        boundary_degree_fallback: Option<bool>,
        missing: &str,
        retain_model: bool,
    ) -> PyResult<Self> {
        validate_outputs(
            outputs.as_ref(),
            &[
                "diagnostics",
                "residuals",
                "weights",
                "gradient",
                "derivative",
                "se",
                "sorted",
            ],
        )?;
        let (cv_fractions, cv_method, cv_k) = parse_cv_options(cv.as_ref())?;
        let (mut builder, _) = map_invalid_arg(shared_parse::apply_builder_options(
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
                return_residuals: has_output(outputs.as_ref(), "residuals"),
                return_robustness_weights: has_output(outputs.as_ref(), "weights"),
                return_diagnostics: has_output(outputs.as_ref(), "diagnostics"),
                parallel: Some(parallel),
                degree: Some(degree),
                dimensions: Some(dimensions),
                distance_metric: Some(distance_metric),
                weighted_metric_weights: weighted_metric_weights.as_deref(),
                surface_mode: Some(surface_mode),
                return_se: has_output(outputs.as_ref(), "se"),
                return_sorted: has_output(outputs.as_ref(), "sorted"),
                cell,
                interpolation_vertices,
                boundary_degree_fallback,
                cv_fractions: cv_fractions.as_deref(),
                cv_method: Some(&cv_method),
                cv_k: Some(cv_k),
                cv_seed: seed,
                missing: Some(missing),
                retain_model: Some(retain_model),
                ..Default::default()
            },
        ))?;
        if has_output(outputs.as_ref(), "gradient") || has_output(outputs.as_ref(), "derivative") {
            builder = builder.return_gradient();
        }

        builder = builder.intervals(parse_intervals(intervals)?.builder());
        Ok(PyLoess {
            builder,
            fraction,
            iterations,
            parallel,
        })
    }

    /// Fit LOESS model to data.
    ///
    /// Parameters
    /// ----------
    /// x : array_like
    ///     Independent variable values.
    /// y : array_like
    ///     Dependent variable values.
    /// custom_weights : array_like, optional
    ///     Per-observation weights (same length as y). Each weight multiplies the
    ///     local kernel weight: w_ij = custom_weights[j] * K(d_ij/h) * rob_j.
    ///     Analogous to the ``weights`` argument in R's ``stats::loess``.
    ///
    /// Returns
    /// -------
    /// LoessResult
    ///     Smoothed values and optional diagnostics.
    #[pyo3(signature = (x, y, custom_weights=None))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        custom_weights: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyLoessResult> {
        // 1. Copy data (Must be done with GIL)
        let x_vec = array_like_to_vec(py, x)?;
        let y_vec = array_like_to_vec(py, y)?;
        let uw_vec = custom_weights
            .map(|weights| array_like_to_vec(py, weights))
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

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLoessResult>()?;
    m.add_class::<PyPredictOutput>()?;
    m.add_class::<PyDiagnostics>()?;
    m.add_class::<PyOnlineOutput>()?;
    m.add_class::<PyLoess>()?;
    m.add_class::<PyStreamingLoess>()?;
    m.add_class::<PyOnlineLoess>()?;
    Ok(())
}
