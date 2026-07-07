//! R bindings for fastLoess.
//!
//! Provides R access to the fastLoess Rust library via extendr.
//!
//! @srrstats {G1.0} Documentation of core R-to-Rust interface.
//! @srrstats {G1.1} Implementation of thin R wrapper for statistical algorithms.

#![allow(non_snake_case)]

use extendr_api::prelude::*;

type Result<T> = std::result::Result<T, Error>;

use fastLoess::binding_support as shared_parse;
use fastLoess::internals::api::{DistanceMetric, PolynomialDegree};
use fastLoess::prelude::{Batch, Loess as LoessBuilder, LoessResult, Online, Streaming};

// Helper Functions

fn to_r_error(err: shared_parse::BindingError) -> Error {
    let prefix = match err.category {
        shared_parse::BindingErrorCategory::InvalidArg => "[invalid-arg]",
        shared_parse::BindingErrorCategory::Runtime => "[runtime]",
    };
    Error::Other(format!("{} {}", prefix, err.message))
}

fn map_invalid_arg<T, E: ToString>(result: std::result::Result<T, E>) -> Result<T> {
    shared_parse::map_invalid_arg(result).map_err(to_r_error)
}

fn map_runtime<T, E: ToString>(result: std::result::Result<T, E>) -> Result<T> {
    shared_parse::map_runtime(result).map_err(to_r_error)
}

fn require_positive_usize(name: &str, value: i32) -> Result<usize> {
    if value <= 0 {
        return Err(to_r_error(shared_parse::BindingError::invalid_arg(
            format!("{} must be greater than 0", name),
        )));
    }
    Ok(value as usize)
}

fn optional_positive_usize(name: &str, value: Nullable<i32>) -> Result<Option<usize>> {
    match value {
        NotNull(v) => Ok(Some(require_positive_usize(name, v)?)),
        Null => Ok(None),
    }
}

// Stateful API: Loess

#[extendr]
pub struct RLoess {
    builder: LoessBuilder<f64>,
    parallel: bool,
}

#[extendr]
impl RLoess {
    // Create a new Loess model
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        iterations: i32,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
        confidence_intervals: Nullable<f64>,
        prediction_intervals: Nullable<f64>,
        return_diagnostics: bool,
        return_residuals: bool,
        return_robustness_weights: bool,
        zero_weight_fallback: &str,
        auto_converge: Nullable<f64>,
        cv_fractions: Nullable<Vec<f64>>,
        cv_method: &str,
        cv_k: i32,
        parallel: bool,
        degree: &str,
        dimensions: i32,
        distance_metric: &str,
        weighted_metric_weights: Nullable<Vec<f64>>,
        surface_mode: &str,
        return_se: bool,
        cell: Nullable<f64>,
        interpolation_vertices: Nullable<i32>,
        boundary_degree_fallback: Nullable<bool>,
        cv_seed: Nullable<i32>,
    ) -> Result<Self> {
        let weighted_weights = match weighted_metric_weights {
            NotNull(v) => Some(v),
            Null => None,
        };
        let fractions = match cv_fractions {
            NotNull(v) => Some(v),
            Null => None,
        };
        let seed = match cv_seed {
            NotNull(s) => Some(s as u64),
            Null => None,
        };
        let iterations = require_positive_usize("iterations", iterations)?;
        let dimensions = require_positive_usize("dimensions", dimensions)?;
        let cv_k = require_positive_usize("cv_k", cv_k)?;
        let interpolation_vertices =
            optional_positive_usize("interpolation_vertices", interpolation_vertices)?;
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
                auto_converge: match auto_converge {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                return_residuals,
                return_robustness_weights,
                return_diagnostics,
                confidence_intervals: match confidence_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                prediction_intervals: match prediction_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                parallel: None,
                degree: Some(degree),
                dimensions: Some(dimensions),
                distance_metric: Some(distance_metric),
                weighted_metric_weights: weighted_weights.as_deref(),
                surface_mode: Some(surface_mode),
                return_se,
                cell: match cell {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                interpolation_vertices,
                boundary_degree_fallback: match boundary_degree_fallback {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                cv_fractions: fractions.as_deref(),
                cv_method: Some(cv_method),
                cv_k: Some(cv_k),
                cv_seed: seed,
            },
        ))?;

        Ok(Self { builder, parallel })
    }

    // Fit the model to data, with optional user-defined case weights.
    //
    // `custom_weights` must be the same length as `y`. Each weight multiplies the
    // local kernel weight: analogous to `weights` in `stats::loess`.
    fn fit(&self, x: &[f64], y: &[f64], custom_weights: Nullable<Vec<f64>>) -> Result<List> {
        let mut builder = self.builder.clone();
        if let NotNull(w) = custom_weights {
            builder = builder.custom_weights(w);
        }
        let result = builder.adapter(Batch).parallel(self.parallel).build();
        let result = map_runtime(result)?.fit(x, y);
        let result = map_runtime(result)?;

        loess_result_to_list(result)
    }
}

// Stateful API: StreamingLoess

#[extendr]
pub struct RStreamingLoess {
    inner: fastLoess::internals::adapters::streaming::ParallelStreamingLoess<f64>,
    fraction: f64,
    iterations: usize,
}

#[extendr]
impl RStreamingLoess {
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        chunk_size: i32,
        overlap: Nullable<i32>,
        iterations: i32,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
        zero_weight_fallback: &str,
        auto_converge: Nullable<f64>,
        return_diagnostics: bool,
        return_residuals: bool,
        return_robustness_weights: bool,
        merge_strategy: &str,
        parallel: bool,
        degree: &str,
        dimensions: i32,
        distance_metric: &str,
        weighted_metric_weights: Nullable<Vec<f64>>,
        surface_mode: &str,
        return_se: bool,
        confidence_intervals: Nullable<f64>,
        prediction_intervals: Nullable<f64>,
        cell: Nullable<f64>,
        interpolation_vertices: Nullable<i32>,
        boundary_degree_fallback: Nullable<bool>,
    ) -> Result<Self> {
        let chunk_size = require_positive_usize("chunk_size", chunk_size)?;
        let overlap_size = match overlap {
            NotNull(o) => require_positive_usize("overlap", o)?,
            Null => (chunk_size / 10).min(chunk_size.saturating_sub(10)).max(1),
        };
        let iterations = require_positive_usize("iterations", iterations)?;
        let dimensions = require_positive_usize("dimensions", dimensions)?;
        let interpolation_vertices =
            optional_positive_usize("interpolation_vertices", interpolation_vertices)?;

        let ms = map_invalid_arg(shared_parse::parse_merge_strategy(merge_strategy))?;
        let weighted_weights = match weighted_metric_weights {
            NotNull(v) => Some(v),
            Null => None,
        };
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
                auto_converge: None,
                return_residuals,
                return_robustness_weights: false,
                return_diagnostics: false,
                confidence_intervals: match confidence_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                prediction_intervals: match prediction_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                parallel: None,
                degree: Some(degree),
                dimensions: Some(dimensions),
                distance_metric: Some(distance_metric),
                weighted_metric_weights: weighted_weights.as_deref(),
                surface_mode: Some(surface_mode),
                return_se,
                cell: match cell {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                interpolation_vertices,
                boundary_degree_fallback: match boundary_degree_fallback {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        ))?;

        let mut s_builder = builder.adapter(Streaming);
        s_builder = s_builder.chunk_size(chunk_size);
        s_builder = s_builder.overlap(overlap_size);
        s_builder = s_builder.parallel(parallel);
        s_builder = s_builder.merge_strategy(ms);

        if let NotNull(tol) = auto_converge {
            s_builder = s_builder.auto_converge(tol);
        }
        if return_diagnostics {
            s_builder = s_builder.return_diagnostics(true);
        }
        if return_robustness_weights {
            s_builder = s_builder.return_robustness_weights(true);
        }

        let model = map_runtime(s_builder.build())?;
        Ok(Self {
            inner: model,
            fraction,
            iterations,
        })
    }

    fn process_chunk(&mut self, x: &[f64], y: &[f64]) -> Result<List> {
        let result = self.inner.process_chunk(x, y);
        let mut result = map_runtime(result)?;
        result.fraction_used = self.fraction;
        result.iterations_used = Some(self.iterations);
        loess_result_to_list(result)
    }

    fn finalize(&mut self) -> Result<List> {
        let result = self.inner.finalize();
        let mut result = map_runtime(result)?;
        result.fraction_used = self.fraction;
        result.iterations_used = Some(self.iterations);
        loess_result_to_list(result)
    }
}

// Stateful API: OnlineLoess

#[extendr]
pub struct ROnlineLoess {
    inner: fastLoess::internals::adapters::online::ParallelOnlineLoess<f64>,
    fraction: f64,
    iterations: usize,
    dimensions: usize,
    degree: PolynomialDegree,
    distance_metric: DistanceMetric<f64>,
}

#[extendr]
impl ROnlineLoess {
    #[allow(clippy::too_many_arguments)]
    fn new(
        fraction: f64,
        window_capacity: i32,
        min_points: i32,
        iterations: i32,
        weight_function: &str,
        robustness_method: &str,
        scaling_method: &str,
        boundary_policy: &str,
        zero_weight_fallback: &str,
        update_mode: &str,
        auto_converge: Nullable<f64>,
        return_robustness_weights: bool,
        parallel: bool,
        degree: &str,
        dimensions: i32,
        distance_metric: &str,
        weighted_metric_weights: Nullable<Vec<f64>>,
        surface_mode: &str,
        return_se: bool,
        confidence_intervals: Nullable<f64>,
        prediction_intervals: Nullable<f64>,
        cell: Nullable<f64>,
        interpolation_vertices: Nullable<i32>,
        boundary_degree_fallback: Nullable<bool>,
    ) -> Result<Self> {
        let um = map_invalid_arg(shared_parse::parse_update_mode(update_mode))?;
        let weighted_weights = match weighted_metric_weights {
            NotNull(v) => Some(v),
            Null => None,
        };
        let configured_dimensions = require_positive_usize("dimensions", dimensions)?;
        let iterations = require_positive_usize("iterations", iterations)?;
        let window_capacity = require_positive_usize("window_capacity", window_capacity)?;
        let min_points = require_positive_usize("min_points", min_points)?;
        let interpolation_vertices =
            optional_positive_usize("interpolation_vertices", interpolation_vertices)?;
        let (builder, applied) = map_invalid_arg(shared_parse::apply_builder_options(
            LoessBuilder::<f64>::new(),
            shared_parse::BuilderOptionSet {
                fraction: Some(fraction),
                iterations: Some(iterations),
                weight_function: Some(weight_function),
                robustness_method: Some(robustness_method),
                zero_weight_fallback: Some(zero_weight_fallback),
                boundary_policy: Some(boundary_policy),
                scaling_method: Some(scaling_method),
                auto_converge: None,
                return_residuals: false,
                return_robustness_weights: false,
                return_diagnostics: false,
                confidence_intervals: match confidence_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                prediction_intervals: match prediction_intervals {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                parallel: None,
                degree: Some(degree),
                dimensions: Some(configured_dimensions),
                distance_metric: Some(distance_metric),
                weighted_metric_weights: weighted_weights.as_deref(),
                surface_mode: Some(surface_mode),
                return_se,
                cell: match cell {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                interpolation_vertices,
                boundary_degree_fallback: match boundary_degree_fallback {
                    NotNull(v) => Some(v),
                    Null => None,
                },
                cv_fractions: None,
                cv_method: None,
                cv_k: None,
                cv_seed: None,
            },
        ))?;

        let deg = applied.degree.ok_or_else(|| {
            to_r_error(shared_parse::BindingError::invalid_arg(
                shared_parse::required_option_message("degree"),
            ))
        })?;
        let dm = applied.distance_metric.ok_or_else(|| {
            to_r_error(shared_parse::BindingError::invalid_arg(
                shared_parse::required_option_message("distance_metric"),
            ))
        })?;

        let mut o_builder = builder.adapter(Online);
        o_builder = o_builder.window_capacity(window_capacity);
        o_builder = o_builder.min_points(min_points);
        o_builder = o_builder.update_mode(um);
        o_builder = o_builder.parallel(parallel);

        if let NotNull(tol) = auto_converge {
            o_builder = o_builder.auto_converge(tol);
        }
        if return_robustness_weights {
            o_builder = o_builder.return_robustness_weights(true);
        }

        let model = map_runtime(o_builder.build())?;
        Ok(Self {
            inner: model,
            fraction,
            iterations,
            dimensions: configured_dimensions,
            degree: deg,
            distance_metric: dm,
        })
    }

    fn add_points(&mut self, x: &[f64], y: &[f64]) -> Result<List> {
        let metadata = shared_parse::OnlineResultMetadata {
            dimensions: self.dimensions,
            degree: self.degree,
            distance_metric: self.distance_metric.clone(),
            fraction_used: self.fraction,
            iterations_used: Some(self.iterations),
        };

        let result = shared_parse::online_add_points_to_result(x, y, &metadata, |xi_chunk, yi| {
            let output = self.inner.add_point(xi_chunk, yi)?;
            Ok(output.map(|o| o.smoothed))
        })
        .map_err(|e| to_r_error(shared_parse::BindingError::invalid_arg(e)))?;

        loess_result_to_list(result)
    }
}

// Helper: Convert LoessResult to R List

fn loess_result_to_list(result: LoessResult<f64>) -> Result<List> {
    let mut list_items: Vec<(&str, Robj)> = vec![
        ("x", result.x.into_robj()),
        ("y", result.y.into_robj()),
        ("fraction_used", result.fraction_used.into_robj()),
    ];

    if let Some(se) = result.standard_errors {
        list_items.push(("standard_errors", se.into_robj()));
    }
    if let Some(cl) = result.confidence_lower {
        list_items.push(("confidence_lower", cl.into_robj()));
    }
    if let Some(cu) = result.confidence_upper {
        list_items.push(("confidence_upper", cu.into_robj()));
    }
    if let Some(pl) = result.prediction_lower {
        list_items.push(("prediction_lower", pl.into_robj()));
    }
    if let Some(pu) = result.prediction_upper {
        list_items.push(("prediction_upper", pu.into_robj()));
    }
    if let Some(res) = result.residuals {
        list_items.push(("residuals", res.into_robj()));
    }
    if let Some(rw) = result.robustness_weights {
        list_items.push(("robustness_weights", rw.into_robj()));
    }
    if let Some(iters) = result.iterations_used {
        list_items.push(("iterations_used", (iters as i32).into_robj()));
    }
    if let Some(cv) = result.cv_scores {
        list_items.push(("cv_scores", cv.into_robj()));
    }
    if let Some(enp) = result.enp {
        list_items.push(("enp", enp.into_robj()));
    }
    if let Some(th) = result.trace_hat {
        list_items.push(("trace_hat", th.into_robj()));
    }
    if let Some(d1) = result.delta1 {
        list_items.push(("delta1", d1.into_robj()));
    }
    if let Some(d2) = result.delta2 {
        list_items.push(("delta2", d2.into_robj()));
    }
    if let Some(rs) = result.residual_scale {
        list_items.push(("residual_scale", rs.into_robj()));
    }
    if let Some(lev) = result.leverage {
        list_items.push(("leverage", lev.into_robj()));
    }
    list_items.push(("dimensions", (result.dimensions as i32).into_robj()));
    if let Some(diag) = result.diagnostics {
        let diag_list = list!(
            rmse = diag.rmse,
            mae = diag.mae,
            r_squared = diag.r_squared,
            aic = diag.aic.unwrap_or(f64::NAN),
            aicc = diag.aicc.unwrap_or(f64::NAN),
            effective_df = diag.effective_df.unwrap_or(f64::NAN),
            residual_sd = diag.residual_sd
        );
        list_items.push(("diagnostics", diag_list.into_robj()));
    }

    // Build the list manually
    let names: Vec<&str> = list_items.iter().map(|(k, _)| *k).collect();
    let values: Vec<Robj> = list_items.into_iter().map(|(_, v)| v).collect();
    let mut list = List::from_names_and_values(names, values)?;
    list.set_class(&["LoessResult"])?;
    Ok(list)
}

// Module Registration

extendr_module! {
    mod rfastloess;
    impl RLoess;
    impl RStreamingLoess;
    impl ROnlineLoess;
}
