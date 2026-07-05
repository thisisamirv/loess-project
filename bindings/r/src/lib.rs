//! R bindings for fastLoess.
//!
//! Provides R access to the fastLoess Rust library via extendr.
//!
//! @srrstats {G1.0} Documentation of core R-to-Rust interface.
//! @srrstats {G1.1} Implementation of thin R wrapper for statistical algorithms.

#![allow(non_snake_case)]

use extendr_api::prelude::*;

type Result<T> = std::result::Result<T, Error>;

use fastLoess::internals::api::{
    BoundaryPolicy, DistanceMetric, MergeStrategy, PolynomialDegree, RobustnessMethod,
    ScalingMethod::{self, Mean, MAD, MAR},
    SurfaceMode, UpdateMode, WeightFunction, ZeroWeightFallback,
};
use fastLoess::prelude::{
    Batch, KFold, Loess as LoessBuilder, LoessResult, Online, Streaming, LOOCV,
};

// Helper Functions

// Parse weight function from string
fn parse_weight_function(name: &str) -> Result<WeightFunction> {
    match name.to_lowercase().as_str() {
        "tricube" => Ok(WeightFunction::Tricube),
        "epanechnikov" => Ok(WeightFunction::Epanechnikov),
        "gaussian" => Ok(WeightFunction::Gaussian),
        "uniform" | "boxcar" => Ok(WeightFunction::Uniform),
        "biweight" | "bisquare" => Ok(WeightFunction::Biweight),
        "triangle" | "triangular" => Ok(WeightFunction::Triangle),
        "cosine" => Ok(WeightFunction::Cosine),
        _ => Err(Error::Other(format!(
            "Unknown weight function: {}. Valid options: tricube, epanechnikov, gaussian, uniform, biweight, triangle, cosine",
            name
        ))),
    }
}

// Parse robustness method from string
fn parse_robustness_method(name: &str) -> Result<RobustnessMethod> {
    match name.to_lowercase().as_str() {
        "bisquare" | "biweight" => Ok(RobustnessMethod::Bisquare),
        "huber" => Ok(RobustnessMethod::Huber),
        "talwar" => Ok(RobustnessMethod::Talwar),
        _ => Err(Error::Other(format!(
            "Unknown robustness method: {}. Valid options: bisquare, huber, talwar",
            name
        ))),
    }
}

// Parse zero weight fallback from string
fn parse_zero_weight_fallback(name: &str) -> Result<ZeroWeightFallback> {
    match name.to_lowercase().as_str() {
        "use_local_mean" | "local_mean" | "mean" => Ok(ZeroWeightFallback::UseLocalMean),
        "return_original" | "original" => Ok(ZeroWeightFallback::ReturnOriginal),
        "return_none" | "none" | "nan" => Ok(ZeroWeightFallback::ReturnNone),
        _ => Err(Error::Other(format!(
            "Unknown zero weight fallback: {}. Valid options: use_local_mean, return_original, return_none",
            name
        ))),
    }
}

// Parse boundary policy from string
fn parse_boundary_policy(name: &str) -> Result<BoundaryPolicy> {
    match name.to_lowercase().as_str() {
        "extend" | "pad" => Ok(BoundaryPolicy::Extend),
        "reflect" | "mirror" => Ok(BoundaryPolicy::Reflect),
        "zero" | "none" => Ok(BoundaryPolicy::Zero),
        "noboundary" => Ok(BoundaryPolicy::NoBoundary),
        _ => Err(Error::Other(format!(
            "Unknown boundary policy: {}. Valid options: extend, reflect, zero, noboundary",
            name
        ))),
    }
}

// Parse scaling method from string
fn parse_scaling_method(name: &str) -> Result<ScalingMethod> {
    match name.to_lowercase().as_str() {
        "mad" => Ok(MAD),
        "mar" => Ok(MAR),
        "mean" => Ok(Mean),
        _ => Err(Error::Other(format!(
            "Unknown scaling method: {}. Valid options: mad, mar, mean",
            name
        ))),
    }
}

// Parse polynomial degree from string
fn parse_polynomial_degree(name: &str) -> Result<PolynomialDegree> {
    match name.to_lowercase().as_str() {
        "constant" | "0" => Ok(PolynomialDegree::Constant),
        "linear" | "1" => Ok(PolynomialDegree::Linear),
        "quadratic" | "2" => Ok(PolynomialDegree::Quadratic),
        "cubic" | "3" => Ok(PolynomialDegree::Cubic),
        "quartic" | "4" => Ok(PolynomialDegree::Quartic),
        _ => Err(Error::Other(format!("Unknown polynomial degree: {}", name))),
    }
}

// Parse distance metric from string
fn parse_distance_metric(name: &str) -> Result<DistanceMetric<f64>> {
    let lower = name.to_lowercase();
    // Handle "minkowski:p" inline format
    if let Some(p_str) = lower.strip_prefix("minkowski:") {
        let p: f64 = p_str
            .parse()
            .map_err(|_| Error::Other(format!("Invalid Minkowski p value: {}", p_str)))?;
        return Ok(DistanceMetric::Minkowski(p));
    }
    match lower.as_str() {
        "normalized" | "norm" => Ok(DistanceMetric::Normalized),
        "euclidean" | "euclid" => Ok(DistanceMetric::Euclidean),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        "chebyshev" | "linf" => Ok(DistanceMetric::Chebyshev),
        "minkowski" => Ok(DistanceMetric::Minkowski(2.0)),
        _ => Err(Error::Other(format!(
            "Unknown distance metric: {}. Valid options: normalized, euclidean, manhattan, chebyshev, minkowski",
            name
        ))),
    }
}

// Parse surface mode from string
fn parse_surface_mode(name: &str) -> Result<SurfaceMode> {
    match name.to_lowercase().as_str() {
        "interpolation" | "interp" => Ok(SurfaceMode::Interpolation),
        "direct" => Ok(SurfaceMode::Direct),
        _ => Err(Error::Other(format!("Unknown surface mode: {}", name))),
    }
}

// Parse update mode from string
fn parse_update_mode(name: &str) -> Result<UpdateMode> {
    match name.to_lowercase().as_str() {
        "full" | "resmooth" => Ok(UpdateMode::Full),
        "incremental" | "single" => Ok(UpdateMode::Incremental),
        _ => Err(Error::Other(format!(
            "Unknown update mode: {}. Valid options: full, incremental",
            name
        ))),
    }
}

// Parse merge strategy from string
fn parse_merge_strategy(name: &str) -> Result<MergeStrategy> {
    match name.to_lowercase().as_str() {
        "average" | "mean" => Ok(MergeStrategy::Average),
        "weighted_average" | "weighted" => Ok(MergeStrategy::WeightedAverage),
        "take_first" | "first" => Ok(MergeStrategy::TakeFirst),
        "take_last" | "last" => Ok(MergeStrategy::TakeLast),
        _ => Err(Error::Other(format!(
            "Unknown merge strategy: {}. Valid options: average, weighted_average, take_first, take_last",
            name
        ))),
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
        surface_mode: &str,
        return_se: bool,
    ) -> Result<Self> {
        let wf = parse_weight_function(weight_function)?;
        let rm = parse_robustness_method(robustness_method)?;
        let sm = parse_scaling_method(scaling_method)?;
        let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
        let bp = parse_boundary_policy(boundary_policy)?;

        let mut builder = LoessBuilder::<f64>::new();
        builder = builder.fraction(fraction);
        builder = builder.iterations(iterations as usize);
        builder = builder.weight_function(wf);
        builder = builder.robustness_method(rm);
        builder = builder.scaling_method(sm);
        builder = builder.zero_weight_fallback(zwf);
        builder = builder.boundary_policy(bp);

        if let NotNull(cl) = confidence_intervals {
            builder = builder.confidence_intervals(cl);
        }
        if let NotNull(pl) = prediction_intervals {
            builder = builder.prediction_intervals(pl);
        }
        if return_diagnostics {
            builder = builder.return_diagnostics();
        }
        if return_residuals {
            builder = builder.return_residuals();
        }
        if return_robustness_weights {
            builder = builder.return_robustness_weights();
        }
        if let NotNull(tol) = auto_converge {
            builder = builder.auto_converge(tol);
        }

        let deg = parse_polynomial_degree(degree)?;
        let dm = parse_distance_metric(distance_metric)?;
        let surf = parse_surface_mode(surface_mode)?;
        builder = builder.degree(deg);
        builder = builder.dimensions(dimensions as usize);
        builder = builder.distance_metric(dm);
        builder = builder.surface_mode(surf);
        if return_se {
            builder = builder.return_se();
        }

        // Cross-validation if fractions are provided
        if let NotNull(fractions) = cv_fractions {
            match cv_method.to_lowercase().as_str() {
                "simple" | "loo" | "loocv" | "leave_one_out" => {
                    builder = builder.cross_validate(LOOCV(&fractions));
                }
                "kfold" | "k_fold" | "k-fold" => {
                    builder = builder.cross_validate(KFold(cv_k as usize, &fractions));
                }
                _ => {
                    return Err(Error::Other(format!(
                        "Unknown CV method: {}. Valid options: loocv, kfold",
                        cv_method
                    )));
                }
            }
        }

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
        let result = builder
            .adapter(Batch)
            .parallel(self.parallel)
            .build()
            .map_err(|e| Error::Other(e.to_string()))?
            .fit(x, y)
            .map_err(|e| Error::Other(e.to_string()))?;

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
        surface_mode: &str,
        return_se: bool,
    ) -> Result<Self> {
        let chunk_size = chunk_size as usize;
        let overlap_size = match overlap {
            NotNull(o) => o as usize,
            Null => (chunk_size / 10).min(chunk_size.saturating_sub(10)).max(1),
        };

        let wf = parse_weight_function(weight_function)?;
        let rm = parse_robustness_method(robustness_method)?;
        let sm = parse_scaling_method(scaling_method)?;
        let bp = parse_boundary_policy(boundary_policy)?;
        let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
        let ms = parse_merge_strategy(merge_strategy)?;

        let mut builder = LoessBuilder::<f64>::new();
        builder = builder.fraction(fraction);
        builder = builder.iterations(iterations as usize);
        builder = builder.weight_function(wf);
        builder = builder.robustness_method(rm);
        builder = builder.scaling_method(sm);
        builder = builder.boundary_policy(bp);
        builder = builder.zero_weight_fallback(zwf);

        let deg = parse_polynomial_degree(degree)?;
        let dm = parse_distance_metric(distance_metric)?;
        let surf = parse_surface_mode(surface_mode)?;
        builder = builder.degree(deg);
        builder = builder.dimensions(dimensions as usize);
        builder = builder.distance_metric(dm);
        builder = builder.surface_mode(surf);
        if return_se {
            builder = builder.return_se();
        }
        if return_residuals {
            builder = builder.return_residuals();
        }

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

        let model = s_builder.build().map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self {
            inner: model,
            fraction,
            iterations: iterations as usize,
        })
    }

    fn process_chunk(&mut self, x: &[f64], y: &[f64]) -> Result<List> {
        let mut result = self
            .inner
            .process_chunk(x, y)
            .map_err(|e| Error::Other(e.to_string()))?;
        result.fraction_used = self.fraction;
        result.iterations_used = Some(self.iterations);
        loess_result_to_list(result)
    }

    fn finalize(&mut self) -> Result<List> {
        let mut result = self
            .inner
            .finalize()
            .map_err(|e| Error::Other(e.to_string()))?;
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
        surface_mode: &str,
        return_se: bool,
    ) -> Result<Self> {
        let wf = parse_weight_function(weight_function)?;
        let rm = parse_robustness_method(robustness_method)?;
        let sm = parse_scaling_method(scaling_method)?;
        let bp = parse_boundary_policy(boundary_policy)?;
        let zwf = parse_zero_weight_fallback(zero_weight_fallback)?;
        let um = parse_update_mode(update_mode)?;

        let mut builder = LoessBuilder::<f64>::new();
        builder = builder.fraction(fraction);
        builder = builder.iterations(iterations as usize);
        builder = builder.weight_function(wf);
        builder = builder.robustness_method(rm);
        builder = builder.scaling_method(sm);
        builder = builder.boundary_policy(bp);
        builder = builder.zero_weight_fallback(zwf);

        let deg = parse_polynomial_degree(degree)?;
        let dm = parse_distance_metric(distance_metric)?;
        let surf = parse_surface_mode(surface_mode)?;
        let configured_dimensions = dimensions as usize;
        builder = builder.degree(deg);
        builder = builder.dimensions(configured_dimensions);
        builder = builder.distance_metric(dm.clone());
        builder = builder.surface_mode(surf);
        if return_se {
            builder = builder.return_se();
        }

        let mut o_builder = builder.adapter(Online);
        o_builder = o_builder.window_capacity(window_capacity as usize);
        o_builder = o_builder.min_points(min_points as usize);
        o_builder = o_builder.update_mode(um);
        o_builder = o_builder.parallel(parallel);

        if let NotNull(tol) = auto_converge {
            o_builder = o_builder.auto_converge(tol);
        }
        if return_robustness_weights {
            o_builder = o_builder.return_robustness_weights(true);
        }

        let model = o_builder.build().map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self {
            inner: model,
            fraction,
            iterations: iterations as usize,
            dimensions: configured_dimensions,
            degree: deg,
            distance_metric: dm,
        })
    }

    fn add_points(&mut self, x: &[f64], y: &[f64]) -> Result<List> {
        let mut smoothed = Vec::with_capacity(y.len());
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let output = self
                .inner
                .add_point(std::slice::from_ref(&xi), yi)
                .map_err(|e| Error::Other(e.to_string()))?;
            smoothed.push(output.as_ref().map_or(yi, |o| o.smoothed));
        }

        let result = LoessResult {
            x: x.to_vec(),
            y: smoothed,
            dimensions: self.dimensions,
            distance_metric: self.distance_metric.clone(),
            polynomial_degree: self.degree,
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals: None,
            robustness_weights: None,
            diagnostics: None,
            iterations_used: Some(self.iterations),
            fraction_used: self.fraction,
            cv_scores: None,
            enp: None,
            trace_hat: None,
            delta1: None,
            delta2: None,
            residual_scale: None,
            leverage: None,
        };

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
