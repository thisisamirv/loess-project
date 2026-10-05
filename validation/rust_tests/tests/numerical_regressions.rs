use loess_rs::internals::api::Batch;
use loess_rs::internals::evaluation::diagnostics::{Diagnostics, DiagnosticsState};
use loess_rs::internals::primitives::policies::{RobustnessMethod, ScalingMethod};
use loess_rs::prelude::*;

fn assert_close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) {
    assert!(actual.is_finite(), "actual value is not finite: {actual}");
    assert!(
        expected.is_finite(),
        "expected value is not finite: {expected}"
    );
    let error = (actual - expected).abs();
    let tolerance = abs_tol + rel_tol * expected.abs();
    assert!(
        error <= tolerance,
        "actual {actual:.17e}, expected {expected:.17e}, error {error:.3e} > {tolerance:.3e}"
    );
}

fn assert_slice_close(actual: &[f64], expected: &[f64], abs_tol: f64, rel_tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert_close(actual_value, expected_value, abs_tol, rel_tol);
        assert!(actual_value.is_finite(), "value[{index}] is not finite");
    }
}

fn gaussian_se_reference(
    x: &[f64],
    y: &[f64],
    y_smooth: &[f64],
    query: f64,
    bandwidth: f64,
    observations: std::ops::Range<usize>,
) -> f64 {
    let (mut sum_w_r2, mut sum_w, mut s1, mut s2) = (0.0, 0.0, 0.0, 0.0);
    let (mut t0, mut t1, mut t2) = (0.0, 0.0, 0.0);
    for index in observations {
        let offset = x[index] - query;
        let u = offset.abs() / bandwidth;
        let weight = (-0.5 * u * u).exp();
        let residual = y[index] - y_smooth[index];
        sum_w_r2 += weight * residual * residual;
        sum_w += weight;
        s1 += weight * offset;
        s2 += weight * offset * offset;
        t0 += weight * weight;
        t1 += weight * weight * offset;
        t2 += weight * weight * offset * offset;
    }

    let determinant = sum_w * s2 - s1 * s1;
    let leverage = (s2 * s2 * t0 - 2.0 * s1 * s2 * t1 + s1 * s1 * t2) / (determinant * determinant);
    let degrees_of_freedom = sum_w - 2.0 + t0 / sum_w;
    (sum_w_r2 / degrees_of_freedom * leverage).sqrt()
}

#[test]
fn gaussian_standard_errors_use_the_full_kernel_neighborhood() {
    let x: Vec<f64> = (0..9).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&value| value * value).collect();
    let fraction = 0.34;
    let result = Loess::new()
        .fraction(fraction)
        .iterations(0)
        .degree("linear")
        .weight_function("gaussian")
        .surface_mode("direct")
        .boundary_policy("noboundary")
        .outputs(["se"])
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let index = 4;
    let bandwidth = 1.0;
    let all_points = gaussian_se_reference(&x, &y, &result.y, x[index], bandwidth, 0..x.len());
    let k_nearest_only = gaussian_se_reference(&x, &y, &result.y, x[index], bandwidth, 3..6);
    assert!((all_points - k_nearest_only).abs() > 1e-3);
    assert_close(
        result.standard_errors.as_ref().unwrap()[index],
        all_points,
        1e-12,
        1e-12,
    );

    let retained = Loess::new()
        .fraction(fraction)
        .iterations(0)
        .degree("linear")
        .weight_function("gaussian")
        .surface_mode("direct")
        .boundary_policy("noboundary")
        .retain_model(true)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    let query = 4.5;
    let prediction = Predict::new()
        .outputs(["se"])
        .build()
        .unwrap()
        .call(&retained, &[query])
        .unwrap();
    let query_all_points = gaussian_se_reference(&x, &y, &retained.y, query, 1.5, 0..x.len());
    assert_close(
        prediction.standard_errors.unwrap()[0],
        query_all_points,
        1e-12,
        1e-12,
    );
}

#[test]
fn even_median_and_mean_are_finite_at_f64_max() {
    let mut mar_values = [f64::MAX; 4];
    let mut mad_values = [f64::MAX; 4];
    let mut mean_values = [f64::MAX; 4];

    assert_eq!(ScalingMethod::MAR.compute(&mut mar_values), f64::MAX);
    assert_eq!(ScalingMethod::MAD.compute(&mut mad_values), 0.0);
    assert_eq!(ScalingMethod::Mean.compute(&mut mean_values), f64::MAX);
}

#[test]
fn bisquare_weights_are_invariant_to_large_finite_residual_scale() {
    let base_residuals = [0.0_f64, 0.0, 4.0, 8.0, 16.0];
    let scaled_residuals: Vec<f64> = base_residuals
        .iter()
        .map(|&residual| residual * 1.0e307)
        .collect();
    let weights_for = |residuals: &[f64]| {
        let mut weights = vec![1.0; residuals.len()];
        let mut scratch = vec![0.0; residuals.len()];
        RobustnessMethod::Bisquare.apply_robustness_weights(
            residuals,
            &mut weights,
            ScalingMethod::MAR,
            &mut scratch,
        );
        weights
    };

    let expected = weights_for(&base_residuals);
    let actual = weights_for(&scaled_residuals);
    assert_slice_close(&actual, &expected, 1e-12, 1e-12);
}

#[test]
fn batch_diagnostics_and_aic_avoid_intermediate_overflow() {
    let scale = 1.0e154_f64;
    let y = [-scale, 0.0, scale];
    let yhat = [0.0; 3];

    assert_close(
        Diagnostics::calculate_rmse(&y, &yhat),
        scale * (2.0_f64 / 3.0).sqrt(),
        1e138,
        1e-12,
    );
    assert_close(
        Diagnostics::calculate_mae(&y, &yhat),
        2.0 * scale / 3.0,
        1e138,
        1e-12,
    );
    assert_close(
        Diagnostics::calculate_r_squared(&y, &yhat),
        0.0,
        1e-12,
        1e-12,
    );

    let residuals = [scale, -scale];
    let expected_aic = 4.0 * scale.ln() + 4.0;
    assert_close(
        Diagnostics::calculate_aic(&residuals, 2.0),
        expected_aic,
        1e-12,
        1e-12,
    );
}

#[test]
fn batch_and_streaming_r_squared_preserve_ulp_variation_at_large_offsets() {
    let base = 9_007_199_254_740_992.0_f64;
    let y = [base, base + 2.0, base + 2.0];
    let yhat = [base; 3];

    assert_close(
        Diagnostics::calculate_r_squared(&y, &yhat),
        -2.0,
        1e-12,
        1e-12,
    );

    let mut state = DiagnosticsState::<f64>::new();
    state.update(&y[..1], &yhat[..1]);
    state.update(&y[1..], &yhat[1..]);
    assert_close(state.finalize().r_squared, -2.0, 1e-12, 1e-12);
}

#[test]
fn streaming_diagnostics_avoid_overflow_in_large_residual_squares() {
    let scale = 1.0e154_f64;
    let y = [-scale, 0.0, scale];
    let yhat = [0.0; 3];
    let mut state = DiagnosticsState::<f64>::new();
    state.update(&y, &yhat);
    let diagnostics = state.finalize();

    assert_close(
        diagnostics.rmse,
        scale * (2.0_f64 / 3.0).sqrt(),
        1e138,
        1e-12,
    );
    assert_close(diagnostics.mae, 2.0 * scale / 3.0, 1e138, 1e-12);
    assert_close(diagnostics.r_squared, 0.0, 1e-12, 1e-12);
    assert_close(diagnostics.residual_sd, scale, 1e138, 1e-12);
}

#[test]
fn custom_weight_scaling_preserves_local_and_all_tied_fits() {
    let x: Vec<f64> = (0..9).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&value| (0.4 * value).sin() + value).collect();
    let fit_with_weights = |x: &[f64], y: &[f64], weights: Vec<f64>| {
        Loess::new()
            .fraction(0.5)
            .iterations(0)
            .custom_weights(weights)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(x, y)
            .unwrap()
            .y
    };

    let unit_weights = fit_with_weights(&x, &y, vec![1.0; x.len()]);
    let large_weights = fit_with_weights(&x, &y, vec![1.0e308; x.len()]);
    assert_slice_close(&large_weights, &unit_weights, 1e-12, 1e-12);

    let tied_x = [3.0_f64; 3];
    let tied_y = [1.0_f64, 2.0, 6.0];
    let tied_unit = fit_with_weights(&tied_x, &tied_y, vec![1.0; tied_x.len()]);
    let tied_large = fit_with_weights(&tied_x, &tied_y, vec![1.0e308; tied_x.len()]);
    assert_slice_close(&tied_large, &tied_unit, 1e-12, 1e-12);
}
