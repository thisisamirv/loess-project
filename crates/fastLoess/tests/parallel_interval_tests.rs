#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLoess::prelude::*;
use ndarray::Array1;

#[test] // Parallel intervals produce inconsistent standard errors compared to Sequential
fn test_parallel_interval_estimation() {
    // Generate sample data
    let n = 100;
    let x_vec: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y_vec: Vec<f64> = x_vec
        .iter()
        .map(|&xi| xi.sin() + 0.1 * (xi * 10.0).sin())
        .collect();

    let x = Array1::from_vec(x_vec);
    let y = Array1::from_vec(y_vec);

    // Run Sequential Intervals
    let seq_model = Loess::new()
        .fraction(0.3)
        .iterations(2)
        .intervals(
            fastLoess::IntervalsBuilder::new()
                .confidence(0.95)
                .prediction(0.95),
        )
        .surface_mode("direct")
        .parallel(false)
        .build()
        .unwrap();

    let seq_result = seq_model.fit(&x, &y).unwrap();

    // Run Parallel Intervals
    let par_model = Loess::new()
        .fraction(0.3)
        .iterations(2)
        .intervals(
            fastLoess::IntervalsBuilder::new()
                .confidence(0.95)
                .prediction(0.95),
        )
        .surface_mode("direct")
        .parallel(true)
        .build()
        .unwrap();

    let par_result = par_model.fit(&x, &y).unwrap();

    // Compare results
    assert_eq!(par_result.y, seq_result.y);

    let par_std_err = par_result.standard_errors.as_ref().unwrap();
    let seq_std_err = seq_result.standard_errors.as_ref().unwrap();

    for (p, s) in par_std_err.iter().zip(seq_std_err.iter()) {
        assert_abs_diff_eq!(p, s, epsilon = 1e-10);
    }

    let par_conf_lower = par_result.confidence_lower.as_ref().unwrap();
    let seq_conf_lower = seq_result.confidence_lower.as_ref().unwrap();
    for (p, s) in par_conf_lower.iter().zip(seq_conf_lower.iter()) {
        assert_abs_diff_eq!(p, s, epsilon = 1e-10);
    }

    let par_pred_lower = par_result.prediction_lower.as_ref().unwrap();
    let seq_pred_lower = seq_result.prediction_lower.as_ref().unwrap();
    for (p, s) in par_pred_lower.iter().zip(seq_pred_lower.iter()) {
        assert_abs_diff_eq!(p, s, epsilon = 1e-10);
    }

    println!("Parallel and Sequential Intervals match exactly!");
}

#[test]
fn test_parallel_interval_keeps_se_for_downweighted_observation() {
    let x = Array1::from_iter((0..21).map(|index| index as f64));
    let y = Array1::from_iter(x.iter().enumerate().map(|(index, &value)| {
        if index == 10 {
            2.0 * value + 100.0
        } else {
            2.0 * value
        }
    }));

    let result = Loess::new()
        .fraction(0.5)
        .iterations(3)
        .intervals(fastLoess::IntervalsBuilder::new().confidence(0.95))
        .return_robustness_weights()
        .surface_mode("direct")
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let robustness = result.robustness_weights.as_ref().unwrap();
    let standard_errors = result.standard_errors.as_ref().unwrap();
    assert!(robustness[10] < 1e-6, "outlier should be downweighted");
    assert!(standard_errors[10].is_finite());
    assert!(
        standard_errors[10] > 0.0,
        "downweighted point should retain SE"
    );
}
