#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLoess::prelude::*;
use loess_rs::internals::engine::predict::PredictOptions;

#[test]
fn test_predict_pass_consistency() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let new_x = vec![10.5, 25.25, 40.75];

    let seq_res = Loess::new()
        .fraction(0.4)
        .iterations(0)
        .parallel(false)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let par_res = Loess::new()
        .fraction(0.4)
        .iterations(0)
        .parallel(true)
        .retain_model(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let options = PredictOptions {
        return_se: true,
        return_derivative: true,
        confidence_level: Some(0.95),
        prediction_level: Some(0.95),
        ..PredictOptions::default()
    };

    let seq_out = seq_res.predict(&new_x, &options).expect("serial predict");
    let par_out = par_res.predict(&new_x, &options).expect("parallel predict");

    for i in 0..new_x.len() {
        assert_abs_diff_eq!(seq_out.y[i], par_out.y[i], epsilon = 1e-12);
        assert_abs_diff_eq!(
            seq_out.standard_errors.as_ref().unwrap()[i],
            par_out.standard_errors.as_ref().unwrap()[i],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            seq_out.confidence_lower.as_ref().unwrap()[i],
            par_out.confidence_lower.as_ref().unwrap()[i],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            seq_out.prediction_upper.as_ref().unwrap()[i],
            par_out.prediction_upper.as_ref().unwrap()[i],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            seq_out.derivative.as_ref().unwrap()[i],
            par_out.derivative.as_ref().unwrap()[i],
            epsilon = 1e-12
        );
    }
}
