#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLoess::prelude::*;

#[test] // Parallel CV produces inconsistent results compared to Sequential
fn test_parallel_cross_validation() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi / 5.0).sin()).collect();

    let fractions = vec![0.1, 0.2, 0.3, 0.5];

    // Sequential CV with Direct surface mode
    let seq_res = Loess::new()
        .iterations(0)
        .surface_mode(Direct)
        .cross_validate(KFold(5, &fractions))
        .adapter(Batch)
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Parallel CV with Direct surface mode
    let par_res = Loess::new()
        .iterations(0)
        .surface_mode(Direct)
        .cross_validate(KFold(5, &fractions))
        .adapter(Batch)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    println!("Parallel best fraction: {}", par_res.fraction_used);
    println!("Sequential best fraction: {}", seq_res.fraction_used);

    if let (Some(ps), Some(ss)) = (&par_res.cv_scores, &seq_res.cv_scores) {
        println!("Parallel scores: {:?}", ps);
        println!("Sequential scores: {:?}", ss);
    }

    // Results should be identical
    assert_abs_diff_eq!(
        par_res.fraction_used,
        seq_res.fraction_used,
        epsilon = 1e-10
    );
    assert_abs_diff_eq!(par_res.y[0], seq_res.y[0], epsilon = 1e-10);
}
