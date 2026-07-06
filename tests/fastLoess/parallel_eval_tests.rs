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
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(fractions.clone())
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
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(fractions.clone())
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

/// Runs parallel LOOCV, exercising the `CVKind::LOOCV` arm inside
/// `evaluate_fraction_cv`.
#[test]
fn test_loocv_cross_validation_parallel() {
    let n = 20;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0 + 1.0).collect();
    let fractions = vec![0.3, 0.5, 0.7];

    let res = Loess::new()
        .cv_method("loocv")
        .cv_fractions(fractions)
        .adapter(Batch)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(!res.y.is_empty());
    assert!(res.fraction_used > 0.0);
}

/// KFold where fold_size = n / k < 2 triggers the fold-size guard inside
/// `evaluate_fraction_cv`.
#[test]
fn test_kfold_fold_size_less_than_2() {
    // n=10, k=10 => fold_size = 10/10 = 1 < 2
    let n = 10;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 1.5).collect();
    let fractions = vec![0.3, 0.5];

    let res = Loess::new()
        .cv_method("kfold")
        .cv_k(10)
        .cv_fractions(fractions)
        .adapter(Batch)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(!res.y.is_empty());
}

/// Multi-dimensional (2-D) KFold CV exercises the n-D prediction branch
/// inside `evaluate_fraction_cv`.
#[test]
fn test_multidim_kfold_cv_parallel() {
    let n = 40;
    // 2-D input: each observation has (x0, x1)
    let x: Vec<f64> = (0..n)
        .flat_map(|i| {
            let xi = i as f64 / n as f64;
            vec![xi, (i % 5) as f64 / 5.0]
        })
        .collect();
    let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 3.0).collect();
    let fractions = vec![0.4, 0.6];

    let res = Loess::new()
        .dimensions(2)
        .cv_method("kfold")
        .cv_k(3)
        .cv_fractions(fractions)
        .adapter(Batch)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(!res.y.is_empty());
    assert!(res.fraction_used > 0.0);
}
