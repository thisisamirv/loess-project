#![cfg(feature = "dev")]
use approx::assert_abs_diff_eq;
use fastLoess::prelude::*;

#[test]
fn test_weighted_cv_matches_serial_for_all_methods_and_dimensions() {
    for dimensions in [1, 2] {
        let mut predictors = Vec::new();
        let mut observations = Vec::new();
        let mut weights = Vec::new();
        for index in (0..25).rev() {
            let coordinate = if dimensions == 1 {
                index as f64
            } else {
                (index % 5) as f64
            };
            predictors.push(coordinate);
            if dimensions == 2 {
                predictors.push((index / 5) as f64);
            }
            observations
                .push(coordinate + (index as f64).sin() + if index == 11 { 80.0 } else { 0.0 });
            weights.push(if index == 11 {
                0.0
            } else {
                0.5 + (index % 3) as f64
            });
        }
        for method in ["kfold", "loocv"] {
            let fit = |parallel| {
                Loess::new()
                    .iterations(0)
                    .dimensions(dimensions)
                    .surface_mode("direct")
                    .boundary_policy("noboundary")
                    .custom_weights(weights.clone())
                    .cv(CVBuilder::new()
                        .method(method)
                        .k(4)
                        .fraction(vec![0.45, 0.8]))
                    .seed(17)
                    .parallel(parallel)
                    .build()
                    .unwrap()
                    .fit(&predictors, &observations)
                    .unwrap()
            };
            let serial = fit(false);
            let parallel = fit(true);
            assert_eq!(parallel.fraction_used, serial.fraction_used);
            for (&actual, &expected) in parallel
                .cv_scores
                .as_ref()
                .unwrap()
                .iter()
                .zip(serial.cv_scores.as_ref().unwrap())
            {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-8);
            }
        }
    }
}

#[test]
fn test_grouped_cross_validation_parallel() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi / 5.0).sin()).collect();
    let fractions = vec![0.3, 0.5];
    let options: fastLoess::CVOptions<f64> = CVBuilder::new()
        .method("kfold")
        .k(3)
        .fraction(fractions.clone());
    let result = Loess::new()
        .iterations(0)
        .cv(options)
        .seed(42)
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(result.cv_scores.unwrap().len(), fractions.len());

    let duplicate = Loess::new()
        .cv(CVBuilder::new().fraction(vec![0.5]))
        .cv(CVBuilder::new().method("kfold").fraction(fractions))
        .build();
    assert!(matches!(
        duplicate,
        Err(LoessError::DuplicateParameter { parameter: "cv" })
    ));
}

#[test]
fn test_grouped_intervals_shared_seed_parallel() {
    let x: Vec<f64> = (0..20).map(|index| index as f64 / 19.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(index, value)| value.sin() + (index % 3) as f64 * 0.1)
        .collect();
    let fit = |seed_first, parallel| {
        let cv = CVBuilder::new().k(3).fraction(vec![0.4, 0.7]);
        let builder = Loess::new()
            .surface_mode("direct")
            .iterations(0)
            .parallel(parallel)
            .retain_model(true)
            .intervals(
                IntervalsBuilder::new()
                    .confidence(0.8)
                    .prediction(0.99)
                    .bootstrap(8),
            );
        let builder = if seed_first {
            builder.seed(7).cv(cv)
        } else {
            builder.cv(cv).seed(7)
        };
        builder.build().unwrap().fit(&x, &y).unwrap()
    };
    let first = fit(true, true);
    let second = fit(false, true);
    assert_eq!(first.cv_scores, second.cv_scores);
    assert_eq!(first.standard_errors, second.standard_errors);
    assert_eq!(first.prediction_lower, second.prediction_lower);
    let serial = Loess::new()
        .surface_mode("direct")
        .iterations(0)
        .parallel(false)
        .fraction(first.fraction_used)
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.8)
                .prediction(0.99)
                .bootstrap(8),
        )
        .seed(7)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    for (&parallel, &sequential) in first
        .standard_errors
        .as_ref()
        .unwrap()
        .iter()
        .zip(serial.standard_errors.as_ref().unwrap())
    {
        assert_abs_diff_eq!(parallel, sequential, epsilon = 1e-8);
    }
    let prediction = Predict::new()
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.8)
                .prediction(0.99)
                .bootstrap(8),
        )
        .seed(7)
        .build()
        .unwrap()
        .call(&first, &[0.25, 0.75])
        .unwrap();
    assert_eq!(prediction.standard_errors.unwrap().len(), 2);
}

#[test]
fn test_grouped_intervals_streaming_online_and_validation() {
    let x: Vec<f64> = (0..20).map(|index| index as f64 / 19.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(index, value)| value.sin() + (index % 3) as f64 * 0.1)
        .collect();
    let mut stream = StreamingLoess::new()
        .surface_mode("direct")
        .iterations(0)
        .chunk_size(20)
        .overlap(3)
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.8)
                .prediction(0.99)
                .bootstrap(8),
        )
        .seed(7)
        .build()
        .unwrap();
    let chunk = stream.process_chunk(&x, &y).unwrap();
    let tail = stream.finalize().unwrap();
    assert_eq!(
        chunk.standard_errors.unwrap().len() + tail.standard_errors.unwrap().len(),
        y.len()
    );
    let mut online = OnlineLoess::new()
        .surface_mode("direct")
        .iterations(0)
        .update_mode("full")
        .window_capacity(20)
        .min_points(3)
        .intervals(
            IntervalsBuilder::new()
                .confidence(0.8)
                .prediction(0.99)
                .bootstrap(8),
        )
        .seed(7)
        .build()
        .unwrap();
    let mut latest = None;
    for (&point, &response) in x.iter().zip(&y) {
        latest = online.add_point(&[point], response).unwrap();
    }
    assert!(latest.unwrap().standard_error.is_some());
    assert!(matches!(
        StreamingLoess::new()
            .intervals(IntervalsBuilder::new().bootstrap(1))
            .build(),
        Err(LoessError::InvalidBootstrapSamples(1))
    ));
    assert!(matches!(
        OnlineLoess::new()
            .update_mode("incremental")
            .intervals(IntervalsBuilder::new().bootstrap(8))
            .build(),
        Err(LoessError::StandardErrorRequiresFullUpdateMode)
    ));
}

#[test] // Parallel CV produces inconsistent results compared to Sequential
fn test_parallel_cross_validation() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| (xi / 5.0).sin()).collect();

    let fractions = vec![0.1, 0.2, 0.3, 0.5];

    // Sequential CV with Direct surface mode
    let seq_res = Loess::new()
        .iterations(0)
        .surface_mode("direct")
        .cv(fastLoess::prelude::CVBuilder::new()
            .method("kfold")
            .k(5)
            .fraction(fractions.clone()))
        .parallel(false)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Parallel CV with Direct surface mode
    let par_res = Loess::new()
        .iterations(0)
        .surface_mode("direct")
        .cv(fastLoess::prelude::CVBuilder::new()
            .method("kfold")
            .k(5)
            .fraction(fractions.clone()))
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
        .cv(fastLoess::prelude::CVBuilder::new()
            .method("loocv")
            .fraction(fractions))
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
        .cv(fastLoess::prelude::CVBuilder::new()
            .method("kfold")
            .k(10)
            .fraction(fractions))
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
        .cv(fastLoess::prelude::CVBuilder::new()
            .method("kfold")
            .k(3)
            .fraction(fractions))
        .parallel(true)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(!res.y.is_empty());
    assert!(res.fraction_used > 0.0);
}
