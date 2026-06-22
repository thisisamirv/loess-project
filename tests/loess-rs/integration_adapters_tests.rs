use loess_rs::prelude::*;

// ============================================================================
// Multi-Dimensional Integration Tests
// ============================================================================

#[test]
fn test_batch_2d_quadratic() {
    // 2D data: z = x^2 + y^2
    let x: Vec<f64> = (0..10)
        .flat_map(|i| (0..10).map(move |j| vec![i as f64, j as f64]))
        .flatten()
        .collect();

    let y: Vec<f64> = (0..10)
        .flat_map(|i| (0..10).map(move |j| (i * i + j * j) as f64))
        .collect();

    let result = Loess::new()
        .degree(Quadratic)
        .dimensions(2)
        .fraction(0.5)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert_eq!(result.y.len(), 100);
    assert!(result.y.iter().all(|v| v.is_finite()));
}

#[test]
fn test_streaming_2d_linear_with_residuals() {
    // 2D linear plane: z = 2x + 3y
    let x: Vec<f64> = (0..20)
        .flat_map(|i| {
            (0..5).map(move |j| vec![i as f64, j as f64]) // 100 points total
        })
        .flatten()
        .collect();

    let y: Vec<f64> = (0..20)
        .flat_map(|i| (0..5).map(move |j| (2 * i + 3 * j) as f64))
        .collect();

    let mut processor = Loess::new()
        .degree(Linear)
        .dimensions(2)
        .fraction(0.3)
        .return_residuals()
        .adapter(Streaming)
        .chunk_size(50)
        .overlap(10)
        .build()
        .unwrap();

    let result1 = processor.process_chunk(&x[0..100], &y[0..50]).unwrap();
    let _result2 = processor.process_chunk(&x[100..200], &y[50..100]).unwrap();
    let _final_res = processor.finalize().unwrap();

    assert!(result1.residuals.is_some());
}

#[test]
fn test_online_3d_constant() {
    // 3D data: w = 10 (constant)
    let x: Vec<f64> = (0..50)
        .flat_map(|i| vec![i as f64, (i + 1) as f64, (i + 2) as f64])
        .collect();

    let y: Vec<f64> = vec![10.0; 50];

    let mut model = Loess::new()
        .degree(Constant)
        .dimensions(3)
        .fraction(0.5)
        .adapter(Online)
        .window_capacity(20)
        .build()
        .unwrap();

    for i in 0..50 {
        let xi = &x[i * 3..(i + 1) * 3];
        let yi = y[i];
        let result = model.add_point(xi, yi).unwrap();

        if let Some(res) = result {
            assert!((res.smoothed - 10.0).abs() < 1e-1);
        }
    }
}

// ============================================================================
// High-Degree Polynomial Tests
// ============================================================================

#[test]
fn test_batch_cubic_1d() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v.powi(3)).collect();

    let result = Loess::new()
        .degree(Cubic)
        .fraction(0.6)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(result.y.len() == 20);
}

#[test]
fn test_batch_quartic_1d() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v.powi(4)).collect();

    let result = Loess::new()
        .degree(Quartic)
        .fraction(0.7)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(result.y.len() == 20);
}

// ============================================================================
// Robustness Tests across Adapters
// ============================================================================

#[test]
fn test_streaming_robust_outliers() {
    let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.clone();
    // Add outliers
    y[50] = 1000.0;
    y[51] = -1000.0;

    let mut processor = Loess::new()
        .degree(Linear)
        .robustness_method(Bisquare)
        .iterations(2)
        .return_robustness_weights()
        .adapter(Streaming)
        .chunk_size(40)
        .overlap(10)
        .build()
        .unwrap();

    let _ = processor.process_chunk(&x, &y).unwrap();
    let fin = processor.finalize().unwrap();

    // Robustness weights check
    if let Some(rw) = fin.robustness_weights {
        // Just verify we got some weights back
        assert!(!rw.is_empty());
    }
}

#[test]
fn test_online_robust_outliers() {
    let mut model = Loess::new()
        .robustness_method(Bisquare)
        .iterations(1)
        .adapter(Online)
        .window_capacity(10)
        .build()
        .unwrap();

    // Feed data
    for i in 0..20 {
        let val = if i == 10 { 1000.0 } else { i as f64 };
        model.add_point(&[i as f64], val).unwrap();
    }
}

// ============================================================================
// Precision Tests (f32)
// ============================================================================

#[test]
fn test_batch_f32_2d() {
    let x: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let y: Vec<f32> = vec![0.0, 1.0, 0.0];

    let result = Loess::<f32>::new()
        .degree(Linear)
        .dimensions(2)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    assert!(result.y.iter().all(|v: &f32| v.is_finite()));
}
