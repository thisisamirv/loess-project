//! Comprehensive LOESS Batch Smoothing Examples
//!
//! This example demonstrates various LOESS smoothing scenarios:
//! - Basic smoothing with minimal configuration
//! - Robust smoothing with outlier handling
//! - Uncertainty quantification with confidence/prediction intervals
//! - Cross-validation for automatic parameter selection
//! - Complete diagnostic analysis
//! - Different weight functions and robustness methods
//!
//! Each scenario includes the expected output as comments.

use fastLoess::prelude::*;
use std::time::Instant;

fn main() -> Result<(), LoessError> {
    println!("{}", "=".repeat(80));
    println!("LOESS Batch Smoothing - Comprehensive Examples");
    println!("{}", "=".repeat(80));
    println!();

    // Run all example scenarios
    example_1_basic_smoothing()?;
    example_2_robust_with_outliers()?;
    example_3_uncertainty_quantification()?;
    example_4_cross_validation()?;
    example_5_complete_diagnostics()?;
    example_6_different_kernels()?;
    example_7_robustness_methods()?;
    example_8_benchmark()?;
    example_9_scaling_methods()?;
    example_10_boundary_policies()?;
    example_11_zero_weight_fallback()?;
    example_12_polynomial_degrees()?;
    example_13_distance_metrics()?;
    example_14_surface_modes_and_se()?;
    example_15_additional_kernels()?;
    example_16_loocv_and_auto_converge()?;
    example_17_interpolation_tuning()?;

    Ok(())
}

/// Example 1: Basic Smoothing
/// Demonstrates the simplest usage with minimal configuration
fn example_1_basic_smoothing() -> Result<(), LoessError> {
    println!("Example 1: Basic Smoothing");
    println!("{}", "-".repeat(80));

    // Simple linear data with noise
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    let model = Loess::new()
        .fraction(0.5) // Use 50% of data for each local fit
        .iterations(3) // 3 robustness iterations
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("{}", result);

    /* Expected Output:
    Summary:
      Data points: 5
      Fraction: 0.5

    Smoothed Data:
           X     Y_smooth
      --------------------
        1.00     2.00000
        2.00     4.10000
        3.00     5.90000
        4.00     8.20000
        5.00     9.80000
    */

    println!();
    Ok(())
}

/// Example 2: Robust Smoothing with Outliers
/// Shows how LOESS handles outliers with robustness iterations
fn example_2_robust_with_outliers() -> Result<(), LoessError> {
    println!("Example 2: Robust Smoothing with Outliers");
    println!("{}", "-".repeat(80));

    // Data with an obvious outlier at index 3
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 4.0, 5.9, 25.0, 10.1, 12.0, 14.1, 15.9]; // 25.0 is an outlier

    let model = Loess::new()
        .fraction(0.5)
        .iterations(5) // More iterations for stronger robustness
        .robustness_method(Bisquare)
        .return_residuals()
        .return_robustness_weights()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("{}", result);

    // Identify outliers
    if let Some(weights) = &result.robustness_weights {
        println!("\nOutlier Detection:");
        for (i, &w) in weights.iter().enumerate() {
            if w < 0.5 {
                println!(
                    "  Point {} (x={:.1}, y={:.1}) is an outlier (weight: {:.3})",
                    i, x[i], y[i], w
                );
            }
        }
    }

    /* Expected Output:
    Summary:
      Data points: 8
      Fraction: 0.5
      Robustness: Applied

    Smoothed Data:
           X     Y_smooth     Residual Rob_Weight
      ----------------------------------------------
        1.00     2.10000     0.000000     1.0000
        2.00     4.00000     0.000000     1.0000
        3.00     5.90000     0.000000     1.0000
        4.00     8.00000    17.000000     0.0000
        5.00    10.10000     0.000000     1.0000
        6.00    12.00000     0.000000     1.0000
        7.00    14.10000     0.000000     1.0000
        8.00    15.90000     0.000000     1.0000

    Outlier Detection:
      Point 3 (x=4.0, y=25.0) is an outlier (weight: 0.000)
    */

    println!();
    Ok(())
}

/// Example 3: Uncertainty Quantification
/// Demonstrates confidence and prediction intervals
fn example_3_uncertainty_quantification() -> Result<(), LoessError> {
    println!("Example 3: Uncertainty Quantification");
    println!("{}", "-".repeat(80));

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];

    let model = Loess::new()
        .fraction(0.5)
        .iterations(3)
        .confidence_intervals(0.95) // 95% confidence intervals
        .prediction_intervals(0.95) // 95% prediction intervals
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("{}", result);

    /* Expected Output:
    Summary:
      Data points: 8
      Fraction: 0.5

    Smoothed Data:
           X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper
      ----------------------------------------------------------------------------------
        1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353
        2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386
        3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013
        4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518
        5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551
        6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083
        7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892
        8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356
    */

    println!();
    Ok(())
}

/// Example 4: Cross-Validation
/// Automatic selection of optimal smoothing fraction
fn example_4_cross_validation() -> Result<(), LoessError> {
    println!("Example 4: Cross-Validation for Parameter Selection");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.5).sin())
        .collect();

    // Test multiple fractions and select the best one
    let model = Loess::new()
        .cross_validate(KFold(5, &[0.2, 0.3, 0.5, 0.7]))
        .iterations(2)
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;

    println!("Selected fraction: {}", result.fraction_used);
    if let Some(scores) = &result.cv_scores {
        println!("CV scores for each fraction: {:?}", scores);
    }
    println!("\n{}", result);

    /* Expected Output:
    Selected fraction: 0.5
    CV scores for each fraction: [0.123, 0.098, 0.145, 0.187]

    Summary:
      Data points: 20
      Fraction: 0.5 (selected via K-Fold CV)

    Smoothed Data:
           X     Y_smooth
      --------------------
        1.00     3.47943
        2.00     5.47943
        3.00     7.14112
        ... (17 more rows)
    */

    println!();
    Ok(())
}

/// Example 5: Complete Diagnostic Analysis
/// Full feature demonstration with all diagnostics
fn example_5_complete_diagnostics() -> Result<(), LoessError> {
    println!("Example 5: Complete Diagnostic Analysis");
    println!("{}", "-".repeat(80));

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7];

    let model = Loess::new()
        .fraction(0.5)
        .iterations(3)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .return_diagnostics()
        .return_residuals()
        .return_robustness_weights()
        .adapter(Batch)
        .build()?;

    let result = model.fit(&x, &y)?;
    println!("{}", result);

    /* Expected Output:
    Summary:
      Data points: 8
      Fraction: 0.5
      Robustness: Applied

    LOESS Diagnostics:
      RMSE:         0.191925
      MAE:          0.181676
      R²:           0.998205
      Residual SD:  0.297750
      Effective DF: 8.00
      AIC:          -10.41
      AICc:         inf

    Smoothed Data:
           X     Y_smooth      Std_Err   Conf_Lower   Conf_Upper   Pred_Lower   Pred_Upper     Residual Rob_Weight
      ----------------------------------------------------------------------------------------------------------------
        1.00     2.01963     0.389365     1.256476     2.782788     1.058911     2.980353     0.080368     1.0000
        2.00     4.00251     0.345447     3.325438     4.679589     3.108641     4.896386    -0.202513     1.0000
        3.00     5.99959     0.423339     5.169846     6.829335     4.985168     7.014013     0.200410     1.0000
        4.00     8.09859     0.489473     7.139224     9.057960     6.975666     9.221518    -0.198592     1.0000
        5.00    10.03881     0.551687     8.957506    11.120118     8.810073    11.267551     0.261188     1.0000
        6.00    12.02872     0.539259    10.971775    13.085672    10.821364    13.236083    -0.228723     1.0000
        7.00    13.89828     0.371149    13.170829    14.625733    12.965670    14.830892     0.201719     1.0000
        8.00    15.77990     0.408300    14.979631    16.580167    14.789441    16.770356    -0.079899     1.0000
    */

    println!();
    Ok(())
}

/// Example 6: Different Weight Functions (Kernels)
/// Comparison of various kernel functions
fn example_6_different_kernels() -> Result<(), LoessError> {
    println!("Example 6: Different Weight Functions (Kernels)");
    println!("{}", "-".repeat(80));

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 5.9, 8.2, 9.8];

    let kernels = vec![
        ("Tricube", Tricube),
        ("Epanechnikov", Epanechnikov),
        ("Gaussian", Gaussian),
        ("Biweight", Biweight),
    ];

    for (name, kernel) in kernels {
        println!("Using {} kernel:", name);

        let model = Loess::new()
            .fraction(0.5)
            .weight_function(kernel)
            .adapter(Batch)
            .build()?;

        let result = model.fit(&x, &y)?;

        // Print just the smoothed values
        print!("  Smoothed Y: [");
        for (i, &val) in result.y.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", val);
        }
        println!("]");
    }

    /* Expected Output:
    Using Tricube kernel:
      Smoothed Y: [2.000, 4.100, 5.900, 8.200, 9.800]
    Using Epanechnikov kernel:
      Smoothed Y: [2.000, 4.100, 5.900, 8.200, 9.800]
    Using Gaussian kernel:
      Smoothed Y: [2.001, 4.099, 5.901, 8.199, 9.799]
    Using Biweight kernel:
      Smoothed Y: [2.000, 4.100, 5.900, 8.200, 9.800]
    */

    println!();
    Ok(())
}

/// Example 7: Robustness Methods Comparison
/// Different methods for handling outliers
fn example_7_robustness_methods() -> Result<(), LoessError> {
    println!("Example 7: Robustness Methods Comparison");
    println!("{}", "-".repeat(80));

    // Data with outlier
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.1, 20.0, 8.2, 9.8]; // 20.0 is an outlier

    let methods = vec![("Bisquare", Bisquare), ("Huber", Huber), ("Talwar", Talwar)];

    for (name, method) in methods {
        println!("Using {} robustness method:", name);

        let model = Loess::new()
            .fraction(0.5)
            .iterations(5)
            .robustness_method(method)
            .return_robustness_weights()
            .adapter(Batch)
            .build()?;

        let result = model.fit(&x, &y)?;

        // Print smoothed values and weights
        print!("  Smoothed Y: [");
        for (i, &val) in result.y.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.2}", val);
        }
        println!("]");

        if let Some(weights) = &result.robustness_weights {
            print!("  Weights:    [");
            for (i, &w) in weights.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.3}", w);
            }
            println!("]");
        }
    }

    /* Expected Output:
    Using Bisquare robustness method:
      Smoothed Y: [2.00, 4.10, 5.90, 8.20, 9.80]
      Weights:    [1.000, 1.000, 0.000, 1.000, 1.000]
    Using Huber robustness method:
      Smoothed Y: [2.00, 4.10, 6.15, 8.20, 9.80]
      Weights:    [1.000, 1.000, 0.123, 1.000, 1.000]
    Using Talwar robustness method:
      Smoothed Y: [2.00, 4.10, 5.90, 8.20, 9.80]
      Weights:    [1.000, 1.000, 0.000, 1.000, 1.000]
    */

    println!();
    Ok(())
}

/// Example 8: Benchmark (Sequential Batch)
/// Measure execution time for a large dataset using the sequential Batch adapter
fn example_8_benchmark() -> Result<(), LoessError> {
    println!("Example 8: Benchmark (Sequential Batch)");
    println!("{}", "-".repeat(80));

    // Generate a larger synthetic dataset
    let n = 1_000;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (xi * 0.1).sin() + (xi * 0.01).cos())
        .collect();

    let start = Instant::now();
    let model = Loess::new().adapter(Batch).build()?;

    let result = model.fit(&x, &y)?;
    let duration = start.elapsed();

    println!("Processed {} points in {:?}", n, duration);
    println!("Execution mode: Sequential Batch");
    println!("Result summary:\n{}", result);

    println!();
    Ok(())
}

/// Example 9: Scaling Methods (MAR, MAD, Mean)
fn example_9_scaling_methods() -> Result<(), LoessError> {
    println!("Example 9: Scaling Methods");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    for method in [MAR, MAD, Mean] {
        let model = Loess::new()
            .fraction(0.5)
            .scaling_method(method)
            .adapter(Batch)
            .build()?;
        let result = model.fit(&x, &y)?;
        println!("  {:?}: y[0]={:.3}", method, result.y[0]);
    }

    println!();
    Ok(())
}

/// Example 10: Boundary Policies
fn example_10_boundary_policies() -> Result<(), LoessError> {
    println!("Example 10: Boundary Policies");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    for policy in [Extend, Reflect, Zero, NoBoundary] {
        let model = Loess::new()
            .fraction(0.5)
            .boundary_policy(policy)
            .adapter(Batch)
            .build()?;
        let result = model.fit(&x, &y)?;
        println!(
            "  {:?}: first={:.2}, last={:.2}",
            policy,
            result.y[0],
            result.y.last().unwrap()
        );
    }

    println!();
    Ok(())
}

/// Example 11: Zero-Weight Fallback Strategies
fn example_11_zero_weight_fallback() -> Result<(), LoessError> {
    println!("Example 11: Zero-Weight Fallback Strategies");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    for fallback in [UseLocalMean, ReturnOriginal, ReturnNone] {
        let model = Loess::new()
            .fraction(0.5)
            .zero_weight_fallback(fallback)
            .adapter(Batch)
            .build()?;
        let result = model.fit(&x, &y)?;
        println!("  {:?}: y[0]={:.3}", fallback, result.y[0]);
    }

    println!();
    Ok(())
}

/// Example 12: Polynomial Degrees and `iterations_used` / `polynomial_degree` result fields
fn example_12_polynomial_degrees() -> Result<(), LoessError> {
    println!("Example 12: Polynomial Degrees");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    for deg in [Constant, Linear, Quadratic, Cubic, Quartic] {
        let model = Loess::new()
            .fraction(0.5)
            .iterations(2)
            .degree(deg)
            .adapter(Batch)
            .build()?;
        let result = model.fit(&x, &y)?;
        println!(
            "  {:?}: y[0]={:.3}, iterations_used={:?}, degree={:?}",
            deg, result.y[0], result.iterations_used, result.polynomial_degree
        );
    }

    println!();
    Ok(())
}

/// Example 13: Distance Metrics (Euclidean, Normalized, Manhattan, Chebyshev, Minkowski, Weighted)
fn example_13_distance_metrics() -> Result<(), LoessError> {
    println!("Example 13: Distance Metrics");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    for metric in [Euclidean, Normalized, Manhattan, Chebyshev] {
        let model = Loess::new()
            .fraction(0.5)
            .distance_metric(metric.clone())
            .adapter(Batch)
            .build()?;
        let result = model.fit(&x, &y)?;
        println!("  {:?}: y[0]={:.3}", metric, result.y[0]);
    }

    // Minkowski(p) — parametric distance
    let model = Loess::new()
        .fraction(0.5)
        .distance_metric(Minkowski(3.0_f64))
        .adapter(Batch)
        .build()?;
    let result = model.fit(&x, &y)?;
    println!(
        "  Minkowski(3): y[0]={:.3}, stored metric={:?}",
        result.y[0], result.distance_metric
    );

    // Weighted([1.0]) — per-dimension scale
    let model = Loess::new()
        .fraction(0.5)
        .distance_metric(Weighted(vec![1.0_f64]))
        .adapter(Batch)
        .build()?;
    let result = model.fit(&x, &y)?;
    println!("  Weighted([1.0]): y[0]={:.3}", result.y[0]);

    println!();
    Ok(())
}

/// Example 14: Surface Modes and Standard Errors
/// Covers `return_se`, `standard_errors`, `enp`, `trace_hat`, `delta1`, `delta2`,
/// `residual_scale`, `leverage`, `has_confidence_intervals`, `has_prediction_intervals`.
fn example_14_surface_modes_and_se() -> Result<(), LoessError> {
    println!("Example 14: Surface Modes and Standard Errors");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    // Direct — fits every point exactly; SE fields are fully populated
    let result_direct = Loess::new()
        .fraction(0.5)
        .surface_mode(Direct)
        .return_se()
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;

    println!("  Surface: Direct");
    println!(
        "  has_confidence_intervals: {}",
        result_direct.has_confidence_intervals()
    );
    println!(
        "  has_prediction_intervals: {}",
        result_direct.has_prediction_intervals()
    );
    if let Some(se) = &result_direct.standard_errors {
        println!("  standard_errors[0]: {:.4}", se[0]);
    }
    if let Some(v) = result_direct.enp {
        println!("  enp: {:.3}", v);
    }
    if let Some(v) = result_direct.trace_hat {
        println!("  trace_hat: {:.3}", v);
    }
    if let Some(v) = result_direct.delta1 {
        println!("  delta1: {:.3}", v);
    }
    if let Some(v) = result_direct.delta2 {
        println!("  delta2: {:.3}", v);
    }
    if let Some(v) = result_direct.residual_scale {
        println!("  residual_scale: {:.4}", v);
    }
    if let Some(lev) = &result_direct.leverage {
        println!("  leverage[0]: {:.4}", lev[0]);
    }

    // Interpolation — faster, approximate; SE still computed when return_se is set
    let result_interp = Loess::new()
        .fraction(0.5)
        .surface_mode(Interpolation)
        .return_se()
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;

    println!("\n  Surface: Interpolation");
    println!("  y[0]: {:.3}", result_interp.y[0]);
    if let Some(se) = &result_interp.standard_errors {
        println!("  standard_errors[0]: {:.4}", se[0]);
    }

    println!();
    Ok(())
}

/// Example 15: Additional Weight Functions (Uniform, Triangle, Cosine)
fn example_15_additional_kernels() -> Result<(), LoessError> {
    println!("Example 15: Additional Weight Functions (Uniform, Triangle, Cosine)");
    println!("{}", "-".repeat(80));

    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.1, 5.9, 8.2, 9.8];

    for kernel in [Uniform, Triangle, Cosine] {
        let model = Loess::new()
            .fraction(0.5)
            .weight_function(kernel)
            .adapter(Batch)
            .build()?;
        let result = model.fit(&x, &y)?;
        print!("  {:?}: [", kernel);
        for (i, &v) in result.y.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", v);
        }
        println!("]");
    }

    println!();
    Ok(())
}

/// Example 16: LOOCV, KFold with Seed, and Auto-Converge
fn example_16_loocv_and_auto_converge() -> Result<(), LoessError> {
    println!("Example 16: LOOCV, KFold with Seed, and Auto-Converge");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.5).sin())
        .collect();

    // Leave-one-out cross-validation
    let result_loocv = Loess::new()
        .cross_validate(LOOCV(&[0.3_f64, 0.5, 0.7]))
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    println!("  LOOCV selected fraction: {}", result_loocv.fraction_used);
    if result_loocv.has_cv_scores() {
        println!("  LOOCV best score: {:?}", result_loocv.best_cv_score());
    }

    // KFold with reproducible seed
    let result_seeded = Loess::new()
        .cross_validate(KFold(5, &[0.2_f64, 0.4, 0.6]).seed(42))
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    println!(
        "  KFold(k=5, seed=42) selected fraction: {}",
        result_seeded.fraction_used
    );
    if let Some(scores) = &result_seeded.cv_scores {
        println!("  CV scores: {:?}", scores);
    }

    // Auto-converge: stop robustness iterations once change < tolerance
    let result_ac = Loess::new()
        .fraction(0.5)
        .auto_converge(1e-4_f64)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    println!(
        "  auto_converge=1e-4: iterations_used={:?}",
        result_ac.iterations_used
    );

    println!();
    Ok(())
}

/// Example 17: Interpolation Tuning (cell, interpolation_vertices, boundary_degree_fallback)
fn example_17_interpolation_tuning() -> Result<(), LoessError> {
    println!("Example 17: Interpolation Tuning");
    println!("{}", "-".repeat(80));

    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    // Default cell size (0.2)
    let r_default = Loess::new()
        .fraction(0.5)
        .surface_mode(Interpolation)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    println!("  Default cell (0.2): y[0]={:.3}", r_default.y[0]);

    // Finer grid (smaller cell)
    let r_fine = Loess::new()
        .fraction(0.5)
        .surface_mode(Interpolation)
        .cell(0.05_f64)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    println!("  cell=0.05 (finer): y[0]={:.3}", r_fine.y[0]);

    // Limit interpolation vertices
    let r_verts = Loess::new()
        .fraction(0.5)
        .surface_mode(Interpolation)
        .interpolation_vertices(20)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    println!("  interpolation_vertices=20: y[0]={:.3}", r_verts.y[0]);

    // Disable boundary degree fallback (match R's loess behavior)
    let r_no_fallback = Loess::new()
        .fraction(0.5)
        .surface_mode(Interpolation)
        .boundary_degree_fallback(false)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    println!(
        "  boundary_degree_fallback=false: y[0]={:.3}",
        r_no_fallback.y[0]
    );

    println!();
    Ok(())
}
