//! Combined Visualization Examples for LOESS
//!
//! This script runs multiple scenarios to generate CSV data for visualization.
//! It covers:
//! 1. Fraction Comparison (Effect of bandwidth)
//! 2. Intervals Comparison (Confidence vs Prediction)
//! 3. Robustness Comparison (With vs Without robustness iterations)
//! 4. LOESS Concept (Local weighting visualization)
//! 5. Kernel Comparison
//! 6. Robustness Method Comparison
//! 7. Boundary Policy Comparison
//! 8. Gap Handling
//! 9. Cross-Validation Comparison
//! 10. Scaling Method Comparison
//! 11. Zero Weight Fallback Comparison
//! 12. Streaming Adapter Comparison
//! 13. Online Adapter Comparison
//! 14. Auto-Convergence Comparison

use fastLoess::prelude::*;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running All Visualization Examples...");
    println!("=====================================");
    println!();

    // Ensure output directory exists
    let output_dir = "../output/visual/";
    std::fs::create_dir_all(output_dir)?;
    println!("Output directory: {}", output_dir);
    println!();

    run_fraction_comparison()?;
    println!();

    run_intervals_comparison()?;
    println!();

    run_robustness_comparison()?;
    println!();

    run_loess_concept()?;
    println!();

    run_kernel_comparison()?;
    println!();

    run_robust_method_comparison()?;
    println!();

    run_boundary_policy_comparison()?;
    println!();

    run_gap_handling()?;
    println!();

    run_cv_comparison()?;
    println!();

    run_surface_mode_comparison()?;
    println!();

    run_scaling_method_comparison()?;
    println!();

    run_zero_weight_fallback_comparison()?;
    println!();

    run_streaming_comparison()?;
    println!();

    run_online_comparison()?;
    println!();

    run_auto_converge_comparison()?;
    println!();

    println!("All examples completed successfully.");
    Ok(())
}

/// 2. Fraction Comparison
fn run_fraction_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Generate data with multiple features
    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = 0.5 * t + 0.3 * t * t - 0.02 * t * t * t
            + 2.0 * (t * 1.5).sin()
            + 0.5 * (t * 5.0).sin();
        let noise = 0.3 * ((i as f64 * 7.0).sin() + (i as f64 * 13.0).cos()) / 2.0;

        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("2. Fraction Comparison");
    println!("----------------------");

    let fractions = [0.2, 0.5, 0.9];
    let mut results = Vec::new();

    for &frac in &fractions {
        let result = Loess::new()
            .fraction(frac)
            .iterations(2)
            .delta(0.0) // Direct equivalent
            .boundary_policy(Reflect)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();

        let mut mse = 0.0;
        for i in 0..n {
            let error = result.y[i] - y_true[i];
            mse += error * error;
        }
        let rmse = (mse / n as f64).sqrt();

        println!("Fraction {:.1}: RMSE = {:.6}", frac, rmse);
        results.push(result);
    }

    let path = "../output/visual/fraction_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_frac_0.2,y_frac_0.5,y_frac_0.9")?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], results[0].y[i], results[1].y[i], results[2].y[i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 3. Intervals Comparison
fn run_intervals_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 8.0 - 1.0;
        let signal = 3.0 + 2.0 * t - 0.3 * t * t + 1.5 * (t * 0.8).sin();
        let noise = 0.5 * ((i as f64 * 7.0).sin() + (i as f64 * 13.0).cos());

        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("3. Intervals Comparison");
    println!("-----------------------");

    // Confidence
    let result_conf = Loess::new()
        .fraction(0.3)
        .iterations(2)
        .confidence_intervals(0.95)
        .delta(0.0)
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Prediction
    let result_pred = Loess::new()
        .fraction(0.3)
        .iterations(2)
        .prediction_intervals(0.95)
        .delta(0.0)
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let conf_lower = result_conf.confidence_lower.as_ref().unwrap();
    let conf_upper = result_conf.confidence_upper.as_ref().unwrap();
    let pred_lower = result_pred.prediction_lower.as_ref().unwrap();
    let pred_upper = result_pred.prediction_upper.as_ref().unwrap();

    let avg_conf_width: f64 = conf_upper
        .iter()
        .zip(conf_lower.iter())
        .map(|(u, l)| u - l)
        .sum::<f64>()
        / n as f64;
    let avg_pred_width: f64 = pred_upper
        .iter()
        .zip(pred_lower.iter())
        .map(|(u, l)| u - l)
        .sum::<f64>()
        / n as f64;

    println!("Avg Confidence Width: {:.3}", avg_conf_width);
    println!("Avg Prediction Width: {:.3}", avg_pred_width);
    println!(
        "Ratio (Pred/Conf):    {:.2}x",
        avg_pred_width / avg_conf_width
    );

    let path = "../output/visual/intervals_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_smooth,conf_lower,conf_upper,pred_lower,pred_upper"
    )?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y[i],
            result_conf.y[i],
            conf_lower[i],
            conf_upper[i],
            pred_lower[i],
            pred_upper[i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 4. Robustness Comparison
fn run_robustness_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = 3.0 * (t * 0.8).sin();
        let mut value = signal + 0.5 * ((i as f64 * 17.0).sin() + (i as f64 * 3.0).cos());

        // Add outliers
        if t <= 4.0 {
            let pseudo_rand = ((i as f64 * 1337.0).sin() * 43758.5453).fract().abs();
            if pseudo_rand > 0.85 {
                value += 10.0 + pseudo_rand * 10.0;
            }
        }

        x.push(t);
        y_true.push(signal);
        y.push(value);
    }

    println!("4. Robustness Comparison");
    println!("------------------------");

    // Non-Robust (0 iterations)
    let result_non_robust = Loess::new()
        .fraction(0.25)
        .iterations(0)
        .delta(0.0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Robust (6 iterations)
    let result_robust = Loess::new()
        .fraction(0.25)
        .iterations(6)
        .delta(0.0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let mut mse_nr = 0.0;
    let mut mse_r = 0.0;
    for i in 0..n {
        let err_nr = result_non_robust.y[i] - y_true[i];
        let err_r = result_robust.y[i] - y_true[i];
        mse_nr += err_nr * err_nr;
        mse_r += err_r * err_r;
    }
    let rmse_nr = (mse_nr / n as f64).sqrt();
    let rmse_r = (mse_r / n as f64).sqrt();

    println!("RMSE (Non-Robust): {:.4}", rmse_nr);
    println!("RMSE (Robust):     {:.4}", rmse_r);
    println!("Improvement:       {:.2}x", rmse_nr / rmse_r);

    let path = "../output/visual/robustness_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_non_robust,y_robust")?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], result_non_robust.y[i], result_robust.y[i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 5. LOESS Concept
fn run_loess_concept() -> Result<(), Box<dyn std::error::Error>> {
    let n = 80;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 2.0 * std::f64::consts::PI;
        let signal = t.sin();
        let noise = 0.3 * ((i as f64 * 7.0).sin() * (i as f64 * 3.0).cos());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    let fraction = 0.35;
    let result = Loess::new()
        .fraction(fraction)
        .iterations(0)
        .delta(0.0)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Focus point visualization
    let focus_idx = 35;
    let x0 = x[focus_idx];

    // Manual neighbor finding and weighting for visualization
    let k = (fraction * n as f64).ceil() as usize;
    let mut distances: Vec<(usize, f64)> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| (i, (xi - x0).abs()))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let max_dist = distances[k - 1].1;

    let mut weights = vec![0.0; n];
    for i in 0..n {
        let dist = (x[i] - x0).abs();
        if dist < max_dist {
            let u = dist / max_dist;
            weights[i] = (1.0 - u * u * u).powi(3);
        }
    }

    // Manual Weighted Least Squares output (simplified linear)
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wx2 = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxy = 0.0;

    for i in 0..n {
        if weights[i] > 0.0 {
            let w = weights[i];
            let dx = x[i] - x0;
            let dx2 = dx * dx;
            sum_w += w;
            sum_wx += w * dx;
            sum_wx2 += w * dx2;
            sum_wy += w * y[i];
            sum_wxy += w * dx * y[i];
        }
    }

    // Linear fit y = a + b*dx
    let det = sum_w * sum_wx2 - sum_wx * sum_wx;
    let a = (sum_wy * sum_wx2 - sum_wx * sum_wxy) / det;
    let b = (sum_w * sum_wxy - sum_wx * sum_wy) / det;

    println!("5. LOESS Concept");
    println!("----------------");
    println!("Focus point: x = {:.2} (Index {})", x0, focus_idx);
    println!("Local Fit: a={:.3}, b={:.3}", a, b);

    let path = "../output/visual/loess_concept.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_noisy,y_smooth,weight,y_local_fit_x0,is_focus")?;

    for i in 0..n {
        let is_focus = if i == focus_idx { 1 } else { 0 };
        let dx = x[i] - x0;
        let local_val = if weights[i] > 0.0 {
            a + b * dx
        } else {
            f64::NAN
        };
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y[i], result.y[i], weights[i], local_val, is_focus
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 7. Kernel Comparison
fn run_kernel_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 6.0;
        let signal = (t).sin() + 0.5 * (t * 3.0).cos();
        let noise = 0.2 * ((i as f64 * 11.0).sin());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("7. Kernel Comparison");
    println!("--------------------");

    let kernels = [
        Tricube,
        Gaussian,
        Uniform,
        Cosine,
        Epanechnikov,
        Biweight,
        Triangle,
    ];
    let mut results = Vec::new();

    for &kernel in &kernels {
        let result = Loess::new()
            .weight_function(kernel)
            .fraction(0.3)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Kernel processed: {}", kernel.name());
    }

    let path = "../output/visual/kernel_comparison.csv";
    let mut file = File::create(path)?;
    write!(file, "x,y_true,y_noisy")?;
    for kernel in &kernels {
        write!(file, ",y_{}", kernel.name().to_lowercase())?;
    }
    writeln!(file)?;

    for i in 0..n {
        write!(file, "{},{},{}", x[i], y_true[i], y[i])?;
        for result in &results {
            write!(file, ",{}", result.y[i])?;
        }
        writeln!(file)?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 8. Robustness Method Comparison
fn run_robust_method_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = (t * 0.5).sin() * 5.0;
        let mut value = signal + 0.3 * (i as f64 * 5.0).cos();

        // Add extreme outliers
        if i % 20 == 0 {
            value += 15.0;
        } else if i % 20 == 10 {
            value -= 15.0;
        }

        x.push(t);
        y_true.push(signal);
        y.push(value);
    }

    println!("8. Robustness Method Comparison");
    println!("-------------------------------");

    let methods = [Bisquare, Huber, Talwar];
    let mut results = Vec::new();

    for &method in &methods {
        let result = Loess::new()
            .robustness_method(method)
            .iterations(5)
            .fraction(0.2)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Method processed: {:?}", method);
    }

    let path = "../output/visual/robust_method_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_bisquare,y_huber,y_talwar")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], results[0].y[i], results[1].y[i], results[2].y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 9. Boundary Policy Comparison
fn run_boundary_policy_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Data with strong trend at boundaries
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let signal = (3.0 * t).exp(); // Exponential growth
        let noise = 0.1 * (i as f64 * 10.0).sin();
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("9. Boundary Policy Comparison");
    println!("-----------------------------");

    let policies = [NoBoundary, Extend, Reflect];
    let mut results = Vec::new();

    for &policy in &policies {
        let result = Loess::new()
            .boundary_policy(policy)
            .fraction(0.4) // Larger fraction highlights boundary bias
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Policy processed: {:?}", policy);
    }

    let path = "../output/visual/boundary_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_none,y_extend,y_reflect")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], results[0].y[i], results[1].y[i], results[2].y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 11. Gap Handling
fn run_gap_handling() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        // Create a large gap in the middle
        if t > 4.0 && t < 7.0 {
            continue;
        }
        let signal = (t).sin();
        let noise = 0.1 * (i as f64 * 7.0).cos();
        x.push(t);
        y.push(signal + noise);
    }

    println!("11. Gap Handling");
    println!("----------------");

    let result = Loess::new()
        .fraction(0.3)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let path = "../output/visual/gap_handling.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_noisy,y_smooth")?;
    for i in 0..x.len() {
        writeln!(file, "{},{},{}", x[i], y[i], result.y[i])?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 12. Cross-Validation Comparison
fn run_cv_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Highly noisy signal on a small scale to make bandwidth selection critical
    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 5.0;
        let signal = (t * 2.5).sin();
        let noise = 0.6 * ((i as f64 * 17.0).sin() * (i as f64 * 13.0).cos());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("12. Cross-Validation Comparison");
    println!("-------------------------------");

    let candidate_fractions = [
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8,
    ];

    // 1. LOOCV
    let loocv_result = Loess::new()
        .cross_validate(LOOCV(&candidate_fractions))
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  LOOCV Best Fraction: {}", loocv_result.fraction_used);

    // 2. K-Fold (5 folds)
    let kfold_result = Loess::new()
        .cross_validate(KFold(5, &candidate_fractions).seed(42))
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  5-Fold Best Fraction: {}", kfold_result.fraction_used);

    // 3. No CV (Fixed bad fraction - too small)
    let fixed_result = Loess::new()
        .fraction(0.1) // Overfitting deliberately
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  No CV Fraction (Fixed): 0.1");

    // Export scores comparison
    let scores_path = "../output/visual/cv_scores.csv";
    let mut score_file = File::create(scores_path)?;
    writeln!(score_file, "fraction,loocv_rmse,kfold_rmse")?;
    let loocv_scores = loocv_result.cv_scores.as_ref().unwrap();
    let kfold_scores = kfold_result.cv_scores.as_ref().unwrap();
    for i in 0..candidate_fractions.len() {
        writeln!(
            score_file,
            "{},{},{}",
            candidate_fractions[i], loocv_scores[i], kfold_scores[i]
        )?;
    }
    println!("CV Scores exported to {}", scores_path);

    // Export fits comparison
    let fits_path = "../output/visual/cv_fits.csv";
    let mut fit_file = File::create(fits_path)?;
    writeln!(fit_file, "x,y_true,y_noisy,y_loocv,y_kfold,y_fixed")?;
    for i in 0..n {
        writeln!(
            fit_file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], loocv_result.y[i], kfold_result.y[i], fixed_result.y[i]
        )?;
    }
    println!("CV Fits exported to {}", fits_path);

    Ok(())
}

/// 13. Surface Mode Comparison
fn run_surface_mode_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = (t * 1.5).sin() + 0.5 * (t * 4.0).cos();
        let noise = 0.2 * ((i as f64 * 11.0).sin());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("13. Surface Mode Comparison (Direct vs Delta)");
    println!("-------------------------------------------");

    let result_direct = Loess::new()
        .delta(0.0) // Direct evaluation
        .fraction(0.2)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result_interp = Loess::new()
        .delta(0.1) // Allow interpolation
        .fraction(0.2)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let path = "../output/visual/surface_mode_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_direct,y_interpolation")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], result_direct.y[i], result_interp.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 14. Scaling Method Comparison
fn run_scaling_method_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 5.0;
        let signal = (t * 0.8).sin() * 2.0;
        let mut value = signal + 0.2 * (i as f64 * 5.0).cos();

        // Add asymmetric outliers
        if i % 15 == 0 {
            value += 8.0; // Large positive outliers
        }

        x.push(t);
        y_true.push(signal);
        y.push(value);
    }

    println!("14. Scaling Method Comparison");
    println!("-----------------------------");

    let result_mad = Loess::new()
        .scaling_method(MAD)
        .iterations(5)
        .fraction(0.3)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result_mar = Loess::new()
        .scaling_method(MAR)
        .iterations(5)
        .fraction(0.3)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let path = "../output/visual/scaling_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_mad,y_mar")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], result_mad.y[i], result_mar.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 15. Zero Weight Fallback Comparison
fn run_zero_weight_fallback_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        let signal = 10.0 + (t * 0.1).sin();
        let mut value = signal;

        // One massive outlier that will cause zero-weight residual
        if i == 50 {
            value = 100.0;
        }

        x.push(t);
        y.push(value);
    }

    println!("15. Zero Weight Fallback Comparison");
    println!("-----------------------------------");

    let result_mean = Loess::new()
        .zero_weight_fallback(UseLocalMean)
        .iterations(3)
        .fraction(0.1)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result_original = Loess::new()
        .zero_weight_fallback(ReturnOriginal)
        .iterations(3)
        .fraction(0.1)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let path = "../output/visual/zero_weight_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_noisy,y_mean,y_original")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{}",
            x[i], y[i], result_mean.y[i], result_original.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 16. Streaming Comparison
fn run_streaming_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = (t).sin();
        let noise = 0.2 * (i as f64 * 7.0).cos();
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("16. Streaming Comparison (Strategies)");
    println!("-----------------------------------");

    let strategies = [WeightedAverage, Average, TakeFirst];
    let mut results = Vec::new();

    for &strat in &strategies {
        let mut adapter = Loess::new()
            .adapter(Streaming)
            .chunk_size(50)
            .overlap(10)
            .merge_strategy(strat)
            .fraction(0.3)
            .build()
            .unwrap();

        // Process in chunks
        let mut final_y = Vec::new();
        for chunk_idx in 0..4 {
            let start = chunk_idx * 50;
            let end = start + 50;
            let result = adapter
                .process_chunk(&x[start..end], &y[start..end])
                .unwrap();
            final_y.extend(result.y);
        }
        let result = adapter.finalize().unwrap();
        final_y.extend(result.y);

        results.push(final_y);
        println!("  Strategy processed: {:?}", strat);
    }

    let path = "../output/visual/streaming_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_weighted,y_average,y_first")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], results[0][i], results[1][i], results[2][i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 17. Online Comparison
fn run_online_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 500;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        let mut signal = (t * 0.05).sin();
        if t > 250.0 {
            signal += 2.0; // Sudden shift
        }
        let noise = 0.5 * (i as f64 * 1.3).sin();
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("17. Online Comparison (Windows)");
    println!("------------------------------");

    // 1. Small window (more responsive, less smooth)
    let mut adapter_small = Loess::new()
        .adapter(Online)
        .window_capacity(50)
        .min_points(10)
        .update_mode(Incremental)
        .fraction(0.5)
        .build()
        .unwrap();

    // 2. Large window (smoother, slower to adapt, includes robustness)
    let mut adapter_large = Loess::new()
        .adapter(Online)
        .window_capacity(200)
        .min_points(10)
        .update_mode(Full)
        .fraction(0.3)
        .iterations(3)
        .build()
        .unwrap();

    let mut y_small = Vec::<f64>::with_capacity(n);
    let mut y_large = Vec::<f64>::with_capacity(n);

    for i in 0..n {
        let x_slice = &x[i..i + 1];
        let y_slice = &y[i..i + 1];
        let r_small = adapter_small.add_points(x_slice, y_slice).unwrap();
        let r_large = adapter_large.add_points(x_slice, y_slice).unwrap();
        y_small.push(r_small[0].as_ref().map(|o| o.smoothed).unwrap_or(y[i]));
        y_large.push(r_large[0].as_ref().map(|o| o.smoothed).unwrap_or(y[i]));
    }

    let path = "../output/visual/online_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_small_window,y_large_window")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], y_small[i], y_large[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 18. Auto-Convergence Comparison
fn run_auto_converge_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        let signal = (t * 0.1).sin();
        let noise = 0.2 * (i as f64 * 7.0).cos();
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("18. Auto-Convergence Comparison");
    println!("-----------------------------");

    let mut file = File::create("../output/visual/auto_converge_comparison.csv")?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_batch_off,y_batch_on,iter_batch_off,iter_batch_on,y_stream_off,y_stream_on,iter_stream_off,iter_stream_on,y_online_off,y_online_on,iter_online_off,iter_online_on"
    )?;

    // --- Batch ---
    let res_b_off = Loess::new()
        .iterations(10)
        .fraction(0.3)
        .adapter(Batch)
        .build()
        .map_err(|e| format!("Failed to build Batch processor (off): {:?}", e))?
        .fit(&x, &y)
        .map_err(|e| format!("Failed to fit Batch (off): {:?}", e))?;
    let res_b_on = Loess::new()
        .iterations(10)
        .fraction(0.3)
        .auto_converge(0.001)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // --- Streaming ---
    let mut ad_s_off = Loess::new()
        .iterations(10)
        .fraction(0.3)
        .adapter(Streaming)
        .chunk_size(50)
        .overlap(10)
        .merge_strategy(WeightedAverage)
        .build()
        .unwrap();
    let mut ad_s_on = Loess::new()
        .iterations(10)
        .fraction(0.3)
        .auto_converge(0.001)
        .adapter(Streaming)
        .chunk_size(50)
        .overlap(10)
        .merge_strategy(WeightedAverage)
        .build()
        .unwrap();

    let mut y_s_off = Vec::<f64>::new();
    let mut y_s_on = Vec::<f64>::new();
    let mut it_s_off = Vec::<usize>::new();
    let mut it_s_on = Vec::<usize>::new();

    for chunk in 0..4 {
        let r_off = ad_s_off
            .process_chunk(
                &x[chunk * 50..(chunk + 1) * 50],
                &y[chunk * 50..(chunk + 1) * 50],
            )
            .unwrap();
        let r_on = ad_s_on
            .process_chunk(
                &x[chunk * 50..(chunk + 1) * 50],
                &y[chunk * 50..(chunk + 1) * 50],
            )
            .unwrap();
        let n_chunk = r_off.y.len();
        y_s_off.extend(&r_off.y);
        y_s_on.extend(&r_on.y);
        it_s_off.extend(std::iter::repeat(r_off.iterations_used.unwrap_or(0)).take(n_chunk));
        it_s_on.extend(std::iter::repeat(r_on.iterations_used.unwrap_or(0)).take(n_chunk));
    }
    let f_off = ad_s_off.finalize().unwrap();
    let f_on = ad_s_on.finalize().unwrap();
    let n_f = f_off.y.len();
    y_s_off.extend(&f_off.y);
    y_s_on.extend(&f_on.y);
    it_s_off.extend(std::iter::repeat(f_off.iterations_used.unwrap_or(0)).take(n_f));
    it_s_on.extend(std::iter::repeat(f_on.iterations_used.unwrap_or(0)).take(n_f));

    // --- Online ---
    let mut ad_o_off = Loess::new()
        .iterations(10)
        .fraction(0.3)
        .adapter(Online)
        .window_capacity(50)
        .min_points(10)
        .update_mode(Full)
        .build()
        .unwrap();
    let mut ad_o_on = Loess::new()
        .iterations(10)
        .fraction(0.3)
        .auto_converge(0.001)
        .adapter(Online)
        .window_capacity(50)
        .min_points(10)
        .update_mode(Full)
        .build()
        .unwrap();

    let mut y_o_off = Vec::new();
    let mut y_o_on = Vec::new();
    let mut it_o_off = Vec::new();
    let mut it_o_on = Vec::new();

    for i in 0..n {
        let x_slice = &x[i..i + 1];
        let y_slice = &y[i..i + 1];
        let r_off = ad_o_off.add_points(x_slice, y_slice).unwrap();
        let r_on = ad_o_on.add_points(x_slice, y_slice).unwrap();
        let o_off = r_off[0].as_ref();
        let o_on = r_on[0].as_ref();
        y_o_off.push(o_off.map(|o| o.smoothed).unwrap_or(y[i]));
        y_o_on.push(o_on.map(|o| o.smoothed).unwrap_or(y[i]));
        it_o_off.push(o_off.and_then(|o| o.iterations_used).unwrap_or(0));
        it_o_on.push(o_on.and_then(|o| o.iterations_used).unwrap_or(0));
    }

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y[i],
            res_b_off.y[i],
            res_b_on.y[i],
            res_b_off.iterations_used.unwrap_or(0),
            res_b_on.iterations_used.unwrap_or(0),
            y_s_off[i],
            y_s_on[i],
            it_s_off[i],
            it_s_on[i],
            y_o_off[i],
            y_o_on[i],
            it_o_off[i],
            it_o_on[i]
        )?;
    }

    println!("Results exported to ../output/visual/auto_converge_comparison.csv");
    Ok(())
}
