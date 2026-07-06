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

    run_degree_comparison()?;
    println!();

    run_fraction_comparison()?;
    println!();

    run_intervals_comparison()?;
    println!();

    run_robust_iter_comparison()?;
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

    run_higher_degree_comparison()?;
    println!();

    run_multivariate_loess()?;
    println!();

    run_merge_comparison()?;
    println!();

    run_online_comparison()?;
    println!();

    run_adapter_comparison()?;
    println!();

    run_degree_interpolation_comparison()?;
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
            .boundary_policy("reflect")
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();

        let mut mse = 0.0;
        for (i, &y_t) in y_true.iter().enumerate().take(n) {
            let error = result.y[i] - y_t;
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
        .boundary_policy("reflect")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Prediction
    let result_pred = Loess::new()
        .fraction(0.3)
        .iterations(2)
        .prediction_intervals(0.95)
        .boundary_policy("reflect")
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
fn run_robust_iter_comparison() -> Result<(), Box<dyn std::error::Error>> {
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
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Robust (6 iterations)
    let result_robust = Loess::new()
        .fraction(0.25)
        .iterations(6)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let mut mse_nr = 0.0;
    let mut mse_r = 0.0;
    for (i, &y_t) in y_true.iter().enumerate().take(n) {
        let err_nr = result_non_robust.y[i] - y_t;
        let err_r = result_robust.y[i] - y_t;
        mse_nr += err_nr * err_nr;
        mse_r += err_r * err_r;
    }
    let rmse_nr = (mse_nr / n as f64).sqrt();
    let rmse_r = (mse_r / n as f64).sqrt();

    println!("RMSE (Non-Robust): {:.4}", rmse_nr);
    println!("RMSE (Robust):     {:.4}", rmse_r);
    println!("Improvement:       {:.2}x", rmse_nr / rmse_r);

    let path = "../output/visual/robust_iter_comparison.csv";
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
        "tricube",
        "gaussian",
        "uniform",
        "cosine",
        "epanechnikov",
        "biweight",
        "triangle",
    ];
    let mut results = Vec::new();

    for kernel in &kernels {
        let result = Loess::new()
            .weight_function(kernel)
            .fraction(0.3)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Kernel processed: {}", kernel);
    }

    let path = "../output/visual/kernel_comparison.csv";
    let mut file = File::create(path)?;
    write!(file, "x,y_true,y_noisy")?;
    for kernel in &kernels {
        write!(file, ",y_{}", kernel)?;
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
    // 250 clean sine points + 75 genuine outliers with random (x, y).
    // Outliers are confined to x ∈ [0, 2] and x ∈ [4, 5] so the scatter
    // looks clearly localised, leaving the middle and right of the sine
    // free of contamination.
    let n_clean = 250usize;
    let n_outliers = 75usize;
    let pi = std::f64::consts::PI;

    let mut x = Vec::with_capacity(n_clean + n_outliers);
    let mut y = Vec::with_capacity(n_clean + n_outliers);
    let mut y_true = Vec::with_capacity(n_clean + n_outliers);

    // Clean sine signal with light noise.
    for i in 0..n_clean {
        let t = (i as f64 / (n_clean - 1) as f64) * 2.0 * pi;
        let signal = t.sin();
        let noise = 0.05 * ((i as f64 * 7.3).sin() + (i as f64 * 13.1).cos()) * 0.5;
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    // Random outliers — xorshift64 gives a good uniform distribution.
    let mut seed: u64 = 0xdead_beef_cafe_f00d;
    let mut rng = || -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64)
    };

    for i in 0..n_outliers {
        // Alternate between the two contaminated bands.
        let ox = if i % 2 == 0 {
            rng() * 2.0 // x ∈ [0, 2]
        } else {
            4.0 + rng() * 1.0 // x ∈ [4, 5]
        };
        let oy = 2.0 + rng() * 2.0; // random y in [2.0, 4.0]
        x.push(ox);
        y_true.push(ox.sin());
        y.push(oy);
    }

    // LOESS requires sorted input.
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let x: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
    let y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
    let y_true: Vec<f64> = idx.iter().map(|&i| y_true[i]).collect();
    let n = x.len();

    println!("8. Robustness Method Comparison");
    println!("-------------------------------");

    let methods = ["bisquare", "huber", "talwar"];
    let mut results = Vec::new();

    for method in &methods {
        let result = Loess::new()
            .robustness_method(method)
            .scaling_method("mean")
            .iterations(5)
            .fraction(0.35)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Method processed: {}", method);
    }

    // Print RMSE for each method
    for (i, name) in ["Bisquare", "Huber", "Talwar"].iter().enumerate() {
        let rmse: f64 = (results[i]
            .y
            .iter()
            .zip(y_true.iter())
            .map(|(f, t)| (f - t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        println!("  RMSE {:8}: {:.4}", name, rmse);
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

    let policies = ["noboundary", "extend", "reflect"];
    let mut results = Vec::new();

    for policy in &policies {
        let result = Loess::new()
            .boundary_policy(*policy)
            .fraction(0.4) // Larger fraction highlights boundary bias
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Policy processed: {}", policy);
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
    let pi = std::f64::consts::PI;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // xorshift64 — truly uncorrelated noise so small-fraction LOESS can't "track" it.
    // With σ=0.8 on a single smooth sine cycle, LOOCV's leverage-inflation penalty
    // makes h=0.1 (15 neighbors) worse than h≈0.25–0.35 (37–52 neighbors).
    let mut seed: u64 = 0xabad_cafe_dead_beef;
    let mut rng = || -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64) * 2.0 - 1.0 // [-1, 1]
    };

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 2.0 * pi;
        let signal = t.sin();
        let noise = 1.0 * rng();
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
        .cv_method("loocv")
        .cv_fractions(candidate_fractions.to_vec())
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  LOOCV Best Fraction: {}", loocv_result.fraction_used);

    // 2. K-Fold (5 folds)
    let kfold_result = Loess::new()
        .cv_method("kfold")
        .cv_k(5)
        .cv_fractions(candidate_fractions.to_vec())
        .cv_seed(42)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  5-Fold Best Fraction: {}", kfold_result.fraction_used);

    // 3. No CV (Fixed bad fraction - too large, over-smoothing)
    let fixed_result = Loess::new()
        .fraction(0.8) // Over-smoothing deliberately
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  No CV Fraction (Fixed): 0.8");

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

    // Two-frequency signal over [0, 4π]: the rapidly-changing second derivative
    // means the LOESS surface varies enough that Hermite cubic interpolation
    // with cell=0.5 (≈ 9 vertices) produces a measurable approximation error.
    // Direct mode evaluates the local polynomial at every one of the 200
    // points; Interpolation re-uses the local fit from the nearest vertex and
    // blends via the Hermite cubic.
    let range = 4.0 * std::f64::consts::PI;
    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * range;
        let signal = t.sin() + 0.4 * (2.0 * t).sin();
        let noise = 0.15 * ((i as f64 * 17.3).sin() + (i as f64 * 5.7).cos()) * 0.5;
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("13. Surface Mode Comparison (Direct vs Interpolation)");
    println!("-------------------------------------------------------");

    // Direct: evaluates the local polynomial at every one of the n points.
    let result_direct = Loess::new()
        .surface_mode("direct")
        .fraction(0.3)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Interpolation with cell=0.5 → ≈ 9 vertices.  The local polynomial is
    // only computed at these vertices; all other x-values are obtained via
    // Hermite cubic blending, introducing a small but measurable error.
    let result_interp = Loess::new()
        .surface_mode("interpolation")
        .cell(0.5)
        .fraction(0.3)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let max_diff = result_direct
        .y
        .iter()
        .zip(result_interp.y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("  max |Direct − Interp| = {:.4}", max_diff);

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
    let n = 200;
    let pi = std::f64::consts::PI;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Two-tier contamination (40 % total, well under the 50 % breakdown point):
    //   • i%5 == 0 → moderate outlier  +1.5  (20 %)
    //   • i%5 == 1 → extreme  outlier  +6.0  (20 %)
    //   • others   → clean, σ ≈ 0.25          (60 %)
    //
    // Scale estimates after the first LOESS pass (fit ≈ signal):
    //   MAR  = median(|r|) ≈ 0.17  (dominated by the 60 % clean half-normal)
    //          → threshold ≈ 0.26 → rejects BOTH outlier tiers
    //   Mean = mean(|r|)   ≈ 1.64  (inflated by extreme outliers)
    //          → threshold ≈ 2.46 → rejects extreme, keeps moderate (~51 % weight)
    //   None → both tiers kept → fit biased ≈ signal + 1.5
    let mut seed: u64 = 0xfeed_face_cafe_b0b0_u64;
    let mut rng = || -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64) * 2.0 - 1.0
    };

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 2.0 * pi;
        let signal = t.sin();
        let mut value = signal + 0.25 * rng();
        if i % 5 == 0 {
            value += 1.5; // moderate outlier
        } else if i % 5 == 1 {
            value += 6.0; // extreme outlier
        }
        x.push(t);
        y_true.push(signal);
        y.push(value);
    }

    println!("14. Scaling Method Comparison");
    println!("-----------------------------");

    // No robustness — baseline.
    let result_none = Loess::new()
        .iterations(0)
        .fraction(0.3)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // MAR: tight threshold (≈ clean noise level) → rejects both outlier tiers.
    let result_mar = Loess::new()
        .scaling_method("mar")
        .iterations(5)
        .fraction(0.3)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // MAD (default): centers on median residual first → same tight threshold as
    // MAR for this data (median(r) ≈ 0 since 60 % clean dominates).
    let result_mad = Loess::new()
        .scaling_method("mad")
        .iterations(5)
        .fraction(0.3)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Mean (MAE): extreme outliers inflate the scale → looser threshold →
    // moderate outliers partially retained → fit between MAR and None.
    let result_mean = Loess::new()
        .scaling_method("mean")
        .iterations(5)
        .fraction(0.3)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    println!(
        "  None MAE from true: {:.3}",
        result_none
            .y
            .iter()
            .zip(y_true.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / n as f64
    );
    println!(
        "  MAD  MAE from true: {:.3}",
        result_mad
            .y
            .iter()
            .zip(y_true.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / n as f64
    );
    println!(
        "  MAR  MAE from true: {:.3}",
        result_mar
            .y
            .iter()
            .zip(y_true.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / n as f64
    );
    println!(
        "  Mean MAE from true: {:.3}",
        result_mean
            .y
            .iter()
            .zip(y_true.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / n as f64
    );

    let path = "../output/visual/scaling_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_none,y_mad,y_mar,y_mean")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y[i],
            result_none.y[i],
            result_mad.y[i],
            result_mar.y[i],
            result_mean.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 15. Zero Weight Fallback Comparison
fn run_zero_weight_fallback_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Compare ZeroWeightFallback policies: UseLocalMean, ReturnOriginal, ReturnNone.
    //
    // When every k-nearest neighbour carries a zero Talwar robustness weight the
    // regression context cannot form a weighted fit and delegates to the fallback:
    //
    //   UseLocalMean   → unweighted mean of the k neighbours' y-values
    //   ReturnOriginal → original y at the query point
    //   ReturnNone     → None; the executor substitutes y_original[i]
    //                    (same result as ReturnOriginal in the Batch executor)
    //
    // Construction:
    //   Signal  : y = sin(π·x / 5)  on [0, 10], n=200
    //   Anomalous zone x ∈ [4, 6] (≈ indices 80–119, 40 points):
    //     even indices → +6 spike above signal    (50 % of zone)
    //     odd  indices → signal + small noise      (50 % clean)
    //   Talwar robustness, fraction=0.10 (k=20), NoBoundary, Direct surface, 2 iterations.
    //   SurfaceMode::Direct forces per-point regression (not Hermite interpolation),
    //   which is required so that each query executes a RegressionContext and the
    //   weight_sum = 0 check can trigger the fallback.
    //
    // After the first OLS pass the fit in the zone sits at ≈ signal+3
    // (midpoint of spike and clean). All zone residuals ≈ ±3.
    //   MAD ≈ 0 (160 clean points dominate) → scale = MAE = (40×3)/200 = 0.6
    //   Talwar threshold = 2.5 × 0.6 = 1.5 < 3  → all 40 zone pts → weight 0
    //
    // For query points whose entire 20-neighbour window lies inside the zone
    // (≈ x ∈ [4.5, 5.5]) weight_sum = 0 → handle_zero_weights_fit() fires:
    //   UseLocalMean   ≈ signal + 3   (smooth intermediate, neighbourhood mean)
    //   ReturnOriginal ≈ signal+6 (even/spike) or signal (odd/clean)  → jagged
    //   ReturnNone     ≈ same as ReturnOriginal via executor fallback

    let n = 200usize;
    let pi = std::f64::consts::PI;

    let mut seed: u64 = 0xdead_beef_cafe_1234;
    let mut rng = || -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64)
    };

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = (pi * t / 5.0).sin();
        let noise = 0.05 * (2.0 * rng() - 1.0);
        // Alternating spike in anomalous zone x ∈ [4, 6]
        let in_zone = (4.0..=6.0_f64).contains(&t);
        let spike = if in_zone && i % 2 == 0 { 6.0 } else { 0.0 };
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise + spike);
    }

    println!("15. Zero Weight Fallback Comparison");
    println!("------------------------------------");

    let make_fit = |fallback: &str| {
        Loess::new()
            .iterations(2) // 1 OLS + 1 WLS — fallback fires before IRLS can propagate
            .fraction(0.10)
            .surface_mode("direct")
            .robustness_method("talwar")
            .zero_weight_fallback(fallback)
            .boundary_policy("noboundary")
            .return_robustness_weights()
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap()
    };

    let fit_lm = make_fit("use_local_mean");
    let fit_ro = make_fit("return_original");
    let fit_rn = make_fit("return_none");

    // Report how many points were driven to zero robustness weight
    if let Some(ref rw) = fit_lm.robustness_weights {
        let n_zero = rw.iter().filter(|&&w| w < 1e-10).count();
        println!("  Points with zero robustness weight: {}/{}", n_zero, n);
    }

    for (name, fit) in [
        ("UseLocalMean  ", &fit_lm),
        ("ReturnOriginal", &fit_ro),
        ("ReturnNone    ", &fit_rn),
    ] {
        let rmse = (fit
            .y
            .iter()
            .zip(y_true.iter())
            .map(|(f, t)| (f - t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        println!("  RMSE {}: {:.4}", name, rmse);
    }

    let path = "../output/visual/zero_weight_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_local_mean,y_return_original,y_return_none"
    )?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], fit_lm.y[i], fit_ro.y[i], fit_rn.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 16. Streaming Comparison
fn run_merge_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 600;
    let chunk_size = 150;
    let overlap = 90; // 60 % overlap — large overlap zone = more points where strategies differ

    // Piecewise signal with hard jumps AT each chunk boundary (i=150, 300, 450).
    // Adjacent chunks therefore fit very different levels in the overlap zone:
    //   • Chunk 1 [0..150]  fits base ≈ 0,   so its overlap prediction ≈ 0
    //   • Chunk 2 [150..300] fits base ≈ 2.5, so its overlap prediction ≈ 2.5
    //   → TakeFirst:        hard step at boundary (takes chunk 1, then chunk 2)
    //   → Average:          midpoint ramp in the 90-pt overlap zone
    //   → WeightedAverage:  smooth distance-weighted ramp
    let mut seed: u64 = 0xcafe_babe_1234_5678;
    let mut rng = || -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64) * 2.0 - 1.0
    };

    let bases = [0.0_f64, 2.5, 0.5, 3.5];
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        let base = bases[i / chunk_size];
        let signal = base + 0.4 * (t * 0.025).sin();
        let noise = 0.35 * rng();
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("16. Streaming Comparison (Strategies)");
    println!("-----------------------------------");

    let mut streaming_weighted = StreamingLoess::new()
        .chunk_size(chunk_size)
        .overlap(overlap)
        .merge_strategy("weighted_average")
        .fraction(0.8)
        .build()?;

    let mut streaming_average = StreamingLoess::new()
        .chunk_size(chunk_size)
        .overlap(overlap)
        .merge_strategy("average")
        .fraction(0.8)
        .build()?;

    let mut streaming_first = StreamingLoess::new()
        .chunk_size(chunk_size)
        .overlap(overlap)
        .merge_strategy("take_first")
        .fraction(0.8)
        .build()?;

    let mut y_weighted = Vec::new();
    let mut y_average = Vec::new();
    let mut y_first = Vec::new();

    for i in (0..n).step_by(chunk_size) {
        let end = (i + chunk_size).min(n);
        let chunk_x = &x[i..end];
        let chunk_y = &y[i..end];

        y_weighted.extend(streaming_weighted.process_chunk(chunk_x, chunk_y)?.y);
        y_average.extend(streaming_average.process_chunk(chunk_x, chunk_y)?.y);
        y_first.extend(streaming_first.process_chunk(chunk_x, chunk_y)?.y);

        if end == n {
            break;
        }
    }

    y_weighted.extend(streaming_weighted.finalize()?.y);
    y_average.extend(streaming_average.finalize()?.y);
    y_first.extend(streaming_first.finalize()?.y);

    let path = "../output/visual/merge_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_weighted,y_average,y_first")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], y_weighted[i], y_average[i], y_first[i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 17. Online Comparison
fn run_online_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 600;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Two sudden regime shifts: +2.5 at t=200, -2.0 at t=400.
    // xorshift64 noise so neither window can "track" it — forcing the lag
    // difference to be purely about window size, not noise structure.
    let mut seed: u64 = 0xc0de_babe_dead_beef;
    let mut rng = || -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64) * 2.0 - 1.0
    };

    for i in 0..n {
        let t = i as f64;
        let mut signal = (t * 0.04).sin();
        if t >= 200.0 {
            signal += 2.5;
        }
        if t >= 400.0 {
            signal -= 2.0;
        }
        let noise = 0.5 * rng();
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("17. Online Comparison (Windows)");
    println!("------------------------------");

    // 1. Small window — fast/responsive but noisier
    let mut adapter_small = OnlineLoess::new()
        .window_capacity(30)
        .min_points(10)
        .update_mode("incremental")
        .fraction(0.9)
        .build()
        .unwrap();

    // 2. Large window — smooth but slow to adapt (lags behind at shifts)
    let mut adapter_large = OnlineLoess::new()
        .window_capacity(300)
        .min_points(10)
        .update_mode("full")
        .fraction(0.3)
        .iterations(3)
        .build()
        .unwrap();

    let mut y_small = Vec::<f64>::with_capacity(n);
    let mut y_large = Vec::<f64>::with_capacity(n);

    for i in 0..n {
        let r_small = adapter_small.add_point(&x[i..i + 1], y[i]).unwrap();
        let r_large = adapter_large.add_point(&x[i..i + 1], y[i]).unwrap();
        y_small.push(r_small.as_ref().map(|o| o.smoothed).unwrap_or(y[i]));
        y_large.push(r_large.as_ref().map(|o| o.smoothed).unwrap_or(y[i]));
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
fn run_adapter_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // x ∈ [0, 4π]: two full sine periods over 200 evenly-spaced points.
    // fraction = 0.15 → window ≈ 30 pts ≈ 1.88 x-units ≈ 0.30 periods, so
    // a local-linear fit captures the sinusoidal shape well.
    // Every 10th point has a +2.5 outlier (10 % contamination) to give
    // Bisquare robustness something to work with across iterations.
    let n = 200;
    let chunk_size = 50;
    let overlap = 10;
    let pi = std::f64::consts::PI;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 4.0 * pi;
        let signal = t.sin();
        let noise = if i % 10 == 0 {
            2.5
        } else {
            0.1 * (i as f64 * 3.7).cos()
        };
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("18. Auto-Convergence Comparison (All Adapters)");
    println!("--------------------------------------------");

    let iterations = 12;
    let tolerance = 0.01;
    let frac = 0.15;

    // 1. Batch
    let res_b_off = Loess::new()
        .fraction(frac)
        .robustness_method("bisquare")
        .iterations(iterations)
        .build()?
        .fit(&x, &y)?;
    let res_b_on = Loess::new()
        .fraction(frac)
        .robustness_method("bisquare")
        .iterations(iterations)
        .auto_converge(tolerance)
        .build()?
        .fit(&x, &y)?;

    // 2. Streaming
    let mut stream_off = StreamingLoess::new()
        .fraction(frac)
        .robustness_method("bisquare")
        .iterations(iterations)
        .chunk_size(chunk_size)
        .overlap(overlap)
        .build()?;
    let mut stream_on = StreamingLoess::new()
        .fraction(frac)
        .robustness_method("bisquare")
        .iterations(iterations)
        .auto_converge(tolerance)
        .chunk_size(chunk_size)
        .overlap(overlap)
        .build()?;

    let mut y_s_off: Vec<f64> = Vec::new();
    let mut y_s_on: Vec<f64> = Vec::new();
    let mut iter_s_off: Vec<usize> = Vec::new();
    let mut iter_s_on: Vec<usize> = Vec::new();

    for i in (0..n).step_by(chunk_size) {
        let end = (i + chunk_size).min(n);
        let out_off = stream_off.process_chunk(&x[i..end], &y[i..end])?;
        let out_on = stream_on.process_chunk(&x[i..end], &y[i..end])?;

        let n_off = out_off.y.len();
        let n_on = out_on.y.len();
        let iters_off = out_off.iterations_used.unwrap_or(0);
        let iters_on = out_on.iterations_used.unwrap_or(0);

        y_s_off.extend(&out_off.y);
        y_s_on.extend(&out_on.y);
        iter_s_off.extend(vec![iters_off; n_off]);
        iter_s_on.extend(vec![iters_on; n_on]);

        if end == n {
            break;
        }
    }
    let fin_off = stream_off.finalize()?;
    let fin_on = stream_on.finalize()?;
    let n_fin = fin_off.y.len();
    let iters_fin_off = fin_off.iterations_used.unwrap_or(0);
    let iters_fin_on = fin_on.iterations_used.unwrap_or(0);
    y_s_off.extend(&fin_off.y);
    y_s_on.extend(&fin_on.y);
    iter_s_off.extend(vec![iters_fin_off; n_fin]);
    iter_s_on.extend(vec![iters_fin_on; n_fin]);

    // 3. Online
    let mut online_off = OnlineLoess::new()
        .fraction(frac)
        .robustness_method("bisquare")
        .iterations(iterations)
        .window_capacity(50)
        .update_mode("full")
        .build()?;
    let mut online_on = OnlineLoess::new()
        .fraction(frac)
        .robustness_method("bisquare")
        .iterations(iterations)
        .auto_converge(tolerance)
        .window_capacity(50)
        .update_mode("full")
        .build()?;

    let mut y_o_off = Vec::new();
    let mut y_o_on = Vec::new();
    // Per-point iteration counts not exposed by online adapter; use constant
    let iter_o_off = vec![0usize; n];
    let iter_o_on = vec![0usize; n];

    for i in 0..n {
        let r_off = online_off.add_point(&x[i..i + 1], y[i])?;
        let r_on = online_on.add_point(&x[i..i + 1], y[i])?;
        y_o_off.push(r_off.as_ref().map(|o| o.smoothed).unwrap_or(y_true[i]));
        y_o_on.push(r_on.as_ref().map(|o| o.smoothed).unwrap_or(y_true[i]));
    }

    let path = "../output/visual/adapter_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_batch_off,y_batch_on,y_stream_off,y_stream_on,y_online_off,y_online_on,iter_batch_off,iter_batch_on,iter_stream_off,iter_stream_on,iter_online_off,iter_online_on"
    )?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y[i],
            res_b_off.y[i],
            res_b_on.y[i],
            y_s_off[i],
            y_s_on[i],
            y_o_off[i],
            y_o_on[i],
            res_b_off.iterations_used.unwrap_or(0),
            res_b_on.iterations_used.unwrap_or(0),
            iter_s_off[i],
            iter_s_on[i],
            iter_o_off[i],
            iter_o_on[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 1. Degree Comparison (Linear vs Quadratic)
fn run_degree_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 2.0 * std::f64::consts::PI;
        let signal = t.sin() + 0.4 * (2.0 * t).cos() + 0.2 * (4.0 * t).sin();
        let noise = 0.3 * ((i as f64 * 7.0).sin() * (i as f64 * 3.7).cos());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("1. Degree Comparison (Linear vs Quadratic)");
    println!("------------------------------------------");

    let result_linear = Loess::new()
        .fraction(0.25)
        .iterations(2)
        .degree("linear")
        .boundary_policy("reflect")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result_quadratic = Loess::new()
        .fraction(0.25)
        .iterations(2)
        .degree("quadratic")
        .boundary_policy("reflect")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let rmse_lin: f64 = (result_linear
        .y
        .iter()
        .zip(y_true.iter())
        .map(|(f, t)| (f - t).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let rmse_quad: f64 = (result_quadratic
        .y
        .iter()
        .zip(y_true.iter())
        .map(|(f, t)| (f - t).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    println!("RMSE Linear:    {:.6}", rmse_lin);
    println!("RMSE Quadratic: {:.6}", rmse_quad);

    let path = "../output/visual/degree_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_linear,y_quadratic")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], result_linear.y[i], result_quadratic.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 6. Higher Degree Comparison (Linear / Quadratic / Cubic / Quartic)
///
/// With uniformly-spaced x the normal equations decouple into orthogonal even/odd
/// subsystems: adding an odd-degree term (cubic) cannot change the intercept, so
/// cubic == quadratic. Non-uniform spacing breaks this symmetry — the odd moments
/// Σ w·(xᵢ−x₀)^k (k odd) become non-zero, and cubic genuinely captures local
/// asymmetry that quadratic cannot.  All four degrees then have distinct RMSEs.
fn run_higher_degree_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 300;

    // Non-uniform x: deterministic xorshift pseudo-random, sorted.
    // This breaks the exact cancellation of odd kernel moments so that
    // cubic ≠ quadratic at every query point.
    let mut seed: u64 = 0x123456789abcdef0;
    let mut rand_f64 = || -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x: Vec<f64> = (0..n).map(|_| -2.5 + rand_f64() * 5.0).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Signal: cubic + quartic polynomial so that each degree family is the
    // limiting factor for a different reason:
    //   Linear   misses x² + x³ + x⁴ → largest RMSE
    //   Quadratic misses x³ + x⁴      → medium-large RMSE
    //   Cubic    misses only x⁴        → small RMSE
    //   Quartic  fits the signal exactly → RMSE ≈ noise floor
    let mut y_true: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let mut noise_seed = 0.0_f64;
    for (i, &t) in x.iter().enumerate() {
        let signal = t.powi(3) / 3.0 - t + t.powi(4) / 12.0;
        noise_seed += 1.0;
        let noise = 0.06 * ((noise_seed * 7.3).sin() + (noise_seed * 13.7).cos()) * 0.5;
        y_true.push(signal);
        y.push(signal + noise);
        let _ = i;
    }

    println!("6. Higher Degree Comparison (Linear / Quadratic / Cubic / Quartic)");
    println!("--------------------------------------------------------------------");

    let degrees = ["linear", "quadratic", "cubic", "quartic"];
    let mut results = Vec::new();

    for &name in &degrees {
        let result = Loess::new()
            .fraction(0.40)
            .iterations(0)
            .degree(name)
            .surface_mode("direct")
            .boundary_policy("noboundary")
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        let rmse: f64 = (result
            .y
            .iter()
            .zip(y_true.iter())
            .map(|(f, t)| (f - t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        println!("  {} RMSE: {:.6}", name, rmse);
        results.push(result);
    }

    let path = "../output/visual/higher_degree_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_linear,y_quadratic,y_cubic,y_quartic"
    )?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y[i],
            results[0].y[i],
            results[1].y[i],
            results[2].y[i],
            results[3].y[i],
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 10. Multivariate LOESS (2-D surface)
fn run_multivariate_loess() -> Result<(), Box<dyn std::error::Error>> {
    let grid_n = 25usize; // 25×25 grid = 625 points
    let mut xs: Vec<[f64; 2]> = Vec::with_capacity(grid_n * grid_n);
    let mut z_true_vec = Vec::with_capacity(grid_n * grid_n);
    let mut z_noisy_vec = Vec::with_capacity(grid_n * grid_n);
    let mut x_flat = Vec::with_capacity(grid_n * grid_n);
    let mut y_flat = Vec::with_capacity(grid_n * grid_n);

    for xi in 0..grid_n {
        for yi in 0..grid_n {
            let xv = (xi as f64 / (grid_n - 1) as f64) * 4.0 - 2.0;
            let yv = (yi as f64 / (grid_n - 1) as f64) * 4.0 - 2.0;
            let signal = (xv.powi(2) + yv.powi(2)).sqrt().cos();
            let noise_seed = (xi * grid_n + yi) as f64;
            let noise = 0.2 * ((noise_seed * 7.1).sin() * (noise_seed * 3.3).cos());
            xs.push([xv, yv]);
            x_flat.push(xv);
            y_flat.push(yv);
            z_true_vec.push(signal);
            z_noisy_vec.push(signal + noise);
        }
    }

    println!(
        "10. Multivariate LOESS (2-D surface, {}×{} grid)",
        grid_n, grid_n
    );
    println!("-----------------------------------------------");

    // The API expects row-major interleaved layout: [x1_0, x2_0, x1_1, x2_1, ...]
    // i.e. x[i * dimensions + d] for point i and dimension d.
    let mut x_combined: Vec<f64> = Vec::with_capacity(xs.len() * 2);
    for p in &xs {
        x_combined.push(p[0]);
        x_combined.push(p[1]);
    }

    let result = Loess::new()
        .fraction(0.15)
        .iterations(1)
        .dimensions(2)
        .distance_metric("euclidean")
        .boundary_policy("noboundary")
        .build()
        .unwrap()
        .fit(&x_combined, &z_noisy_vec)
        .unwrap();

    let rmse: f64 = (result
        .y
        .iter()
        .zip(z_true_vec.iter())
        .map(|(f, t)| (f - t).powi(2))
        .sum::<f64>()
        / result.y.len() as f64)
        .sqrt();
    println!("RMSE: {:.6}", rmse);

    let path = "../output/visual/multivariate_loess.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y,z_true,z_smooth")?;
    for i in 0..result.y.len() {
        writeln!(
            file,
            "{},{},{},{}",
            x_flat[i], y_flat[i], z_true_vec[i], result.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

fn run_degree_interpolation_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);
    let mut y_noisy = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 6.0 * std::f64::consts::PI;
        let signal = t.sin() + 0.4 * (2.0 * t).sin();
        let noise = 0.25 * ((i as f64 * 17.3).sin() + (i as f64 * 5.7).cos());
        x.push(t);
        y_true.push(signal);
        y_noisy.push(signal + noise);
    }

    println!("Degree x Surface Mode Comparison");
    println!("---------------------------------");

    let degrees = [
        ("linear", "Linear"),
        ("quadratic", "Quadratic"),
        ("cubic", "Cubic"),
        ("quartic", "Quartic"),
    ];

    let mut results: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for (deg, name) in &degrees {
        let direct = Loess::new()
            .fraction(0.3)
            .degree(*deg)
            .surface_mode("direct")
            .build()?
            .fit(&x, &y_noisy)?;
        let interp = Loess::new()
            .fraction(0.3)
            .degree(*deg)
            .surface_mode("interpolation")
            .build()?
            .fit(&x, &y_noisy)?;
        let max_diff = direct
            .y
            .iter()
            .zip(interp.y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!("{}: direct/interp max diff = {:.2e}", name, max_diff);
        results.push((direct.y, interp.y));
    }

    let path = "../output/visual/degree_interpolation_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_lin_direct,y_lin_interp,y_quad_direct,y_quad_interp,y_cubic_direct,y_cubic_interp,y_quartic_direct,y_quartic_interp"
    )?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y_noisy[i],
            results[0].0[i],
            results[0].1[i],
            results[1].0[i],
            results[1].1[i],
            results[2].0[i],
            results[2].1[i],
            results[3].0[i],
            results[3].1[i],
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}
