#!/usr/bin/env julia
"""
FastLOESS Batch Smoothing Example

This example demonstrates batch LOESS smoothing features:
- Basic smoothing with different parameters
- Robustness iterations for outlier handling
- Confidence and prediction intervals
- Diagnostics and cross-validation

The Loess class is the primary interface for
processing complete datasets that fit in memory.
"""

using Random
using Printf

# Handle package loading - check if we're already in the FastLOESS project
using Pkg
project_name = Pkg.project().name
if project_name != "FastLOESS"
    # Not in the FastLOESS project, need to develop it
    script_dir = @__DIR__
    julia_pkg_dir = joinpath(dirname(script_dir), "julia")
    if !haskey(Pkg.project().dependencies, "FastLOESS")
        Pkg.develop(path = julia_pkg_dir)
    end
end

using FastLOESS

function generate_sample_data(n_points = 1000)
    """
    Generate complex sample data with a trend, seasonality, and outliers.
    """
    Random.seed!(42)
    x = collect(range(0, 50, length = n_points))

    # Trend + Seasonality
    y_true = 0.5 .* x .+ 5 .* sin.(x .* 0.5)

    # Gaussian noise
    y = y_true .+ randn(n_points) .* 1.5

    # Add significant outliers (10% of data)
    n_outliers = Int(round(n_points * 0.1))
    outlier_indices = randperm(n_points)[1:n_outliers]
    for i in outlier_indices
        y[i] += rand(10:20) * rand([-1, 1])
    end

    return x, y, y_true
end

function main()
    println("=== FastLOESS Batch Smoothing Example ===")

    # 1. Generate Data
    x, y, y_true = generate_sample_data(1000)
    println("Generated $(length(x)) data points with outliers.")

    # 2. Basic Smoothing (Default parameters)
    println("Running basic smoothing...")
    # 2. Basic Smoothing (Default parameters)
    println("Running basic smoothing...")
    # Use a smaller fraction (0.05) to capture the sine wave seasonality
    l_basic = Loess(iterations = 0, fraction = 0.05)
    res_basic = fit(l_basic, x, y)

    # 3. Robust Smoothing (IRLS)
    println("Running robust smoothing (3 iterations)...")
    l_robust = Loess(
        fraction = 0.05,
        iterations = 3,
        robustness_method = "bisquare",
        return_robustness_weights = true,
    )
    res_robust = fit(l_robust, x, y)

    # 4. Uncertainty Quantification
    println("Computing confidence and prediction intervals...")
    l_intervals = Loess(
        fraction = 0.05,
        confidence_intervals = 0.95,
        prediction_intervals = 0.95,
        return_diagnostics = true,
    )
    res_intervals = fit(l_intervals, x, y)

    # 5. Cross-Validation for optimal fraction
    println("Running cross-validation to find optimal fraction...")
    cv_fractions = [0.05, 0.1, 0.2, 0.4]
    l_cv = Loess(cv_fractions = cv_fractions, cv_method = "kfold", cv_k = 5)
    res_cv = fit(l_cv, x, y)
    println("Optimal fraction found: $(res_cv.fraction_used)")

    # Diagnostics Printout
    if res_intervals.diagnostics !== nothing
        diag = res_intervals.diagnostics
        println("\nFit Statistics (Intervals Model):")
        @printf(" - R²:   %.4f\n", diag.r_squared)
        @printf(" - RMSE: %.4f\n", diag.rmse)
        @printf(" - MAE:  %.4f\n", diag.mae)
    end

    # 6. Boundary Policy Comparison
    println("\nDemonstrating boundary policy effects on linear data...")
    xl = collect(range(0, 10, length = 50))
    yl = 2 .* xl .+ 1

    # Compare policies
    r_ext = fit(Loess(fraction = 0.6, boundary_policy = "extend"), xl, yl)
    r_ref = fit(Loess(fraction = 0.6, boundary_policy = "reflect"), xl, yl)
    r_zr = fit(Loess(fraction = 0.6, boundary_policy = "zero"), xl, yl)

    println("Boundary policy comparison:")
    println(
        " - Extend (Default): first=$(round(r_ext.y[1], digits=2)), last=$(round(r_ext.y[end], digits=2))",
    )
    println(
        " - Reflect:          first=$(round(r_ref.y[1], digits=2)), last=$(round(r_ref.y[end], digits=2))",
    )
    println(
        " - Zero:             first=$(round(r_zr.y[1], digits=2)), last=$(round(r_zr.y[end], digits=2))",
    )

    println("\n=== Batch Smoothing Example Complete ===")
end

main()
