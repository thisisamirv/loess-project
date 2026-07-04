#!/usr/bin/env julia
"""
FastLOESS Batch Smoothing - Comprehensive Examples

 1. Basic smoothing
 2. Robust smoothing with outliers
 3. Uncertainty quantification (confidence/prediction intervals)
 4. Cross-validation (K-Fold)
 5. Complete diagnostic analysis
 6. Different weight functions (kernels)
 7. Robustness methods comparison
 8. Benchmark
 9. Scaling methods (MAR, MAD, Mean)
10. Boundary policies
11. Zero-weight fallback strategies
12. Polynomial degrees + iterations_used
13. Distance metrics
14. Surface modes and standard errors
15. Additional weight functions
16. LOOCV and auto-converge
17. Interpolation tuning (surface_mode effects)
"""

using Printf

# Handle package loading
using Pkg
project_name = Pkg.project().name
if project_name != "FastLOESS"
    script_dir = @__DIR__
    julia_pkg_dir = joinpath(dirname(script_dir), "julia")
    if !haskey(Pkg.project().dependencies, "FastLOESS")
        Pkg.develop(path = julia_pkg_dir)
    end
end

using FastLOESS

make_linear(n) = (collect(Float64, 0:(n-1)), collect(Float64, 0:(n-1)) .* 2 .+ 1)

# ── Example 1: Basic Smoothing ───────────────────────────────────────────────
function example_1_basic_smoothing()
    println("Example 1: Basic Smoothing")
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.1, 5.9, 8.2, 9.8]
    result = fit(Loess(fraction = 0.5, iterations = 3), x, y)
    println("  fraction_used=$(result.fraction_used)")
    println("  Smoothed: [$(join(round.(result.y, digits=3), ", "))]")
    println()
end

# ── Example 2: Robust Smoothing with Outliers ────────────────────────────────
function example_2_robust_with_outliers()
    println("Example 2: Robust Smoothing with Outliers")
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [2.1, 4.0, 5.9, 25.0, 10.1, 12.0, 14.1, 15.9]  # 25.0 is outlier
    result = fit(
        Loess(
            fraction = 0.5,
            iterations = 5,
            robustness_method = "bisquare",
            return_robustness_weights = true,
            return_residuals = true,
        ),
        x,
        y,
    )
    if result.robustness_weights !== nothing
        for (i, w) ∈ enumerate(result.robustness_weights)
            w < 0.5 && @printf("  Outlier at index %d (y=%.1f): weight=%.3f\n", i, y[i], w)
        end
    end
    println("  Smoothed: [$(join(round.(result.y, digits=2), ", "))]")
    println()
end

# ── Example 3: Uncertainty Quantification ───────────────────────────────────
function example_3_uncertainty_quantification()
    println("Example 3: Uncertainty Quantification")
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7]
    result = fit(
        Loess(
            fraction = 0.5,
            iterations = 3,
            confidence_intervals = 0.95,
            prediction_intervals = 0.95,
        ),
        x,
        y,
    )
    println("  x  y_smooth  conf_low  conf_high  pred_low  pred_high")
    for i ∈ eachindex(result.y)
        @printf(
            "  %d  %.4f  %.4f  %.4f  %.4f  %.4f\n",
            Int(result.x[i]),
            result.y[i],
            result.confidence_lower[i],
            result.confidence_upper[i],
            result.prediction_lower[i],
            result.prediction_upper[i]
        )
    end
    println()
end

# ── Example 4: Cross-Validation ──────────────────────────────────────────────
function example_4_cross_validation()
    println("Example 4: Cross-Validation for Parameter Selection")
    x = collect(Float64, 1:20)
    y = 2 .* x .+ 1 .+ sin.(x .* 0.5)
    result = fit(
        Loess(
            cv_fractions = [0.2, 0.3, 0.5, 0.7],
            cv_method = "kfold",
            cv_k = 5,
            iterations = 2,
            return_diagnostics = true,
        ),
        x,
        y,
    )
    println("  Selected fraction: $(result.fraction_used)")
    if result.robustness_weights === nothing  # cv_scores stored separately
        scores = [0.0]  # placeholder; use result inspection
    end
    # Note: cv_scores accessible via LoessResult (not yet mapped to Julia struct)
    println()
end

# ── Example 5: Complete Diagnostic Analysis ──────────────────────────────────
function example_5_complete_diagnostics()
    println("Example 5: Complete Diagnostic Analysis")
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7]
    result = fit(
        Loess(
            fraction = 0.5,
            iterations = 3,
            confidence_intervals = 0.95,
            prediction_intervals = 0.95,
            return_diagnostics = true,
            return_residuals = true,
            return_robustness_weights = true,
        ),
        x,
        y,
    )
    if result.diagnostics !== nothing
        d = result.diagnostics
        println("  Diagnostics:")
        @printf("    RMSE:        %.6f\n", d.rmse)
        @printf("    MAE:         %.6f\n", d.mae)
        @printf("    R²:          %.6f\n", d.r_squared)
        @printf("    Residual SD: %.6f\n", d.residual_sd)
        !isnan(d.aic) && @printf("    AIC:         %.2f\n", d.aic)
        !isnan(d.aicc) && @printf("    AICc:        %.2f\n", d.aicc)
        !isnan(d.effective_df) && @printf("    Eff. DF:     %.2f\n", d.effective_df)
    end
    @printf("  smoothed[1]: %.5f\n", result.y[1])
    result.residuals !== nothing && @printf("  residuals[1]: %.5f\n", result.residuals[1])
    result.robustness_weights !== nothing &&
        @printf("  rob_weight[1]: %.4f\n", result.robustness_weights[1])
    println()
end

# ── Example 6: Different Weight Functions (Kernels) ──────────────────────────
function example_6_different_kernels()
    println("Example 6: Different Weight Functions (Kernels)")
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.1, 5.9, 8.2, 9.8]
    for kernel ∈ ["tricube", "epanechnikov", "gaussian", "biweight"]
        result = fit(Loess(fraction = 0.5, weight_function = kernel), x, y)
        println("  $kernel: [$(join(round.(result.y, digits=3), ", "))]")
    end
    println()
end

# ── Example 7: Robustness Methods Comparison ─────────────────────────────────
function example_7_robustness_methods()
    println("Example 7: Robustness Methods Comparison")
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.1, 20.0, 8.2, 9.8]  # 20.0 is an outlier
    for method ∈ ["bisquare", "huber", "talwar"]
        result = fit(
            Loess(
                fraction = 0.5,
                iterations = 5,
                robustness_method = method,
                return_robustness_weights = true,
            ),
            x,
            y,
        )
        println("  $method:")
        println("    Smoothed: [$(join(round.(result.y, digits=2), ", "))]")
        if result.robustness_weights !== nothing
            println(
                "    Weights:  [$(join(round.(result.robustness_weights, digits=3), ", "))]",
            )
        end
    end
    println()
end

# ── Example 8: Benchmark ─────────────────────────────────────────────────────
function example_8_benchmark()
    println("Example 8: Benchmark")
    n = 1000
    x = collect(Float64, 0:(n-1))
    y = sin.(x .* 0.1) .+ cos.(x .* 0.01)
    t0 = time()
    result = fit(Loess(parallel = true), x, y)
    ms = (time() - t0) * 1000
    @printf("  %d points in %.2fms\n", n, ms)
    @printf("  fraction_used=%g, y[1]=%.4f\n", result.fraction_used, result.y[1])
    println()
end

# ── Example 9: Scaling Methods (MAR, MAD, Mean) ──────────────────────────────
function example_9_scaling_methods()
    println("Example 9: Scaling Methods")
    x, y = make_linear(20)
    for method ∈ ["mar", "mad", "mean"]
        result = fit(Loess(fraction = 0.5, scaling_method = method), x, y)
        @printf("  %s: y[1]=%.3f\n", method, result.y[1])
    end
    println()
end

# ── Example 10: Boundary Policies ────────────────────────────────────────────
function example_10_boundary_policies()
    println("Example 10: Boundary Policies")
    x, y = make_linear(30)
    for policy ∈ ["extend", "reflect", "zero", "noboundary"]
        result = fit(Loess(fraction = 0.5, boundary_policy = policy), x, y)
        @printf("  %s: first=%.2f, last=%.2f\n", policy, result.y[1], result.y[end])
    end
    println()
end

# ── Example 11: Zero-Weight Fallback Strategies ───────────────────────────────
function example_11_zero_weight_fallback()
    println("Example 11: Zero-Weight Fallback Strategies")
    x, y = make_linear(20)
    for fb ∈ ["use_local_mean", "return_original", "return_none"]
        result = fit(Loess(fraction = 0.5, zero_weight_fallback = fb), x, y)
        @printf("  %s: y[1]=%.3f\n", fb, result.y[1])
    end
    println()
end

# ── Example 12: Polynomial Degrees + iterations_used ──────────────────────────
function example_12_polynomial_degrees()
    println("Example 12: Polynomial Degrees")
    x, y = make_linear(30)
    for deg ∈ ["constant", "linear", "quadratic", "cubic", "quartic"]
        result = fit(Loess(fraction = 0.5, iterations = 2, degree = deg), x, y)
        @printf(
            "  %s: y[1]=%.3f, iterations_used=%d\n",
            deg,
            result.y[1],
            result.iterations_used
        )
    end
    println()
end

# ── Example 13: Distance Metrics ─────────────────────────────────────────────
function example_13_distance_metrics()
    println("Example 13: Distance Metrics")
    x, y = make_linear(20)
    for metric ∈ ["euclidean", "normalized", "manhattan", "chebyshev"]
        result = fit(Loess(fraction = 0.5, distance_metric = metric), x, y)
        @printf("  %s: y[1]=%.3f\n", metric, result.y[1])
    end
    # Minkowski with custom p via "minkowski:p" string format
    result = fit(Loess(fraction = 0.5, distance_metric = "minkowski:3"), x, y)
    @printf("  minkowski(p=3): y[1]=%.3f\n", result.y[1])
    println()
end

# ── Example 14: Surface Modes and Standard Errors ────────────────────────────
function example_14_surface_modes_and_se()
    println("Example 14: Surface Modes and Standard Errors")
    x, y = make_linear(30)

    # Direct surface — fits every point; SE fields fully populated
    r = fit(
        Loess(
            fraction = 0.5,
            surface_mode = "direct",
            return_se = true,
            confidence_intervals = 0.95,
            prediction_intervals = 0.95,
        ),
        x,
        y,
    )
    println("  surface_mode=direct:")
    println("    confidence_lower non-null: $(r.confidence_lower !== nothing)")
    println("    prediction_lower non-null: $(r.prediction_lower !== nothing)")
    r.standard_errors !== nothing &&
        @printf("    standard_errors[1]: %.4f\n", r.standard_errors[1])
    r.enp !== nothing && @printf("    enp: %.3f\n", r.enp)
    r.trace_hat !== nothing && @printf("    trace_hat: %.3f\n", r.trace_hat)
    r.delta1 !== nothing && @printf("    delta1: %.3f\n", r.delta1)
    r.delta2 !== nothing && @printf("    delta2: %.3f\n", r.delta2)
    r.residual_scale !== nothing && @printf("    residual_scale: %.4f\n", r.residual_scale)
    r.leverage !== nothing && @printf("    leverage[1]: %.4f\n", r.leverage[1])

    # Interpolation surface — faster, approximate
    r2 = fit(Loess(fraction = 0.5, surface_mode = "interpolation", return_se = true), x, y)
    println("  surface_mode=interpolation:")
    @printf("    y[1]: %.3f\n", r2.y[1])
    r2.standard_errors !== nothing &&
        @printf("    standard_errors[1]: %.4f\n", r2.standard_errors[1])
    println()
end

# ── Example 15: Additional Weight Functions (Uniform, Triangle, Cosine) ───────
function example_15_additional_kernels()
    println("Example 15: Additional Weight Functions (Uniform, Triangle, Cosine)")
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.1, 5.9, 8.2, 9.8]
    for kernel ∈ ["uniform", "triangle", "cosine"]
        result = fit(Loess(fraction = 0.5, weight_function = kernel), x, y)
        println("  $kernel: [$(join(round.(result.y, digits=3), ", "))]")
    end
    println()
end

# ── Example 16: LOOCV, K-Fold, and Auto-Converge ─────────────────────────────
function example_16_loocv_and_auto_converge()
    println("Example 16: LOOCV, K-Fold, and Auto-Converge")
    x = collect(Float64, 1:20)
    y = 2 .* x .+ 1 .+ sin.(x .* 0.5)

    # Leave-one-out cross-validation
    r_loocv = fit(Loess(cv_fractions = [0.3, 0.5, 0.7], cv_method = "loocv"), x, y)
    @printf("  LOOCV selected fraction: %g\n", r_loocv.fraction_used)

    # K-Fold cross-validation
    r_kfold =
        fit(Loess(cv_fractions = [0.2, 0.4, 0.6], cv_method = "kfold", cv_k = 5), x, y)
    @printf("  KFold(k=5) selected fraction: %g\n", r_kfold.fraction_used)

    # Auto-converge: stop robustness iterations when change < tolerance
    r_ac = fit(Loess(fraction = 0.5, auto_converge = 1e-4), x, y)
    @printf("  auto_converge=1e-4: iterations_used=%d\n", r_ac.iterations_used)
    println()
end

# ── Example 17: Interpolation Tuning (surface_mode effects) ──────────────────
function example_17_interpolation_tuning()
    println("Example 17: Interpolation Tuning (surface_mode effects)")
    n = 50
    x, y = make_linear(n)

    r_interp = fit(Loess(fraction = 0.5, surface_mode = "interpolation"), x, y)
    @printf("  interpolation: y[1]=%.3f, y[end]=%.3f\n", r_interp.y[1], r_interp.y[end])

    r_direct = fit(Loess(fraction = 0.5, surface_mode = "direct"), x, y)
    @printf("  direct:        y[1]=%.3f, y[end]=%.3f\n", r_direct.y[1], r_direct.y[end])

    for frac ∈ [0.2, 0.5, 0.8]
        r = fit(Loess(fraction = frac, surface_mode = "direct"), x, y)
        @printf("  direct fraction=%.1f: y[1]=%.3f\n", frac, r.y[1])
    end

    r_se =
        fit(Loess(fraction = 0.5, surface_mode = "interpolation", return_se = true), x, y)
    r_se.enp !== nothing && @printf("  interpolation+SE enp: %.3f\n", r_se.enp)
    println()
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()
    println("=" ^ 60)
    println("FastLOESS Batch Smoothing - Comprehensive Examples")
    println("=" ^ 60)
    println()

    example_1_basic_smoothing()
    example_2_robust_with_outliers()
    example_3_uncertainty_quantification()
    example_4_cross_validation()
    example_5_complete_diagnostics()
    example_6_different_kernels()
    example_7_robustness_methods()
    example_8_benchmark()
    example_9_scaling_methods()
    example_10_boundary_policies()
    example_11_zero_weight_fallback()
    example_12_polynomial_degrees()
    example_13_distance_metrics()
    example_14_surface_modes_and_se()
    example_15_additional_kernels()
    example_16_loocv_and_auto_converge()
    example_17_interpolation_tuning()

    println("=== Batch Smoothing Examples Complete ===")
end

main()
