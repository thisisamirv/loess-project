#!/usr/bin/env python3
"""
fastloess Batch Smoothing - Comprehensive Examples

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

import time

import numpy as np
from fastloess import Loess


def make_linear(n: int):
    x = np.arange(n, dtype=float)
    y = 2.0 * x + 1.0
    return x, y


# ── Example 1: Basic Smoothing ───────────────────────────────────────────────
def example_1_basic_smoothing():
    print("Example 1: Basic Smoothing")

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

    result = Loess(fraction=0.5, iterations=3).fit(x, y)

    print(f"  fraction_used={result.fraction_used}")
    print(f"  Smoothed: [{', '.join(f'{v:.3f}' for v in result.y)}]")
    print()


# ── Example 2: Robust Smoothing with Outliers ────────────────────────────────
def example_2_robust_with_outliers():
    print("Example 2: Robust Smoothing with Outliers")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y = np.array([2.1, 4.0, 5.9, 25.0, 10.1, 12.0, 14.1, 15.9])  # 25.0 outlier

    result = Loess(
        fraction=0.5,
        iterations=5,
        robustness_method="bisquare",
        return_robustness_weights=True,
        return_residuals=True,
    ).fit(x, y)

    if result.robustness_weights is not None:
        for i, w in enumerate(result.robustness_weights):
            if w < 0.5:
                print(f"  Outlier at index {i} (y={y[i]}): weight={w:.3f}")
    print(f"  Smoothed: [{', '.join(f'{v:.2f}' for v in result.y)}]")
    print()


# ── Example 3: Uncertainty Quantification ───────────────────────────────────
def example_3_uncertainty_quantification():
    print("Example 3: Uncertainty Quantification")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y = np.array([2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7])

    result = Loess(
        fraction=0.5,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95,
    ).fit(x, y)

    print("  x  y_smooth  conf_low  conf_high  pred_low  pred_high")
    cl = result.confidence_lower
    cu = result.confidence_upper
    pl = result.prediction_lower
    pu = result.prediction_upper
    if cl is not None and cu is not None and pl is not None and pu is not None:
        for i in range(len(result.y)):
            print(
                f"  {result.x[i]:.0f}  {result.y[i]:.4f}  "
                f"{cl[i]:.4f}  {cu[i]:.4f}  "
                f"{pl[i]:.4f}  {pu[i]:.4f}"
            )
    print()


# ── Example 4: Cross-Validation ──────────────────────────────────────────────
def example_4_cross_validation():
    print("Example 4: Cross-Validation for Parameter Selection")

    x = np.arange(1, 21, dtype=float)
    y = 2 * x + 1 + np.sin(x * 0.5)

    result = Loess(
        cv_fractions=[0.2, 0.3, 0.5, 0.7],
        cv_method="kfold",
        cv_k=5,
        iterations=2,
        return_diagnostics=True,
    ).fit(x, y)

    print(f"  Selected fraction: {result.fraction_used}")
    if result.cv_scores is not None:
        fracs = [0.2, 0.3, 0.5, 0.7]
        print("  CV Scores (RMSE per fraction):")
        for frac, score in zip(fracs, result.cv_scores):
            print(f"    fraction={frac}: {score:.4f}")
    print()


# ── Example 5: Complete Diagnostic Analysis ──────────────────────────────────
def example_5_complete_diagnostics():
    print("Example 5: Complete Diagnostic Analysis")

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y = np.array([2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7])

    result = Loess(
        fraction=0.5,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95,
        return_diagnostics=True,
        return_residuals=True,
        return_robustness_weights=True,
    ).fit(x, y)

    if result.diagnostics is not None:
        d = result.diagnostics
        print("  Diagnostics:")
        print(f"    RMSE:        {d.rmse:.6f}")
        print(f"    MAE:         {d.mae:.6f}")
        print(f"    R²:          {d.r_squared:.6f}")
        print(f"    Residual SD: {d.residual_sd:.6f}")
        if d.aic is not None:
            print(f"    AIC:         {d.aic:.2f}")
        if d.aicc is not None:
            print(f"    AICc:        {d.aicc:.2f}")
        if d.effective_df is not None:
            print(f"    Eff. DF:     {d.effective_df:.2f}")

    print(f"  smoothed[0]: {result.y[0]:.5f}")
    if result.residuals is not None:
        print(f"  residuals[0]: {result.residuals[0]:.5f}")
    if result.robustness_weights is not None:
        print(f"  rob_weight[0]: {result.robustness_weights[0]:.4f}")
    print()


# ── Example 6: Different Weight Functions (Kernels) ──────────────────────────
def example_6_different_kernels():
    print("Example 6: Different Weight Functions (Kernels)")

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

    for kernel in ["tricube", "epanechnikov", "gaussian", "biweight"]:
        result = Loess(fraction=0.5, weight_function=kernel).fit(x, y)
        print(f"  {kernel}: [{', '.join(f'{v:.3f}' for v in result.y)}]")
    print()


# ── Example 7: Robustness Methods Comparison ─────────────────────────────────
def example_7_robustness_methods():
    print("Example 7: Robustness Methods Comparison")

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2.0, 4.1, 20.0, 8.2, 9.8])  # 20.0 is an outlier

    for method in ["bisquare", "huber", "talwar"]:
        result = Loess(
            fraction=0.5,
            iterations=5,
            robustness_method=method,
            return_robustness_weights=True,
        ).fit(x, y)
        print(f"  {method}:")
        print(f"    Smoothed: [{', '.join(f'{v:.2f}' for v in result.y)}]")
        if result.robustness_weights is not None:
            print(
                f"    Weights:  [{', '.join(f'{w:.3f}' for w in result.robustness_weights)}]"
            )
    print()


# ── Example 8: Benchmark ─────────────────────────────────────────────────────
def example_8_benchmark():
    print("Example 8: Benchmark")

    n = 1000
    x = np.arange(n, dtype=float)
    y = np.sin(x * 0.1) + np.cos(x * 0.01)

    t0 = time.perf_counter()
    result = Loess(parallel=True).fit(x, y)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  {n} points in {ms:.2f}ms")
    print(f"  fraction_used={result.fraction_used}, y[0]={result.y[0]:.4f}")
    print()


# ── Example 9: Scaling Methods (MAR, MAD, Mean) ──────────────────────────────
def example_9_scaling_methods():
    print("Example 9: Scaling Methods")

    x, y = make_linear(20)

    for method in ["mar", "mad", "mean"]:
        result = Loess(fraction=0.5, scaling_method=method).fit(x, y)
        print(f"  {method}: y[0]={result.y[0]:.3f}")
    print()


# ── Example 10: Boundary Policies ────────────────────────────────────────────
def example_10_boundary_policies():
    print("Example 10: Boundary Policies")

    x, y = make_linear(30)

    for policy in ["extend", "reflect", "zero", "noboundary"]:
        result = Loess(fraction=0.5, boundary_policy=policy).fit(x, y)
        print(f"  {policy}: first={result.y[0]:.2f}, last={result.y[-1]:.2f}")
    print()


# ── Example 11: Zero-Weight Fallback Strategies ───────────────────────────────
def example_11_zero_weight_fallback():
    print("Example 11: Zero-Weight Fallback Strategies")

    x, y = make_linear(20)

    for fb in ["use_local_mean", "return_original", "return_none"]:
        result = Loess(fraction=0.5, zero_weight_fallback=fb).fit(x, y)
        print(f"  {fb}: y[0]={result.y[0]:.3f}")
    print()


# ── Example 12: Polynomial Degrees + iterations_used ──────────────────────────
def example_12_polynomial_degrees():
    print("Example 12: Polynomial Degrees")

    x, y = make_linear(30)

    for deg in ["constant", "linear", "quadratic", "cubic", "quartic"]:
        result = Loess(fraction=0.5, iterations=2, degree=deg).fit(x, y)
        print(
            f"  {deg}: y[0]={result.y[0]:.3f}, iterations_used={result.iterations_used}"
        )
    print()


# ── Example 13: Distance Metrics ─────────────────────────────────────────────
def example_13_distance_metrics():
    print("Example 13: Distance Metrics")

    x, y = make_linear(20)

    for metric in ["euclidean", "normalized", "manhattan", "chebyshev"]:
        result = Loess(fraction=0.5, distance_metric=metric).fit(x, y)
        print(f"  {metric}: y[0]={result.y[0]:.3f}")

    # Minkowski with custom p via the dedicated minkowski_p parameter
    result_mink = Loess(fraction=0.5, distance_metric="minkowski", minkowski_p=3.0).fit(
        x, y
    )
    print(f"  minkowski(p=3): y[0]={result_mink.y[0]:.3f}")
    print()


# ── Example 14: Surface Modes and Standard Errors ────────────────────────────
def example_14_surface_modes_and_se():
    print("Example 14: Surface Modes and Standard Errors")

    x, y = make_linear(30)

    # Direct surface — fits every point exactly; SE fields fully populated
    r_direct = Loess(
        fraction=0.5,
        surface_mode="direct",
        return_se=True,
        confidence_intervals=0.95,
        prediction_intervals=0.95,
    ).fit(x, y)

    print("  surface_mode=direct:")
    print(f"    confidence_lower non-null: {r_direct.confidence_lower is not None}")
    print(f"    prediction_lower non-null: {r_direct.prediction_lower is not None}")
    if r_direct.standard_errors is not None:
        print(f"    standard_errors[0]: {r_direct.standard_errors[0]:.4f}")
    if r_direct.enp is not None:
        print(f"    enp: {r_direct.enp:.3f}")
    if r_direct.trace_hat is not None:
        print(f"    trace_hat: {r_direct.trace_hat:.3f}")
    if r_direct.delta1 is not None:
        print(f"    delta1: {r_direct.delta1:.3f}")
    if r_direct.delta2 is not None:
        print(f"    delta2: {r_direct.delta2:.3f}")
    if r_direct.residual_scale is not None:
        print(f"    residual_scale: {r_direct.residual_scale:.4f}")
    if r_direct.leverage is not None:
        print(f"    leverage[0]: {r_direct.leverage[0]:.4f}")

    # Interpolation surface — faster, approximate
    r_interp = Loess(fraction=0.5, surface_mode="interpolation", return_se=True).fit(
        x, y
    )

    print("  surface_mode=interpolation:")
    print(f"    y[0]: {r_interp.y[0]:.3f}")
    if r_interp.standard_errors is not None:
        print(f"    standard_errors[0]: {r_interp.standard_errors[0]:.4f}")
    print()


# ── Example 15: Additional Weight Functions (Uniform, Triangle, Cosine) ───────
def example_15_additional_kernels():
    print("Example 15: Additional Weight Functions (Uniform, Triangle, Cosine)")

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

    for kernel in ["uniform", "triangle", "cosine"]:
        result = Loess(fraction=0.5, weight_function=kernel).fit(x, y)
        print(f"  {kernel}: [{', '.join(f'{v:.3f}' for v in result.y)}]")
    print()


# ── Example 16: LOOCV, K-Fold, and Auto-Converge ─────────────────────────────
def example_16_loocv_and_auto_converge():
    print("Example 16: LOOCV, K-Fold, and Auto-Converge")

    x = np.arange(1, 21, dtype=float)
    y = 2 * x + 1 + np.sin(x * 0.5)

    # Leave-one-out cross-validation
    r_loocv = Loess(cv_fractions=[0.3, 0.5, 0.7], cv_method="loocv").fit(x, y)
    print(f"  LOOCV selected fraction: {r_loocv.fraction_used}")
    if r_loocv.cv_scores is not None:
        print(f"  LOOCV scores: [{', '.join(f'{s:.4f}' for s in r_loocv.cv_scores)}]")

    # K-Fold cross-validation
    r_kfold = Loess(cv_fractions=[0.2, 0.4, 0.6], cv_method="kfold", cv_k=5).fit(x, y)
    print(f"  KFold(k=5) selected fraction: {r_kfold.fraction_used}")
    if r_kfold.cv_scores is not None:
        print(f"  KFold scores: [{', '.join(f'{s:.4f}' for s in r_kfold.cv_scores)}]")

    # Auto-converge: stop robustness iterations when change < tolerance
    r_ac = Loess(fraction=0.5, auto_converge=1e-4).fit(x, y)
    print(f"  auto_converge=1e-4: iterations_used={r_ac.iterations_used}")
    print()


# ── Example 17: Interpolation Tuning (surface_mode effects) ──────────────────
def example_17_interpolation_tuning():
    print("Example 17: Interpolation Tuning (surface_mode effects)")

    n = 50
    x, y = make_linear(n)

    # Default (interpolation) — fastest, uses a spatial grid
    r_interp = Loess(fraction=0.5, surface_mode="interpolation").fit(x, y)
    print(f"  interpolation: y[0]={r_interp.y[0]:.3f}, y[-1]={r_interp.y[-1]:.3f}")

    # Direct — fits every point exactly, more accurate but slower
    r_direct = Loess(fraction=0.5, surface_mode="direct").fit(x, y)
    print(f"  direct:        y[0]={r_direct.y[0]:.3f}, y[-1]={r_direct.y[-1]:.3f}")

    # Fraction sweep with direct surface
    for frac in [0.2, 0.5, 0.8]:
        r = Loess(fraction=frac, surface_mode="direct").fit(x, y)
        print(f"  direct fraction={frac}: y[0]={r.y[0]:.3f}")

    # Interpolation + SE for hat-matrix statistics
    r_se = Loess(fraction=0.5, surface_mode="interpolation", return_se=True).fit(x, y)
    if r_se.enp is not None:
        print(f"  interpolation+SE enp: {r_se.enp:.3f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("fastloess Batch Smoothing - Comprehensive Examples")
    print("=" * 60)
    print()

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

    print("=== Batch Smoothing Examples Complete ===")


if __name__ == "__main__":
    main()
