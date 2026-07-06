const fastloess = require('../../bindings/nodejs');

/**
 * fastloess Batch Smoothing - Comprehensive Examples
 *
 * 17 examples covering the full Loess batch API:
 *  1. Basic smoothing
 *  2. Robust smoothing with outliers
 *  3. Uncertainty quantification (confidence/prediction intervals)
 *  4. Cross-validation (K-Fold)
 *  5. Complete diagnostic analysis
 *  6. Different weight functions (kernels)
 *  7. Robustness methods comparison
 *  8. Benchmark
 *  9. Scaling methods (MAR, MAD, Mean)
 * 10. Boundary policies
 * 11. Zero-weight fallback strategies
 * 12. Polynomial degrees + iterationsUsed
 * 13. Distance metrics
 * 14. Surface modes and standard errors
 * 15. Additional weight functions
 * 16. LOOCV and auto-converge
 * 17. Interpolation tuning (surface_mode effects)
 */

function makeLinear(n) {
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = i; y[i] = 2 * i + 1; }
    return { x, y };
}

// ── Example 1: Basic Smoothing ──────────────────────────────────────────────
function example_1_basic_smoothing() {
    console.log("Example 1: Basic Smoothing");

    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2.0, 4.1, 5.9, 8.2, 9.8]);

    const result = new fastloess.Loess({ fraction: 0.5, iterations: 3 }).fit(x, y);

    console.log(`  fractionUsed=${result.fractionUsed}`);
    console.log(`  Smoothed: [${Array.from(result.y).map(v => v.toFixed(3)).join(', ')}]`);
    console.log();
}

// ── Example 2: Robust Smoothing with Outliers ────────────────────────────────
function example_2_robust_with_outliers() {
    console.log("Example 2: Robust Smoothing with Outliers");

    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const y = new Float64Array([2.1, 4.0, 5.9, 25.0, 10.1, 12.0, 14.1, 15.9]); // 25.0 is an outlier

    const result = new fastloess.Loess({
        fraction: 0.5,
        iterations: 5,
        robustness_method: "bisquare",
        return_robustness_weights: true,
        return_residuals: true,
    }).fit(x, y);

    const weights = result.robustnessWeights;
    if (weights) {
        for (let i = 0; i < weights.length; i++) {
            if (weights[i] < 0.5) {
                console.log(`  Outlier at index ${i} (y=${y[i]}): weight=${weights[i].toFixed(3)}`);
            }
        }
    }
    console.log(`  Smoothed: [${Array.from(result.y).map(v => v.toFixed(2)).join(', ')}]`);
    console.log();
}

// ── Example 3: Uncertainty Quantification ───────────────────────────────────
function example_3_uncertainty_quantification() {
    console.log("Example 3: Uncertainty Quantification");

    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const y = new Float64Array([2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7]);

    const result = new fastloess.Loess({
        fraction: 0.5,
        iterations: 3,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
    }).fit(x, y);

    const cLow = result.confidenceLower;
    const cHigh = result.confidenceUpper;
    const pLow = result.predictionLower;
    const pHigh = result.predictionUpper;

    console.log("  x\t  y_smooth\t  conf[low, high]\t  pred[low, high]");
    for (let i = 0; i < result.y.length; i++) {
        console.log(
            `  ${result.x[i].toFixed(0)}\t  ${result.y[i].toFixed(4)}\t` +
            `  [${cLow[i].toFixed(4)}, ${cHigh[i].toFixed(4)}]\t` +
            `  [${pLow[i].toFixed(4)}, ${pHigh[i].toFixed(4)}]`
        );
    }
    console.log();
}

// ── Example 4: Cross-Validation ──────────────────────────────────────────────
function example_4_cross_validation() {
    console.log("Example 4: Cross-Validation for Parameter Selection");

    const n = 20;
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        x[i] = i + 1;
        y[i] = 2 * x[i] + 1 + Math.sin(x[i] * 0.5);
    }

    const result = new fastloess.Loess({
        cvFractions: [0.2, 0.3, 0.5, 0.7],
        cvMethod: "kfold",
        cvK: 5,
        iterations: 2,
        return_diagnostics: true,
    }).fit(x, y);

    console.log(`  Selected fraction: ${result.fractionUsed}`);
    const scores = result.cvScores;
    if (scores) {
        const fracs = [0.2, 0.3, 0.5, 0.7];
        console.log("  CV Scores (RMSE per fraction):");
        for (let i = 0; i < fracs.length; i++) {
            console.log(`    fraction=${fracs[i]}: ${scores[i].toFixed(4)}`);
        }
    }
    console.log();
}

// ── Example 5: Complete Diagnostic Analysis ──────────────────────────────────
function example_5_complete_diagnostics() {
    console.log("Example 5: Complete Diagnostic Analysis");

    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const y = new Float64Array([2.1, 3.8, 6.2, 7.9, 10.3, 11.8, 14.1, 15.7]);

    const result = new fastloess.Loess({
        fraction: 0.5,
        iterations: 3,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
        return_diagnostics: true,
        return_residuals: true,
        return_robustness_weights: true,
    }).fit(x, y);

    const diag = result.diagnostics;
    if (diag) {
        console.log("  Diagnostics:");
        console.log(`    RMSE:        ${diag.rmse.toFixed(6)}`);
        console.log(`    MAE:         ${diag.mae.toFixed(6)}`);
        console.log(`    R²:          ${diag.r_squared.toFixed(6)}`);
        console.log(`    Residual SD: ${diag.residual_sd.toFixed(6)}`);
        if (diag.aic != null) console.log(`    AIC:         ${diag.aic.toFixed(2)}`);
        if (diag.aicc != null) console.log(`    AICc:        ${diag.aicc.toFixed(2)}`);
        if (diag.effective_df != null) console.log(`    Eff. DF:     ${diag.effective_df.toFixed(2)}`);
    }
    console.log(`  Smoothed[0]: ${result.y[0].toFixed(5)}`);
    if (result.residuals) console.log(`  residuals[0]: ${result.residuals[0].toFixed(5)}`);
    if (result.robustnessWeights) console.log(`  robWeight[0]: ${result.robustnessWeights[0].toFixed(4)}`);
    console.log();
}

// ── Example 6: Different Weight Functions (Kernels) ──────────────────────────
function example_6_different_kernels() {
    console.log("Example 6: Different Weight Functions (Kernels)");

    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2.0, 4.1, 5.9, 8.2, 9.8]);

    for (const kernel of ["tricube", "epanechnikov", "gaussian", "biweight"]) {
        const result = new fastloess.Loess({ fraction: 0.5, weight_function: kernel }).fit(x, y);
        console.log(`  ${kernel}: [${Array.from(result.y).map(v => v.toFixed(3)).join(', ')}]`);
    }
    console.log();
}

// ── Example 7: Robustness Methods Comparison ─────────────────────────────────
function example_7_robustness_methods() {
    console.log("Example 7: Robustness Methods Comparison");

    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2.0, 4.1, 20.0, 8.2, 9.8]); // 20.0 is an outlier

    for (const method of ["bisquare", "huber", "talwar"]) {
        const result = new fastloess.Loess({
            fraction: 0.5,
            iterations: 5,
            robustness_method: method,
            return_robustness_weights: true,
        }).fit(x, y);
        const wStr = result.robustnessWeights
            ? Array.from(result.robustnessWeights).map(v => v.toFixed(3)).join(', ')
            : 'N/A';
        console.log(`  ${method}:`);
        console.log(`    Smoothed: [${Array.from(result.y).map(v => v.toFixed(2)).join(', ')}]`);
        console.log(`    Weights:  [${wStr}]`);
    }
    console.log();
}

// ── Example 8: Benchmark ─────────────────────────────────────────────────────
function example_8_benchmark() {
    console.log("Example 8: Benchmark");

    const n = 1000;
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        x[i] = i;
        y[i] = Math.sin(i * 0.1) + Math.cos(i * 0.01);
    }

    const t0 = process.hrtime.bigint();
    const result = new fastloess.Loess({ parallel: true }).fit(x, y);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;

    console.log(`  ${n} points in ${ms.toFixed(2)}ms`);
    console.log(`  fractionUsed=${result.fractionUsed}, y[0]=${result.y[0].toFixed(4)}`);
    console.log();
}

// ── Example 9: Scaling Methods (MAR, MAD, Mean) ──────────────────────────────
function example_9_scaling_methods() {
    console.log("Example 9: Scaling Methods");

    const { x, y } = makeLinear(20);

    for (const method of ["mar", "mad", "mean"]) {
        const result = new fastloess.Loess({ fraction: 0.5, scaling_method: method }).fit(x, y);
        console.log(`  ${method}: y[0]=${result.y[0].toFixed(3)}`);
    }
    console.log();
}

// ── Example 10: Boundary Policies ────────────────────────────────────────────
function example_10_boundary_policies() {
    console.log("Example 10: Boundary Policies");

    const { x, y } = makeLinear(30);

    for (const policy of ["extend", "reflect", "zero", "noboundary"]) {
        const result = new fastloess.Loess({ fraction: 0.5, boundary_policy: policy }).fit(x, y);
        console.log(
            `  ${policy}: first=${result.y[0].toFixed(2)}, last=${result.y[result.y.length - 1].toFixed(2)}`
        );
    }
    console.log();
}

// ── Example 11: Zero-Weight Fallback Strategies ───────────────────────────────
function example_11_zero_weight_fallback() {
    console.log("Example 11: Zero-Weight Fallback Strategies");

    const { x, y } = makeLinear(20);

    for (const fb of ["use_local_mean", "return_original", "return_none"]) {
        const result = new fastloess.Loess({ fraction: 0.5, zero_weight_fallback: fb }).fit(x, y);
        console.log(`  ${fb}: y[0]=${result.y[0].toFixed(3)}`);
    }
    console.log();
}

// ── Example 12: Polynomial Degrees + iterationsUsed ──────────────────────────
function example_12_polynomial_degrees() {
    console.log("Example 12: Polynomial Degrees");

    const { x, y } = makeLinear(30);

    for (const deg of ["constant", "linear", "quadratic", "cubic", "quartic"]) {
        const result = new fastloess.Loess({
            fraction: 0.5,
            iterations: 2,
            degree: deg,
        }).fit(x, y);
        console.log(
            `  ${deg}: y[0]=${result.y[0].toFixed(3)}, iterationsUsed=${result.iterationsUsed}`
        );
    }
    console.log();
}

// ── Example 13: Distance Metrics ─────────────────────────────────────────────
function example_13_distance_metrics() {
    console.log("Example 13: Distance Metrics");

    const { x, y } = makeLinear(20);

    for (const metric of ["euclidean", "normalized", "manhattan", "chebyshev"]) {
        const result = new fastloess.Loess({ fraction: 0.5, distance_metric: metric }).fit(x, y);
        console.log(`  ${metric}: y[0]=${result.y[0].toFixed(3)}`);
    }

    // Minkowski with custom p via "minkowski:p" format
    const rMink = new fastloess.Loess({ fraction: 0.5, distance_metric: "minkowski:3" }).fit(x, y);
    console.log(`  minkowski(p=3): y[0]=${rMink.y[0].toFixed(3)}`);

    console.log();
}

// ── Example 14: Surface Modes and Standard Errors ────────────────────────────
function example_14_surface_modes_and_se() {
    console.log("Example 14: Surface Modes and Standard Errors");

    const { x, y } = makeLinear(30);

    // Direct surface — fits every point exactly; SE fields fully populated
    const rDirect = new fastloess.Loess({
        fraction: 0.5,
        surface_mode: "direct",
        returnSe: true,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
    }).fit(x, y);

    console.log("  surface_mode=direct:");
    console.log(`    confidenceLower non-null: ${rDirect.confidenceLower != null}`);
    console.log(`    predictionLower non-null: ${rDirect.predictionLower != null}`);
    if (rDirect.standardErrors) console.log(`    standardErrors[0]: ${rDirect.standardErrors[0].toFixed(4)}`);
    if (rDirect.enp != null) console.log(`    enp: ${rDirect.enp.toFixed(3)}`);
    if (rDirect.traceHat != null) console.log(`    traceHat: ${rDirect.traceHat.toFixed(3)}`);
    if (rDirect.delta1 != null) console.log(`    delta1: ${rDirect.delta1.toFixed(3)}`);
    if (rDirect.delta2 != null) console.log(`    delta2: ${rDirect.delta2.toFixed(3)}`);
    if (rDirect.residualScale != null) console.log(`    residualScale: ${rDirect.residualScale.toFixed(4)}`);
    if (rDirect.leverage) console.log(`    leverage[0]: ${rDirect.leverage[0].toFixed(4)}`);

    // Interpolation surface — faster, approximate
    const rInterp = new fastloess.Loess({
        fraction: 0.5,
        surface_mode: "interpolation",
        returnSe: true,
    }).fit(x, y);

    console.log("  surface_mode=interpolation:");
    console.log(`    y[0]: ${rInterp.y[0].toFixed(3)}`);
    if (rInterp.standardErrors) console.log(`    standardErrors[0]: ${rInterp.standardErrors[0].toFixed(4)}`);
    console.log();
}

// ── Example 15: Additional Weight Functions (Uniform, Triangle, Cosine) ───────
function example_15_additional_kernels() {
    console.log("Example 15: Additional Weight Functions (Uniform, Triangle, Cosine)");

    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2.0, 4.1, 5.9, 8.2, 9.8]);

    for (const kernel of ["uniform", "triangle", "cosine"]) {
        const result = new fastloess.Loess({ fraction: 0.5, weight_function: kernel }).fit(x, y);
        console.log(`  ${kernel}: [${Array.from(result.y).map(v => v.toFixed(3)).join(', ')}]`);
    }
    console.log();
}

// ── Example 16: LOOCV, K-Fold, and Auto-Converge ─────────────────────────────
function example_16_loocv_and_auto_converge() {
    console.log("Example 16: LOOCV, K-Fold, and Auto-Converge");

    const n = 20;
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        x[i] = i + 1;
        y[i] = 2 * x[i] + 1 + Math.sin(x[i] * 0.5);
    }

    // Leave-one-out cross-validation
    const rLoocv = new fastloess.Loess({
        cvFractions: [0.3, 0.5, 0.7],
        cvMethod: "loocv",
    }).fit(x, y);
    console.log(`  LOOCV selected fraction: ${rLoocv.fractionUsed}`);
    if (rLoocv.cvScores) {
        console.log(`  LOOCV scores: [${Array.from(rLoocv.cvScores).map(v => v.toFixed(4)).join(', ')}]`);
    }

    // K-Fold cross-validation
    const rKfold = new fastloess.Loess({
        cvFractions: [0.2, 0.4, 0.6],
        cvMethod: "kfold",
        cvK: 5,
    }).fit(x, y);
    console.log(`  KFold(k=5) selected fraction: ${rKfold.fractionUsed}`);
    if (rKfold.cvScores) {
        console.log(`  KFold scores: [${Array.from(rKfold.cvScores).map(v => v.toFixed(4)).join(', ')}]`);
    }

    // Auto-converge: stop robustness iterations when change < tolerance
    const rAc = new fastloess.Loess({
        fraction: 0.5,
        auto_converge: 1e-4,
    }).fit(x, y);
    console.log(`  auto_converge=1e-4: iterationsUsed=${rAc.iterationsUsed}`);
    console.log();
}

// ── Example 17: Interpolation Tuning (surface_mode effects) ───────────────────
function example_17_interpolation_tuning() {
    console.log("Example 17: Interpolation Tuning (surface_mode effects)");

    const n = 50;
    const { x, y } = makeLinear(n);

    // Default (interpolation) — fastest, uses a spatial grid
    const rInterp = new fastloess.Loess({
        fraction: 0.5,
        surface_mode: "interpolation",
    }).fit(x, y);
    console.log(`  interpolation: y[0]=${rInterp.y[0].toFixed(3)}, y[-1]=${rInterp.y[n - 1].toFixed(3)}`);

    // Direct — fits every point exactly, more accurate but slower
    const rDirect = new fastloess.Loess({
        fraction: 0.5,
        surface_mode: "direct",
    }).fit(x, y);
    console.log(`  direct:        y[0]=${rDirect.y[0].toFixed(3)}, y[-1]=${rDirect.y[n - 1].toFixed(3)}`);

    // Fraction sweep with direct surface
    for (const frac of [0.2, 0.5, 0.8]) {
        const r = new fastloess.Loess({ fraction: frac, surface_mode: "direct" }).fit(x, y);
        console.log(`  direct fraction=${frac}: y[0]=${r.y[0].toFixed(3)}`);
    }

    // Interpolation + SE for hat-matrix statistics
    const rSe = new fastloess.Loess({
        fraction: 0.5,
        surface_mode: "interpolation",
        returnSe: true,
    }).fit(x, y);
    if (rSe.enp != null) console.log(`  interpolation+SE enp: ${rSe.enp.toFixed(3)}`);
    console.log();
}

// ── Main ──────────────────────────────────────────────────────────────────────
function main() {
    console.log("=".repeat(60));
    console.log("fastloess Batch Smoothing - Comprehensive Examples");
    console.log("=".repeat(60));
    console.log();

    example_1_basic_smoothing();
    example_2_robust_with_outliers();
    example_3_uncertainty_quantification();
    example_4_cross_validation();
    example_5_complete_diagnostics();
    example_6_different_kernels();
    example_7_robustness_methods();
    example_8_benchmark();
    example_9_scaling_methods();
    example_10_boundary_policies();
    example_11_zero_weight_fallback();
    example_12_polynomial_degrees();
    example_13_distance_metrics();
    example_14_surface_modes_and_se();
    example_15_additional_kernels();
    example_16_loocv_and_auto_converge();
    example_17_interpolation_tuning();

    console.log("=== Batch Smoothing Examples Complete ===");
}

main();

