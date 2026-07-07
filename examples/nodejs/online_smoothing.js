const fastloess = require('../../bindings/nodejs');

/**
 * fastloess Online Smoothing - Comprehensive Examples
 *
 * 9 examples covering the full OnlineLoess API:
 *  1. Basic incremental processing
 *  2. Real-time sensor data simulation
 *  3. Outlier handling in online mode
 *  4. Window size comparison
 *  5. Memory-bounded processing (embedded systems)
 *  6. Sliding window behavior
 *  7. Benchmark (sequential online)
 *  8. Update modes (Full vs Incremental) and min_points
 *  9. Advanced online options
 */

// ── Example 1: Basic Incremental Processing ──────────────────────────────────
function example_1_basic_streaming() {
    console.log("Example 1: Basic Incremental Processing");

    const data = [
        [1, 3.1], [2, 5.0], [3, 7.2], [4, 8.9], [5, 11.1],
        [6, 13.0], [7, 15.2], [8, 16.8], [9, 19.1], [10, 21.0],
    ];

    const model = new fastloess.OnlineLoess(
        { fraction: 0.5, iterations: 2, return_residuals: true },
        { window_capacity: 5 }
    );

    console.log(`  ${"X".padStart(8)} ${"Y_obs".padStart(12)} ${"Y_smooth".padStart(12)}`);
    for (const [x, y] of data) {
        const res = model.add_point(x, y);
        const smoothed = res !== null ? res.smoothed.toFixed(2) : "(buffering)";
        console.log(`  ${x.toFixed(2).padStart(8)} ${y.toFixed(2).padStart(12)} ${smoothed.padStart(12)}`);
    }
    console.log();
}

// ── Example 2: Real-Time Sensor Data Simulation ───────────────────────────────
function example_2_sensor_data_simulation() {
    console.log("Example 2: Real-Time Sensor Data Simulation");
    console.log("  Simulating temperature sensor readings with noise...");

    const n = 24; // 24 hours
    const model = new fastloess.OnlineLoess(
        { fraction: 0.4, iterations: 3, robustness_method: "bisquare", return_residuals: true },
        { window_capacity: 12 }
    );

    console.log(`  ${"Hour".padStart(6)} ${"Raw".padStart(12)} ${"Smoothed".padStart(12)}`);
    for (let hour = 0; hour < n; hour++) {
        const baseTemp = 20.0;
        const cycle = 5.0 * Math.sin(hour * Math.PI / 12.0);
        const noise = ((hour * 7) % 11) * 0.3 - 1.5;
        const temp = baseTemp + cycle + noise;

        const res = model.add_point(hour, temp);
        if (res !== null) {
            console.log(
                `  ${hour.toString().padStart(6)} ${temp.toFixed(2).padStart(12)}°C ${res.smoothed.toFixed(2).padStart(12)}°C`
            );
        } else {
            console.log(`  ${hour.toString().padStart(6)} ${temp.toFixed(2).padStart(12)}°C ${"(warming up)".padStart(13)}`);
        }
    }
    console.log();
}

// ── Example 3: Outlier Handling in Online Mode ────────────────────────────────
function example_3_outlier_handling() {
    console.log("Example 3: Outlier Handling in Online Mode");

    const data = [
        [1, 2.0], [2, 4.1], [3, 5.9],
        [4, 25.0], // Outlier!
        [5, 10.1], [6, 12.0], [7, 14.1],
        [8, 50.0], // Outlier!
        [9, 18.0], [10, 20.1],
    ];

    for (const method of ["bisquare", "talwar"]) {
        const model = new fastloess.OnlineLoess(
            { fraction: 0.5, iterations: 5, robustness_method: method, return_residuals: true },
            { window_capacity: 6 }
        );
        const smoothed = [];
        for (const [x, y] of data) {
            const res = model.add_point(x, y);
            if (res !== null) smoothed.push(res.smoothed.toFixed(1));
        }
        console.log(`  ${method}: [${smoothed.join(', ')}]`);
    }
    console.log();
}

// ── Example 4: Window Size Comparison ────────────────────────────────────────
function example_4_window_size_comparison() {
    console.log("Example 4: Window Size Comparison");

    const data = Array.from({ length: 20 }, (_, i) => {
        const x = i + 1;
        return [x, 2 * x + Math.sin(x * 0.5) * 3];
    });

    for (const windowSize of [5, 10, 15]) {
        const model = new fastloess.OnlineLoess(
            { fraction: 0.5, iterations: 2 },
            { window_capacity: windowSize }
        );
        const smoothed = [];
        for (const [x, y] of data) {
            const res = model.add_point(x, y);
            if (res !== null) smoothed.push(res.smoothed);
        }
        const last5 = smoothed.slice(-5).map(v => v.toFixed(2));
        console.log(`  window_capacity=${windowSize}: last 5 = [${last5.join(', ')}]`);
    }
    console.log();
}

// ── Example 5: Memory-Bounded Processing ──────────────────────────────────────
function example_5_memory_bounded_processing() {
    console.log("Example 5: Memory-Bounded Processing (Embedded Systems)");

    const total = 1000;
    const model = new fastloess.OnlineLoess(
        { fraction: 0.3, iterations: 1 },
        { window_capacity: 20 }
    );

    let count = 0;
    let lastSmoothed = 0;
    for (let i = 0; i < total; i++) {
        const x = i;
        const y = 2 * x + Math.sin(x * 0.1) * 5 + ((i % 7) - 3) * 0.5;
        const res = model.add_point(x, y);
        if (res !== null) {
            count++;
            lastSmoothed = res.smoothed;
            if (count % 200 === 0) {
                console.log(`  Processed: ${count.toString().padStart(4)} pts | smoothed=${lastSmoothed.toFixed(2)}`);
            }
        }
    }
    console.log(`  Total processed: ${count}, final smoothed: ${lastSmoothed.toFixed(2)}`);
    console.log(`  Memory: constant (window=20)`);
    console.log();
}

// ── Example 6: Sliding Window Behavior ───────────────────────────────────────
function example_6_sliding_window_behavior() {
    console.log("Example 6: Sliding Window Behavior");

    const data = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16]];
    const model = new fastloess.OnlineLoess(
        { fraction: 0.6, iterations: 0, return_residuals: true },
        { window_capacity: 4 }
    );

    console.log(`  ${"Pt".padStart(4)} ${"X".padStart(6)} ${"Y".padStart(8)} ${"Smoothed".padStart(10)} ${"Status".padStart(22)}`);
    data.forEach(([x, y], i) => {
        const res = model.add_point(x, y);
        if (res !== null) {
            console.log(`  ${(i + 1).toString().padStart(4)} ${x.toFixed(0).padStart(6)} ${y.toFixed(0).padStart(8)} ${res.smoothed.toFixed(2).padStart(10)} ${"Window full (sliding)".padStart(22)}`);
        } else {
            console.log(`  ${(i + 1).toString().padStart(4)} ${x.toFixed(0).padStart(6)} ${y.toFixed(0).padStart(8)} ${"-".padStart(10)} ${`Filling (${i + 1}/4)`.padStart(22)}`);
        }
    });
    console.log("  Output starts after window fills (4 pts), then slides.");
    console.log();
}

// ── Example 7: Benchmark (Sequential Online) ──────────────────────────────────
function example_7_benchmark() {
    console.log("Example 7: Benchmark (Sequential Online)");

    const n = 1000;
    const model = new fastloess.OnlineLoess(
        { fraction: 0.5, iterations: 3 },
        { window_capacity: 10 }
    );

    const t0 = process.hrtime.bigint();
    let count = 0;
    for (let i = 0; i < n; i++) {
        const x = i;
        const y = Math.sin(x * 0.1) + Math.cos(x * 0.01);
        const res = model.add_point(x, y);
        if (res !== null) count++;
    }
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;

    console.log(`  ${count} pts processed in ${ms.toFixed(2)}ms`);
    console.log(`  window_capacity=10`);
    console.log();
}

// ── Example 8: Update Modes (Full vs Incremental) and min_points ───────────────
function example_8_update_modes() {
    console.log("Example 8: Update Modes (Full vs Incremental) and min_points");

    const data = Array.from({ length: 30 }, (_, i) => [i, 2 * i + 1]);

    for (const mode of ["full", "incremental"]) {
        const model = new fastloess.OnlineLoess(
            { fraction: 0.5, iterations: 2 },
            { window_capacity: 15, min_points: 5, update_mode: mode }
        );
        let emitted = 0;
        for (const [x, y] of data) {
            const res = model.add_point(x, y);
            if (res !== null) emitted++;
        }
        console.log(`  ${mode}: ${emitted} points emitted (out of ${data.length})`);
    }

    // Show iterations_used from the returned OnlineOutput
    const model = new fastloess.OnlineLoess(
        { fraction: 0.5, iterations: 2, return_residuals: true, return_robustness_weights: true },
        { window_capacity: 10, min_points: 3 }
    );
    let lastSmoothed = null;
    let lastIterations = null;
    for (const [x, y] of data) {
        const res = model.add_point(x, y);
        if (res !== null) { lastSmoothed = res.smoothed; lastIterations = res.iterationsUsed; }
    }
    if (lastSmoothed !== null) {
        console.log(`  last smoothed: ${lastSmoothed.toFixed(3)}`);
        if (lastIterations !== null) console.log(`  iterations_used: ${lastIterations}`);
    }
    console.log();
}

// ── Example 9: Advanced Online Options ────────────────────────────────────────
function example_9_advanced_online_options() {
    console.log("Example 9: Advanced Online Options");

    const data = Array.from({ length: 30 }, (_, i) => [i, 2 * i + 1]);

    const model = new fastloess.OnlineLoess(
        {
            fraction: 0.5,
            iterations: 2,
            degree: "quadratic",
            scaling_method: "mar",
            boundary_policy: "reflect",
            zero_weight_fallback: "return_original",
            distance_metric: "chebyshev",
            auto_converge: 1e-3,
            return_residuals: true,
            return_robustness_weights: true,
        },
        { window_capacity: 15, min_points: 5 }
    );

    let emitted = 0;
    let lastSmoothed = null;
    for (const [x, y] of data) {
        const res = model.add_point(x, y);
        if (res !== null) { emitted++; lastSmoothed = res.smoothed; }
    }

    console.log(`  emitted: ${emitted}`);
    if (lastSmoothed !== null) {
        console.log(`  last smoothed: ${lastSmoothed.toFixed(3)}`);
    }
    console.log();
}

// ── Main ──────────────────────────────────────────────────────────────────────
function main() {
    console.log("=".repeat(60));
    console.log("fastloess Online Smoothing - Comprehensive Examples");
    console.log("=".repeat(60));
    console.log();

    example_1_basic_streaming();
    example_2_sensor_data_simulation();
    example_3_outlier_handling();
    example_4_window_size_comparison();
    example_5_memory_bounded_processing();
    example_6_sliding_window_behavior();
    example_7_benchmark();
    example_8_update_modes();
    example_9_advanced_online_options();

    console.log("=== Online Smoothing Examples Complete ===");
}

main();

