const fastloess = require('../../bindings/nodejs');

/**
 * fastloess Streaming Smoothing - Comprehensive Examples
 *
 * 9 examples covering the full StreamingLoess API:
 *  1. Basic chunked processing
 *  2. Chunk size comparison
 *  3. Overlap strategies
 *  4. Large dataset processing
 *  5. Outlier handling in streaming mode
 *  6. File-based streaming simulation
 *  7. Benchmark (sequential streaming)
 *  8. Merge strategies (average, weighted_average, take_first, take_last)
 *  9. Advanced streaming options
 */

function makeLinear(n) {
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = i; y[i] = 2 * i + 1; }
    return { x, y };
}

// ── Example 1: Basic Chunked Processing ─────────────────────────────────────
function example_1_basic_chunked_processing() {
    console.log("Example 1: Basic Chunked Processing");

    const n = 50;
    const { x, y } = makeLinear(n);
    const chunkSize = 15;
    const overlap = 5;

    const streamer = new fastloess.StreamingLoess(
        { fraction: 0.5, iterations: 2, returnResiduals: true },
        { chunkSize, overlap }
    );

    console.log(`  Dataset: ${n} points, chunk=${chunkSize}, overlap=${overlap}`);

    let totalProcessed = 0;
    let chunkIdx = 0;
    for (let start = 0; start < n; start += chunkSize - overlap) {
        const end = Math.min(start + chunkSize, n);
        const res = streamer.processChunk(x.subarray(start, end), y.subarray(start, end));
        if (res.x.length > 0) {
            totalProcessed += res.x.length;
            console.log(`  Chunk ${chunkIdx}: ${res.x.length} pts (x: ${res.x[0].toFixed(0)}..${res.x[res.x.length - 1].toFixed(0)})`);
        }
        chunkIdx++;
    }
    const fin = streamer.finalize();
    if (fin.x.length > 0) {
        totalProcessed += fin.x.length;
        console.log(`  Finalize: ${fin.x.length} remaining pts`);
    }
    console.log(`  Total: ${totalProcessed}/${n}`);
    console.log();
}

// ── Example 2: Chunk Size Comparison ────────────────────────────────────────
function example_2_chunk_size_comparison() {
    console.log("Example 2: Chunk Size Comparison");

    const n = 100;
    const { x, y } = makeLinear(n);

    for (const [chunkSize, overlap, label] of [[20, 5, "Small"], [50, 10, "Medium"], [80, 15, "Large"]]) {
        const streamer = new fastloess.StreamingLoess(
            { fraction: 0.5, iterations: 1 },
            { chunkSize, overlap }
        );
        let chunks = 0, total = 0;
        for (let start = 0; start < n; start += chunkSize - overlap) {
            const end = Math.min(start + chunkSize, n);
            const res = streamer.processChunk(x.subarray(start, end), y.subarray(start, end));
            if (res.x.length > 0) { chunks++; total += res.x.length; }
        }
        const fin = streamer.finalize();
        if (fin.x.length > 0) { chunks++; total += fin.x.length; }
        console.log(`  ${label} (size=${chunkSize}, overlap=${overlap}): chunks=${chunks}, total=${total}`);
    }
    console.log();
}

// ── Example 3: Overlap Strategies ────────────────────────────────────────────
function example_3_overlap_strategies() {
    console.log("Example 3: Overlap Strategies");

    const n = 100;
    const { x, y } = makeLinear(n);

    for (const [overlap, label] of [[0, "No overlap"], [10, "10-pt overlap"], [20, "20-pt overlap"]]) {
        const chunkSize = 40;
        const streamer = new fastloess.StreamingLoess(
            { fraction: 0.5 },
            { chunkSize, overlap }
        );
        let total = 0;
        const step = chunkSize - overlap;
        // Feed only full-size chunks; finalize() handles remaining data
        for (let start = 0; start + chunkSize <= n; start += step) {
            const res = streamer.processChunk(x.subarray(start, start + chunkSize), y.subarray(start, start + chunkSize));
            total += res.x.length;
        }
        total += streamer.finalize().x.length;
        console.log(`  ${label} (overlap=${overlap}): total points output=${total}`);
    }
    console.log();
}

// ── Example 4: Large Dataset Processing ──────────────────────────────────────
function example_4_large_dataset_processing() {
    console.log("Example 4: Large Dataset Processing");

    const n = 10000;
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) { x[i] = i; y[i] = Math.sin(i * 0.01) + i * 0.001; }

    const chunkSize = 500;
    const overlap = 50;

    const streamer = new fastloess.StreamingLoess(
        { fraction: 0.05, iterations: 2 },
        { chunkSize, overlap }
    );

    let total = 0;
    const step = chunkSize - overlap;
    for (let start = 0; start < n; start += step) {
        const end = Math.min(start + chunkSize, n);
        const res = streamer.processChunk(x.subarray(start, end), y.subarray(start, end));
        total += res.x.length;
        if (total > 0 && total % 2000 < step) {
            console.log(`  Progress: ~${total} points smoothed`);
        }
    }
    total += streamer.finalize().x.length;
    console.log(`  Total processed: ${total}/${n}`);
    console.log(`  Memory efficiency: constant (chunk=${chunkSize})`);
    console.log();
}

// ── Example 5: Outlier Handling in Streaming Mode ─────────────────────────────
function example_5_outlier_handling() {
    console.log("Example 5: Outlier Handling in Streaming Mode");

    const n = 100;
    const x = new Float64Array(n);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        x[i] = i;
        y[i] = 2 * i + 1 + Math.sin(i * 0.2) * 2;
        if (i === 25 || i === 50 || i === 75) y[i] += 50; // Outliers
    }

    for (const method of ["bisquare", "huber", "talwar"]) {
        const streamer = new fastloess.StreamingLoess(
            { fraction: 0.5, iterations: 5, robustnessMethod: method, returnResiduals: true },
            { chunkSize: 30, overlap: 10 }
        );
        let largeResiduals = 0;
        for (let start = 0; start < n; start += 20) {
            const end = Math.min(start + 30, n);
            const res = streamer.processChunk(x.subarray(start, end), y.subarray(start, end));
            if (res.residuals) {
                for (const r of res.residuals) { if (Math.abs(r) > 10) largeResiduals++; }
            }
        }
        const fin = streamer.finalize();
        if (fin.residuals) {
            for (const r of fin.residuals) { if (Math.abs(r) > 10) largeResiduals++; }
        }
        console.log(`  ${method}: points with |residual|>10: ${largeResiduals}`);
    }
    console.log();
}

// ── Example 6: File-Based Streaming Simulation ───────────────────────────────
function example_6_file_simulation() {
    console.log("Example 6: File-Based Streaming Simulation");
    console.log("  Simulating: Read from input.csv -> Smooth -> Write to output.csv");

    const totalLines = 200;
    const chunkSize = 50;
    const overlap = 10;

    const streamer = new fastloess.StreamingLoess(
        { fraction: 0.5, iterations: 2, returnResiduals: true },
        { chunkSize, overlap }
    );

    let outputLines = 0;
    for (let ci = 0; ci < Math.ceil(totalLines / (chunkSize - overlap)); ci++) {
        const start = ci * (chunkSize - overlap);
        const end = Math.min(start + chunkSize, totalLines);

        // Simulate reading a chunk from a file
        const xChunk = new Float64Array(end - start);
        const yChunk = new Float64Array(end - start);
        for (let j = 0; j < end - start; j++) {
            xChunk[j] = start + j;
            yChunk[j] = 2 * xChunk[j] + 1 + Math.sin(xChunk[j] * 0.1) * 3;
        }

        console.log(`  Reading chunk ${ci} (lines ${start}..${end - 1})`);
        const res = streamer.processChunk(xChunk, yChunk);
        if (res.x.length > 0) {
            outputLines += res.x.length;
            console.log(`    -> Writing ${res.x.length} smoothed pts (total: ${outputLines})`);
        }
    }

    const fin = streamer.finalize();
    if (fin.x.length > 0) {
        outputLines += fin.x.length;
        console.log(`  Finalizing: Writing ${fin.x.length} remaining pts`);
    }
    console.log(`  Input: ${totalLines}, Output: ${outputLines}`);
    console.log();
}

// ── Example 7: Benchmark (Sequential Streaming) ───────────────────────────────
function example_7_benchmark() {
    console.log("Example 7: Benchmark (Sequential Streaming)");

    const n = 1000;
    const chunkSize = 100;
    const overlap = 10;

    const streamer = new fastloess.StreamingLoess(
        { fraction: 0.5, iterations: 3 },
        { chunkSize, overlap }
    );

    const t0 = process.hrtime.bigint();
    let total = 0;
    const step = chunkSize - overlap;
    for (let start = 0; start < n; start += step) {
        const end = Math.min(start + chunkSize, n);
        const xc = new Float64Array(end - start);
        const yc = new Float64Array(end - start);
        for (let j = 0; j < end - start; j++) {
            xc[j] = start + j;
            yc[j] = Math.sin(xc[j] * 0.1) + Math.cos(xc[j] * 0.01);
        }
        total += streamer.processChunk(xc, yc).x.length;
    }
    total += streamer.finalize().x.length;
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;

    console.log(`  ${total} points in ${ms.toFixed(2)}ms`);
    console.log(`  chunk=${chunkSize}, overlap=${overlap}`);
    console.log();
}

// ── Example 8: Merge Strategies ──────────────────────────────────────────────
function example_8_merge_strategies() {
    console.log("Example 8: Merge Strategies");

    const n = 50;
    const { x, y } = makeLinear(n);

    for (const strategy of ["average", "weighted_average", "take_first", "take_last"]) {
        const streamer = new fastloess.StreamingLoess(
            { fraction: 0.5, iterations: 2 },
            { chunkSize: 20, overlap: 5, mergeStrategy: strategy }
        );
        let total = 0;
        for (let start = 0; start < n; start += 15) {
            const end = Math.min(start + 20, n);
            total += streamer.processChunk(x.subarray(start, end), y.subarray(start, end)).x.length;
        }
        total += streamer.finalize().x.length;
        console.log(`  ${strategy}: total=${total}`);
    }
    console.log();
}

// ── Example 9: Advanced Streaming Options ─────────────────────────────────────
function example_9_advanced_options() {
    console.log("Example 9: Advanced Streaming Options");

    const n = 50;
    const { x, y } = makeLinear(n);

    const streamer = new fastloess.StreamingLoess(
        {
            fraction: 0.5,
            iterations: 2,
            degree: "quadratic",
            scalingMethod: "mar",
            boundaryPolicy: "reflect",
            zeroWeightFallback: "return_original",
            distanceMetric: "manhattan",
            surfaceMode: "direct",
            returnSe: true,
            returnDiagnostics: true,
            returnRobustnessWeights: true,
            autoConverge: 1e-3,
        },
        { chunkSize: 20, overlap: 5 }
    );

    let total = 0;
    for (let start = 0; start < n; start += 15) {
        const end = Math.min(start + 20, n);
        total += streamer.processChunk(x.subarray(start, end), y.subarray(start, end)).x.length;
    }
    const fin = streamer.finalize();
    total += fin.x.length;

    console.log(`  total points: ${total}`);
    if (fin.standardErrors && fin.standardErrors.length > 0) {
        console.log(`  standardErrors[0]: ${fin.standardErrors[0].toFixed(4)}`);
    }
    if (fin.diagnostics) {
        console.log(`  diagnostics.rmse: ${fin.diagnostics.rmse.toFixed(3)}`);
        console.log(`  diagnostics.rSquared: ${fin.diagnostics.rSquared.toFixed(3)}`);
        if (fin.diagnostics.aic != null) console.log(`  diagnostics.aic: ${fin.diagnostics.aic.toFixed(3)}`);
    }
    if (fin.robustnessWeights && fin.robustnessWeights.length > 0) {
        console.log(`  robustnessWeights[0]: ${fin.robustnessWeights[0].toFixed(4)}`);
    }
    console.log();
}

// ── Main ──────────────────────────────────────────────────────────────────────
function main() {
    console.log("=".repeat(60));
    console.log("fastloess Streaming Smoothing - Comprehensive Examples");
    console.log("=".repeat(60));
    console.log();

    example_1_basic_chunked_processing();
    example_2_chunk_size_comparison();
    example_3_overlap_strategies();
    example_4_large_dataset_processing();
    example_5_outlier_handling();
    example_6_file_simulation();
    example_7_benchmark();
    example_8_merge_strategies();
    example_9_advanced_options();

    console.log("=== Streaming Smoothing Examples Complete ===");
}

main();

