const test = require('node:test');
const assert = require('node:assert');

// Import WASM bindings using require (works in Node with generated pkg)
const fastloess = require('../pkg/fastloess_wasm.js');

test('WASM batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastloess.Loess({
        fraction: 0.3,
        return_diagnostics: true
    }).fit(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    // Check diagnostics using getters
    assert.ok(result.diagnostics.rmse < 0.1);
});

test('WASM return_sorted defaults to original input order', () => {
    const x = new Float64Array([3, 1, 5, 2, 4]);
    const y = new Float64Array([6, 2, 10, 4, 8]);

    const result = new fastloess.Loess({ fraction: 0.7 }).fit(x, y);

    assert.deepStrictEqual(Array.from(result.x), Array.from(x));
});

test('WASM return_sorted = true returns results sorted ascending by x', () => {
    const x = new Float64Array([3, 1, 5, 2, 4]);
    const y = new Float64Array([6, 2, 10, 4, 8]);

    const result = new fastloess.Loess({
        fraction: 0.7,
        return_residuals: true,
        return_robustness_weights: true,
        return_sorted: true
    }).fit(x, y);

    for (let i = 1; i < result.x.length; i++) {
        assert.ok(result.x[i - 1] <= result.x[i]);
    }
    assert.notDeepStrictEqual(Array.from(result.x), Array.from(x));

    const unsortedResult = new fastloess.Loess({
        fraction: 0.7,
        return_residuals: true,
        return_robustness_weights: true
    }).fit(x, y);

    const sortedPairs = Array.from(result.x).map((xv, i) => [xv, result.y[i]]).sort();
    const unsortedPairs = Array.from(unsortedResult.x).map((xv, i) => [xv, unsortedResult.y[i]]).sort();
    assert.deepStrictEqual(sortedPairs, unsortedPairs);

    assert.strictEqual(result.residuals.length, x.length);
    assert.strictEqual(result.robustness_weights.length, x.length);
});

test('WASM streaming smoothing', () => {
    const streamer = new fastloess.StreamingLoess({
        fraction: 0.3
    }, {
        chunk_size: 10,
        overlap: 2
    });

    const x = new Float64Array(Array.from({ length: 20 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 2));

    const result = streamer.process_chunk(x, y);
    // WASM process_chunk returns a struct, safe to check .y existence/length if populated
    if (result) {
        assert.ok(result.y.length >= 0);
    }

    const finalResult = streamer.finalize();
    if (finalResult) {
        assert.ok(finalResult.y.length > 0);
    }
});

test('WASM online smoothing', () => {
    const online = new fastloess.OnlineLoess({
        fraction: 0.5
    }, {
        window_capacity: 10,
        min_points: 2
    });

    let lastSmoothed;
    for (let i = 0; i < 10; i++) {
        const res = online.add_point(i, i * 2);
        if (res !== undefined && res !== null) {
            lastSmoothed = res.y;
        }
    }

    assert.ok(lastSmoothed !== undefined && lastSmoothed !== null);
    assert.ok(Math.abs(lastSmoothed - 18) < 1.0);
});

test('WASM options parsing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastloess.Loess({
        weight_function: 'tricube',
        robustness_method: 'bisquare',
        boundary_policy: 'extend',
        scaling_method: 'mad'
    }).fit(x, y);

    assert.strictEqual(result.y.length, 5);
});

// ---- Parameter coverage tests ----

test('WASM smooth: iterations, zero_weight_fallback, return_residuals, return_robustness_weights', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastloess.Loess({
        fraction: 0.7,
        iterations: 5,
        zero_weight_fallback: 'return_original',
        return_residuals: true,
        return_robustness_weights: true,
    }).fit(x, y);
    assert.strictEqual(result.y.length, 5);
});

test('WASM smooth: confidence_intervals, prediction_intervals', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const result = new fastloess.Loess({
        fraction: 0.5,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
    }).fit(x, y);
    assert.ok(result.confidence_lower !== null);
    assert.ok(result.prediction_upper !== null);
});

test('WASM smooth: degree, surface_mode, distance_metric', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([1, 4, 9, 16, 25]);

    for (const deg of ['constant', 'linear', 'quadratic']) {
        const r = new fastloess.Loess({ fraction: 0.9, degree: deg }).fit(x, y);
        assert.strictEqual(r.y.length, 5);
    }
    for (const dm of ['normalized', 'euclidean', 'manhattan', 'chebyshev', 'minkowski']) {
        const r = new fastloess.Loess({ fraction: 0.5, distance_metric: dm }).fit(x, y);
        assert.strictEqual(r.y.length, 5);
    }
    const r2 = new fastloess.Loess({ fraction: 0.5, surface_mode: 'direct' }).fit(x, y);
    assert.strictEqual(r2.y.length, 5);
});

test('WASM smooth: return_se', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const result = new fastloess.Loess({ fraction: 0.5, return_se: true, surface_mode: 'direct' }).fit(x, y);
    assert.ok(result.enp !== null);
    assert.ok(result.trace_hat !== null);
});

test('WASM smooth: auto_converge, parallel', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastloess.Loess({
        fraction: 0.5,
        auto_converge: 1e-4,
        parallel: false,
    }).fit(x, y);
    assert.strictEqual(result.y.length, 5);
});

test('WASM streaming: merge_strategy', () => {
    const x = new Float64Array(Array.from({ length: 40 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 40 }, (_, i) => i * 2));

    for (const ms of ['average', 'weighted_average', 'take_first', 'take_last']) {
        const s = new fastloess.StreamingLoess(
            { fraction: 0.3 },
            { chunk_size: 20, overlap: 2, merge_strategy: ms }
        );
        s.process_chunk(x, y);
        const r = s.finalize();
        assert.ok(r.y.length >= 0);
    }
});

test('WASM online: update mode via options', () => {
    const online = new fastloess.OnlineLoess(
        { fraction: 0.5, degree: 'linear', distance_metric: 'euclidean' },
        { window_capacity: 10, min_points: 2 }
    );

    let lastSmoothed;
    for (let i = 0; i < 10; i++) {
        const res = online.add_point(i, i * 2);
        if (res !== undefined && res !== null) {
            lastSmoothed = res.y;
        }
    }
    assert.ok(lastSmoothed !== undefined && lastSmoothed !== null);
});

test('WASM smooth: missing default ("error") rejects NaN', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, NaN, 6, 8, 10]);

    assert.throws(() => new fastloess.Loess({ fraction: 0.5 }).fit(x, y));
});

test('WASM smooth: missing = "drop" removes non-finite rows', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, NaN, 6, 8, 10]);

    const result = new fastloess.Loess({ fraction: 0.5, missing: 'drop' }).fit(x, y);
    assert.strictEqual(result.y.length, 4);
});

test('WASM streaming: missing = "drop" removes non-finite rows', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, NaN, 8, 10, 12, 14, 16, 18, 20]);

    const streamer = new fastloess.StreamingLoess(
        { fraction: 0.5, missing: 'drop' },
        { chunk_size: 10 }
    );
    const r1 = streamer.process_chunk(x, y);
    const r2 = streamer.finalize();
    assert.strictEqual(r1.y.length + r2.y.length, 9);
});

test('WASM online: missing = "drop" ignores non-finite point', () => {
    const online = new fastloess.OnlineLoess(
        { fraction: 0.5, missing: 'drop' },
        { window_capacity: 10 }
    );
    const r = online.add_point(1.0, NaN);
    assert.ok(r === undefined || r === null);
});

test('WASM smooth: invalid missing policy throws', () => {
    const x = new Float64Array([1, 2, 3]);
    const y = new Float64Array([1, 2, 3]);
    assert.throws(() => new fastloess.Loess({ fraction: 0.5, missing: 'invalid' }).fit(x, y));
});

test('WASM custom weights: zero on outlier reduces error', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const yOutlier = new Float64Array([1, 2, 3, 100, 5, 6, 7]);
    const yTrue = [1, 2, 3, 4, 5, 6, 7];
    const wZero = [1, 1, 1, 0, 1, 1, 1];

    const rNoW = new fastloess.Loess({ fraction: 0.6 }).fit(x, yOutlier);
    const rW = new fastloess.Loess({ fraction: 0.6 }).fit(x, yOutlier, wZero);

    const nonOutlier = [0, 1, 2, 4, 5, 6];
    const errNoW = nonOutlier.reduce((s, i) => s + Math.abs(rNoW.y[i] - yTrue[i]), 0) / nonOutlier.length;
    const errW = nonOutlier.reduce((s, i) => s + Math.abs(rW.y[i] - yTrue[i]), 0) / nonOutlier.length;
    assert.ok(errW < errNoW, `expected ${errW} < ${errNoW}`);
});

test('WASM custom weights: uniform equals no weights', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const wUniform = [1, 1, 1, 1, 1, 1, 1];

    const rNoW = new fastloess.Loess({ fraction: 0.6 }).fit(x, y);
    const rW = new fastloess.Loess({ fraction: 0.6 }).fit(x, y, wUniform);

    for (let i = 0; i < rNoW.y.length; i++) {
        assert.ok(Math.abs(rW.y[i] - rNoW.y[i]) < 1e-6, `mismatch at index ${i}`);
    }
});

test('WASM custom weights: wrong length throws', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const wBad = [1, 1, 1];

    assert.throws(() => new fastloess.Loess({ fraction: 0.6 }).fit(x, y, wBad));
});

test('WASM custom weights: negative weight throws', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const wNeg = [1, -1, 1, 1, 1, 1, 1];

    assert.throws(() => new fastloess.Loess({ fraction: 0.6 }).fit(x, y, wNeg));
});
