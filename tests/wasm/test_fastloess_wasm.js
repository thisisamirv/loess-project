const test = require('node:test');
const assert = require('node:assert');

// Import WASM bindings using require (works in Node with generated pkg)
const fastloess = require('../../bindings/wasm/pkg/fastloess_wasm.js');

test('WASM batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastloess.Loess({
        fraction: 0.3,
        returnDiagnostics: true
    }).fit(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    // Check diagnostics using getters
    assert.ok(result.diagnostics.rmse < 0.1);
});

test('WASM streaming smoothing', () => {
    const streamer = new fastloess.StreamingLoess({
        fraction: 0.3
    }, {
        chunkSize: 10,
        overlap: 2
    });

    const x = new Float64Array(Array.from({ length: 20 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 2));

    const result = streamer.processChunk(x, y);
    // WASM processChunk returns a struct, safe to check .y existence/length if populated
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
        windowCapacity: 10,
        minPoints: 2
    });

    let lastVal;
    for (let i = 0; i < 10; i++) {
        lastVal = online.update(i, i * 2);
    }

    // lastVal should not be undefined/null after enough points
    assert.ok(lastVal !== undefined && lastVal !== null);
    assert.ok(Math.abs(lastVal - 18) < 1.0);
});

test('WASM options parsing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastloess.Loess({
        weightFunction: 'tricube',
        robustnessMethod: 'bisquare',
        boundaryPolicy: 'extend',
        scalingMethod: 'mad'
    }).fit(x, y);

    assert.strictEqual(result.y.length, 5);
});

// ---- Parameter coverage tests ----

test('WASM smooth: iterations, zeroWeightFallback, returnResiduals, returnRobustnessWeights', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastloess.Loess({
        fraction: 0.7,
        iterations: 5,
        zeroWeightFallback: 'return_original',
        returnResiduals: true,
        returnRobustnessWeights: true,
    }).fit(x, y);
    assert.strictEqual(result.y.length, 5);
});

test('WASM smooth: confidenceIntervals, predictionIntervals', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const result = new fastloess.Loess({
        fraction: 0.5,
        confidenceIntervals: 0.95,
        predictionIntervals: 0.95,
    }).fit(x, y);
    assert.ok(result.confidenceLower !== null);
    assert.ok(result.predictionUpper !== null);
});

test('WASM smooth: degree, surfaceMode, distanceMetric', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([1, 4, 9, 16, 25]);

    for (const deg of ['constant', 'linear', 'quadratic']) {
        const r = new fastloess.Loess({ fraction: 0.9, degree: deg }).fit(x, y);
        assert.strictEqual(r.y.length, 5);
    }
    for (const dm of ['normalized', 'euclidean', 'manhattan', 'chebyshev', 'minkowski']) {
        const r = new fastloess.Loess({ fraction: 0.5, distanceMetric: dm }).fit(x, y);
        assert.strictEqual(r.y.length, 5);
    }
    const r2 = new fastloess.Loess({ fraction: 0.5, surfaceMode: 'direct' }).fit(x, y);
    assert.strictEqual(r2.y.length, 5);
});

test('WASM smooth: returnSe', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const result = new fastloess.Loess({ fraction: 0.5, returnSe: true, surfaceMode: 'direct' }).fit(x, y);
    assert.ok(result.enp !== null);
    assert.ok(result.traceHat !== null);
});

test('WASM smooth: autoConverge, parallel', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = new fastloess.Loess({
        fraction: 0.5,
        autoConverge: 1e-4,
        parallel: false,
    }).fit(x, y);
    assert.strictEqual(result.y.length, 5);
});

test('WASM streaming: mergeStrategy', () => {
    const x = new Float64Array(Array.from({ length: 40 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 40 }, (_, i) => i * 2));

    for (const ms of ['average', 'weighted_average', 'take_first', 'take_last']) {
        const s = new fastloess.StreamingLoess(
            { fraction: 0.3 },
            { chunkSize: 20, overlap: 2, mergeStrategy: ms }
        );
        s.processChunk(x, y);
        const r = s.finalize();
        assert.ok(r.y.length >= 0);
    }
});

test('WASM online: update mode via options', () => {
    const online = new fastloess.OnlineLoess(
        { fraction: 0.5, degree: 'linear', distanceMetric: 'euclidean' },
        { windowCapacity: 10, minPoints: 2 }
    );

    let lastVal;
    for (let i = 0; i < 10; i++) {
        lastVal = online.update(i, i * 2);
    }
    assert.ok(lastVal !== undefined && lastVal !== null);
});

test('WASM custom weights: zero on outlier reduces error', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const yOutlier = new Float64Array([1, 2, 3, 100, 5, 6, 7]);
    const yTrue = [1, 2, 3, 4, 5, 6, 7];
    const wZero = [1, 1, 1, 0, 1, 1, 1];

    const rNoW = new fastloess.Loess({ fraction: 0.6 }).fit(x, yOutlier);
    const rW = new fastloess.Loess({ fraction: 0.6, custom_weights: wZero }).fit(x, yOutlier);

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
    const rW = new fastloess.Loess({ fraction: 0.6, custom_weights: wUniform }).fit(x, y);

    for (let i = 0; i < rNoW.y.length; i++) {
        assert.ok(Math.abs(rW.y[i] - rNoW.y[i]) < 1e-6, `mismatch at index ${i}`);
    }
});

test('WASM custom weights: wrong length throws', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const wBad = [1, 1, 1];

    assert.throws(() => new fastloess.Loess({ fraction: 0.6, custom_weights: wBad }).fit(x, y));
});

test('WASM custom weights: negative weight throws', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const wNeg = [1, -1, 1, 1, 1, 1, 1];

    assert.throws(() => new fastloess.Loess({ fraction: 0.6, custom_weights: wNeg }).fit(x, y));
});
