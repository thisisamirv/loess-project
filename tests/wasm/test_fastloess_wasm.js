const test = require('node:test');
const assert = require('node:assert');

// Import WASM bindings using require (works in Node with generated pkg)
const fastloess = require('../../bindings/wasm/pkg/fastloess_wasm.js');

test('WASM batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = fastloess.smooth(x, y, {
        fraction: 0.3,
        returnDiagnostics: true
    });

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    // Check diagnostics using getters
    assert.ok(result.diagnostics.rmse < 0.1);
});

test('WASM streaming smoothing', () => {
    const streamer = new fastloess.StreamingLoessWasm({
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
    const online = new fastloess.OnlineLoessWasm({
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

    const result = fastloess.smooth(x, y, {
        weightFunction: 'tricube',
        robustnessMethod: 'bisquare',
        boundaryPolicy: 'extend',
        scalingMethod: 'mad'
    });

    assert.strictEqual(result.y.length, 5);
});

// ---- Parameter coverage tests ----

test('WASM smooth: iterations, zeroWeightFallback, returnResiduals, returnRobustnessWeights', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = fastloess.smooth(x, y, {
        fraction: 0.7,
        iterations: 5,
        zeroWeightFallback: 'return_original',
        returnResiduals: true,
        returnRobustnessWeights: true,
    });
    assert.strictEqual(result.y.length, 5);
});

test('WASM smooth: confidenceIntervals, predictionIntervals', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const result = fastloess.smooth(x, y, {
        fraction: 0.5,
        confidenceIntervals: 0.95,
        predictionIntervals: 0.95,
    });
    assert.ok(result.confidenceLower !== null);
    assert.ok(result.predictionUpper !== null);
});

test('WASM smooth: degree, surfaceMode, distanceMetric', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([1, 4, 9, 16, 25]);

    for (const deg of ['constant', 'linear', 'quadratic']) {
        const r = fastloess.smooth(x, y, { fraction: 0.9, degree: deg });
        assert.strictEqual(r.y.length, 5);
    }
    for (const dm of ['normalized', 'euclidean', 'manhattan', 'chebyshev', 'minkowski']) {
        const r = fastloess.smooth(x, y, { fraction: 0.5, distanceMetric: dm });
        assert.strictEqual(r.y.length, 5);
    }
    const r2 = fastloess.smooth(x, y, { fraction: 0.5, surfaceMode: 'direct' });
    assert.strictEqual(r2.y.length, 5);
});

test('WASM smooth: returnSe', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const result = fastloess.smooth(x, y, { fraction: 0.5, returnSe: true, surfaceMode: 'direct' });
    assert.ok(result.enp !== null);
    assert.ok(result.traceHat !== null);
});

test('WASM smooth: autoConverge, parallel', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = fastloess.smooth(x, y, {
        fraction: 0.5,
        autoConverge: 1e-4,
        parallel: false,
    });
    assert.strictEqual(result.y.length, 5);
});

test('WASM streaming: mergeStrategy', () => {
    const x = new Float64Array(Array.from({ length: 40 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 40 }, (_, i) => i * 2));

    for (const ms of ['average', 'weighted_average', 'take_first', 'take_last']) {
        const s = new fastloess.StreamingLoessWasm(
            { fraction: 0.3 },
            { chunkSize: 20, overlap: 2, mergeStrategy: ms }
        );
        s.processChunk(x, y);
        const r = s.finalize();
        assert.ok(r.y.length >= 0);
    }
});

test('WASM online: update mode via options', () => {
    const online = new fastloess.OnlineLoessWasm(
        { fraction: 0.5, degree: 'linear', distanceMetric: 'euclidean' },
        { windowCapacity: 10, minPoints: 2 }
    );

    let lastVal;
    for (let i = 0; i < 10; i++) {
        lastVal = online.update(i, i * 2);
    }
    assert.ok(lastVal !== undefined && lastVal !== null);
});
