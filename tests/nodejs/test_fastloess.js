const test = require('node:test');
const assert = require('node:assert');

const fastloess = require('../../bindings/nodejs');

test('batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastloess.Loess({
        fraction: 0.3,
        return_diagnostics: true
    });

    const result = model.fit(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    assert.ok(result.diagnostics.rmse < 0.1);
});

test('streaming smoothing', () => {
    const streamer = new fastloess.StreamingLoess({
        fraction: 0.3
    }, {
        chunk_size: 10,
        overlap: 2
    });

    const x = new Float64Array(Array.from({ length: 20 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 2));

    const result = streamer.processChunk(x, y);
    assert.ok(result.y.length >= 0);

    const finalResult = streamer.finalize();
    assert.ok(finalResult.y.length > 0);
});

test('online smoothing', () => {
    const online = new fastloess.OnlineLoess({
        fraction: 0.5
    }, {
        window_capacity: 10,
        min_points: 2
    });

    let lastVal = null;
    for (let i = 0; i < 10; i++) {
        const xArr = new Float64Array([i]);
        const yArr = new Float64Array([i * 2]);
        const res = online.add_points(xArr, yArr);

        if (res.y.length > 0) {
            lastVal = res.y[0];
        }
    }

    assert.ok(lastVal !== null);
    assert.ok(Math.abs(lastVal - 18) < 1.0);
});

test('options parsing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastloess.Loess({
        weight_function: 'tricube',
        robustness_method: 'bisquare',
        boundary_policy: 'extend',
        scaling_method: 'mad'
    });

    const result = model.fit(x, y);

    assert.strictEqual(result.y.length, 5);
});

test('async batch smoothing', async () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastloess.Loess({
        fraction: 0.3
    });

    if (typeof model.fitAsync !== 'function') {
        console.error('Available properties on model:', Object.getOwnPropertyNames(Object.getPrototypeOf(model)));
        throw new Error('model.fitAsync is not a function');
    }
    const result = await model.fitAsync(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    assert.ok(result.y[0] > 0);
});

// ---- Parameter coverage tests ----

test('SmoothOptions: iterations, zero_weight_fallback, return_residuals, return_robustness_weights', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastloess.Loess({
        fraction: 0.7,
        iterations: 5,
        zero_weight_fallback: 'return_original',
        return_residuals: true,
        return_robustness_weights: true,
    });
    const result = model.fit(x, y);

    assert.ok(result.residuals !== null);
    assert.ok(result.robustnessWeights !== null);
});

test('SmoothOptions: confidence_intervals, prediction_intervals', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const model = new fastloess.Loess({
        fraction: 0.5,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
    });
    const result = model.fit(x, y);

    assert.ok(result.confidenceLower !== null);
    assert.ok(result.confidenceUpper !== null);
    assert.ok(result.predictionLower !== null);
    assert.ok(result.predictionUpper !== null);
});

test('SmoothOptions: returnSe', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const model = new fastloess.Loess({ fraction: 0.5, returnSe: true, surface_mode: 'direct' });
    const result = model.fit(x, y);

    assert.ok(result.enp !== null);
    assert.ok(result.traceHat !== null);
    assert.ok(result.leverage !== null);
});

test('SmoothOptions: degree, surface_mode', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([1, 4, 9, 16, 25]);

    for (const degree of ['constant', 'linear', 'quadratic']) {
        const r = new fastloess.Loess({ fraction: 0.9, degree }).fit(x, y);
        assert.strictEqual(r.y.length, 5);
    }
    const r2 = new fastloess.Loess({ fraction: 0.5, surface_mode: 'direct' }).fit(x, y);
    assert.strictEqual(r2.y.length, 5);
});

test('SmoothOptions: distance_metric variants', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    for (const dm of ['normalized', 'euclidean', 'manhattan', 'chebyshev']) {
        const r = new fastloess.Loess({ fraction: 0.5, distance_metric: dm }).fit(x, y);
        assert.strictEqual(r.y.length, 10);
    }
});

test('SmoothOptions: minkowski via string format', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const r = new fastloess.Loess({
        fraction: 0.5,
        distance_metric: 'minkowski:3',
    }).fit(x, y);
    assert.strictEqual(r.y.length, 10);
});

test('SmoothOptions: weightedMetricWeights', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = new Float64Array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

    const r = new fastloess.Loess({
        fraction: 0.5,
        distance_metric: 'weighted',
        weightedMetricWeights: [1.0],
    }).fit(x, y);
    assert.strictEqual(r.y.length, 10);
});

test('SmoothOptions: auto_converge, parallel', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const r = new fastloess.Loess({
        fraction: 0.5,
        auto_converge: 1e-4,
        parallel: false,
    }).fit(x, y);
    assert.strictEqual(r.y.length, 5);
});

test('SmoothOptions: cvFractions, cvMethod, cvK', () => {
    const x = new Float64Array(Array.from({ length: 30 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 30 }, (_, i) => i * 2));

    const r = new fastloess.Loess({
        cvFractions: [0.3, 0.5, 0.7],
        cvMethod: 'kfold',
        cvK: 3,
    }).fit(x, y);
    assert.ok(r.cvScores !== null);
    assert.strictEqual(r.cvScores.length, 3);
});

test('StreamingOptions: merge_strategy', () => {
    const x = new Float64Array(Array.from({ length: 40 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 40 }, (_, i) => i * 2));

    for (const ms of ['average', 'weighted_average', 'take_first', 'take_last']) {
        const s = new fastloess.StreamingLoess(
            { fraction: 0.3 },
            { chunk_size: 20, overlap: 2, merge_strategy: ms }
        );
        s.processChunk(x, y);
        const r = s.finalize();
        assert.ok(r.y.length >= 0);
    }
});

test('OnlineOptions: update_mode', () => {
    const x = new Float64Array(Array.from({ length: 15 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 15 }, (_, i) => i * 2));

    for (const um of ['full', 'incremental']) {
        const o = new fastloess.OnlineLoess(
            { fraction: 0.5 },
            { window_capacity: 10, min_points: 3, update_mode: um }
        );
        const r = o.add_points(x, y);
        assert.ok(r.y.length > 0);
    }
});

test('custom weights: zero on outlier reduces error', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const yOutlier = new Float64Array([1, 2, 3, 100, 5, 6, 7]);
    const yTrue = [1, 2, 3, 4, 5, 6, 7];
    const wZero = new Float64Array([1, 1, 1, 0, 1, 1, 1]);

    const model = new fastloess.Loess({ fraction: 0.6 });
    const rNoW = model.fit(x, yOutlier);
    const rW = model.fit(x, yOutlier, wZero);

    const nonOutlier = [0, 1, 2, 4, 5, 6];
    const errNoW = nonOutlier.reduce((s, i) => s + Math.abs(rNoW.y[i] - yTrue[i]), 0) / nonOutlier.length;
    const errW = nonOutlier.reduce((s, i) => s + Math.abs(rW.y[i] - yTrue[i]), 0) / nonOutlier.length;
    assert.ok(errW < errNoW, `expected ${errW} < ${errNoW}`);
});

test('custom weights: uniform equals no weights', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const wUniform = new Float64Array([1, 1, 1, 1, 1, 1, 1]);

    const model = new fastloess.Loess({ fraction: 0.6 });
    const rNoW = model.fit(x, y);
    const rW = model.fit(x, y, wUniform);

    for (let i = 0; i < rNoW.y.length; i++) {
        assert.ok(Math.abs(rW.y[i] - rNoW.y[i]) < 1e-6, `mismatch at index ${i}`);
    }
});

test('custom weights: wrong length throws', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const wBad = new Float64Array([1, 1, 1]);

    const model = new fastloess.Loess({ fraction: 0.6 });
    assert.throws(() => model.fit(x, y, wBad));
});

test('custom weights: negative weight throws', () => {
    const x = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const y = new Float64Array([1, 2, 3, 4, 5, 6, 7]);
    const wNeg = new Float64Array([1, -1, 1, 1, 1, 1, 1]);

    const model = new fastloess.Loess({ fraction: 0.6 });
    assert.throws(() => model.fit(x, y, wNeg));
});
