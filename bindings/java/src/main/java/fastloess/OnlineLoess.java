package fastloess;

import java.util.Optional;

/**
 * An online LOESS model that updates incrementally as points arrive. Operations
 * on one instance are synchronized because it owns mutable native state.
 */
public final class OnlineLoess implements AutoCloseable {

    private long handle;
    private final int dimensions;

    /**
     * Creates a new online model from the given options.
     *
     * @param options the model configuration
     */
    public OnlineLoess(OnlineOptions options) {
        Options c = options.common;
        this.dimensions = c.dimensions;
        this.handle = NativeBridge.onlineNew(
                c.fraction,
                c.iterations,
                c.weightFunction,
                c.robustnessMethod,
                c.scalingMethod,
                c.boundaryPolicy,
                c.returnRobustnessWeights,
                c.returnGradient,
                c.confidenceIntervals,
                c.predictionIntervals,
                c.returnSe,
                c.zeroWeightFallback,
                c.autoConverge,
                options.windowCapacity,
                options.minPoints,
                options.updateMode,
                c.degree,
                c.dimensions,
                c.distanceMetric,
                c.surfaceMode,
                c.cell,
                c.interpolationVertices,
                NativeBridge.boolSentinel(c.boundaryDegreeFallback),
                c.weightedMetricWeights,
                c.missing);
    }

    /**
     * Adds a one-dimensional point to the model. For multivariate models, use
     * {@link #addPoint(double[], double)}.
     *
     * @param x the x value
     * @param y the y value
     * @return the smoothed output, or {@link Optional#empty()} if not enough
     * points have been seen yet
     */
    public synchronized Optional<PointResult> addPoint(double x, double y) {
        return addPoint(new double[]{x}, y, 1.0);
    }

    /**
     * Adds a one-dimensional observation with a finite, non-negative case
     * weight.
     *
     * @param x the predictor value
     * @param y the response value
     * @param weight the case weight
     * @return the smoothed output, or empty while the window is filling
     */
    public synchronized Optional<PointResult> addPoint(double x, double y, double weight) {
        return addPoint(new double[]{x}, y, weight);
    }

    /**
     * Adds a point with one coordinate per configured predictor dimension.
     * Returns a smoothed output once at least {@code minPoints} have been seen,
     * or {@link Optional#empty()} otherwise.
     *
     * @param x predictor coordinates
     * @param y response value
     * @return the smoothed output, or empty while the window is filling
     */
    public synchronized Optional<PointResult> addPoint(double[] x, double y) {
        return addPoint(x, y, 1.0);
    }

    /**
     * Adds a point with one coordinate per configured predictor dimension and a
     * case weight.
     *
     * @param x predictor coordinates
     * @param y response value
     * @param weight finite, non-negative case weight
     * @return the smoothed output, or empty while the window is filling
     */
    public synchronized Optional<PointResult> addPoint(double[] x, double y, double weight) {
        checkOpen();
        if (x == null || x.length != dimensions) {
            throw new IllegalArgumentException("x must contain exactly " + dimensions + " predictor coordinates");
        }
        NativeOnlineOutput o = NativeBridge.onlineAddPoint(handle, x, y, weight);
        return o.hasValue ? Optional.of(PointResult.fromNative(o)) : Optional.empty();
    }

    /**
     * Computes diagnostics for the current window on demand.
     *
     * @return the diagnostics, or empty before the window reaches
     * {@code minPoints}
     */
    public synchronized Optional<Diagnostics> windowDiagnostics() {
        checkOpen();
        double[] values = NativeBridge.onlineWindowDiagnostics(handle);
        if (values == null || values.length == 0) {
            return Optional.empty();
        }
        return Optional.of(Diagnostics.fromWindow(values));
    }

    /**
     * Predicts query points using a full fit of the current window.
     *
     * @param newX flattened query points, one coordinate per dimension
     * @return the prediction result
     */
    public synchronized PredictResult predictWindow(double[] newX) {
        return predictWindow(newX, PredictOptions.builder().build());
    }

    /**
     * Predicts query points using a full fit of the current window.
     *
     * @param newX flattened query points, one coordinate per dimension
     * @param options prediction options
     * @return the prediction result
     */
    public synchronized PredictResult predictWindow(double[] newX, PredictOptions options) {
        checkOpen();
        NativePredictResult result = NativeBridge.onlinePredictWindow(
                handle,
                newX,
                options.outputs().contains("se"),
                options.intervals() == null ? Double.NaN : options.intervals().confidence(),
                options.intervals() == null ? Double.NaN : options.intervals().prediction(),
                options.outputs().contains("gradient") || options.outputs().contains("derivative"),
                options.extrapolation(),
                options.maxExtrapolationDistance(),
                options.maxNeighborDistance());
        return PredictResult.fromNative(result);
    }

    private void checkOpen() {
        if (handle == 0) {
            throw new IllegalStateException("OnlineLoess has already been closed");
        }
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            NativeBridge.onlineFree(handle);
            handle = 0;
        }
    }
}
