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
        return addPoint(new double[]{x}, y);
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
        checkOpen();
        if (x == null || x.length != dimensions) {
            throw new IllegalArgumentException("x must contain exactly " + dimensions + " predictor coordinates");
        }
        NativeOnlineOutput o = NativeBridge.onlineAddPoint(handle, x, y);
        return o.hasValue ? Optional.of(PointResult.fromNative(o)) : Optional.empty();
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
