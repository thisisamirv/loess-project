package fastloess;

/**
 * A batch LOESS model. Operations on one instance are synchronized; each
 * instance wraps a native handle that must be freed.
 */
public final class Loess implements AutoCloseable {

    private final NativeHandles.Owned nativeHandle;
    private final int dimensions;

    /**
     * Creates a new batch model from the given options.
     *
     * @param options the model configuration
     */
    public Loess(Options options) {
        this.dimensions = options.dimensions;
        long handle = NativeBridge.loessNew(
                options.fraction,
                options.iterations,
                options.weightFunction,
                options.robustnessMethod,
                options.scalingMethod,
                options.boundaryPolicy,
                options.confidenceIntervals,
                options.predictionIntervals,
                options.returnDiagnostics,
                options.returnResiduals,
                options.returnRobustnessWeights,
                options.returnGradient,
                options.zeroWeightFallback,
                options.autoConverge,
                options.cvFractions,
                options.cvMethod,
                options.cvK,
                options.parallel,
                options.degree,
                options.dimensions,
                options.distanceMetric,
                options.surfaceMode,
                options.returnSe,
                options.returnSorted,
                options.cell,
                options.interpolationVertices,
                NativeBridge.boolSentinel(options.boundaryDegreeFallback),
                options.weightedMetricWeights,
                options.missing,
                options.retainModel);
        this.nativeHandle = new NativeHandles.Owned(handle, NativeBridge::loessFree);
        if (options.cvSeed != null) {
            NativeBridge.loessSetCvSeed(nativeHandle.get(), options.cvSeed);
        }
    }

    /**
     * Fits the model to {@code x}/{@code y}, using uniform weights.
     *
     * @param x the x values
     * @param y the y values
     * @return the fit result (close it, or close {@link Result#predictModel()},
     * when {@code retainModel} was set)
     */
    public synchronized Result fit(double[] x, double[] y) {
        return fit(x, y, null);
    }

    /**
     * Fits the model to {@code x}/{@code y}, using the given per-point weights.
     *
     * @param x the x values
     * @param y the y values
     * @param customWeights non-negative per-observation weights, or
     * {@code null}
     * @return the fit result (close it, or close {@link Result#predictModel()},
     * when {@code retainModel} was set)
     */
    public synchronized Result fit(double[] x, double[] y, double[] customWeights) {
        checkOpen();
        Inputs.requireXy(x, y, dimensions);
        Inputs.requireCustomWeights(customWeights, y.length);
        NativeResult r = NativeBridge.loessFit(nativeHandle.get(), x, y, customWeights);
        return Result.fromNative(r);
    }

    private void checkOpen() {
        if (!nativeHandle.isOpen()) {
            throw new IllegalStateException("Loess has already been closed");
        }
    }

    @Override
    public synchronized void close() {
        nativeHandle.close();
    }
}
