package fastloess;

/**
 * A streaming LOESS model that processes data in chunks. Operations on one
 * instance are synchronized because it owns mutable native state.
 */
public final class StreamingLoess implements AutoCloseable {

    private final NativeHandles.Owned nativeHandle;
    private final int dimensions;

    /**
     * Creates a new streaming model from the given options.
     *
     * @param options the model configuration
     */
    public StreamingLoess(StreamingOptions options) {
        Options c = options.common;
        this.dimensions = c.dimensions;
        long handle = NativeBridge.streamingNew(
                c.fraction,
                c.iterations,
                c.weightFunction,
                c.robustnessMethod,
                c.scalingMethod,
                c.boundaryPolicy,
                c.returnDiagnostics,
                c.returnResiduals,
                c.returnRobustnessWeights,
                c.returnGradient,
                c.confidenceIntervals,
                c.predictionIntervals,
                c.returnSe,
                c.zeroWeightFallback,
                c.autoConverge,
                c.parallel,
                options.chunkSize,
                options.overlap,
                options.mergeStrategy,
                c.degree,
                c.dimensions,
                c.distanceMetric,
                c.surfaceMode,
                c.cell,
                c.interpolationVertices,
                NativeBridge.boolSentinel(c.boundaryDegreeFallback),
                c.weightedMetricWeights,
                c.missing);
        this.nativeHandle = new NativeHandles.Owned(handle, NativeBridge::streamingFree);
    }

    /**
     * Processes the next chunk of data, returning a partial fit for it.
     *
     * @param x the x values of this chunk
     * @param y the y values of this chunk
     * @return the partial fit result for this chunk
     */
    public synchronized Result processChunk(double[] x, double[] y) {
        checkOpen();
        Inputs.requireXy(x, y, dimensions);
        NativeResult r = NativeBridge.streamingProcess(nativeHandle.get(), x, y);
        return Result.fromNative(r);
    }

    /**
     * Processes a chunk with one case weight per observation.
     *
     * @param x the predictor values
     * @param y the response values
     * @param customWeights finite, non-negative weights, one per response
     * @return the partial fit result for this chunk
     */
    public synchronized Result processChunk(double[] x, double[] y, double[] customWeights) {
        checkOpen();
        Inputs.requireXy(x, y, dimensions);
        Inputs.requireCustomWeights(customWeights, y.length);
        NativeResult r = NativeBridge.streamingProcessWeighted(nativeHandle.get(), x, y, customWeights);
        return Result.fromNative(r);
    }

    /**
     * Merges all processed chunks into a final result.
     *
     * @return the final merged result
     */
    public synchronized Result finish() {
        checkOpen();
        NativeResult r = NativeBridge.streamingFinalize(nativeHandle.get());
        return Result.fromNative(r);
    }

    private void checkOpen() {
        if (!nativeHandle.isOpen()) {
            throw new IllegalStateException("StreamingLoess has already been closed");
        }
    }

    @Override
    public synchronized void close() {
        nativeHandle.close();
    }
}
