package fastloess;

/**
 * Configuration for a {@link StreamingLoess} model. Construct via
 * {@link #builder()}.
 */
public final class StreamingOptions {

    final Options common;
    final int chunkSize;
    final int overlap;
    final String mergeStrategy;

    StreamingOptions(Builder b) {
        this.common = b.common.build();
        this.chunkSize = b.chunkSize;
        this.overlap = b.overlap;
        this.mergeStrategy = b.mergeStrategy;
    }

    /**
     * Creates a new builder.
     *
     * @return a new {@link Builder}
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for {@link StreamingOptions}.
     */
    public static final class Builder {

        private final Options.Builder common = Options.builder();
        int chunkSize = 5000;
        int overlap = -1;
        String mergeStrategy = null;

        Builder() {
        }

        /**
         * Sets the smoothing fraction.
         *
         * @param fraction the fraction of points used to compute each local
         * regression
         * @return this builder, for chaining
         * @see Options.Builder#fraction(double)
         */
        public Builder fraction(double fraction) {
            common.fraction(fraction);
            return this;
        }

        /**
         * Sets the number of robustness iterations.
         *
         * @param iterations the number of robustifying iterations
         * @return this builder, for chaining
         * @see Options.Builder#iterations(int)
         */
        public Builder iterations(int iterations) {
            common.iterations(iterations);
            return this;
        }

        /**
         * Sets the kernel weight function.
         *
         * @param weightFunction the weight function name
         * @return this builder, for chaining
         * @see Options.Builder#weightFunction(String)
         */
        public Builder weightFunction(String weightFunction) {
            common.weightFunction(weightFunction);
            return this;
        }

        /**
         * Sets the outlier robustness method.
         *
         * @param robustnessMethod the robustness method name
         * @return this builder, for chaining
         * @see Options.Builder#robustnessMethod(String)
         */
        public Builder robustnessMethod(String robustnessMethod) {
            common.robustnessMethod(robustnessMethod);
            return this;
        }

        /**
         * Sets the residual scaling method.
         *
         * @param scalingMethod the residual scaling method name
         * @return this builder, for chaining
         * @see Options.Builder#scalingMethod(String)
         */
        public Builder scalingMethod(String scalingMethod) {
            common.scalingMethod(scalingMethod);
            return this;
        }

        /**
         * Sets the boundary handling policy.
         *
         * @param boundaryPolicy the boundary handling policy name
         * @return this builder, for chaining
         * @see Options.Builder#boundaryPolicy(String)
         */
        public Builder boundaryPolicy(String boundaryPolicy) {
            common.boundaryPolicy(boundaryPolicy);
            return this;
        }

        /**
         * Sets the fallback for zero-weight neighborhoods.
         *
         * @param zeroWeightFallback the zero-weight handling strategy name
         * @return this builder, for chaining
         * @see Options.Builder#zeroWeightFallback(String)
         */
        public Builder zeroWeightFallback(String zeroWeightFallback) {
            common.zeroWeightFallback(zeroWeightFallback);
            return this;
        }

        /**
         * Sets the missing-value policy for each chunk.
         *
         * @param missing the missing-value policy name
         * @return this builder, for chaining
         * @see Options.Builder#missing(String)
         */
        public Builder missing(String missing) {
            common.missing(missing);
            return this;
        }

        /**
         * Sets the tolerance for stopping robustness iterations early.
         *
         * @param autoConverge the auto-convergence tolerance
         * @return this builder, for chaining
         * @see Options.Builder#autoConverge(double)
         */
        public Builder autoConverge(double autoConverge) {
            common.autoConverge(autoConverge);
            return this;
        }

        /**
         * Requests fit diagnostics in the result.
         *
         * @param returnDiagnostics whether to compute diagnostics
         * @return this builder, for chaining
         * @see Options.Builder#returnDiagnostics(boolean)
         */
        public Builder returnDiagnostics(boolean returnDiagnostics) {
            common.returnDiagnostics(returnDiagnostics);
            return this;
        }

        /**
         * Requests per-point residuals in the result.
         *
         * @param returnResiduals whether to include residuals in the result
         * @return this builder, for chaining
         * @see Options.Builder#returnResiduals(boolean)
         */
        public Builder returnResiduals(boolean returnResiduals) {
            common.returnResiduals(returnResiduals);
            return this;
        }

        /**
         * Requests robustness weights in the result.
         *
         * @param returnRobustnessWeights whether to include robustness weights
         * in the result
         * @return this builder, for chaining
         * @see Options.Builder#returnRobustnessWeights(boolean)
         */
        public Builder returnRobustnessWeights(boolean returnRobustnessWeights) {
            common.returnRobustnessWeights(returnRobustnessWeights);
            return this;
        }

        /**
         * Requests per-point local-fit gradients.
         *
         * @param returnGradient whether to include the per-point local fit
         * gradient in the result
         * @return this builder, for chaining
         * @see Options.Builder#returnGradient(boolean)
         */
        public Builder returnGradient(boolean returnGradient) {
            common.returnGradient(returnGradient);
            return this;
        }

        /**
         * Sets the confidence interval level for each chunk.
         *
         * @param confidenceIntervals the confidence level for confidence
         * intervals, computed per chunk and merged across overlap boundaries
         * via {@code mergeStrategy}
         * @return this builder, for chaining
         * @see Options.Builder#confidenceIntervals(double)
         */
        public Builder confidenceIntervals(double confidenceIntervals) {
            common.confidenceIntervals(confidenceIntervals);
            return this;
        }

        /**
         * Sets the prediction interval level for each chunk.
         *
         * @param predictionIntervals the confidence level for prediction
         * intervals; same per-chunk computation and overlap-merging as
         * {@code confidenceIntervals}
         * @return this builder, for chaining
         * @see Options.Builder#predictionIntervals(double)
         */
        public Builder predictionIntervals(double predictionIntervals) {
            common.predictionIntervals(predictionIntervals);
            return this;
        }

        /**
         * Requests standard errors for each chunk.
         *
         * @param returnSe whether to return standard errors in the result,
         * computed per chunk and merged across overlap boundaries via
         * {@code mergeStrategy}
         * @return this builder, for chaining
         * @see Options.Builder#returnSe(boolean)
         */
        public Builder returnSe(boolean returnSe) {
            common.returnSe(returnSe);
            return this;
        }

        /**
         * Selects optional result components: {@code "diagnostics"},
         * {@code "residuals"}, {@code "weights"}, {@code "gradient"} (or
         * {@code "derivative"}), and {@code "se"}.
         *
         * @param outputs optional output component names
         * @return this builder, for chaining
         * @throws IllegalArgumentException if an output is not supported here
         */
        public Builder outputs(String... outputs) {
            for (String output : outputs) {
                switch (output) {
                    case "diagnostics", "residuals", "weights", "gradient", "derivative", "se" ->
                        common.outputs(output);
                    default ->
                        throw new IllegalArgumentException("Unknown output: " + output);
                }
            }
            return this;
        }

        /**
         * Controls parallel chunk processing.
         *
         * @param parallel whether to enable parallel execution
         * @return this builder, for chaining
         * @see Options.Builder#parallel(boolean)
         */
        public Builder parallel(boolean parallel) {
            common.parallel(parallel);
            return this;
        }

        /**
         * Sets the local polynomial degree.
         *
         * @param degree the local polynomial degree name
         * @return this builder, for chaining
         * @see Options.Builder#degree(String)
         */
        public Builder degree(String degree) {
            common.degree(degree);
            return this;
        }

        /**
         * Sets the number of predictor dimensions.
         *
         * @param dimensions the number of predictor dimensions
         * @return this builder, for chaining
         * @see Options.Builder#dimensions(int)
         */
        public Builder dimensions(int dimensions) {
            common.dimensions(dimensions);
            return this;
        }

        /**
         * Sets the neighborhood distance metric.
         *
         * @param distanceMetric the distance metric name
         * @return this builder, for chaining
         * @see Options.Builder#distanceMetric(String)
         */
        public Builder distanceMetric(String distanceMetric) {
            common.distanceMetric(distanceMetric);
            return this;
        }

        /**
         * Sets per-dimension weights for the weighted distance metric.
         *
         * @param weightedMetricWeights the per-dimension weights
         * @return this builder, for chaining
         * @see Options.Builder#weightedMetricWeights(double[])
         */
        public Builder weightedMetricWeights(double[] weightedMetricWeights) {
            common.weightedMetricWeights(weightedMetricWeights);
            return this;
        }

        /**
         * Sets how the fitted surface is evaluated.
         *
         * @param surfaceMode the surface mode name
         * @return this builder, for chaining
         * @see Options.Builder#surfaceMode(String)
         */
        public Builder surfaceMode(String surfaceMode) {
            common.surfaceMode(surfaceMode);
            return this;
        }

        /**
         * Sets the interpolation cell size.
         *
         * @param cell the interpolation cell size
         * @return this builder, for chaining
         * @see Options.Builder#cell(double)
         */
        public Builder cell(double cell) {
            common.cell(cell);
            return this;
        }

        /**
         * Limits the number of interpolation vertices.
         *
         * @param interpolationVertices the maximum number of interpolation
         * vertices
         * @return this builder, for chaining
         * @see Options.Builder#interpolationVertices(int)
         */
        public Builder interpolationVertices(int interpolationVertices) {
            common.interpolationVertices(interpolationVertices);
            return this;
        }

        /**
         * Configures lower-degree fallback near boundaries.
         *
         * @param boundaryDegreeFallback whether to fall back to a lower degree
         * near boundary vertices
         * @return this builder, for chaining
         * @see Options.Builder#boundaryDegreeFallback(boolean)
         */
        public Builder boundaryDegreeFallback(boolean boundaryDegreeFallback) {
            common.boundaryDegreeFallback(boundaryDegreeFallback);
            return this;
        }

        /**
         * Number of points processed per chunk (default {@code 5000}).
         *
         * @param chunkSize the chunk size
         * @return this builder, for chaining
         */
        public Builder chunkSize(int chunkSize) {
            this.chunkSize = chunkSize;
            return this;
        }

        /**
         * Number of points overlapped between consecutive chunks (default:
         * library default of {@code chunk_size / 10}, clamped to
         * {@code [1, chunk_size - 10]}). Any negative value means "use the
         * library default".
         *
         * @param overlap the overlap size
         * @return this builder, for chaining
         */
        public Builder overlap(int overlap) {
            this.overlap = overlap;
            return this;
        }

        /**
         * One of
         * {@code "average"}, {@code "weighted_average"}, {@code "take_first"}, {@code "take_last"}
         * (default {@code "weighted_average"}).
         *
         * @param mergeStrategy the merge strategy name
         * @return this builder, for chaining
         */
        public Builder mergeStrategy(String mergeStrategy) {
            this.mergeStrategy = mergeStrategy;
            return this;
        }

        /**
         * Builds the immutable {@link StreamingOptions}.
         *
         * @return the constructed options
         */
        public StreamingOptions build() {
            return new StreamingOptions(this);
        }
    }
}
