package fastloess;

/**
 * Configuration for an {@link OnlineLoess} model. Construct via
 * {@link #builder()}.
 */
public final class OnlineOptions {

    final Options common;
    final int windowCapacity;
    final int minPoints;
    final String updateMode;

    OnlineOptions(Builder b) {
        this.common = b.common.build();
        this.windowCapacity = b.windowCapacity;
        this.minPoints = b.minPoints;
        this.updateMode = b.updateMode;
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
     * Fluent builder for {@link OnlineOptions}.
     */
    public static final class Builder {

        private final Options.Builder common = Options.builder();
        int windowCapacity = 1000;
        int minPoints = 2;
        String updateMode = null;

        Builder() {
            common.iterations(0);
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
         * Policy for handling a non-finite (NaN/Inf) x or y value passed to
         * {@code addPoint} (default {@code "error"}). {@code "drop"} silently
         * ignores the point instead of adding it to the window.
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
         * Sets the confidence interval level for full updates.
         *
         * @param confidenceIntervals the confidence level for confidence
         * intervals; only computed under {@code updateMode("full")}
         * @return this builder, for chaining
         * @see Options.Builder#confidenceIntervals(double)
         */
        public Builder confidenceIntervals(double confidenceIntervals) {
            common.confidenceIntervals(confidenceIntervals);
            return this;
        }

        /**
         * Sets the prediction interval level for full updates.
         *
         * @param predictionIntervals the confidence level for prediction
         * intervals; only computed under {@code updateMode("full")}
         * @return this builder, for chaining
         * @see Options.Builder#predictionIntervals(double)
         */
        public Builder predictionIntervals(double predictionIntervals) {
            common.predictionIntervals(predictionIntervals);
            return this;
        }

        /**
         * Selects optional result components: {@code "weights"},
         * {@code "gradient"} (or {@code "derivative"}), and {@code "se"}.
         *
         * @param outputs optional output component names
         * @return this builder, for chaining
         * @throws IllegalArgumentException if an output is not supported here
         */
        public Builder outputs(String... outputs) {
            for (String output : outputs) {
                switch (output) {
                    case "weights", "gradient", "derivative", "se" ->
                        common.outputs(output);
                    default ->
                        throw new IllegalArgumentException("Unknown output: " + output);
                }
            }
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
         * Maximum number of points retained in the sliding window (default
         * {@code 1000}).
         *
         * @param windowCapacity the maximum window size
         * @return this builder, for chaining
         */
        public Builder windowCapacity(int windowCapacity) {
            this.windowCapacity = windowCapacity;
            return this;
        }

        /**
         * Minimum number of points required before a fit is produced (default
         * {@code 2}).
         *
         * @param minPoints the minimum point count
         * @return this builder, for chaining
         */
        public Builder minPoints(int minPoints) {
            this.minPoints = minPoints;
            return this;
        }

        /**
         * One of {@code "incremental"}, {@code "full"} (default
         * {@code "incremental"}).
         *
         * @param updateMode the update mode name
         * @return this builder, for chaining
         */
        public Builder updateMode(String updateMode) {
            this.updateMode = updateMode;
            return this;
        }

        /**
         * Builds the immutable {@link OnlineOptions}.
         *
         * @return the constructed options
         */
        public OnlineOptions build() {
            return new OnlineOptions(this);
        }
    }
}
