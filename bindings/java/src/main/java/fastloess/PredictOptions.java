package fastloess;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Options for {@link PredictModel#predict}. Construct via {@link #builder()}.
 *
 * @param outputs optional prediction components such as {@code "se"} and
 * {@code "gradient"}
 * @param confidenceLevel confidence interval coverage level (e.g.
 * {@code 0.95}), or {@code Double.NaN} to skip
 * @param predictionLevel prediction interval coverage level (e.g.
 * {@code 0.95}), or {@code Double.NaN} to skip
 * @param extrapolation behavior for query points outside the training range:
 * one of {@code "clamp"} (default), {@code "linear"}, {@code "error"}
 * @param maxExtrapolationDistance under {@code "linear"} extrapolation, the
 * maximum allowed distance beyond the training boundary before {@code predict}
 * throws instead of returning an unbounded value, or {@code Double.NaN} to
 * disable
 * @param maxNeighborDistance maximum allowed distance to the farthest point in
 * a query's neighbor window before {@code predict} throws, catching
 * in-range-but-sparse query points, or {@code Double.NaN} to disable
 */
public record PredictOptions(
        List<String> outputs,
        double confidenceLevel,
        double predictionLevel,
        String extrapolation,
        double maxExtrapolationDistance,
        double maxNeighborDistance) {

    /**
     * Creates a new builder.
     *
     * @return a new {@link Builder}
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent builder for {@link PredictOptions}.
     */
    public static final class Builder {

        final Set<String> outputs = new LinkedHashSet<>();
        double confidenceLevel = Double.NaN;
        double predictionLevel = Double.NaN;
        String extrapolation = "clamp";
        double maxExtrapolationDistance = Double.NaN;
        double maxNeighborDistance = Double.NaN;

        Builder() {
        }

        /**
         * Requests confidence intervals at the given level (e.g. {@code 0.95}).
         *
         * @param confidenceLevel the confidence level
         * @return this builder, for chaining
         */
        public Builder confidenceLevel(double confidenceLevel) {
            this.confidenceLevel = confidenceLevel;
            return this;
        }

        /**
         * Requests prediction intervals at the given level (e.g. {@code 0.95}).
         *
         * @param predictionLevel the prediction level
         * @return this builder, for chaining
         */
        public Builder predictionLevel(double predictionLevel) {
            this.predictionLevel = predictionLevel;
            return this;
        }

        /**
         * Selects optional prediction components: {@code "se"},
         * {@code "gradient"}, or {@code "derivative"}.
         *
         * @param outputs optional prediction component names
         * @return this builder, for chaining
         * @throws IllegalArgumentException if an output name is unknown
         */
        public Builder outputs(String... outputs) {
            for (String output : outputs) {
                switch (output) {
                    case "se", "gradient", "derivative" ->
                        this.outputs.add(output);
                    default ->
                        throw new IllegalArgumentException("Unknown output: " + output);
                }
            }
            return this;
        }

        /**
         * Behavior for query points outside the training range: one of
         * {@code "clamp"} (default), {@code "linear"}, {@code "error"}.
         *
         * @param extrapolation the extrapolation policy name
         * @return this builder, for chaining
         */
        public Builder extrapolation(String extrapolation) {
            this.extrapolation = extrapolation;
            return this;
        }

        /**
         * Under {@code "linear"} extrapolation, the maximum allowed distance
         * beyond the training boundary before {@code predict} throws instead of
         * returning an unbounded value.
         *
         * @param maxExtrapolationDistance the maximum extrapolation distance
         * @return this builder, for chaining
         */
        public Builder maxExtrapolationDistance(double maxExtrapolationDistance) {
            this.maxExtrapolationDistance = maxExtrapolationDistance;
            return this;
        }

        /**
         * Maximum allowed distance to the farthest point in a query's neighbor
         * window before {@code predict} throws, catching in-range-but-sparse
         * query points.
         *
         * @param maxNeighborDistance the maximum neighbor distance
         * @return this builder, for chaining
         */
        public Builder maxNeighborDistance(double maxNeighborDistance) {
            this.maxNeighborDistance = maxNeighborDistance;
            return this;
        }

        /**
         * Builds the immutable {@link PredictOptions}.
         *
         * @return the constructed options
         */
        public PredictOptions build() {
            return new PredictOptions(
                    List.copyOf(outputs),
                    confidenceLevel,
                    predictionLevel,
                    extrapolation,
                    maxExtrapolationDistance,
                    maxNeighborDistance);
        }
    }
}
