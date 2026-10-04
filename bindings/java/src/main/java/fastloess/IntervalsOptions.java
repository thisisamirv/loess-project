package fastloess;

/**
 * Grouped interval coverage levels for fitting and prediction.
 *
 * @param confidence confidence coverage level, or {@code Double.NaN} to disable
 * @param prediction prediction coverage level, or {@code Double.NaN} to disable
 */
public record IntervalsOptions(double confidence, double prediction) {

    /**
     * Creates a builder with intervals disabled.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent interval configuration builder.
     */
    public static final class Builder {

        double confidence = Double.NaN;
        double prediction = Double.NaN;

        Builder() {
        }

        /**
         * Sets the confidence coverage level.
         *
         * @param level coverage level
         * @return this builder
         */
        public Builder confidence(double level) {
            confidence = level;
            return this;
        }

        /**
         * Sets the prediction coverage level.
         *
         * @param level coverage level
         * @return this builder
         */
        public Builder prediction(double level) {
            prediction = level;
            return this;
        }

        /**
         * Builds the immutable interval settings.
         *
         * @return interval settings
         */
        public IntervalsOptions build() {
            return new IntervalsOptions(confidence, prediction);
        }
    }
}
