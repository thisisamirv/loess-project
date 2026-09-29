package fastloess;

/**
 * Grouped cross-validation configuration for
 * {@link Options.Builder#cv(CVOptions)}.
 */
public final class CVOptions {

    final double[] fractions;
    final String method;
    final int k;
    final Long seed;

    private CVOptions(Builder builder) {
        this.fractions = builder.fractions.clone();
        this.method = builder.method;
        this.k = builder.k;
        this.seed = builder.seed;
    }

    /**
     * Creates a cross-validation settings builder.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Fluent cross-validation configuration builder.
     */
    public static final class Builder {

        double[] fractions;
        String method = "kfold";
        int k = 5;
        Long seed;

        Builder() {
        }

        /**
         * Sets candidate smoothing fractions.
         *
         * @param fractions candidate smoothing fractions
         * @return this builder
         */
        public Builder fractions(double... fractions) {
            this.fractions = fractions.clone();
            return this;
        }

        /**
         * Sets the cross-validation method.
         *
         * @param method "kfold" or "loocv"
         * @return this builder
         */
        public Builder method(String method) {
            this.method = method;
            return this;
        }

        /**
         * Sets the number of k-fold splits.
         *
         * @param k number of folds
         * @return this builder
         */
        public Builder k(int k) {
            this.k = k;
            return this;
        }

        /**
         * Sets the reproducible fold-assignment seed.
         *
         * @param seed random seed
         * @return this builder
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Builds the immutable cross-validation settings.
         *
         * @return the constructed settings
         */
        public CVOptions build() {
            if (fractions == null) {
                throw new IllegalStateException("CV fractions must be provided");
            }
            return new CVOptions(this);
        }
    }
}
