package fastloess;

/**
 * Raw online-update output constructed directly by the native layer. See
 * {@link PointResult}.
 */
final class NativeOnlineOutput {

    final boolean hasValue;
    final double y;
    final double standardError;
    final double residual;
    final double robustnessWeight;
    final int iterationsUsed;
    final double confidenceLower;
    final double confidenceUpper;
    final double predictionLower;
    final double predictionUpper;
    final double[] gradient;

    @SuppressWarnings("unused") // called by JNI
    NativeOnlineOutput(
            boolean hasValue,
            double y,
            double standardError,
            double residual,
            double robustnessWeight,
            int iterationsUsed,
            double confidenceLower,
            double confidenceUpper,
            double predictionLower,
            double predictionUpper,
            double[] gradient) {
        this.hasValue = hasValue;
        this.y = y;
        this.standardError = standardError;
        this.residual = residual;
        this.robustnessWeight = robustnessWeight;
        this.iterationsUsed = iterationsUsed;
        this.confidenceLower = confidenceLower;
        this.confidenceUpper = confidenceUpper;
        this.predictionLower = predictionLower;
        this.predictionUpper = predictionUpper;
        this.gradient = gradient;
    }
}
