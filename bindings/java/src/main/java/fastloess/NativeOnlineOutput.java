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
    final double[] gradient;

    @SuppressWarnings("unused") // called by JNI
    NativeOnlineOutput(
            boolean hasValue,
            double y,
            double standardError,
            double residual,
            double robustnessWeight,
            int iterationsUsed,
            double[] gradient) {
        this.hasValue = hasValue;
        this.y = y;
        this.standardError = standardError;
        this.residual = residual;
        this.robustnessWeight = robustnessWeight;
        this.iterationsUsed = iterationsUsed;
        this.gradient = gradient;
    }
}
