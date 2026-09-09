package fastloess;

import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * A single smoothed output produced by
 * {@link OnlineLoess#addPoint(double, double)}.
 *
 * @param y the smoothed value
 * @param standardError the standard error of the smoothed value, if computed
 * @param residual the residual (observed minus smoothed), if computed
 * @param robustnessWeight the final robustness weight applied to this point, if
 * computed
 * @param iterationsUsed the number of robustness iterations performed, if
 * applicable
 * @param gradient the local fit's gradient at this point, length
 * {@code dimensions}, if computed (only populated when {@code surfaceMode} is
 * {@code "direct"})
 */
public record PointResult(
        double y,
        OptionalDouble standardError,
        OptionalDouble residual,
        OptionalDouble robustnessWeight,
        OptionalInt iterationsUsed,
        Optional<double[]> gradient) {

    static PointResult fromNative(NativeOnlineOutput o) {
        return new PointResult(
                o.y,
                optionalDouble(o.standardError),
                optionalDouble(o.residual),
                optionalDouble(o.robustnessWeight),
                o.iterationsUsed < 0 ? OptionalInt.empty() : OptionalInt.of(o.iterationsUsed),
                Optional.ofNullable(o.gradient));
    }

    private static OptionalDouble optionalDouble(double value) {
        return Double.isNaN(value) ? OptionalDouble.empty() : OptionalDouble.of(value);
    }
}
