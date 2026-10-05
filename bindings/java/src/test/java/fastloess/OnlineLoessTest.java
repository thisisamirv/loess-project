package fastloess;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

class OnlineLoessTest {

    @Test
    void weightedWindowDiagnosticsAndPrediction() {
        OnlineOptions options = OnlineOptions.builder()
                .fraction(1.0)
                .iterations(0)
                .windowCapacity(10)
                .minPoints(10)
                .updateMode("full")
                .surfaceMode("direct")
                .build();
        try (OnlineLoess weighted = new OnlineLoess(options); OnlineLoess plain = new OnlineLoess(options)) {
            for (int i = 0; i < 10; i++) {
                double y = i == 5 ? 100.0 : 2.0 * i + 1.0;
                weighted.addPoint((double) i, y, i == 5 ? 0.0 : 1.0);
                plain.addPoint((double) i, y);
            }
            Diagnostics diagnostics = weighted.windowDiagnostics().orElseThrow();
            assertTrue(diagnostics.rmse() > 0.0);
            RuntimeException error = assertThrows(
                    RuntimeException.class,
                    () -> weighted.addPoint(10.0, 21.0, -1.0));
            assertTrue(error.getMessage() != null && !error.getMessage().isEmpty());

            double weightedPrediction = weighted.predictWindow(new double[]{5.0}).y()[0];
            double plainPrediction = plain.predictWindow(new double[]{5.0}).y()[0];
            assertTrue(Math.abs(weightedPrediction - 11.0) < Math.abs(plainPrediction - 11.0));
        }
    }

    @Test
    void acceptsMultivariatePoints() {
        try (OnlineLoess model = new OnlineLoess(
                OnlineOptions.builder()
                        .fraction(1.0)
                        .windowCapacity(10)
                        .minPoints(3)
                        .dimensions(2)
                        .surfaceMode("direct")
                        .outputs("gradient")
                        .build())) {
            Optional<PointResult> last = Optional.empty();
            double[][] points = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
            double[] responses = {0.0, 1.0, 2.0, 3.0};
            for (int i = 0; i < points.length; i++) {
                Optional<PointResult> result = model.addPoint(points[i], responses[i]);
                if (result.isPresent()) {
                    last = result;
                }
            }
            assertTrue(last.isPresent());
            assertEquals(2, last.orElseThrow().gradient().orElseThrow().length);
            IllegalArgumentException error = assertThrows(
                    IllegalArgumentException.class,
                    () -> model.addPoint(new double[]{1.0}, 2.0));
            assertTrue(error.getMessage().contains("exactly 2 predictor coordinates"));
        }
    }

    @Test
    void addsPointsAndEventuallyProducesOutput() {
        try (OnlineLoess model = new OnlineLoess(OnlineOptions.builder().minPoints(5).build())) {
            boolean sawValue = false;
            for (int i = 0; i < 20; i++) {
                Optional<PointResult> point = model.addPoint(i, i * 2.0);
                if (point.isPresent()) {
                    sawValue = true;
                }
            }
            assertTrue(sawValue, "expected at least one point result once minPoints was reached");
        }
    }

    @Test
    void missingDropIgnoresNaNPoint() {
        try (OnlineLoess model = new OnlineLoess(
                OnlineOptions.builder().fraction(0.5).windowCapacity(10).missing("drop").build())) {
            Optional<PointResult> point = model.addPoint(1.0, Double.NaN);
            assertFalse(point.isPresent(), "expected the NaN point to be silently ignored");
        }
    }

    @Test
    void returnSeRequiresFullUpdateMode() {
        RuntimeException ex = assertThrows(RuntimeException.class, () -> new OnlineLoess(
                OnlineOptions.builder().fraction(0.5).windowCapacity(10).outputs("se").build()));
        assertTrue(ex.getMessage() != null && !ex.getMessage().isEmpty());
    }

    @Test
    void groupedOnlineOutputsPopulatePointResult() {
        try (OnlineLoess model = new OnlineLoess(
                OnlineOptions.builder()
                        .fraction(1.0)
                        .windowCapacity(10)
                        .minPoints(3)
                        .updateMode("full")
                        .surfaceMode("direct")
                        .outputs("weights", "gradient", "se")
                        .build())) {
            Optional<PointResult> last = Optional.empty();
            for (int i = 0; i < 6; i++) {
                Optional<PointResult> current = model.addPoint(i, 2.0 * i + 1.0);
                if (current.isPresent()) {
                    last = current;
                }
            }
            PointResult result = last.orElseThrow();
            assertTrue(result.robustnessWeight().isPresent());
            assertTrue(result.standardError().isPresent());
            assertTrue(result.gradient().isPresent());
        }
    }

    @Test
    void groupedOnlineOutputsRejectDiagnosticsName() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> OnlineOptions.builder().outputs("diagnostics"));
        assertTrue(error.getMessage().contains("Unknown output: diagnostics"));
    }

    @Test
    void confidenceIntervalsRequiresFullUpdateMode() {
        RuntimeException ex = assertThrows(RuntimeException.class, () -> new OnlineLoess(
                OnlineOptions.builder().fraction(0.5).windowCapacity(10)
                        .intervals(IntervalsOptions.builder().confidence(0.95).build()).build()));
        assertTrue(ex.getMessage() != null && !ex.getMessage().isEmpty());
    }

    @Test
    void confidenceAndPredictionIntervalsUnderFullUpdateMode() {
        try (OnlineLoess model = new OnlineLoess(
                OnlineOptions.builder()
                        .fraction(1.0)
                        .windowCapacity(10)
                        .minPoints(3)
                        .updateMode("full")
                        .intervals(IntervalsOptions.builder().confidence(0.95).prediction(0.95).build())
                        .build())) {
            Optional<PointResult> last = Optional.empty();
            for (int i = 0; i < 6; i++) {
                Optional<PointResult> point = model.addPoint(i, 2.0 * i + 1.0);
                if (point.isPresent()) {
                    last = point;
                }
            }
            assertTrue(last.isPresent());
            PointResult result = last.get();
            assertTrue(result.confidenceLower().isPresent());
            assertTrue(result.predictionLower().isPresent());
            assertTrue(result.confidenceLower().getAsDouble() <= result.confidenceUpper().getAsDouble());
            assertTrue(result.predictionLower().getAsDouble() <= result.predictionUpper().getAsDouble());
        }
    }
}
