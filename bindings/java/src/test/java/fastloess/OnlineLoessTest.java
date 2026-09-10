package fastloess;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

class OnlineLoessTest {

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
                OnlineOptions.builder().fraction(0.5).windowCapacity(10).returnSe(true).build()));
        assertTrue(ex.getMessage() != null && !ex.getMessage().isEmpty());
    }

    @Test
    void confidenceIntervalsRequiresFullUpdateMode() {
        RuntimeException ex = assertThrows(RuntimeException.class, () -> new OnlineLoess(
                OnlineOptions.builder().fraction(0.5).windowCapacity(10).confidenceIntervals(0.95).build()));
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
                        .confidenceIntervals(0.95)
                        .predictionIntervals(0.95)
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
