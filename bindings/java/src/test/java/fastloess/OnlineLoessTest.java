package fastloess;

import java.util.Optional;

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
}
