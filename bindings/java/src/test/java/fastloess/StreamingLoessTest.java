package fastloess;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

class StreamingLoessTest {

    @Test
    void processesChunksAndFinalizes() {
        try (StreamingLoess model = new StreamingLoess(StreamingOptions.builder().chunkSize(10).overlap(5).build())) {
            double[] x1 = new double[10];
            double[] y1 = new double[10];
            for (int i = 0; i < 10; i++) {
                x1[i] = i;
                y1[i] = i * 2.0;
            }
            Result chunk1 = model.processChunk(x1, y1);
            assertEquals(5, chunk1.x().length);

            double[] x2 = new double[10];
            double[] y2 = new double[10];
            for (int i = 0; i < 10; i++) {
                x2[i] = i + 10;
                y2[i] = (i + 10) * 2.0;
            }
            Result chunk2 = model.processChunk(x2, y2);
            assertEquals(10, chunk2.x().length);

            Result finalResult = model.finish();
            assertTrue(finalResult.x().length > 0);
        }
    }

    @Test
    void missingDropRemovesNaNRows() {
        double[] x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        double[] y = {2, 4, Double.NaN, 8, 10, 12, 14, 16, 18, 20};

        try (StreamingLoess model = new StreamingLoess(
                StreamingOptions.builder().fraction(0.5).chunkSize(10).missing("drop").build())) {
            Result chunkResult = model.processChunk(x, y);
            Result finalResult = model.finish();
            assertEquals(x.length - 1, chunkResult.x().length + finalResult.x().length);
        }
    }
}
