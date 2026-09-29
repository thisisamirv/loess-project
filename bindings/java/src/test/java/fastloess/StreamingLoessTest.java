package fastloess;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
    void returnSePopulatesStandardErrors() {
        try (StreamingLoess model = new StreamingLoess(
                StreamingOptions.builder().fraction(0.3).chunkSize(10).returnSe(true).build())) {
            double[] x = new double[20];
            double[] y = new double[20];
            for (int i = 0; i < 20; i++) {
                x[i] = i;
                y[i] = Math.sin(i);
            }
            Result chunk = model.processChunk(x, y);
            assertTrue(chunk.standardErrors().isPresent());
            assertTrue(chunk.confidenceLower().isEmpty());
        }
    }

    @Test
    void groupedOutputsPopulateStreamingResult() {
        double[] x = new double[20];
        double[] y = new double[20];
        for (int i = 0; i < x.length; i++) {
            x[i] = i;
            y[i] = Math.sin(i);
        }

        try (StreamingLoess model = new StreamingLoess(
                StreamingOptions.builder()
                        .fraction(0.5)
                        .chunkSize(20)
                        .overlap(0)
                        .surfaceMode("direct")
                        .outputs("diagnostics", "residuals", "weights", "gradient", "se")
                        .build())) {
            Result result = model.processChunk(x, y);
            assertTrue(result.diagnostics().isPresent());
            assertTrue(result.residuals().isPresent());
            assertTrue(result.robustnessWeights().isPresent());
            assertEquals(result.y().length, result.gradient().orElseThrow().length);
            assertEquals(result.y().length, result.standardErrors().orElseThrow().length);
        }
    }

    @Test
    void groupedStreamingOutputsRejectSortedName() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> StreamingOptions.builder().outputs("sorted"));
        assertTrue(error.getMessage().contains("Unknown output: sorted"));
    }

    @Test
    void confidenceAndPredictionIntervals() {
        try (StreamingLoess model = new StreamingLoess(
                StreamingOptions.builder()
                        .fraction(0.3)
                        .chunkSize(10)
                        .confidenceIntervals(0.95)
                        .predictionIntervals(0.95)
                        .build())) {
            double[] x = new double[20];
            double[] y = new double[20];
            for (int i = 0; i < 20; i++) {
                x[i] = i;
                y[i] = Math.sin(i);
            }
            Result chunk = model.processChunk(x, y);
            assertTrue(chunk.confidenceLower().isPresent());
            assertTrue(chunk.predictionLower().isPresent());
            double[] cl = chunk.confidenceLower().get();
            double[] cu = chunk.confidenceUpper().get();
            double[] pl = chunk.predictionLower().get();
            double[] pu = chunk.predictionUpper().get();
            for (int i = 0; i < cl.length; i++) {
                assertTrue(cl[i] <= cu[i]);
                assertTrue((pu[i] - pl[i]) >= (cu[i] - cl[i]) - 1e-9);
            }
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
