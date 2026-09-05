package fastloess;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Declares the JNI native methods implemented by the Rust
 * {@code fastloess_java} crate, and loads the platform-specific native library
 * backing them.
 */
final class NativeBridge {

    private NativeBridge() {
    }

    static {
        loadNativeLibrary();
    }

    private static void loadNativeLibrary() {
        String explicitPath = System.getProperty("fastloess.native.path");
        if (explicitPath != null) {
            System.load(explicitPath);
            return;
        }

        String nativeDir = System.getProperty("fastloess.native.dir");
        if (nativeDir != null) {
            File candidate = new File(nativeDir, mapLibraryName());
            if (candidate.isFile()) {
                System.load(candidate.getAbsolutePath());
                return;
            }
        }

        try {
            System.loadLibrary("fastloess_java");
            return;
        } catch (UnsatisfiedLinkError ignored) {
            // Fall through to the bundled-resource lookup below.
        }

        loadFromBundledResource();
    }

    // Extracts a platform-specific native library bundled inside the JAR (under
    // /native/<os>-<arch>/<libname>) to a temp file, then loads it.
    private static void loadFromBundledResource() {
        String resourcePath = "/native/" + osArchDir() + "/" + mapLibraryName();
        try (InputStream in = NativeBridge.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new UnsatisfiedLinkError(
                        "Could not locate the fastloess native library (looked for classpath "
                        + "resource '" + resourcePath + "', system property "
                        + "'fastloess.native.path', 'fastloess.native.dir', and "
                        + "java.library.path). Build it with `cargo build -p fastloess-java`.");
            }
            Path tempFile = Files.createTempFile("fastloess_java", suffix());
            tempFile.toFile().deleteOnExit();
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
            System.load(tempFile.toAbsolutePath().toString());
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to extract the fastloess native library", e);
        }
    }

    private static String mapLibraryName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return "fastloess_java.dll";
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return "libfastloess_java.dylib";
        }
        return "libfastloess_java.so";
    }

    private static String suffix() {
        String name = mapLibraryName();
        int dot = name.lastIndexOf('.');
        return dot >= 0 ? name.substring(dot) : "";
    }

    private static String osArchDir() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        String osName = os.contains("win") ? "windows" : (os.contains("mac") ? "macos" : "linux");
        String archName = (arch.contains("aarch64") || arch.contains("arm64")) ? "aarch64" : "x86_64";
        return osName + "-" + archName;
    }

    // Converts a nullable Boolean option into the native sentinel convention
    // used for boundaryDegreeFallback: -1 = unset (use the library default),
    // 0 = false, 1 = true.
    static int boolSentinel(Boolean value) {
        return value == null ? -1 : (value ? 1 : 0);
    }

    static native long loessNew(
            double fraction,
            int iterations,
            String weightFunction,
            String robustnessMethod,
            String scalingMethod,
            String boundaryPolicy,
            double confidenceIntervals,
            double predictionIntervals,
            boolean returnDiagnostics,
            boolean returnResiduals,
            boolean returnRobustnessWeights,
            String zeroWeightFallback,
            double autoConverge,
            double[] cvFractions,
            String cvMethod,
            int cvK,
            boolean parallel,
            String degree,
            int dimensions,
            String distanceMetric,
            String surfaceMode,
            boolean returnSe,
            boolean returnSorted,
            double cell,
            int interpolationVertices,
            int boundaryDegreeFallback,
            double[] weightedMetricWeights,
            String missing);

    static native void loessSetCvSeed(long handle, long seed);

    static native NativeResult loessFit(long handle, double[] x, double[] y, double[] customWeights);

    static native void loessFree(long handle);

    static native long streamingNew(
            double fraction,
            int iterations,
            String weightFunction,
            String robustnessMethod,
            String scalingMethod,
            String boundaryPolicy,
            boolean returnDiagnostics,
            boolean returnResiduals,
            boolean returnRobustnessWeights,
            String zeroWeightFallback,
            double autoConverge,
            boolean parallel,
            int chunkSize,
            int overlap,
            String mergeStrategy,
            String degree,
            int dimensions,
            String distanceMetric,
            String surfaceMode,
            double cell,
            int interpolationVertices,
            int boundaryDegreeFallback,
            double[] weightedMetricWeights,
            String missing);

    static native NativeResult streamingProcess(long handle, double[] x, double[] y);

    static native NativeResult streamingFinalize(long handle);

    static native void streamingFree(long handle);

    static native long onlineNew(
            double fraction,
            int iterations,
            String weightFunction,
            String robustnessMethod,
            String scalingMethod,
            String boundaryPolicy,
            boolean returnRobustnessWeights,
            String zeroWeightFallback,
            double autoConverge,
            int windowCapacity,
            int minPoints,
            String updateMode,
            String degree,
            int dimensions,
            String distanceMetric,
            String surfaceMode,
            double cell,
            int interpolationVertices,
            int boundaryDegreeFallback,
            double[] weightedMetricWeights,
            String missing);

    static native NativeOnlineOutput onlineAddPoint(long handle, double x, double y);

    static native void onlineFree(long handle);
}
