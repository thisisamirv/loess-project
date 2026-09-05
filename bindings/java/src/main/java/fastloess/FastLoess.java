package fastloess;

/**
 * Static utility methods for the fastloess native library.
 */
public final class FastLoess {

    private FastLoess() {
    }

    /**
     * The released version of this Java binding (bindings/java), tracked
     * independently of the underlying fastLoess Rust core's crate version.
     */
    public static final String VERSION = "2.0.0";

    /**
     * Returns the version of this Java binding.
     *
     * @return the version string
     */
    public static String version() {
        return VERSION;
    }
}
