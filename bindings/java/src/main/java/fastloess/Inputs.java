package fastloess;

/**
 * Shared argument checks for public fit / stream / predict entry points.
 */
final class Inputs {

    private Inputs() {
    }

    static void requireXy(double[] x, double[] y, int dimensions) {
        if (x == null || y == null) {
            throw new IllegalArgumentException("x and y must not be null");
        }
        if (y.length == 0) {
            throw new IllegalArgumentException("x and y must not be empty");
        }
        if (x.length != y.length * dimensions) {
            throw new IllegalArgumentException(
                    "x length must equal y.length * dimensions (" + (y.length * dimensions) + ")");
        }
    }

    static void requireQueryX(double[] newX) {
        if (newX == null || newX.length == 0) {
            throw new IllegalArgumentException("newX must not be null or empty");
        }
    }

    static void requireCustomWeights(double[] customWeights, int yLength) {
        if (customWeights == null) {
            return;
        }
        if (customWeights.length == 0) {
            throw new IllegalArgumentException("customWeights must not be empty");
        }
        if (customWeights.length != yLength) {
            throw new IllegalArgumentException("customWeights must match y length");
        }
    }
}
