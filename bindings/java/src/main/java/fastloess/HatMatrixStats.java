package fastloess;

/**
 * Hat-matrix statistics computed for a batch fit when {@code returnSe} is
 * requested.
 *
 * @param enp equivalent number of parameters
 * @param traceHat trace of the hat matrix
 * @param delta1 first delta statistic (used for the residual scale estimate)
 * @param delta2 second delta statistic (used for GCV/AIC-style criteria)
 * @param residualScale residual scale estimate derived from the hat matrix
 * @param leverage per-point hat-matrix diagonal (leverage) values
 */
public record HatMatrixStats(
        double enp,
        double traceHat,
        double delta1,
        double delta2,
        double residualScale,
        double[] leverage) {

    static HatMatrixStats fromNative(NativeResult r) {
        return new HatMatrixStats(r.enp, r.traceHat, r.delta1, r.delta2, r.residualScale, r.leverage);
    }
}
