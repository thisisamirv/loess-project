# Stored golden fixtures

These CSV files pin LOESS engine paths that have no independent reference
implementation: the default extended boundary, intervals and gradients,
robustness weights, Streaming, and Online full-update mode.

`test-golden.R` re-runs each case and compares the fresh result with the
committed values using a `1e-10` tolerance. Regenerate them explicitly with:

```sh
Rscript validation/fixture_tests/fixtures/make_reference.R
```

The generator records the R version, package version, platform, seed, RNG
kind, tolerance, and file checksums in `PROVENANCE.txt`.
The existing provenance retains the generator path used before these fixtures
were moved; regenerated fixtures record the current repository path.
