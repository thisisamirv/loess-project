# Validation

## Numerical Tests

R randomized properties and fixed reference cases live in
[`property_tests/`](property_tests/). Golden-output tests and their committed
fixtures live in [`fixture_tests/`](fixture_tests/). Run both suites, with R lint,
the Python boundary properties, and direct Rust-core regressions from the
repository root:

```sh
make r
make validate
```

Direct Rust-core numerical regressions live in
[`rust_tests/tests/numerical_regressions.rs`](rust_tests/tests/numerical_regressions.rs).
They cover Gaussian standard errors, high-range scaling and diagnostics, and
case-weight invariants. `make validate` runs them alongside the binding-level
suites.

`make validate` creates the repository Python virtual environment when needed and
installs NumPy, pytest, Hypothesis, and the local Python binding independently of
`make python-dev`. Its boundary property generates 40 cases per policy across
varied input lengths, irregular x spacing, response shapes, and fractions.
Extend, Reflect, and Zero are compared with explicitly padded direct linear fits
that preserve the original neighbor count. This isolates padding and output
slicing; it is not an independent oracle for the shared LOESS engine.

The R property suite compares direct and interpolated fits against `stats::loess`,
including input order, sorted outputs, repeated predictors, robustness, and sparse
outliers. Fixed cases cover noiseless and degenerate inputs and long-run robust
fits. Golden fixtures pin boundary, interval, gradient, robustness-weight,
Streaming, and Online behavior. `make validate` installs `quickcheck`, `testthat`,
and `lintr` when missing; `quickcheck` is validation-only, not a package dependency.
The R binding must already be installed, as above or by `make r-dev`.

`make all-dev` runs validation last, after all component checks. `make r-dev` and
`make r-tests` run package-only checks; they do not run these repository suites.

## Visual Validation

[`rust_tests/`](rust_tests/) generates CSV data for explanatory plots and
contains the direct Rust numerical tests. The visual comparisons help inspect
degrees, kernels, boundaries, intervals, adapters, multivariate surfaces, and
other behaviors; they are not correctness oracles.

From the repository root:

```sh
make -C validation visual
make -C validation plot PYTHON=python
```

For plotting, choose an interpreter with NumPy, Pandas, and Matplotlib installed.
Use an absolute interpreter path when passing `PYTHON`, or one resolved on `PATH`;
for example, `make -C validation plot PYTHON=python`. Generated CSVs and SVGs are
kept in [`rust_tests/output/`](rust_tests/output/).

## Reference Sources

[`reference/`](reference/) contains R's LOESS R/C/Fortran sources and Cleveland's
original LOESS Fortran reference. These are distinct from LOWESS reference sources.
