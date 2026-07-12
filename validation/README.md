# Validation

Validates `fastLoess` (Rust) output against R's `stats::loess` as the reference implementation across 20 scenarios covering a wide range of inputs and parameter combinations.

## Scenarios

| # | Name | n | Fraction | Degree | Iterations | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Tiny Linear | 10 | 0.8 | 1 | 0 | Minimal dataset |
| 02 | Quadratic Degree 2 | 50 | 0.4 | 2 | 0 | Quadratic fit on quadratic data |
| 03 | Sine Standard | 100 | 0.3 | 1 | 0 | Noisy sine wave |
| 04 | Sine Robust | 100 | 0.3 | 1 | 4 | Sine with 5% outliers, bisquare reweighting |
| 05 | Degree 2 | 100 | 0.4 | 2 | 0 | Quadratic local polynomial |
| 06 | Large Scale | 500 | 0.1 | 1 | 0 | Narrow bandwidth, 500 points |
| 07 | High Smoothness | 100 | 0.9 | 1 | 0 | Very wide bandwidth |
| 08 | Low Smoothness | 100 | 0.1 | 1 | 0 | Very narrow bandwidth, direct surface |
| 09 | Sine Degree 2 | 100 | 0.3 | 2 | 0 | Quadratic fit on sine data |
| 10 | Constant | 50 | 0.5 | 1 | 0 | Constant y signal |
| 11 | Step Function | 100 | 0.4 | 1 | 0 | Discontinuous step signal |
| 12 | End-effects Left | 50 | 0.3 | 1 | 0 | Left boundary behavior |
| 13 | End-effects Right | 50 | 0.3 | 1 | 0 | Right boundary behavior |
| 14 | Sparse Data | 20 | 0.6 | 1 | 0 | Wide x-range, only 20 points |
| 15 | Dense Data | 500 | 0.05 | 1 | 0 | Very narrow bandwidth, 500 points |
| 16 | Degree 2 Robust | 100 | 0.3 | 2 | 4 | Quadratic + bisquare on outlier data |
| 17 | Degree 2 Direct | 100 | 0.2 | 2 | 0 | Quadratic, exact computation at all points |
| 18 | Iter 2 Check | 100 | 0.4 | 1 | 2 | Two robustness iterations |
| 19 | Interpolate Exact | 50 | 0.5 | 1 | 0 | Interpolation surface check |
| 20 | Zero Variance | 10 | 0.5 | 1 | 0 | Constant y, minimal n |

## Running

```sh
# Generate R reference outputs (writes output/r/)
make r-validate

# Run fastLoess validation (writes output/fastLoess/)
make fastloess-validate

# Run fastLoess visual output (writes output/fastLoess/)
make fastloess-visual

# Compare R and fastLoess outputs
make compare

# Generate plots
make plot
```

Output JSON files are written to `output/r/` and `output/fastLoess/`.
