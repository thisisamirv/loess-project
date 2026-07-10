use criterion::{BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main};
use fastLoess::prelude::*;
use std::hint::black_box;
use std::time::Duration;

/// Read LOESS_PARALLEL env var (default: true).
/// Set LOESS_PARALLEL=false to run in serial mode.
fn use_parallel() -> bool {
    std::env::var("LOESS_PARALLEL")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

// Deterministic noise: values in [-1, 1], mean ≈ 0
fn noise(i: usize, scale: f64) -> f64 {
    scale * (i as f64 * 13.37).sin()
}

// x in [0, 10], y = sin(x) + noise(sd≈0.2)
fn generate_sine_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() + noise(i, 0.2))
        .collect();
    (x, y)
}

// Same as sine but ~5% of points have ±5 added
fn generate_outlier_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let base = xi.sin() + noise(i, 0.2);
            if i % 20 == 0 {
                base + noise(i, 5.0)
            } else {
                base
            }
        })
        .collect();
    (x, y)
}

// x = 0..n-1, y = cumulative price starting at 100 with small returns
fn generate_financial_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut y = vec![100.0f64; n];
    for i in 1..n {
        let ret = 0.0005 + noise(i, 0.02);
        y[i] = y[i - 1] * (1.0 + ret);
    }
    (x, y)
}

// x in [0, 10], y = exp(-x*0.3)*cos(x*2π) + noise(sd≈0.05)
fn generate_scientific_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            (-xi * 0.3).exp() * (xi * 2.0 * std::f64::consts::PI).cos() + noise(i, 0.05)
        })
        .collect();
    (x, y)
}

// x = 0..n-1 * 1000, y = clamp(0.5 + sin(x/50000)*0.3 + noise(sd≈0.1), 0, 1)
fn generate_genomic_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 1000.0).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let v = 0.5 + (xi / 50000.0).sin() * 0.3 + noise(i, 0.1);
            v.clamp(0.0, 1.0)
        })
        .collect();
    (x, y)
}

// Clustered x: groups of 100 tightly packed, y = sin(x) + noise(sd≈0.1)
fn generate_clustered_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n)
        .map(|i| (i / 100) as f64 + (i % 100) as f64 * 1e-6)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() + noise(i, 0.1))
        .collect();
    (x, y)
}

// x in [0, 10], y = sin(x)*0.5 + noise(sd≈2.0)
fn generate_high_noise_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 / (n - 1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi.sin() * 0.5 + noise(i, 2.0))
        .collect();
    (x, y)
}

fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for size in [1_000, 5_000, 10_000].iter() {
        let (x, y) = generate_sine_data(*size);
        group.bench_with_input(BenchmarkId::new("scale", size), size, |b, _| {
            let model = Loess::new()
                .fraction(0.1)
                .iterations(3)
                .parallel(use_parallel())
                .build()
                .unwrap();
            b.iter(|| model.clone().fit(black_box(&x), black_box(&y)).unwrap());
        });
    }
    group.finish();
}

fn bench_fraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("fraction");
    let (x, y) = generate_sine_data(5_000);

    for frac in [0.05, 0.1, 0.2, 0.3, 0.5, 0.67].iter() {
        group.bench_with_input(BenchmarkId::new("fraction", frac), frac, |b, &f| {
            let model = Loess::new()
                .fraction(f)
                .iterations(3)
                .parallel(use_parallel())
                .build()
                .unwrap();
            b.iter(|| model.clone().fit(black_box(&x), black_box(&y)).unwrap());
        });
    }
    group.finish();
}

fn bench_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterations");
    let (x, y) = generate_outlier_data(5_000);

    for iters in [0usize, 1, 2, 3, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("iterations", iters), iters, |b, &i| {
            let model = Loess::new()
                .fraction(0.2)
                .iterations(i)
                .parallel(use_parallel())
                .build()
                .unwrap();
            b.iter(|| model.clone().fit(black_box(&x), black_box(&y)).unwrap());
        });
    }
    group.finish();
}

fn bench_financial(c: &mut Criterion) {
    let mut group = c.benchmark_group("financial");
    group.sample_size(10);

    for size in [500, 1_000, 5_000].iter() {
        let (x, y) = generate_financial_data(*size);
        group.bench_with_input(BenchmarkId::new("financial", size), size, |b, _| {
            let model = Loess::new()
                .fraction(0.1)
                .iterations(2)
                .parallel(use_parallel())
                .build()
                .unwrap();
            b.iter(|| model.clone().fit(black_box(&x), black_box(&y)).unwrap());
        });
    }
    group.finish();
}

fn bench_scientific(c: &mut Criterion) {
    let mut group = c.benchmark_group("scientific");
    group.sample_size(10);

    for size in [500, 1_000, 5_000].iter() {
        let (x, y) = generate_scientific_data(*size);
        group.bench_with_input(BenchmarkId::new("scientific", size), size, |b, _| {
            let model = Loess::new()
                .fraction(0.15)
                .iterations(3)
                .parallel(use_parallel())
                .build()
                .unwrap();
            b.iter(|| model.clone().fit(black_box(&x), black_box(&y)).unwrap());
        });
    }
    group.finish();
}

fn bench_genomic(c: &mut Criterion) {
    let mut group = c.benchmark_group("genomic");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for size in [1_000, 5_000, 100_000].iter() {
        let (x, y) = generate_genomic_data(*size);
        group.bench_with_input(BenchmarkId::new("genomic", size), size, |b, _| {
            let model = Loess::new()
                .fraction(0.1)
                .iterations(3)
                .parallel(use_parallel())
                .build()
                .unwrap();
            b.iter(|| model.clone().fit(black_box(&x), black_box(&y)).unwrap());
        });
    }
    group.finish();
}

fn bench_pathological(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathological");
    group.sample_size(10);
    let size = 5_000;

    let (xc, yc) = generate_clustered_data(size);
    group.bench_function("clustered", |b| {
        let model = Loess::new()
            .fraction(0.3)
            .iterations(2)
            .parallel(use_parallel())
            .build()
            .unwrap();
        b.iter(|| model.clone().fit(black_box(&xc), black_box(&yc)).unwrap());
    });

    let (xn, yn) = generate_high_noise_data(size);
    group.bench_function("high_noise", |b| {
        let model = Loess::new()
            .fraction(0.5)
            .iterations(5)
            .parallel(use_parallel())
            .build()
            .unwrap();
        b.iter(|| model.clone().fit(black_box(&xn), black_box(&yn)).unwrap());
    });

    let (xo, yo) = generate_outlier_data(size);
    group.bench_function("extreme_outliers", |b| {
        let model = Loess::new()
            .fraction(0.2)
            .iterations(10)
            .parallel(use_parallel())
            .build()
            .unwrap();
        b.iter(|| model.clone().fit(black_box(&xo), black_box(&yo)).unwrap());
    });

    let xk: Vec<f64> = (1..=size).map(|i| i as f64).collect();
    let yk: Vec<f64> = vec![5.0; size];
    group.bench_function("constant_y", |b| {
        let model = Loess::new()
            .fraction(0.2)
            .iterations(2)
            .parallel(use_parallel())
            .build()
            .unwrap();
        b.iter(|| model.clone().fit(black_box(&xk), black_box(&yk)).unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scalability,
    bench_fraction,
    bench_iterations,
    bench_financial,
    bench_scientific,
    bench_genomic,
    bench_pathological
);
criterion_main!(benches);
