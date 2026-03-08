use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
};
use std::hint::black_box;
use fastLoess::prelude::*;
use std::time::Duration;

// Helper to generate sine wave data with some noise
fn generate_sine_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let signal = (t * std::f64::consts::PI * 4.0).sin();
        let noise = 0.1 * (i as f64 * 13.37).sin();
        x.push(t);
        y.push(signal + noise);
    }
    (x, y)
}

fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for size in [1_000, 5_000, 10_000, 50_000].iter() {
        let (x, y) = generate_sine_data(*size);

        // CPU Serial
        group.bench_with_input(BenchmarkId::new("cpu_serial", size), size, |b, _| {
            let model = Loess::new()
                .fraction(0.3)
                .parallel(false)
                .adapter(Batch)
                .build()
                .unwrap();
            b.iter(|| (&model).fit(black_box(&x), black_box(&y)).unwrap());
        });

        // CPU Parallel
        group.bench_with_input(BenchmarkId::new("cpu_parallel", size), size, |b, _| {
            let model = Loess::new()
                .fraction(0.3)
                .parallel(true)
                .adapter(Batch)
                .build()
                .unwrap();
            b.iter(|| (&model).fit(black_box(&x), black_box(&y)).unwrap());
        });

        // GPU (optional feature)
        #[cfg(feature = "gpu")]
        {
            group.bench_with_input(BenchmarkId::new("gpu", size), size, |b, _| {
                // We'd need to ensure wgpu is initialized or benchmark a build that includes it
                let model = Loess::new()
                    .fraction(0.3)
                    .backend(Backend::Gpu)
                    .adapter(Batch)
                    .build()
                    .unwrap();
                b.iter(|| (&model).fit(black_box(&x), black_box(&y)).unwrap());
            });
        }
    }
    group.finish();
}

fn bench_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameters");
    let (x, y) = generate_sine_data(5_000);

    // Iterations Effect
    for iters in [0, 1, 4, 10].iter() {
        group.bench_with_input(BenchmarkId::new("iterations", iters), iters, |b, &i| {
            let model = Loess::new()
                .fraction(0.3)
                .iterations(i)
                .adapter(Batch)
                .build()
                .unwrap();
            b.iter(|| (&model).fit(black_box(&x), black_box(&y)).unwrap());
        });
    }

    // Fraction Effect
    for frac in [0.1, 0.3, 0.7].iter() {
        group.bench_with_input(BenchmarkId::new("fraction", frac), frac, |b, &f| {
            let model = Loess::new()
                .fraction(f)
                .adapter(Batch)
                .build()
                .unwrap();
            b.iter(|| (&model).fit(black_box(&x), black_box(&y)).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_scalability, bench_parameters);
criterion_main!(benches);
