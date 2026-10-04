#![cfg(feature = "dev")]
//! Empirical calibration check for confidence-interval standard errors.
//!
//! On linear truth the local-linear estimator is unbiased, so the Monte Carlo
//! standard deviation of fitted values can be compared with reported SEs.

use loess_rs::prelude::*;

struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }

    fn normal(&mut self) -> f64 {
        let u1 = (self.next_u32() as f64) / (u32::MAX as f64) + 1e-12;
        let u2 = (self.next_u32() as f64) / (u32::MAX as f64);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[test]
fn reported_se_is_calibrated_on_linear_truth() {
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| 10.0 * i as f64 / (n - 1) as f64).collect();
    let truth: Vec<f64> = x.iter().map(|&value| 2.0 * value + 1.0).collect();
    let sigma = 0.3;
    let repetitions = 100;

    for fraction in [0.1, 0.2, 0.3, 0.5] {
        let mut fits = vec![vec![0.0f64; n]; repetitions];
        let mut standard_errors = vec![vec![0.0f64; n]; repetitions];
        let mut rng = Rng(20260910);

        for fit_index in 0..repetitions {
            let mut y = truth.clone();
            for value in &mut y {
                *value += sigma * rng.normal();
            }

            let result = Loess::new()
                .fraction(fraction)
                .iterations(0)
                .intervals(loess_rs::IntervalsBuilder::new().confidence(0.95))
                .surface_mode("direct")
                .parallel(false)
                .build()
                .expect("builder should succeed")
                .fit(&x, &y)
                .expect("fit should succeed");
            fits[fit_index] = result.y;
            standard_errors[fit_index] = result.standard_errors.expect("SEs requested");
        }

        let lower = n / 10;
        let upper = n - lower;
        let mut ratio_sum = 0.0;
        let mut count = 0;
        for index in lower..upper {
            let mean = fits.iter().map(|fit| fit[index]).sum::<f64>() / repetitions as f64;
            let variance = fits
                .iter()
                .map(|fit| (fit[index] - mean).powi(2))
                .sum::<f64>()
                / (repetitions - 1) as f64;
            let true_se = variance.sqrt();
            let reported = standard_errors
                .iter()
                .map(|errors| errors[index])
                .sum::<f64>()
                / repetitions as f64;
            ratio_sum += reported / true_se;
            count += 1;
        }

        let ratio = ratio_sum / count as f64;
        assert!(
            (ratio - 1.0).abs() < 0.20,
            "reported/true SE ratio at fraction {fraction} is {ratio:.4}, expected ~1.0"
        );
    }
}
