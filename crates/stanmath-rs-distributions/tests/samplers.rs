//! The newer samplers against their own CDFs (a Kolmogorov-Smirnov test) or known moments. Both
//! sides are written independently: a sampler that disagrees with its CDF has a bug in one of
//! them.

#![cfg(feature = "rand")]

use rand::SeedableRng;
use rand::rngs::StdRng;
use stanmath_rs_distributions::*;

const N: usize = 5_000;

/// Kolmogorov-Smirnov statistic of `draws` against the CDF `cdf`.
fn ks(mut draws: Vec<f64>, cdf: impl Fn(f64) -> f64) -> f64 {
    draws.sort_by(f64::total_cmp);
    let n = draws.len() as f64;
    draws
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let f = cdf(x);
            (f - i as f64 / n).max((i + 1) as f64 / n - f)
        })
        .fold(0.0, f64::max)
}

macro_rules! ks_check {
    ($rng:ident, $sampler:ident($($p:expr),*), $lcdf:ident) => {{
        let draws: Vec<f64> = (0..N).map(|_| $sampler($($p,)* &mut $rng).unwrap()).collect();
        let d = ks(draws, |y| $lcdf(y, $($p),*).unwrap().exp());
        // The 0.1% critical value is 1.95 / sqrt(N).
        assert!(d < 1.95 / (N as f64).sqrt(), "{}: D = {d}", stringify!($sampler));
    }};
}

#[test]
fn continuous_samplers_match_their_cdfs() {
    let mut rng = StdRng::seed_from_u64(20261112);
    ks_check!(rng, skew_normal_rng(0.5, 1.5, -2.0), skew_normal_lcdf);
    ks_check!(
        rng,
        double_exponential_rng(1.0, 0.7),
        double_exponential_lcdf
    );
    ks_check!(rng, von_mises_rng(0.0, 2.0), von_mises_lcdf);
    ks_check!(rng, rayleigh_rng(1.4), rayleigh_lcdf);
    ks_check!(rng, pareto_rng(0.5, 3.0), pareto_lcdf);
    ks_check!(rng, pareto_type_2_rng(0.0, 2.0, 3.5), pareto_type_2_lcdf);
    ks_check!(rng, frechet_rng(3.0, 1.2), frechet_lcdf);
    ks_check!(rng, gumbel_rng(1.0, 2.0), gumbel_lcdf);
    ks_check!(rng, chi_square_rng(4.5), chi_square_lcdf);
    ks_check!(rng, inv_chi_square_rng(5.0), inv_chi_square_lcdf);
    ks_check!(
        rng,
        scaled_inv_chi_square_rng(6.0, 0.8),
        scaled_inv_chi_square_lcdf
    );
    ks_check!(rng, exp_mod_normal_rng(0.0, 1.0, 0.8), exp_mod_normal_lcdf);
    ks_check!(
        rng,
        skew_double_exponential_rng(0.0, 1.0, 0.3),
        skew_double_exponential_lcdf
    );
    ks_check!(rng, loglogistic_rng(2.0, 3.0), loglogistic_lcdf);
    ks_check!(rng, beta_proportion_rng(0.3, 8.0), beta_proportion_lcdf);
}

/// Sample mean within five standard errors and sample variance within 5%.
fn moments(name: &str, xs: &[f64], mean: f64, var: f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    assert!(
        (m - mean).abs() < 5.0 * (var / n).sqrt(),
        "{name}: mean {m} vs {mean}"
    );
    assert!(
        (v - var).abs() < 0.05 * var,
        "{name}: variance {v} vs {var}"
    );
}

#[test]
fn discrete_samplers_have_the_right_moments() {
    let mut rng = StdRng::seed_from_u64(20261113);
    let n = 40_000;
    let mut draw = |f: &mut dyn FnMut(&mut StdRng) -> i64| -> Vec<f64> {
        (0..n).map(|_| f(&mut rng) as f64).collect()
    };
    // Negative binomial (shape alpha, inverse scale beta): mean alpha / beta,
    // variance alpha (beta + 1) / beta^2.
    let xs = draw(&mut |r| neg_binomial_rng(3.0, 0.5, r).unwrap());
    moments("neg_binomial", &xs, 6.0, 18.0);
    // Beta binomial: mean N a / (a + b), variance N a b (a + b + N) / ((a + b)^2 (a + b + 1)).
    let xs = draw(&mut |r| beta_binomial_rng(20, 2.0, 3.0, r).unwrap());
    moments("beta_binomial", &xs, 8.0, 20.0 * 6.0 * 25.0 / (25.0 * 6.0));
    // Discrete uniform on 3..=9: mean 6, variance (7^2 - 1) / 12.
    let xs = draw(&mut |r| discrete_range_rng(3, 9, r).unwrap());
    moments("discrete_range", &xs, 6.0, 4.0);
    // Yule-Simon: mean alpha / (alpha - 1) and variance alpha^2 / ((alpha - 1)^2 (alpha - 2)).
    // Its kurtosis is too large for the 5% variance check, so only the mean is tested.
    let xs = draw(&mut |r| yule_simon_rng(6.0, r).unwrap());
    let m = xs.iter().sum::<f64>() / n as f64;
    assert!(
        (m - 1.2).abs() < 5.0 * (0.36 / n as f64).sqrt(),
        "yule_simon: mean {m}"
    );
    // Hypergeometric with N draws from a successes and b failures: mean N a / (a + b),
    // variance N a b (a + b - N) / ((a + b)^2 (a + b - 1)).
    let xs = draw(&mut |r| hypergeometric_rng(10, 12, 18, r).unwrap());
    moments(
        "hypergeometric",
        &xs,
        4.0,
        10.0 * 12.0 * 18.0 * 20.0 / (900.0 * 29.0),
    );
    // Binomial with logit success probability.
    let p = 1.0 / (1.0 + (-0.4f64).exp());
    let xs = draw(&mut |r| binomial_logit_rng(15, 0.4, r).unwrap());
    moments("binomial_logit", &xs, 15.0 * p, 15.0 * p * (1.0 - p));
    // Poisson with log rate.
    let xs = draw(&mut |r| poisson_log_rng(1.2, r).unwrap());
    moments("poisson_log", &xs, 1.2f64.exp(), 1.2f64.exp());
}
