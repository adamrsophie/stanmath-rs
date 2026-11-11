//! Random number generation, using samplers from `rand_distr`.
//!
//! Each `*_rng` function validates its parameters with the same rules as the corresponding log
//! density and draws one value. Pass any `rand::Rng`, for example a seeded `rand::rngs::StdRng`.

use probgrad_core::{Error, Result};
use rand::Rng;
use rand::distr::{Bernoulli, Distribution, Uniform, weighted::WeightedIndex};
use rand_distr as rd;

use crate::check::param;

fn sample<D: Distribution<T>, T, R: Rng + ?Sized>(d: D, rng: &mut R) -> T {
    d.sample(rng)
}

macro_rules! build {
    ($f:expr, $e:expr) => {
        $e.map_err(|_| Error::parameter($f, "parameters", f64::NAN, "valid for the sampler"))?
    };
}

/// Draw from `normal(mu, sigma)`.
pub fn normal_rng<R: Rng + ?Sized>(mu: f64, sigma: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "normal_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "sigma", sigma)?;
    Ok(sample(build!(F, rd::Normal::new(mu, sigma)), rng))
}

/// Draw from `student_t(nu, mu, sigma)`.
pub fn student_t_rng<R: Rng + ?Sized>(nu: f64, mu: f64, sigma: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "student_t_rng";
    param::positive_finite(F, "nu", nu)?;
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "sigma", sigma)?;
    let t: f64 = sample(build!(F, rd::StudentT::new(nu)), rng);
    Ok(mu + sigma * t)
}

/// Draw from `cauchy(mu, sigma)`.
pub fn cauchy_rng<R: Rng + ?Sized>(mu: f64, sigma: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "cauchy_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "sigma", sigma)?;
    Ok(sample(build!(F, rd::Cauchy::new(mu, sigma)), rng))
}

/// Draw from `logistic(mu, sigma)` by inversion.
pub fn logistic_rng<R: Rng + ?Sized>(mu: f64, sigma: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "logistic_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "sigma", sigma)?;
    let u: f64 = sample(build!(F, Uniform::new(0.0, 1.0)), rng);
    Ok(mu + sigma * probgrad_core::special::logit(u))
}

/// Draw from `exponential(beta)` (rate `beta`).
pub fn exponential_rng<R: Rng + ?Sized>(beta: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "exponential_rng";
    param::positive_finite(F, "beta", beta)?;
    Ok(sample(build!(F, rd::Exp::new(beta)), rng))
}

/// Draw from `gamma(alpha, beta)` (shape, rate).
pub fn gamma_rng<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "gamma_rng";
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "beta", beta)?;
    Ok(sample(build!(F, rd::Gamma::new(alpha, 1.0 / beta)), rng))
}

/// Draw from `inv_gamma(alpha, beta)` (shape, scale).
pub fn inv_gamma_rng<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "inv_gamma_rng";
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "beta", beta)?;
    let g: f64 = sample(build!(F, rd::Gamma::new(alpha, 1.0 / beta)), rng);
    Ok(1.0 / g)
}

/// Draw from `beta(alpha, beta)`.
pub fn beta_rng<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "beta_rng";
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "beta", beta)?;
    Ok(sample(build!(F, rd::Beta::new(alpha, beta)), rng))
}

/// Draw from `lognormal(mu, sigma)`.
pub fn lognormal_rng<R: Rng + ?Sized>(mu: f64, sigma: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "lognormal_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "sigma", sigma)?;
    Ok(sample(build!(F, rd::LogNormal::new(mu, sigma)), rng))
}

/// Draw from `uniform(alpha, beta)`.
pub fn uniform_rng<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "uniform_rng";
    param::finite(F, "alpha", alpha)?;
    param::finite(F, "beta", beta)?;
    if beta <= alpha {
        return Err(Error::parameter(F, "beta", beta, "greater than alpha"));
    }
    Ok(sample(build!(F, Uniform::new(alpha, beta)), rng))
}

/// Draw from `weibull(alpha, sigma)` (shape, scale).
pub fn weibull_rng<R: Rng + ?Sized>(alpha: f64, sigma: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "weibull_rng";
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "sigma", sigma)?;
    Ok(sample(build!(F, rd::Weibull::new(sigma, alpha)), rng))
}

/// Draw from `bernoulli(theta)`.
pub fn bernoulli_rng<R: Rng + ?Sized>(theta: f64, rng: &mut R) -> Result<i64> {
    const F: &str = "bernoulli_rng";
    param::probability(F, "theta", theta)?;
    let b: bool = sample(build!(F, Bernoulli::new(theta)), rng);
    Ok(i64::from(b))
}

/// Draw from `bernoulli(inv_logit(alpha))`.
pub fn bernoulli_logit_rng<R: Rng + ?Sized>(alpha: f64, rng: &mut R) -> Result<i64> {
    param::finite("bernoulli_logit_rng", "alpha", alpha)?;
    bernoulli_rng(probgrad_core::special::inv_logit(alpha), rng)
}

/// Draw from `binomial(trials, theta)`.
pub fn binomial_rng<R: Rng + ?Sized>(trials: i64, theta: f64, rng: &mut R) -> Result<i64> {
    const F: &str = "binomial_rng";
    if trials < 0 {
        return Err(Error::parameter(F, "N", trials as f64, "nonnegative"));
    }
    param::probability(F, "theta", theta)?;
    let n: u64 = sample(build!(F, rd::Binomial::new(trials as u64, theta)), rng);
    Ok(n as i64)
}

/// Draw from `poisson(lambda)`.
pub fn poisson_rng<R: Rng + ?Sized>(lambda: f64, rng: &mut R) -> Result<i64> {
    const F: &str = "poisson_rng";
    param::nonnegative(F, "lambda", lambda)?;
    param::finite(F, "lambda", lambda)?;
    if lambda == 0.0 {
        return Ok(0);
    }
    let n: f64 = sample(build!(F, rd::Poisson::new(lambda)), rng);
    Ok(n as i64)
}

/// Draw from `poisson(exp(alpha))`.
pub fn poisson_log_rng<R: Rng + ?Sized>(alpha: f64, rng: &mut R) -> Result<i64> {
    param::not_nan("poisson_log_rng", "alpha", alpha)?;
    poisson_rng(alpha.exp(), rng)
}

/// Draw from `neg_binomial_2(mu, phi)` as a gamma mixture of Poissons.
pub fn neg_binomial_2_rng<R: Rng + ?Sized>(mu: f64, phi: f64, rng: &mut R) -> Result<i64> {
    const F: &str = "neg_binomial_2_rng";
    param::positive_finite(F, "mu", mu)?;
    param::positive_finite(F, "phi", phi)?;
    let rate = gamma_rng(phi, phi / mu, rng)?;
    poisson_rng(rate, rng)
}

/// Draw a category in `1..=K` with probabilities `theta`.
pub fn categorical_rng<R: Rng + ?Sized>(theta: &[f64], rng: &mut R) -> Result<i64> {
    const F: &str = "categorical_rng";
    let sum: f64 = theta.iter().sum();
    if theta.iter().any(|t| t.is_nan() || *t < 0.0) || (sum - 1.0).abs() > 1e-8 {
        return Err(Error::parameter(
            F,
            "theta",
            sum,
            "a simplex (nonnegative, summing to 1)",
        ));
    }
    let k: usize = sample(build!(F, WeightedIndex::new(theta)), rng);
    Ok(k as i64 + 1)
}
