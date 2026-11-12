//! Random number generation for the further discrete distributions.
//!
//! Each `*_rng` function validates its parameters with the same rules as the corresponding log
//! mass function and draws one value (or one vector of counts).

use rand::Rng;
use rand::distr::{Distribution, Uniform};
use stanmath_rs_core::special::{Phi, inv_logit, log_sum_exp_slice};
use stanmath_rs_core::{Error, Result};

use crate::check::param;
use crate::rng::{
    bernoulli_rng, beta_rng, binomial_rng, categorical_rng, gamma_rng, neg_binomial_2_rng,
    poisson_rng,
};

fn nonnegative_int(f: &'static str, name: &'static str, v: i64) -> Result<()> {
    if v < 0 {
        return Err(Error::parameter(f, name, v as f64, "nonnegative"));
    }
    Ok(())
}

fn check_simplex(f: &'static str, name: &'static str, theta: &[f64]) -> Result<()> {
    let sum: f64 = theta.iter().sum();
    if theta.iter().any(|t| t.is_nan() || *t < 0.0) || (sum - 1.0).abs() > 1e-8 {
        return Err(Error::parameter(
            f,
            name,
            sum,
            "a simplex (nonnegative, summing to 1)",
        ));
    }
    Ok(())
}

fn check_all_finite(f: &'static str, name: &'static str, xs: &[f64]) -> Result<()> {
    match xs.iter().find(|v| !v.is_finite()) {
        Some(&v) => Err(Error::parameter(f, name, v, "finite")),
        None => Ok(()),
    }
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let lse = log_sum_exp_slice(x);
    x.iter().map(|&v| (v - lse).exp()).collect()
}

/// Draw from `beta_binomial(N, alpha, beta)`: a binomial count with a beta distributed success
/// probability.
pub fn beta_binomial_rng<R: Rng + ?Sized>(
    trials: i64,
    alpha: f64,
    beta: f64,
    rng: &mut R,
) -> Result<i64> {
    const F: &str = "beta_binomial_rng";
    nonnegative_int(F, "N", trials)?;
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "beta", beta)?;
    let p = beta_rng(alpha, beta, rng)?;
    binomial_rng(trials, p, rng)
}

/// Draw from `hypergeometric(N, a, b)`: the number of successes in `N` draws without replacement
/// from `a` successes and `b` failures.
pub fn hypergeometric_rng<R: Rng + ?Sized>(
    draws: i64,
    successes: i64,
    failures: i64,
    rng: &mut R,
) -> Result<i64> {
    const F: &str = "hypergeometric_rng";
    nonnegative_int(F, "a", successes)?;
    nonnegative_int(F, "b", failures)?;
    if !(0..=successes + failures).contains(&draws) {
        return Err(Error::parameter(F, "N", draws as f64, "in [0, a + b]"));
    }
    let unit = Uniform::new(0.0, 1.0)
        .map_err(|_| Error::parameter(F, "parameters", f64::NAN, "valid for the sampler"))?;
    // ponytail: one uniform per draw, O(N); switch to an inversion or ratio sampler for huge N.
    let (mut a, mut b, mut k) = (successes, failures, 0);
    for _ in 0..draws {
        let u: f64 = unit.sample(rng);
        if u * ((a + b) as f64) < a as f64 {
            k += 1;
            a -= 1;
        } else {
            b -= 1;
        }
    }
    Ok(k)
}

/// Draw from `neg_binomial(alpha, beta)` (shape, inverse scale) as a gamma mixture of Poissons.
pub fn neg_binomial_rng<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<i64> {
    const F: &str = "neg_binomial_rng";
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "beta", beta)?;
    let rate = gamma_rng(alpha, beta, rng)?;
    poisson_rng(rate, rng)
}

/// Draw from `beta_neg_binomial(r, alpha, beta)`: failures before `r` successes with a beta
/// distributed success probability.
pub fn beta_neg_binomial_rng<R: Rng + ?Sized>(
    r: f64,
    alpha: f64,
    beta: f64,
    rng: &mut R,
) -> Result<i64> {
    const F: &str = "beta_neg_binomial_rng";
    param::positive_finite(F, "r", r)?;
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "beta", beta)?;
    let p = beta_rng(alpha, beta, rng)?;
    if p >= 1.0 {
        return Ok(0);
    }
    // Failures before r successes: Poisson with a gamma(r, p / (1 - p)) distributed rate.
    let odds = p / (1.0 - p);
    if !(odds > 0.0) {
        return Err(Error::parameter(
            F,
            "success probability",
            p,
            "positive (draw underflowed)",
        ));
    }
    let rate = gamma_rng(r, odds, rng)?;
    poisson_rng(rate, rng)
}

/// Draw uniformly from the integers `lower..=upper`.
pub fn discrete_range_rng<R: Rng + ?Sized>(lower: i64, upper: i64, rng: &mut R) -> Result<i64> {
    const F: &str = "discrete_range_rng";
    if lower > upper {
        return Err(Error::parameter(
            F,
            "lower",
            lower as f64,
            "not greater than upper",
        ));
    }
    let d = Uniform::new_inclusive(lower, upper)
        .map_err(|_| Error::parameter(F, "parameters", f64::NAN, "valid for the sampler"))?;
    Ok(d.sample(rng))
}

/// Draw from `poisson_binomial(theta)`: the number of successes among independent trials.
pub fn poisson_binomial_rng<R: Rng + ?Sized>(theta: &[f64], rng: &mut R) -> Result<i64> {
    let mut k = 0;
    for &t in theta {
        param::probability("poisson_binomial_rng", "theta", t)?;
        k += bernoulli_rng(t, rng)?;
    }
    Ok(k)
}

/// Draws a category in `1..=K` from cumulative probabilities `P(y <= k)`, `k = 1..K-1`.
fn draw_ordered<R: Rng + ?Sized>(cdf: &[f64], rng: &mut R) -> Result<i64> {
    let mut probs = Vec::with_capacity(cdf.len() + 1);
    let mut prev = 0.0;
    for &c in cdf.iter().chain(core::iter::once(&1.0)) {
        probs.push((c - prev).max(0.0));
        prev = c;
    }
    categorical_rng(&probs, rng)
}

fn check_ordered_args(f: &'static str, eta: f64, c: &[f64]) -> Result<()> {
    param::finite(f, "eta", eta)?;
    check_all_finite(f, "c", c)?;
    if let Some(w) = c.windows(2).find(|w| !(w[0] < w[1])) {
        return Err(Error::parameter(f, "c", w[1], "strictly increasing"));
    }
    Ok(())
}

/// Draw from `ordered_logistic(eta, c)`, a category in `1..=K` with `K = size(c) + 1`.
pub fn ordered_logistic_rng<R: Rng + ?Sized>(eta: f64, c: &[f64], rng: &mut R) -> Result<i64> {
    check_ordered_args("ordered_logistic_rng", eta, c)?;
    let cdf: Vec<f64> = c.iter().map(|&ck| inv_logit(ck - eta)).collect();
    draw_ordered(&cdf, rng)
}

/// Draw from `ordered_probit(eta, c)`, a category in `1..=K` with `K = size(c) + 1`.
pub fn ordered_probit_rng<R: Rng + ?Sized>(eta: f64, c: &[f64], rng: &mut R) -> Result<i64> {
    check_ordered_args("ordered_probit_rng", eta, c)?;
    let cdf: Vec<f64> = c.iter().map(|&ck| Phi(ck - eta)).collect();
    draw_ordered(&cdf, rng)
}

/// Draw counts from `multinomial(theta)` with `trials` total, by sequential conditional
/// binomials.
pub fn multinomial_rng<R: Rng + ?Sized>(
    theta: &[f64],
    trials: i64,
    rng: &mut R,
) -> Result<Vec<i64>> {
    const F: &str = "multinomial_rng";
    check_simplex(F, "theta", theta)?;
    nonnegative_int(F, "N", trials)?;
    let mut counts = vec![0i64; theta.len()];
    let (mut remaining, mut mass) = (trials, 1.0f64);
    for (k, &t) in theta.iter().enumerate() {
        if remaining == 0 {
            break;
        }
        if k + 1 == theta.len() {
            counts[k] = remaining;
            break;
        }
        let p = if mass > 0.0 {
            (t / mass).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let n = binomial_rng(remaining, p, rng)?;
        counts[k] = n;
        remaining -= n;
        mass -= t;
    }
    Ok(counts)
}

/// Draw counts from `multinomial_logit(gamma)`, that is `multinomial(softmax(gamma))`.
pub fn multinomial_logit_rng<R: Rng + ?Sized>(
    gamma: &[f64],
    trials: i64,
    rng: &mut R,
) -> Result<Vec<i64>> {
    check_all_finite("multinomial_logit_rng", "gamma", gamma)?;
    multinomial_rng(&softmax(gamma), trials, rng)
}

/// Draw counts from `dirichlet_multinomial(alpha)` with `trials` total: a multinomial with
/// Dirichlet distributed probabilities.
pub fn dirichlet_multinomial_rng<R: Rng + ?Sized>(
    alpha: &[f64],
    trials: i64,
    rng: &mut R,
) -> Result<Vec<i64>> {
    const F: &str = "dirichlet_multinomial_rng";
    for &a in alpha {
        param::positive_finite(F, "alpha", a)?;
    }
    nonnegative_int(F, "N", trials)?;
    if trials == 0 {
        return Ok(vec![0; alpha.len()]);
    }
    let mut g = Vec::with_capacity(alpha.len());
    for &a in alpha {
        g.push(gamma_rng(a, 1.0, rng)?);
    }
    let total: f64 = g.iter().sum();
    let theta: Vec<f64> = g.iter().map(|v| v / total).collect();
    multinomial_rng(&theta, trials, rng)
}

/// Draw from `binomial(trials, inv_logit(alpha))`.
pub fn binomial_logit_rng<R: Rng + ?Sized>(trials: i64, alpha: f64, rng: &mut R) -> Result<i64> {
    param::finite("binomial_logit_rng", "alpha", alpha)?;
    binomial_rng(trials, inv_logit(alpha), rng)
}

/// Draw from `neg_binomial_2(exp(eta), phi)`.
pub fn neg_binomial_2_log_rng<R: Rng + ?Sized>(eta: f64, phi: f64, rng: &mut R) -> Result<i64> {
    const F: &str = "neg_binomial_2_log_rng";
    param::finite(F, "eta", eta)?;
    param::positive_finite(F, "phi", phi)?;
    neg_binomial_2_rng(eta.exp(), phi, rng)
}

/// Draw a category in `1..=K` with probabilities `softmax(beta)`.
pub fn categorical_logit_rng<R: Rng + ?Sized>(beta: &[f64], rng: &mut R) -> Result<i64> {
    check_all_finite("categorical_logit_rng", "beta", beta)?;
    categorical_rng(&softmax(beta), rng)
}
