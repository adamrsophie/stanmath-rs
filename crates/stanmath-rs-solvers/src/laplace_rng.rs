//! Draws of the latent vector from the embedded Laplace approximation.
//!
//! The approximate posterior of `theta` is `normal(theta_hat, (K^(-1) + W)^(-1))`, with the mode
//! `theta_hat` and the curvature `W` described in the Laplace module. The covariance is computed as
//! `(I + K W)^(-1) K`, which needs no inverse of `K`. Each function validates its arguments like
//! the corresponding `laplace_marginal` function and draws one latent vector (without the offset
//! `m` of the built in likelihoods). Pass any `rand::Rng`.

use rand::Rng;
use rand::distr::Distribution;
use rand_distr as rd;
use stanmath_rs_core::Result;

use crate::laplace::{
    Builtin, Family, General, LaplaceCovariance, LaplaceLikelihood, LaplaceOptions, Model,
    latent_gaussian,
};

fn draw<M: Model, K: LaplaceCovariance, R: Rng + ?Sized>(
    function: &'static str,
    model: &M,
    params: &[f64],
    covariance: &K,
    phi: &[f64],
    o: &LaplaceOptions,
    rng: &mut R,
) -> Result<Vec<f64>> {
    let (theta, l) = latent_gaussian(function, model, params, covariance, phi, o)?;
    let n = theta.len();
    let z: Vec<f64> = (0..n).map(|_| rd::StandardNormal.sample(rng)).collect();
    Ok((0..n)
        .map(|i| theta[i] + (0..=i).map(|j| l[(i, j)] * z[j]).sum::<f64>())
        .collect())
}

/// Draw of the latent vector for a user supplied likelihood, with default [`LaplaceOptions`].
/// Stan's `laplace_latent_rng`.
///
/// # Errors
///
/// As [`laplace_marginal_tol`](crate::laplace_marginal_tol).
pub fn laplace_latent_rng<L: LaplaceLikelihood, K: LaplaceCovariance, R: Rng + ?Sized>(
    likelihood: &L,
    eta: &[f64],
    covariance: &K,
    phi: &[f64],
    rng: &mut R,
) -> Result<Vec<f64>> {
    let o = LaplaceOptions::default();
    laplace_latent_tol_rng(likelihood, eta, covariance, phi, &o, rng)
}

/// [`laplace_latent_rng`] with explicit options. Stan's `laplace_latent_tol_rng`.
///
/// # Errors
///
/// As [`laplace_marginal_tol`](crate::laplace_marginal_tol).
pub fn laplace_latent_tol_rng<L: LaplaceLikelihood, K: LaplaceCovariance, R: Rng + ?Sized>(
    likelihood: &L,
    eta: &[f64],
    covariance: &K,
    phi: &[f64],
    options: &LaplaceOptions,
    rng: &mut R,
) -> Result<Vec<f64>> {
    let model = General {
        likelihood,
        block: options.hessian_block_size,
    };
    draw(
        "laplace_latent_rng",
        &model,
        eta,
        covariance,
        phi,
        options,
        rng,
    )
}

/// Draw of the latent vector of the Poisson log model of
/// [`laplace_marginal_poisson_log_lpmf`](crate::laplace_marginal_poisson_log_lpmf). Stan's
/// `laplace_latent_poisson_log_rng`.
///
/// # Errors
///
/// As [`laplace_marginal_tol_poisson_log_lpmf`](crate::laplace_marginal_tol_poisson_log_lpmf).
pub fn laplace_latent_poisson_log_rng<K: LaplaceCovariance, R: Rng + ?Sized>(
    y: &[i64],
    y_index: &[usize],
    m: &[f64],
    covariance: &K,
    phi: &[f64],
    rng: &mut R,
) -> Result<Vec<f64>> {
    let o = LaplaceOptions::default();
    laplace_latent_tol_poisson_log_rng(y, y_index, m, covariance, phi, &o, rng)
}

/// [`laplace_latent_poisson_log_rng`] with explicit options. Stan's
/// `laplace_latent_tol_poisson_log_rng`.
///
/// # Errors
///
/// As [`laplace_marginal_tol_poisson_log_lpmf`](crate::laplace_marginal_tol_poisson_log_lpmf).
pub fn laplace_latent_tol_poisson_log_rng<K: LaplaceCovariance, R: Rng + ?Sized>(
    y: &[i64],
    y_index: &[usize],
    m: &[f64],
    covariance: &K,
    phi: &[f64],
    options: &LaplaceOptions,
    rng: &mut R,
) -> Result<Vec<f64>> {
    const F: &str = "laplace_latent_poisson_log_rng";
    let (model, params) = Builtin::new(F, Family::PoissonLog, y, y_index, m, None)?;
    draw(F, &model, &params, covariance, phi, options, rng)
}

/// Draw of the latent vector of the negative binomial model of
/// [`laplace_marginal_neg_binomial_2_log_lpmf`](crate::laplace_marginal_neg_binomial_2_log_lpmf).
/// Stan's `laplace_latent_neg_binomial_2_log_rng`.
///
/// # Errors
///
/// As [`laplace_marginal_tol_neg_binomial_2_log_lpmf`](crate::laplace_marginal_tol_neg_binomial_2_log_lpmf).
pub fn laplace_latent_neg_binomial_2_log_rng<K: LaplaceCovariance, R: Rng + ?Sized>(
    y: &[i64],
    y_index: &[usize],
    eta: f64,
    m: &[f64],
    covariance: &K,
    phi: &[f64],
    rng: &mut R,
) -> Result<Vec<f64>> {
    let o = LaplaceOptions::default();
    laplace_latent_tol_neg_binomial_2_log_rng(y, y_index, eta, m, covariance, phi, &o, rng)
}

/// [`laplace_latent_neg_binomial_2_log_rng`] with explicit options. Stan's
/// `laplace_latent_tol_neg_binomial_2_log_rng`.
///
/// # Errors
///
/// As [`laplace_marginal_tol_neg_binomial_2_log_lpmf`](crate::laplace_marginal_tol_neg_binomial_2_log_lpmf).
#[allow(clippy::too_many_arguments)] // mirrors Stan's argument list
pub fn laplace_latent_tol_neg_binomial_2_log_rng<K: LaplaceCovariance, R: Rng + ?Sized>(
    y: &[i64],
    y_index: &[usize],
    eta: f64,
    m: &[f64],
    covariance: &K,
    phi: &[f64],
    options: &LaplaceOptions,
    rng: &mut R,
) -> Result<Vec<f64>> {
    const F: &str = "laplace_latent_neg_binomial_2_log_rng";
    let (model, params) = Builtin::new(F, Family::NegBinomial2Log, y, y_index, m, Some(eta))?;
    draw(F, &model, &params, covariance, phi, options, rng)
}

/// Draw of the latent vector of the logistic model of
/// [`laplace_marginal_bernoulli_logit_lpmf`](crate::laplace_marginal_bernoulli_logit_lpmf).
/// Stan's `laplace_latent_bernoulli_logit_rng`.
///
/// # Errors
///
/// As [`laplace_marginal_tol_bernoulli_logit_lpmf`](crate::laplace_marginal_tol_bernoulli_logit_lpmf).
pub fn laplace_latent_bernoulli_logit_rng<K: LaplaceCovariance, R: Rng + ?Sized>(
    y: &[i64],
    y_index: &[usize],
    m: &[f64],
    covariance: &K,
    phi: &[f64],
    rng: &mut R,
) -> Result<Vec<f64>> {
    let o = LaplaceOptions::default();
    laplace_latent_tol_bernoulli_logit_rng(y, y_index, m, covariance, phi, &o, rng)
}

/// [`laplace_latent_bernoulli_logit_rng`] with explicit options. Stan's
/// `laplace_latent_tol_bernoulli_logit_rng`.
///
/// # Errors
///
/// As [`laplace_marginal_tol_bernoulli_logit_lpmf`](crate::laplace_marginal_tol_bernoulli_logit_lpmf).
pub fn laplace_latent_tol_bernoulli_logit_rng<K: LaplaceCovariance, R: Rng + ?Sized>(
    y: &[i64],
    y_index: &[usize],
    m: &[f64],
    covariance: &K,
    phi: &[f64],
    options: &LaplaceOptions,
    rng: &mut R,
) -> Result<Vec<f64>> {
    const F: &str = "laplace_latent_bernoulli_logit_rng";
    let (model, params) = Builtin::new(F, Family::BernoulliLogit, y, y_index, m, None)?;
    draw(F, &model, &params, covariance, phi, options, rng)
}
