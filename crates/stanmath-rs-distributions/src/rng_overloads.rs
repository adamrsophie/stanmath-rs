//! Random number generators for Stan signatures whose argument shapes the base generators cannot
//! take: an array of locations in the multivariate normal and Student t families, and a single
//! predictor row in `bernoulli_logit_glm_rng`.
//!
//! Each function reshapes its arguments and calls the base generator. The array forms return one
//! independent draw per location, in order.

use rand::Rng;
use stanmath_rs_autodiff::Arg;
use stanmath_rs_core::{Matrix, Result};
use stanmath_rs_linalg::cholesky_decompose;

use crate::overloads::repeat_row;
use crate::rng_multivariate::{multi_normal_cholesky_rng, multi_student_t_cholesky_rng};
use crate::rng_wave2::bernoulli_logit_glm_rng;

/// Stan's `multi_normal_rng(array[] vector mu, matrix Sigma)`: one draw per location, with `sigma`
/// factored once.
pub fn multi_normal_rng_array<R: Rng + ?Sized>(
    mu: &[Vec<f64>],
    sigma: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<Vec<f64>>> {
    let l = cholesky_decompose(sigma)?;
    mu.iter()
        .map(|m| multi_normal_cholesky_rng(m, &l, rng))
        .collect()
}

/// Stan's `multi_normal_cholesky_rng(array[] vector mu, matrix L)`: one draw per location.
pub fn multi_normal_cholesky_rng_array<R: Rng + ?Sized>(
    mu: &[Vec<f64>],
    l: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<Vec<f64>>> {
    mu.iter()
        .map(|m| multi_normal_cholesky_rng(m, l, rng))
        .collect()
}

/// Stan's `multi_student_t_rng(real nu, array[] vector mu, matrix Sigma)`: one draw per location,
/// with `sigma` factored once.
pub fn multi_student_t_rng_array<R: Rng + ?Sized>(
    nu: f64,
    mu: &[Vec<f64>],
    sigma: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<Vec<f64>>> {
    let l = cholesky_decompose(sigma)?;
    mu.iter()
        .map(|m| multi_student_t_cholesky_rng(nu, m, &l, rng))
        .collect()
}

/// Stan's `multi_student_t_cholesky_rng(real nu, array[] vector mu, matrix L)`: one draw per
/// location.
pub fn multi_student_t_cholesky_rng_array<R: Rng + ?Sized>(
    nu: f64,
    mu: &[Vec<f64>],
    l: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<Vec<f64>>> {
    mu.iter()
        .map(|m| multi_student_t_cholesky_rng(nu, m, l, rng))
        .collect()
}

/// [`bernoulli_logit_glm_rng`] with `row_vector x`: every draw uses the predictors `x`. The
/// number of draws is the length of `alpha`, or one for a scalar `alpha`.
pub fn bernoulli_logit_glm_rng_row<A, R>(
    x: &[f64],
    alpha: A,
    beta: &[f64],
    rng: &mut R,
) -> Result<Vec<i64>>
where
    A: Arg<Elem = f64>,
    R: Rng + ?Sized,
{
    let n = alpha.arg_len().unwrap_or(1);
    bernoulli_logit_glm_rng(&repeat_row(x, n), alpha, beta, rng)
}
