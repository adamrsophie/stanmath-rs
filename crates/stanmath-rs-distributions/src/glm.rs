//! Generalized linear models with Stan's signatures.
//!
//! Each GLM forms the linear predictor `eta = x beta + alpha` with
//! [`multiply_vector`](stanmath_rs_linalg::multiply_vector) (one structured reverse-mode node) and
//! passes it to the fused vectorized log density of the response distribution, so the likelihood
//! adds a single further node. The design matrix `x` is usually data (`Matrix<f64>`); `alpha` is a
//! scalar or a vector with one entry per row; `beta` fixes the scalar type.

use stanmath_rs_autodiff::functions::log_sum_exp_slice;
use stanmath_rs_autodiff::{Arg, IntArg, Lift, Real, broadcast_len};
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::{multiply, multiply_vector};

use crate::check::param;
use crate::continuous::normal_lpdf;
use crate::discrete::{
    bernoulli_logit_lpmf, binomial_logit_lpmf, neg_binomial_2_log_lpmf, poisson_log_lpmf,
};
use crate::discrete_extra::{check_cutpoints, ordered_logistic_lpmf};

fn check_finite<T: Real>(f: &'static str, name: &'static str, xs: &[T]) -> Result<()> {
    match xs.iter().find(|v| !v.is_finite()) {
        Some(v) => Err(Error::parameter(f, name, v.value(), "finite")),
        None => Ok(()),
    }
}

fn lift_matrix<T: Real, X: Lift<T> + Copy>(x: &Matrix<X>) -> Matrix<T> {
    x.map(|&v| v.lift())
}

/// A scalar or length `n` argument as a vector of length `n`.
fn broadcast_arg<T, A>(f: &'static str, name: &'static str, a: &A, n: usize) -> Result<Vec<T>>
where
    T: Real,
    A: Arg,
    A::Elem: Lift<T>,
{
    match a.arg_len() {
        None => {
            let v: T = a.arg_at(0).lift();
            Ok(vec![v; n])
        }
        Some(len) if len == n => Ok((0..n)
            .map(|i| {
                let v: T = a.arg_at(i).lift();
                v
            })
            .collect()),
        Some(len) => Err(Error::dimension(
            f,
            format!("{name} has length {len}, but x has {n} rows"),
        )),
    }
}

/// `x beta`, after checking sizes and finiteness.
fn x_beta<T, X>(f: &'static str, x: &Matrix<X>, beta: &[T]) -> Result<Vec<T>>
where
    T: Real,
    X: Lift<T> + Copy,
{
    if x.ncols() != beta.len() {
        return Err(Error::dimension(
            f,
            format!(
                "x has {} columns, but beta has length {}",
                x.ncols(),
                beta.len()
            ),
        ));
    }
    let xt = lift_matrix(x);
    check_finite(f, "x", xt.as_slice())?;
    check_finite(f, "beta", beta)?;
    multiply_vector(&xt, beta)
}

/// `x beta + alpha`.
fn linear_predictor<T, X, A>(
    f: &'static str,
    x: &Matrix<X>,
    alpha: &A,
    beta: &[T],
) -> Result<Vec<T>>
where
    T: Real,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
{
    let alpha: Vec<T> = broadcast_arg(f, "alpha", alpha, x.nrows())?;
    check_finite(f, "alpha", &alpha)?;
    let xb = x_beta(f, x, beta)?;
    Ok(xb.into_iter().zip(alpha).map(|(a, b)| a + b).collect())
}

/// Requires a sequence response to have one entry per row of `x`.
fn check_rows(f: &'static str, response: Option<usize>, n: usize) -> Result<()> {
    match response {
        Some(len) if len != n => Err(Error::dimension(
            f,
            format!("the response has length {len}, but x has {n} rows"),
        )),
        _ => Ok(()),
    }
}

/// Poisson regression with log link: `poisson_log_lpmf(y | x beta + alpha)`.
pub fn poisson_log_glm_lpmf<N, X, A, T>(y: N, x: &Matrix<X>, alpha: A, beta: &[T]) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
{
    const F: &str = "poisson_log_glm_lpmf";
    check_rows(F, y.int_len(), x.nrows())?;
    if x.nrows() == 0 {
        // Stan returns zero before checking alpha and beta (checked against Stan Math 5.4.0).
        return Ok(T::zero());
    }
    let eta = linear_predictor(F, x, &alpha, beta)?;
    poisson_log_lpmf(y, &eta[..])
}

/// Logistic regression: `bernoulli_logit_lpmf(y | x beta + alpha)`.
pub fn bernoulli_logit_glm_lpmf<N, X, A, T>(y: N, x: &Matrix<X>, alpha: A, beta: &[T]) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
{
    const F: &str = "bernoulli_logit_glm_lpmf";
    check_rows(F, y.int_len(), x.nrows())?;
    if x.nrows() == 0 {
        // Stan returns zero before checking alpha and beta (checked against Stan Math 5.4.0).
        return Ok(T::zero());
    }
    let eta = linear_predictor(F, x, &alpha, beta)?;
    bernoulli_logit_lpmf(y, &eta[..])
}

/// Binomial regression with logit link: `binomial_logit_lpmf(n | N, x beta + alpha)`.
pub fn binomial_logit_glm_lpmf<N, M, X, A, T>(
    n: N,
    trials: M,
    x: &Matrix<X>,
    alpha: A,
    beta: &[T],
) -> Result<T>
where
    N: IntArg,
    M: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
{
    const F: &str = "binomial_logit_glm_lpmf";
    check_rows(F, n.int_len(), x.nrows())?;
    check_rows(F, trials.int_len(), x.nrows())?;
    let eta = linear_predictor(F, x, &alpha, beta)?;
    binomial_logit_lpmf(n, trials, &eta[..])
}

/// Negative binomial regression with log link:
/// `neg_binomial_2_log_lpmf(y | x beta + alpha, phi)`; `phi` is a scalar or one value per row.
pub fn neg_binomial_2_log_glm_lpmf<N, X, A, T, P>(
    y: N,
    x: &Matrix<X>,
    alpha: A,
    beta: &[T],
    phi: P,
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    P: Arg,
    P::Elem: Lift<T>,
{
    const F: &str = "neg_binomial_2_log_glm_lpmf";
    check_rows(F, y.int_len(), x.nrows())?;
    // Stan checks a scalar alpha even with no rows, but not phi (checked against Stan Math 5.4.0).
    if let Some(a) = alpha.arg_scalar() {
        param::finite(F, "alpha", Lift::<T>::lift(a).value())?;
    }
    let eta = linear_predictor(F, x, &alpha, beta)?;
    let phi: Vec<T> = broadcast_arg(F, "phi", &phi, x.nrows())?;
    neg_binomial_2_log_lpmf(y, &eta[..], &phi[..])
}

/// Linear regression: `normal_lpdf(y | x beta + alpha, sigma)`; `sigma` is a scalar or one value
/// per row.
pub fn normal_id_glm_lpdf<Y, X, A, T, S>(
    y: &[Y],
    x: &Matrix<X>,
    alpha: A,
    beta: &[T],
    sigma: S,
) -> Result<T>
where
    Y: Lift<T> + Copy,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    S: Arg,
    S::Elem: Lift<T>,
{
    const F: &str = "normal_id_glm_lpdf";
    check_rows(F, Some(y.len()), x.nrows())?;
    // Stan checks a scalar sigma even with no rows (checked against Stan Math 5.4.0).
    if let Some(s) = sigma.arg_scalar() {
        param::positive_finite(F, "sigma", Lift::<T>::lift(s).value())?;
    }
    let mu = linear_predictor(F, x, &alpha, beta)?;
    let y: Vec<T> = y.iter().map(|&v| v.lift()).collect();
    let sigma: Vec<T> = broadcast_arg(F, "sigma", &sigma, x.nrows())?;
    normal_lpdf(&y[..], &mu[..], &sigma[..])
}

/// Multi-logit regression: category `y_i` in `1..=C` has log probability
/// `log_softmax(x_i beta + alpha)[y_i]`, with `beta` of size `K` by `C` and `alpha` of length `C`.
pub fn categorical_logit_glm_lpmf<N, X, A, T>(
    y: N,
    x: &Matrix<X>,
    alpha: &[A],
    beta: &Matrix<T>,
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Lift<T> + Copy,
    T: Real,
{
    const F: &str = "categorical_logit_glm_lpmf";
    if x.ncols() != beta.nrows() {
        return Err(Error::dimension(
            F,
            format!(
                "x has {} columns, but beta has {} rows",
                x.ncols(),
                beta.nrows()
            ),
        ));
    }
    let c = beta.ncols();
    if alpha.len() != c {
        return Err(Error::dimension(
            F,
            format!("alpha has length {}, but beta has {c} columns", alpha.len()),
        ));
    }
    let n = x.nrows();
    check_rows(F, y.int_len(), n)?;
    let len = broadcast_len(F, &[y.int_len(), Some(n)])?;
    for i in 0..len {
        let v = y.int_at(i);
        if v < 1 || v as usize > c {
            return Err(Error::observation(F, "y", v as f64, "in [1, C]"));
        }
    }
    if len == 0 {
        // Stan returns zero before checking x, alpha, and beta (checked against Stan Math 5.4.0).
        return Ok(T::zero());
    }
    let alpha: Vec<T> = alpha.iter().map(|&a| a.lift()).collect();
    let xt = lift_matrix(x);
    check_finite(F, "x", xt.as_slice())?;
    check_finite(F, "alpha", &alpha)?;
    check_finite(F, "beta", beta.as_slice())?;
    let eta = multiply(&xt, beta)?;
    let mut lp = T::zero();
    for i in 0..n {
        let row: Vec<T> = (0..c).map(|j| eta[(i, j)] + alpha[j]).collect();
        let k = y.int_at(i) as usize - 1;
        lp += row[k] - log_sum_exp_slice(&row);
    }
    Ok(lp)
}

/// Ordered logistic regression: `ordered_logistic_lpmf(y | x beta, cuts)`.
pub fn ordered_logistic_glm_lpmf<N, X, T, C>(
    y: N,
    x: &Matrix<X>,
    beta: &[T],
    cuts: &[C],
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    T: Real,
    C: Lift<T> + Copy,
{
    const F: &str = "ordered_logistic_glm_lpmf";
    check_rows(F, y.int_len(), x.nrows())?;
    let eta = x_beta(F, x, beta)?;
    let cuts: Vec<T> = cuts.iter().map(|&c| c.lift()).collect();
    // Unlike ordered_logistic_lpmf, Stan checks the cutpoints even with no rows (checked against
    // Stan Math 5.4.0).
    check_cutpoints(F, &cuts)?;
    ordered_logistic_lpmf(y, &eta[..], &cuts)
}
