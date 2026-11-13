//! Generalized linear models with Stan's signatures.
//!
//! Each GLM forms the linear predictor `eta = x beta + alpha` with
//! [`multiply_vector`](stanmath_rs_linalg::multiply_vector) (one structured reverse-mode node) and
//! passes it to the fused vectorized log density of the response distribution, so the likelihood
//! adds a single further node. The design matrix `x` is usually data (`Matrix<f64>`); `alpha` is a
//! scalar or a vector with one entry per row. As in Stan, the result's scalar type is the promotion
//! of every argument's, so data `beta` with an autodiff `x` or `alpha` works.

use std::borrow::Cow;

use stanmath_rs_autodiff::functions::log_sum_exp_slice;
use stanmath_rs_autodiff::{Arg, Broadcast, IntArg, Lift, Real, broadcast_len};
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::{multiply, multiply_vector};

use crate::check::param;
use crate::continuous::Normal;
use crate::discrete::{BernoulliLogit, BinomialLogit, NegBinomial2LogGlm, PoissonLog};
use crate::discrete_extra::{check_cutpoints, ordered_logistic_lpmf};
use crate::kernel::sum_kernel_masked;

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
fn x_beta<T, X, B>(f: &'static str, x: &Matrix<X>, beta: &[B]) -> Result<Vec<T>>
where
    T: Real,
    X: Lift<T> + Copy,
    B: Lift<T> + Copy,
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
    let beta: Vec<T> = beta.iter().map(|&b| b.lift()).collect();
    check_finite(f, "x", xt.as_slice())?;
    check_finite(f, "beta", &beta)?;
    multiply_vector(&xt, &beta)
}

/// `x beta + alpha`.
fn linear_predictor<T, X, A, B>(
    f: &'static str,
    x: &Matrix<X>,
    alpha: &A,
    beta: &[B],
) -> Result<Vec<T>>
where
    T: Real,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    B: Lift<T> + Copy,
{
    let alpha: Vec<T> = broadcast_arg(f, "alpha", alpha, x.nrows())?;
    check_finite(f, "alpha", &alpha)?;
    let xb = x_beta(f, x, beta)?;
    Ok(xb.into_iter().zip(alpha).map(|(a, b)| a + b).collect())
}

/// Whether the linear predictor `x beta + alpha` depends only on data.
fn eta_data<T, X, A, B>() -> bool
where
    T: Real,
    X: Lift<T>,
    A: Arg,
    A::Elem: Lift<T>,
    B: Lift<T>,
{
    <X as Lift<T>>::DATA && <A::Elem as Lift<T>>::DATA && <B as Lift<T>>::DATA
}

/// Stan broadcasts a one row `x` (a `row_vector` argument) to every observation of a longer
/// response (checked against Stan Math 5.4.0 for every GLM).
fn broadcast_rows<'a, X: Copy>(
    x: &'a Matrix<X>,
    responses: &[Option<usize>],
) -> Cow<'a, Matrix<X>> {
    match responses.iter().flatten().max() {
        Some(&n) if x.nrows() == 1 && n > 1 => {
            let cols = (0..x.ncols()).flat_map(|c| std::iter::repeat_n(x[(0, c)], n));
            Cow::Owned(Matrix::from_column_major(n, x.ncols(), cols.collect()))
        }
        _ => Cow::Borrowed(x),
    }
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

/// [`poisson_log_glm_lpmf`] and, with `propto`, [`poisson_log_glm_lupmf`].
fn poisson_log_glm_lpmf_with<N, X, A, B, T>(
    y: N,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    propto: bool,
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    const F: &str = "poisson_log_glm_lpmf";
    let x = &*broadcast_rows(x, &[y.int_len()]);
    check_rows(F, y.int_len(), x.nrows())?;
    if x.nrows() == 0 {
        // Stan returns zero before checking alpha and beta (checked against Stan Math 5.4.0).
        return Ok(T::zero());
    }
    let eta = linear_predictor(F, x, &alpha, beta)?;
    let data = [eta_data::<T, X, A, B>()];
    sum_kernel_masked::<PoissonLog, _, 1, 1>([&y], (&eta[..],), propto, data)
}

/// Poisson regression with log link: `poisson_log_lpmf(y | x beta + alpha)`.
pub fn poisson_log_glm_lpmf<N, X, A, B, T>(y: N, x: &Matrix<X>, alpha: A, beta: &[B]) -> Result<T>
where
    (X, A, B): Broadcast<3, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    poisson_log_glm_lpmf_with(y, x, alpha, beta, false)
}

/// [`poisson_log_glm_lpmf`] without the terms that depend only on data (Stan's `poisson_log_glm_lupmf`).
pub fn poisson_log_glm_lupmf<N, X, A, B, T>(y: N, x: &Matrix<X>, alpha: A, beta: &[B]) -> Result<T>
where
    (X, A, B): Broadcast<3, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    poisson_log_glm_lpmf_with(y, x, alpha, beta, true)
}

/// [`bernoulli_logit_glm_lpmf`] and, with `propto`, [`bernoulli_logit_glm_lupmf`].
fn bernoulli_logit_glm_lpmf_with<N, X, A, B, T>(
    y: N,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    propto: bool,
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    const F: &str = "bernoulli_logit_glm_lpmf";
    let x = &*broadcast_rows(x, &[y.int_len()]);
    check_rows(F, y.int_len(), x.nrows())?;
    if x.nrows() == 0 {
        // Stan returns zero before checking alpha and beta (checked against Stan Math 5.4.0).
        return Ok(T::zero());
    }
    let eta = linear_predictor(F, x, &alpha, beta)?;
    let data = [eta_data::<T, X, A, B>()];
    sum_kernel_masked::<BernoulliLogit, _, 1, 1>([&y], (&eta[..],), propto, data)
}

/// Logistic regression: `bernoulli_logit_lpmf(y | x beta + alpha)`.
pub fn bernoulli_logit_glm_lpmf<N, X, A, B, T>(
    y: N,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
) -> Result<T>
where
    (X, A, B): Broadcast<3, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    bernoulli_logit_glm_lpmf_with(y, x, alpha, beta, false)
}

/// [`bernoulli_logit_glm_lpmf`] without the terms that depend only on data (Stan's `bernoulli_logit_glm_lupmf`).
pub fn bernoulli_logit_glm_lupmf<N, X, A, B, T>(
    y: N,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
) -> Result<T>
where
    (X, A, B): Broadcast<3, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    bernoulli_logit_glm_lpmf_with(y, x, alpha, beta, true)
}

/// [`binomial_logit_glm_lpmf`] and, with `propto`, [`binomial_logit_glm_lupmf`].
fn binomial_logit_glm_lpmf_with<N, M, X, A, B, T>(
    n: N,
    trials: M,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    propto: bool,
) -> Result<T>
where
    N: IntArg,
    M: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    const F: &str = "binomial_logit_glm_lpmf";
    let x = &*broadcast_rows(x, &[n.int_len(), trials.int_len()]);
    check_rows(F, n.int_len(), x.nrows())?;
    check_rows(F, trials.int_len(), x.nrows())?;
    let eta = linear_predictor(F, x, &alpha, beta)?;
    let data = [eta_data::<T, X, A, B>()];
    sum_kernel_masked::<BinomialLogit, _, 2, 1>([&n, &trials], (&eta[..],), propto, data)
}

/// Binomial regression with logit link: `binomial_logit_lpmf(n | N, x beta + alpha)`.
pub fn binomial_logit_glm_lpmf<N, M, X, A, B, T>(
    n: N,
    trials: M,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
) -> Result<T>
where
    (X, A, B): Broadcast<3, Out = T>,
    N: IntArg,
    M: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    binomial_logit_glm_lpmf_with(n, trials, x, alpha, beta, false)
}

/// [`binomial_logit_glm_lpmf`] without the terms that depend only on data (Stan's `binomial_logit_glm_lupmf`).
pub fn binomial_logit_glm_lupmf<N, M, X, A, B, T>(
    n: N,
    trials: M,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
) -> Result<T>
where
    (X, A, B): Broadcast<3, Out = T>,
    N: IntArg,
    M: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
{
    binomial_logit_glm_lpmf_with(n, trials, x, alpha, beta, true)
}

/// [`neg_binomial_2_log_glm_lpmf`] and, with `propto`, [`neg_binomial_2_log_glm_lupmf`].
fn neg_binomial_2_log_glm_lpmf_with<N, X, A, B, T, P>(
    y: N,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    phi: P,
    propto: bool,
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
    P: Arg,
    P::Elem: Lift<T>,
{
    const F: &str = "neg_binomial_2_log_glm_lpmf";
    let x = &*broadcast_rows(x, &[y.int_len()]);
    check_rows(F, y.int_len(), x.nrows())?;
    // Stan checks a scalar alpha even with no rows, but not phi (checked against Stan Math 5.4.0).
    if let Some(a) = alpha.arg_scalar() {
        param::finite(F, "alpha", Lift::<T>::lift(a).value())?;
    }
    let eta = linear_predictor(F, x, &alpha, beta)?;
    let phi: Vec<T> = broadcast_arg(F, "phi", &phi, x.nrows())?;
    let data = [eta_data::<T, X, A, B>(), <P::Elem as Lift<T>>::DATA];
    sum_kernel_masked::<NegBinomial2LogGlm, _, 1, 2>([&y], (&eta[..], &phi[..]), propto, data)
}

/// Negative binomial regression with log link:
/// `neg_binomial_2_log_lpmf(y | x beta + alpha, phi)`; `phi` is a scalar or one value per row.
pub fn neg_binomial_2_log_glm_lpmf<N, X, A, B, T, P>(
    y: N,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    phi: P,
) -> Result<T>
where
    (X, A, B, P): Broadcast<4, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
    P: Arg,
    P::Elem: Lift<T>,
{
    neg_binomial_2_log_glm_lpmf_with(y, x, alpha, beta, phi, false)
}

/// [`neg_binomial_2_log_glm_lpmf`] without the terms that depend only on data (Stan's `neg_binomial_2_log_glm_lupmf`).
pub fn neg_binomial_2_log_glm_lupmf<N, X, A, B, T, P>(
    y: N,
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    phi: P,
) -> Result<T>
where
    (X, A, B, P): Broadcast<4, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
    P: Arg,
    P::Elem: Lift<T>,
{
    neg_binomial_2_log_glm_lpmf_with(y, x, alpha, beta, phi, true)
}

/// [`normal_id_glm_lpdf`] and, with `propto`, [`normal_id_glm_lupdf`].
fn normal_id_glm_lpdf_with<Y, X, A, B, T, S>(
    y: &[Y],
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    sigma: S,
    propto: bool,
) -> Result<T>
where
    Y: Lift<T> + Copy,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
    S: Arg,
    S::Elem: Lift<T>,
{
    const F: &str = "normal_id_glm_lpdf";
    let x = &*broadcast_rows(x, &[Some(y.len())]);
    // A real response (a one element slice) is every observation's (checked against Stan Math
    // 5.4.0).
    let y = if y.len() == 1 {
        &vec![y[0]; x.nrows()][..]
    } else {
        y
    };
    check_rows(F, Some(y.len()), x.nrows())?;
    // Stan checks a scalar sigma even with no rows (checked against Stan Math 5.4.0).
    if let Some(s) = sigma.arg_scalar() {
        param::positive_finite(F, "sigma", Lift::<T>::lift(s).value())?;
    }
    let mu = linear_predictor(F, x, &alpha, beta)?;
    let y: Vec<T> = y.iter().map(|&v| v.lift()).collect();
    let sigma: Vec<T> = broadcast_arg(F, "sigma", &sigma, x.nrows())?;
    let data = [
        <Y as Lift<T>>::DATA,
        eta_data::<T, X, A, B>(),
        <S::Elem as Lift<T>>::DATA,
    ];
    sum_kernel_masked::<Normal, _, 0, 3>([], (&y[..], &mu[..], &sigma[..]), propto, data)
}

/// Linear regression: `normal_lpdf(y | x beta + alpha, sigma)`; `sigma` is a scalar or one value
/// per row.
pub fn normal_id_glm_lpdf<Y, X, A, B, T, S>(
    y: &[Y],
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    sigma: S,
) -> Result<T>
where
    (Y, X, A, B, S): Broadcast<5, Out = T>,
    Y: Lift<T> + Copy,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
    S: Arg,
    S::Elem: Lift<T>,
{
    normal_id_glm_lpdf_with(y, x, alpha, beta, sigma, false)
}

/// [`normal_id_glm_lpdf`] without the terms that depend only on data (Stan's `normal_id_glm_lupdf`).
pub fn normal_id_glm_lupdf<Y, X, A, B, T, S>(
    y: &[Y],
    x: &Matrix<X>,
    alpha: A,
    beta: &[B],
    sigma: S,
) -> Result<T>
where
    (Y, X, A, B, S): Broadcast<5, Out = T>,
    Y: Lift<T> + Copy,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    B: Lift<T> + Copy,
    S: Arg,
    S::Elem: Lift<T>,
{
    normal_id_glm_lpdf_with(y, x, alpha, beta, sigma, true)
}

/// [`categorical_logit_glm_lpmf`] and, with `propto`, [`categorical_logit_glm_lupmf`].
fn categorical_logit_glm_lpmf_with<N, X, A, B, T>(
    y: N,
    x: &Matrix<X>,
    alpha: &[A],
    beta: &Matrix<B>,
    propto: bool,
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Lift<T> + Copy,
    T: Real,
    B: Lift<T> + Copy,
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
    let x = &*broadcast_rows(x, &[y.int_len()]);
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
    let beta: Matrix<T> = lift_matrix(beta);
    let xt = lift_matrix(x);
    check_finite(F, "x", xt.as_slice())?;
    check_finite(F, "alpha", &alpha)?;
    check_finite(F, "beta", beta.as_slice())?;
    if propto && <X as Lift<T>>::DATA && <A as Lift<T>>::DATA && <B as Lift<T>>::DATA {
        return Ok(T::zero());
    }
    let eta = multiply(&xt, &beta)?;
    let mut lp = T::zero();
    for i in 0..n {
        let row: Vec<T> = (0..c).map(|j| eta[(i, j)] + alpha[j]).collect();
        let k = y.int_at(i) as usize - 1;
        lp += row[k] - log_sum_exp_slice(&row);
    }
    Ok(lp)
}

/// Multi-logit regression: category `y_i` in `1..=C` has log probability
/// `log_softmax(x_i beta + alpha)[y_i]`, with `beta` of size `K` by `C` and `alpha` of length `C`.
pub fn categorical_logit_glm_lpmf<N, X, A, B, T>(
    y: N,
    x: &Matrix<X>,
    alpha: &[A],
    beta: &Matrix<B>,
) -> Result<T>
where
    (X, A, B): Broadcast<3, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    A: Lift<T> + Copy,
    T: Real,
    B: Lift<T> + Copy,
{
    categorical_logit_glm_lpmf_with(y, x, alpha, beta, false)
}

/// [`categorical_logit_glm_lpmf`] without the terms that depend only on data (Stan's `categorical_logit_glm_lupmf`).
pub fn categorical_logit_glm_lupmf<N, X, A, B, T>(
    y: N,
    x: &Matrix<X>,
    alpha: &[A],
    beta: &Matrix<B>,
) -> Result<T>
where
    (X, A, B): Broadcast<3, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    A: Lift<T> + Copy,
    T: Real,
    B: Lift<T> + Copy,
{
    categorical_logit_glm_lpmf_with(y, x, alpha, beta, true)
}

/// [`ordered_logistic_glm_lpmf`] and, with `propto`, [`ordered_logistic_glm_lupmf`].
fn ordered_logistic_glm_lpmf_with<N, X, B, T, C>(
    y: N,
    x: &Matrix<X>,
    beta: &[B],
    cuts: &[C],
    propto: bool,
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    T: Real,
    B: Lift<T> + Copy,
    C: Lift<T> + Copy,
{
    const F: &str = "ordered_logistic_glm_lpmf";
    let x = &*broadcast_rows(x, &[y.int_len()]);
    check_rows(F, y.int_len(), x.nrows())?;
    let eta = x_beta(F, x, beta)?;
    let cuts: Vec<T> = cuts.iter().map(|&c| c.lift()).collect();
    // Unlike ordered_logistic_lpmf, Stan checks the cutpoints even with no rows (checked against
    // Stan Math 5.4.0).
    check_cutpoints(F, &cuts)?;
    let lp = ordered_logistic_lpmf(y, &eta[..], &cuts)?;
    // No term of the ordered logistic depends only on data, so propto drops all or nothing.
    if propto && <X as Lift<T>>::DATA && <B as Lift<T>>::DATA && <C as Lift<T>>::DATA {
        return Ok(T::zero());
    }
    Ok(lp)
}

/// Ordered logistic regression: `ordered_logistic_lpmf(y | x beta, cuts)`.
pub fn ordered_logistic_glm_lpmf<N, X, B, T, C>(
    y: N,
    x: &Matrix<X>,
    beta: &[B],
    cuts: &[C],
) -> Result<T>
where
    (X, B, C): Broadcast<3, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    T: Real,
    B: Lift<T> + Copy,
    C: Lift<T> + Copy,
{
    ordered_logistic_glm_lpmf_with(y, x, beta, cuts, false)
}

/// [`ordered_logistic_glm_lpmf`] without the terms that depend only on data (Stan's `ordered_logistic_glm_lupmf`).
pub fn ordered_logistic_glm_lupmf<N, X, B, T, C>(
    y: N,
    x: &Matrix<X>,
    beta: &[B],
    cuts: &[C],
) -> Result<T>
where
    (X, B, C): Broadcast<3, Out = T>,
    N: IntArg,
    X: Lift<T> + Copy,
    T: Real,
    B: Lift<T> + Copy,
    C: Lift<T> + Copy,
{
    ordered_logistic_glm_lpmf_with(y, x, beta, cuts, true)
}
