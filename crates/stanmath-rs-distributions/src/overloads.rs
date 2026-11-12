//! Stan distribution signatures whose argument shapes the base functions cannot take: arrays of
//! vectors in the multivariate densities, one cutpoint vector per observation in the ordered
//! models, a single predictor row in the GLMs, a diagonal observation covariance in the Gaussian
//! dynamic linear model, and the Wiener arities that add a `precision` argument.
//!
//! Each function reshapes its arguments and calls the base kernel, so values, checks, and
//! derivatives come from one place. Array forms return the sum of the per observation log
//! densities, and zero for empty arrays.
//!
//! Naming follows [`multi_normal_lpdf_array`](crate::multi_normal_lpdf_array): the Stan name, then
//! a suffix for the shape. `_array` takes an array of observations with the location (or
//! concentration) given once for all of them or once per observation; `_mu_array` and
//! `_alpha_array` take one observation vector and an array of locations or concentrations;
//! `_row` takes a single predictor row shared by every observation.

use stanmath_rs_autodiff::{Arg, IntArg, Lift, Real, broadcast_len};
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::cholesky_decompose;

use crate::discrete_extra::{ordered_logistic_lpmf, ordered_probit_lpmf};
use crate::glm::{
    bernoulli_logit_glm_lpmf, binomial_logit_glm_lpmf, categorical_logit_glm_lpmf,
    neg_binomial_2_log_glm_lpmf, normal_id_glm_lpdf, ordered_logistic_glm_lpmf,
    poisson_log_glm_lpmf,
};
use crate::multivariate::dirichlet_lpdf;
use crate::multivariate_extra::{
    gaussian_dlm_obs_lpdf, multi_normal_cholesky_lpdf_array, multi_normal_lpdf_array,
    multi_normal_prec_lpdf, multi_student_t_cholesky_lpdf,
};
use crate::wiener_full::{
    WienerArgs, WienerOut, wiener_full_prec_lccdf_unnorm, wiener_full_prec_lcdf_unnorm,
    wiener_full_prec_lpdf,
};

// Helpers.

/// Sum of `term(y_i, p_i)` over the observations `y`, where `p` holds one entry shared by every
/// observation or one entry per observation.
fn sum_terms<Y, P, T: Real>(
    function: &'static str,
    y: &[Y],
    p: &[P],
    mut term: impl FnMut(&Y, &P) -> Result<T>,
) -> Result<T> {
    if !(p.len() == 1 || p.len() == y.len()) {
        return Err(Error::dimension(
            function,
            format!("{} parameter vectors for {} observations", p.len(), y.len()),
        ));
    }
    let mut lp = T::zero();
    for (i, yi) in y.iter().enumerate() {
        lp += term(yi, &p[if p.len() == 1 { 0 } else { i }])?;
    }
    Ok(lp)
}

/// `n` copies of the vector `y`.
fn repeat_vec<Y: Copy>(y: &[Y], n: usize) -> Vec<Vec<Y>> {
    vec![y.to_vec(); n]
}

/// The `n` by `x.len()` matrix whose rows all equal `x`.
pub(crate) fn repeat_row<X: Copy>(x: &[X], n: usize) -> Matrix<X> {
    Matrix::from_fn(n, x.len(), |_, j| x[j])
}

/// Sum of an ordered model over observations that each carry their own cutpoint vector.
fn ordered_per_cutpoints<E, T>(
    function: &'static str,
    y: &[i64],
    eta: E,
    c: &[Vec<T>],
    term: impl Fn(i64, T, &[T]) -> Result<T>,
) -> Result<T>
where
    E: Arg,
    E::Elem: Lift<T>,
    T: Real,
{
    let n = broadcast_len(function, &[Some(y.len()), eta.arg_len(), Some(c.len())])?;
    let mut lp = T::zero();
    for i in 0..n {
        let e: T = eta.arg_at(i).lift();
        lp += term(y[i], e, &c[i])?;
    }
    Ok(lp)
}

// Multivariate normal family.

/// Stan's `multi_normal_lpdf(vector y | array[] vector mu, matrix Sigma)`: the sum over the
/// locations `mu` of the multivariate normal log density of the single observation `y`.
pub fn multi_normal_lpdf_mu_array<T, Y, M>(y: &[Y], mu: &[Vec<M>], sigma: &Matrix<T>) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    M: Lift<T> + Copy,
{
    multi_normal_lpdf_array(&repeat_vec(y, mu.len()), mu, sigma)
}

/// Stan's `multi_normal_cholesky_lpdf(vector y | array[] vector mu, matrix L)`: the sum over the
/// locations `mu` of the log density of the single observation `y`.
pub fn multi_normal_cholesky_lpdf_mu_array<T, Y, M>(
    y: &[Y],
    mu: &[Vec<M>],
    l: &Matrix<T>,
) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    M: Lift<T> + Copy,
{
    multi_normal_cholesky_lpdf_array(&repeat_vec(y, mu.len()), mu, l)
}

/// Sum of multivariate normal log densities with precision matrix `omega` over an array of
/// observations `y`. Stan's `multi_normal_prec_lpdf` with `array[] vector y`.
///
/// `mu` holds either one location shared by every observation or one location per observation.
pub fn multi_normal_prec_lpdf_array<T, Y, M>(
    y: &[Vec<Y>],
    mu: &[Vec<M>],
    omega: &Matrix<T>,
) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    M: Lift<T> + Copy,
{
    // ponytail: the log determinant of omega is recomputed per observation; factor it once if
    // long arrays become a bottleneck.
    sum_terms("multi_normal_prec_lpdf", y, mu, |yi, mi| {
        multi_normal_prec_lpdf(&yi[..], &mi[..], omega)
    })
}

/// Stan's `multi_normal_prec_lpdf(vector y | array[] vector mu, matrix Omega)`: the sum over the
/// locations `mu` of the log density of the single observation `y`.
pub fn multi_normal_prec_lpdf_mu_array<T, Y, M>(
    y: &[Y],
    mu: &[Vec<M>],
    omega: &Matrix<T>,
) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    M: Lift<T> + Copy,
{
    multi_normal_prec_lpdf_array(&repeat_vec(y, mu.len()), mu, omega)
}

/// Sum of multivariate Student t log densities over an array of observations `y`, with the scale
/// matrix `sigma` factored once. Stan's `multi_student_t_lpdf` with `array[] vector y`.
///
/// `mu` holds either one location shared by every observation or one location per observation.
pub fn multi_student_t_lpdf_array<T, Y, N, M>(
    y: &[Vec<Y>],
    nu: N,
    mu: &[Vec<M>],
    sigma: &Matrix<T>,
) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    N: Lift<T>,
    M: Lift<T> + Copy,
{
    let l = cholesky_decompose(sigma)?;
    let nu: T = nu.lift();
    sum_terms("multi_student_t_lpdf", y, mu, |yi, mi| {
        multi_student_t_cholesky_lpdf(&yi[..], nu, &mi[..], &l)
    })
}

/// Stan's `multi_student_t_lpdf(vector y | real nu, array[] vector mu, matrix Sigma)`: the sum
/// over the locations `mu` of the log density of the single observation `y`.
pub fn multi_student_t_lpdf_mu_array<T, Y, N, M>(
    y: &[Y],
    nu: N,
    mu: &[Vec<M>],
    sigma: &Matrix<T>,
) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    N: Lift<T>,
    M: Lift<T> + Copy,
{
    multi_student_t_lpdf_array(&repeat_vec(y, mu.len()), nu, mu, sigma)
}

/// Sum of multivariate Student t log densities over an array of observations `y`, with the scale
/// matrix given by its lower Cholesky factor `l`. Stan's `multi_student_t_cholesky_lpdf` with
/// `array[] vector y`.
///
/// `mu` holds either one location shared by every observation or one location per observation.
pub fn multi_student_t_cholesky_lpdf_array<T, Y, N, M>(
    y: &[Vec<Y>],
    nu: N,
    mu: &[Vec<M>],
    l: &Matrix<T>,
) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    N: Lift<T>,
    M: Lift<T> + Copy,
{
    let nu: T = nu.lift();
    sum_terms("multi_student_t_cholesky_lpdf", y, mu, |yi, mi| {
        multi_student_t_cholesky_lpdf(&yi[..], nu, &mi[..], l)
    })
}

/// Stan's `multi_student_t_cholesky_lpdf(vector y | real nu, array[] vector mu, matrix L)`: the
/// sum over the locations `mu` of the log density of the single observation `y`.
pub fn multi_student_t_cholesky_lpdf_mu_array<T, Y, N, M>(
    y: &[Y],
    nu: N,
    mu: &[Vec<M>],
    l: &Matrix<T>,
) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    N: Lift<T>,
    M: Lift<T> + Copy,
{
    multi_student_t_cholesky_lpdf_array(&repeat_vec(y, mu.len()), nu, mu, l)
}

// Dirichlet.

/// Sum of Dirichlet log densities over an array of simplexes `theta`. Stan's `dirichlet_lpdf`
/// with `array[] vector theta`.
///
/// `alpha` holds either one concentration vector shared by every simplex or one per simplex.
pub fn dirichlet_lpdf_array<T, A>(theta: &[Vec<T>], alpha: &[Vec<A>]) -> Result<T>
where
    T: Real,
    A: Lift<T> + Copy,
{
    sum_terms("dirichlet_lpdf", theta, alpha, |t, a| {
        dirichlet_lpdf(&t[..], &a[..])
    })
}

/// Stan's `dirichlet_lpdf(vector theta | array[] vector alpha)`: the sum over the concentration
/// vectors `alpha` of the Dirichlet log density of the single simplex `theta`.
pub fn dirichlet_lpdf_alpha_array<T, A>(theta: &[T], alpha: &[Vec<A>]) -> Result<T>
where
    T: Real,
    A: Lift<T> + Copy,
{
    dirichlet_lpdf_array(&repeat_vec(theta, alpha.len()), alpha)
}

// Ordered models with one cutpoint vector per observation.

/// Stan's `ordered_logistic_lpmf(array[] int y | vector eta, array[] vector c)`: observation `i`
/// uses the cutpoints `c[i]`. `eta` is a scalar or one value per observation, and `c` has one
/// vector per observation.
pub fn ordered_logistic_lpmf_array<E, T>(y: &[i64], eta: E, c: &[Vec<T>]) -> Result<T>
where
    E: Arg,
    E::Elem: Lift<T>,
    T: Real,
{
    ordered_per_cutpoints("ordered_logistic_lpmf", y, eta, c, |yi, e, ci| {
        ordered_logistic_lpmf(yi, e, ci)
    })
}

/// Stan's `ordered_probit_lpmf(array[] int y | real eta, array[] vector c)` and its `vector eta`
/// form: observation `i` uses the cutpoints `c[i]`. `eta` is a scalar or one value per
/// observation, and `c` has one vector per observation.
pub fn ordered_probit_lpmf_array<E, T>(y: &[i64], eta: E, c: &[Vec<T>]) -> Result<T>
where
    E: Arg,
    E::Elem: Lift<T>,
    T: Real,
{
    ordered_per_cutpoints("ordered_probit_lpmf", y, eta, c, |yi, e, ci| {
        ordered_probit_lpmf(yi, e, ci)
    })
}

// GLMs with a single predictor row shared by every observation.

/// [`poisson_log_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`. The
/// number of observations is the length of `y` or of `alpha`.
pub fn poisson_log_glm_lpmf_row<N, X, A, T>(y: N, x: &[X], alpha: A, beta: &[T]) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
{
    let n = broadcast_len("poisson_log_glm_lpmf", &[y.int_len(), alpha.arg_len()])?;
    poisson_log_glm_lpmf(y, &repeat_row(x, n), alpha, beta)
}

/// [`bernoulli_logit_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`.
/// The number of observations is the length of `y` or of `alpha`.
pub fn bernoulli_logit_glm_lpmf_row<N, X, A, T>(y: N, x: &[X], alpha: A, beta: &[T]) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
{
    let n = broadcast_len("bernoulli_logit_glm_lpmf", &[y.int_len(), alpha.arg_len()])?;
    bernoulli_logit_glm_lpmf(y, &repeat_row(x, n), alpha, beta)
}

/// [`binomial_logit_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`.
/// The number of observations is the common length of `n`, `trials`, and `alpha`.
pub fn binomial_logit_glm_lpmf_row<N, M, X, A, T>(
    n: N,
    trials: M,
    x: &[X],
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
    let rows = broadcast_len(
        "binomial_logit_glm_lpmf",
        &[n.int_len(), trials.int_len(), alpha.arg_len()],
    )?;
    binomial_logit_glm_lpmf(n, trials, &repeat_row(x, rows), alpha, beta)
}

/// [`neg_binomial_2_log_glm_lpmf`] with `row_vector x`: every observation uses the predictors
/// `x`. The number of observations is the common length of `y`, `alpha`, and `phi`.
pub fn neg_binomial_2_log_glm_lpmf_row<N, X, A, T, P>(
    y: N,
    x: &[X],
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
    let n = broadcast_len(
        "neg_binomial_2_log_glm_lpmf",
        &[y.int_len(), alpha.arg_len(), phi.arg_len()],
    )?;
    neg_binomial_2_log_glm_lpmf(y, &repeat_row(x, n), alpha, beta, phi)
}

/// [`normal_id_glm_lpdf`] with `real y`: the scalar response is used for every row of `x`.
pub fn normal_id_glm_lpdf_scalar_y<Y, X, A, T, S>(
    y: Y,
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
    normal_id_glm_lpdf(&vec![y; x.nrows()][..], x, alpha, beta, sigma)
}

/// [`normal_id_glm_lpdf`] with `row_vector x`: every observation uses the predictors `x`. `y` is a
/// scalar or a vector; the number of observations is the common length of `y`, `alpha`, and
/// `sigma`.
pub fn normal_id_glm_lpdf_row<Y, X, A, T, S>(
    y: Y,
    x: &[X],
    alpha: A,
    beta: &[T],
    sigma: S,
) -> Result<T>
where
    Y: Arg,
    Y::Elem: Lift<T>,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    S: Arg,
    S::Elem: Lift<T>,
{
    let n = broadcast_len(
        "normal_id_glm_lpdf",
        &[y.arg_len(), alpha.arg_len(), sigma.arg_len()],
    )?;
    let ys: Vec<T> = (0..n)
        .map(|i| {
            let v: T = y.arg_at(i).lift();
            v
        })
        .collect();
    normal_id_glm_lpdf(&ys[..], &repeat_row(x, n), alpha, beta, sigma)
}

/// [`normal_id_glm_lpdf`] with `vector x`, new in Stan 2.40. `x` is read as an `N` by 1 design
/// matrix, one predictor value per observation, so `beta` has one element; `y` is a scalar or a
/// vector of length `N`. This reading is **unconfirmed**: the Functions Reference only says
/// "location alpha + x * beta".
///
/// # Errors
///
/// As [`normal_id_glm_lpdf`], and [`Error::DimensionMismatch`] when a vector `y` does not have
/// one element per entry of `x`.
pub fn normal_id_glm_lpdf_vector_x<Y, X, A, T, S>(
    y: Y,
    x: &[X],
    alpha: A,
    beta: &[T],
    sigma: S,
) -> Result<T>
where
    Y: Arg,
    Y::Elem: Lift<T>,
    X: Lift<T> + Copy,
    A: Arg,
    A::Elem: Lift<T>,
    T: Real,
    S: Arg,
    S::Elem: Lift<T>,
{
    let n = broadcast_len("normal_id_glm_lpdf", &[y.arg_len(), Some(x.len())])?;
    let ys: Vec<T> = (0..n)
        .map(|i| {
            let v: T = y.arg_at(i).lift();
            v
        })
        .collect();
    let design = Matrix::from_fn(n, 1, |i, _| x[i]);
    normal_id_glm_lpdf(&ys[..], &design, alpha, beta, sigma)
}

/// [`categorical_logit_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`.
/// The number of observations is the length of `y`, or one for a scalar `y`.
pub fn categorical_logit_glm_lpmf_row<N, X, A, T>(
    y: N,
    x: &[X],
    alpha: &[A],
    beta: &Matrix<T>,
) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    A: Lift<T> + Copy,
    T: Real,
{
    let n = y.int_len().unwrap_or(1);
    categorical_logit_glm_lpmf(y, &repeat_row(x, n), alpha, beta)
}

/// [`ordered_logistic_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`.
/// The number of observations is the length of `y`, or one for a scalar `y`.
pub fn ordered_logistic_glm_lpmf_row<N, X, T, C>(y: N, x: &[X], beta: &[T], cuts: &[C]) -> Result<T>
where
    N: IntArg,
    X: Lift<T> + Copy,
    T: Real,
    C: Lift<T> + Copy,
{
    let n = y.int_len().unwrap_or(1);
    ordered_logistic_glm_lpmf(y, &repeat_row(x, n), beta, cuts)
}

// Gaussian dynamic linear model with a diagonal observation covariance.

/// [`gaussian_dlm_obs_lpdf`] with `vector V`: the observation covariance is the diagonal matrix
/// with diagonal `v`.
pub fn gaussian_dlm_obs_lpdf_vector_v<T: Real>(
    y: &Matrix<T>,
    f: &Matrix<T>,
    g: &Matrix<T>,
    v: &[T],
    w: &Matrix<T>,
    m0: &[T],
    c0: &Matrix<T>,
) -> Result<T> {
    let r = v.len();
    let v_diag = Matrix::from_fn(r, r, |i, j| if i == j { v[i] } else { T::zero() });
    gaussian_dlm_obs_lpdf(y, f, g, &v_diag, w, m0, c0)
}

// Wiener arities with an explicit precision.

/// Stan's seven argument `wiener_lpdf(y | a, t0, w, v, sv, precision)`: the five parameter
/// density [`wiener5_lpdf`](crate::wiener5_lpdf) with an explicit `precision`, as for
/// [`wiener_full_prec_lpdf`].
pub fn wiener5_prec_lpdf<Y, A, T0, W, V, SV>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    sv: SV,
    precision: f64,
) -> Result<WienerOut<(Y, A, T0, W, V, SV, f64, f64), 8>>
where
    (Y, A, T0, W, V, SV, f64, f64): WienerArgs<8>,
{
    wiener_full_prec_lpdf(y, a, t0, w, v, sv, 0.0, 0.0, precision)
}

/// Stan's six argument `wiener_lcdf_unnorm(y | a, t0, w, v, precision)`: the four parameter
/// unnormalized log CDF [`wiener4_lcdf_unnorm`](crate::wiener4_lcdf_unnorm) with an explicit
/// `precision`, as for [`wiener_full_prec_lpdf`].
pub fn wiener4_prec_lcdf_unnorm<Y, A, T0, W, V>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    precision: f64,
) -> Result<WienerOut<(Y, A, T0, W, V, f64, f64, f64), 8>>
where
    (Y, A, T0, W, V, f64, f64, f64): WienerArgs<8>,
{
    wiener_full_prec_lcdf_unnorm(y, a, t0, w, v, 0.0, 0.0, 0.0, precision)
}

/// Stan's six argument `wiener_lccdf_unnorm(y | a, t0, w, v, precision)`: the four parameter
/// unnormalized log CCDF [`wiener4_lccdf_unnorm`](crate::wiener4_lccdf_unnorm) with an explicit
/// `precision`, as for [`wiener_full_prec_lpdf`].
pub fn wiener4_prec_lccdf_unnorm<Y, A, T0, W, V>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    precision: f64,
) -> Result<WienerOut<(Y, A, T0, W, V, f64, f64, f64), 8>>
where
    (Y, A, T0, W, V, f64, f64, f64): WienerArgs<8>,
{
    wiener_full_prec_lccdf_unnorm(y, a, t0, w, v, 0.0, 0.0, 0.0, precision)
}
