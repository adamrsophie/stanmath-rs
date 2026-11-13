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

// The return types name one generic parameter per Wiener argument; aliases would only hide them.
#![allow(clippy::type_complexity)]

use stanmath_rs_autodiff::{Arg, IntArg, Lift, Real, broadcast_len};
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::cholesky_decompose;

use crate::discrete_extra::{
    ordered_logistic_lpmf, ordered_logistic_lupmf, ordered_probit_lpmf, ordered_probit_lupmf,
};
use crate::glm::{
    bernoulli_logit_glm_lpmf, bernoulli_logit_glm_lupmf, binomial_logit_glm_lpmf,
    binomial_logit_glm_lupmf, categorical_logit_glm_lpmf, categorical_logit_glm_lupmf,
    neg_binomial_2_log_glm_lpmf, neg_binomial_2_log_glm_lupmf, normal_id_glm_lpdf,
    normal_id_glm_lupdf, ordered_logistic_glm_lpmf, ordered_logistic_glm_lupmf,
    poisson_log_glm_lpmf, poisson_log_glm_lupmf,
};
use crate::multivariate::{dirichlet_lpdf, dirichlet_lupdf};
use crate::multivariate_extra::{
    gaussian_dlm_obs_lpdf, gaussian_dlm_obs_lupdf, multi_normal_cholesky_lpdf_array,
    multi_normal_cholesky_lupdf_array, multi_normal_lpdf_array, multi_normal_lupdf_array,
    multi_normal_prec_lpdf, multi_normal_prec_lupdf, multi_student_t_cholesky_lpdf,
    multi_student_t_cholesky_lupdf,
};
use crate::wiener_full::{
    WienerArgs, WienerOut, wiener_full_prec_lccdf_unnorm, wiener_full_prec_lcdf_unnorm,
    wiener_full_prec_lpdf, wiener_full_prec_lupdf,
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

propto_pair! {
    /// Stan's `multi_normal_lpdf(vector y | array[] vector mu, matrix Sigma)`: the sum over the
    /// locations `mu` of the multivariate normal log density of the single observation `y`.
    multi_normal_lpdf_mu_array, multi_normal_lupdf_mu_array <T, Y, M>(
        y: &[Y],
        mu: &[Vec<M>],
        sigma: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    base(multi_normal_lpdf_array, multi_normal_lupdf_array) |base| {
        base(&repeat_vec(y, mu.len()), mu, sigma)
    }
}

propto_pair! {
    /// Stan's `multi_normal_cholesky_lpdf(vector y | array[] vector mu, matrix L)`: the sum over the
    /// locations `mu` of the log density of the single observation `y`.
    multi_normal_cholesky_lpdf_mu_array, multi_normal_cholesky_lupdf_mu_array <T, Y, M>(
        y: &[Y],
        mu: &[Vec<M>],
        l: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    base(multi_normal_cholesky_lpdf_array, multi_normal_cholesky_lupdf_array) |base| {
        base(&repeat_vec(y, mu.len()), mu, l)
    }
}

propto_pair! {
    /// Sum of multivariate normal log densities with precision matrix `omega` over an array of
    /// observations `y`. Stan's `multi_normal_prec_lpdf` with `array[] vector y`.
    ///
    /// `mu` holds either one location shared by every observation or one location per observation.
    multi_normal_prec_lpdf_array, multi_normal_prec_lupdf_array <T, Y, M>(
        y: &[Vec<Y>],
        mu: &[Vec<M>],
        omega: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    base(multi_normal_prec_lpdf, multi_normal_prec_lupdf) |base| {
        // ponytail: the log determinant of omega is recomputed per observation; factor it once if
        // long arrays become a bottleneck.
        sum_terms("multi_normal_prec_lpdf", y, mu, |yi, mi| {
            base(&yi[..], &mi[..], omega)
        })
    }
}

propto_pair! {
    /// Stan's `multi_normal_prec_lpdf(vector y | array[] vector mu, matrix Omega)`: the sum over the
    /// locations `mu` of the log density of the single observation `y`.
    multi_normal_prec_lpdf_mu_array, multi_normal_prec_lupdf_mu_array <T, Y, M>(
        y: &[Y],
        mu: &[Vec<M>],
        omega: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    base(multi_normal_prec_lpdf_array, multi_normal_prec_lupdf_array) |base| {
        base(&repeat_vec(y, mu.len()), mu, omega)
    }
}

propto_pair! {
    /// Sum of multivariate Student t log densities over an array of observations `y`, with the scale
    /// matrix `sigma` factored once. Stan's `multi_student_t_lpdf` with `array[] vector y`.
    ///
    /// `mu` holds either one location shared by every observation or one location per observation.
    multi_student_t_lpdf_array, multi_student_t_lupdf_array <T, Y, N, M>(
        y: &[Vec<Y>],
        nu: N,
        mu: &[Vec<M>],
        sigma: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, N: Lift<T>, M: Lift<T> + Copy }
    base(multi_student_t_cholesky_lpdf, multi_student_t_cholesky_lupdf) |base| {
        let l = cholesky_decompose(sigma)?;
        let nu: T = nu.lift();
        sum_terms("multi_student_t_lpdf", y, mu, |yi, mi| {
            base(&yi[..], nu, &mi[..], &l)
        })
    }
}

propto_pair! {
    /// Stan's `multi_student_t_lpdf(vector y | real nu, array[] vector mu, matrix Sigma)`: the sum
    /// over the locations `mu` of the log density of the single observation `y`.
    multi_student_t_lpdf_mu_array, multi_student_t_lupdf_mu_array <T, Y, N, M>(
        y: &[Y],
        nu: N,
        mu: &[Vec<M>],
        sigma: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, N: Lift<T>, M: Lift<T> + Copy }
    base(multi_student_t_lpdf_array, multi_student_t_lupdf_array) |base| {
        base(&repeat_vec(y, mu.len()), nu, mu, sigma)
    }
}

propto_pair! {
    /// Sum of multivariate Student t log densities over an array of observations `y`, with the scale
    /// matrix given by its lower Cholesky factor `l`. Stan's `multi_student_t_cholesky_lpdf` with
    /// `array[] vector y`.
    ///
    /// `mu` holds either one location shared by every observation or one location per observation.
    multi_student_t_cholesky_lpdf_array, multi_student_t_cholesky_lupdf_array <T, Y, N, M>(
        y: &[Vec<Y>],
        nu: N,
        mu: &[Vec<M>],
        l: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, N: Lift<T>, M: Lift<T> + Copy }
    base(multi_student_t_cholesky_lpdf, multi_student_t_cholesky_lupdf) |base| {
        let nu: T = nu.lift();
        sum_terms("multi_student_t_cholesky_lpdf", y, mu, |yi, mi| {
            base(&yi[..], nu, &mi[..], l)
        })
    }
}

propto_pair! {
    /// Stan's `multi_student_t_cholesky_lpdf(vector y | real nu, array[] vector mu, matrix L)`: the
    /// sum over the locations `mu` of the log density of the single observation `y`.
    multi_student_t_cholesky_lpdf_mu_array, multi_student_t_cholesky_lupdf_mu_array <T, Y, N, M>(
        y: &[Y],
        nu: N,
        mu: &[Vec<M>],
        l: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, N: Lift<T>, M: Lift<T> + Copy }
    base(multi_student_t_cholesky_lpdf_array, multi_student_t_cholesky_lupdf_array) |base| {
        base(&repeat_vec(y, mu.len()), nu, mu, l)
    }
}

// Dirichlet.

propto_pair! {
    /// Sum of Dirichlet log densities over an array of simplexes `theta`. Stan's `dirichlet_lpdf`
    /// with `array[] vector theta`.
    ///
    /// `alpha` holds either one concentration vector shared by every simplex or one per simplex.
    dirichlet_lpdf_array, dirichlet_lupdf_array <T, A>(
        theta: &[Vec<T>],
        alpha: &[Vec<A>],
    ) -> Result<T>
    where { T: Real, A: Lift<T> + Copy }
    base(dirichlet_lpdf, dirichlet_lupdf) |base| {
        sum_terms("dirichlet_lpdf", theta, alpha, |t, a| {
            base(&t[..], &a[..])
        })
    }
}

propto_pair! {
    /// Stan's `dirichlet_lpdf(vector theta | array[] vector alpha)`: the sum over the concentration
    /// vectors `alpha` of the Dirichlet log density of the single simplex `theta`.
    dirichlet_lpdf_alpha_array, dirichlet_lupdf_alpha_array <T, A>(
        theta: &[T],
        alpha: &[Vec<A>],
    ) -> Result<T>
    where { T: Real, A: Lift<T> + Copy }
    base(dirichlet_lpdf_array, dirichlet_lupdf_array) |base| {
        base(&repeat_vec(theta, alpha.len()), alpha)
    }
}

// Ordered models with one cutpoint vector per observation.

propto_pair! {
    /// Stan's `ordered_logistic_lpmf(array[] int y | vector eta, array[] vector c)`: observation `i`
    /// uses the cutpoints `c[i]`. `eta` is a scalar or one value per observation, and `c` has one
    /// vector per observation.
    ordered_logistic_lpmf_array, ordered_logistic_lupmf_array <E, T>(
        y: &[i64],
        eta: E,
        c: &[Vec<T>],
    ) -> Result<T>
    where { E: Arg, E::Elem: Lift<T>, T: Real }
    base(ordered_logistic_lpmf, ordered_logistic_lupmf) |base| {
        ordered_per_cutpoints("ordered_logistic_lpmf", y, eta, c, |yi, e, ci| {
            base(yi, e, ci)
        })
    }
}

propto_pair! {
    /// Stan's `ordered_probit_lpmf(array[] int y | real eta, array[] vector c)` and its `vector eta`
    /// form: observation `i` uses the cutpoints `c[i]`. `eta` is a scalar or one value per
    /// observation, and `c` has one vector per observation.
    ordered_probit_lpmf_array, ordered_probit_lupmf_array <E, T>(
        y: &[i64],
        eta: E,
        c: &[Vec<T>],
    ) -> Result<T>
    where { E: Arg, E::Elem: Lift<T>, T: Real }
    base(ordered_probit_lpmf, ordered_probit_lupmf) |base| {
        ordered_per_cutpoints("ordered_probit_lpmf", y, eta, c, |yi, e, ci| {
            base(yi, e, ci)
        })
    }
}

// GLMs with a single predictor row shared by every observation.

propto_pair! {
    /// [`poisson_log_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`. The
    /// number of observations is the length of `y` or of `alpha`.
    poisson_log_glm_lpmf_row, poisson_log_glm_lupmf_row <N, X, A, T>(
        y: N,
        x: &[X],
        alpha: A,
        beta: &[T],
    ) -> Result<T>
    where { N: IntArg, X: Lift<T> + Copy, A: Arg, A::Elem: Lift<T>, T: Real }
    base(poisson_log_glm_lpmf, poisson_log_glm_lupmf) |base| {
        let n = broadcast_len("poisson_log_glm_lpmf", &[y.int_len(), alpha.arg_len()])?;
        base(y, &repeat_row(x, n), alpha, beta)
    }
}

propto_pair! {
    /// [`bernoulli_logit_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`.
    /// The number of observations is the length of `y` or of `alpha`.
    bernoulli_logit_glm_lpmf_row, bernoulli_logit_glm_lupmf_row <N, X, A, T>(
        y: N,
        x: &[X],
        alpha: A,
        beta: &[T],
    ) -> Result<T>
    where { N: IntArg, X: Lift<T> + Copy, A: Arg, A::Elem: Lift<T>, T: Real }
    base(bernoulli_logit_glm_lpmf, bernoulli_logit_glm_lupmf) |base| {
        let n = broadcast_len("bernoulli_logit_glm_lpmf", &[y.int_len(), alpha.arg_len()])?;
        base(y, &repeat_row(x, n), alpha, beta)
    }
}

propto_pair! {
    /// [`binomial_logit_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`.
    /// The number of observations is the common length of `n`, `trials`, and `alpha`.
    binomial_logit_glm_lpmf_row, binomial_logit_glm_lupmf_row <N, M, X, A, T>(
        n: N,
        trials: M,
        x: &[X],
        alpha: A,
        beta: &[T],
    ) -> Result<T>
    where { N: IntArg, M: IntArg, X: Lift<T> + Copy, A: Arg, A::Elem: Lift<T>, T: Real }
    base(binomial_logit_glm_lpmf, binomial_logit_glm_lupmf) |base| {
        let rows = broadcast_len(
            "binomial_logit_glm_lpmf",
            &[n.int_len(), trials.int_len(), alpha.arg_len()],
        )?;
        base(n, trials, &repeat_row(x, rows), alpha, beta)
    }
}

propto_pair! {
    /// [`neg_binomial_2_log_glm_lpmf`] with `row_vector x`: every observation uses the predictors
    /// `x`. The number of observations is the common length of `y`, `alpha`, and `phi`.
    neg_binomial_2_log_glm_lpmf_row, neg_binomial_2_log_glm_lupmf_row <N, X, A, T, P>(
        y: N,
        x: &[X],
        alpha: A,
        beta: &[T],
        phi: P,
    ) -> Result<T>
    where {
        N: IntArg,
        X: Lift<T> + Copy,
        A: Arg,
        A::Elem: Lift<T>,
        T: Real,
        P: Arg,
        P::Elem: Lift<T>,
    }
    base(neg_binomial_2_log_glm_lpmf, neg_binomial_2_log_glm_lupmf) |base| {
        let n = broadcast_len(
            "neg_binomial_2_log_glm_lpmf",
            &[y.int_len(), alpha.arg_len(), phi.arg_len()],
        )?;
        base(y, &repeat_row(x, n), alpha, beta, phi)
    }
}

propto_pair! {
    /// [`normal_id_glm_lpdf`] with `real y`: the scalar response is used for every row of `x`.
    normal_id_glm_lpdf_scalar_y, normal_id_glm_lupdf_scalar_y <Y, X, A, T, S>(
        y: Y,
        x: &Matrix<X>,
        alpha: A,
        beta: &[T],
        sigma: S,
    ) -> Result<T>
    where {
        Y: Lift<T> + Copy,
        X: Lift<T> + Copy,
        A: Arg,
        A::Elem: Lift<T>,
        T: Real,
        S: Arg,
        S::Elem: Lift<T>,
    }
    base(normal_id_glm_lpdf, normal_id_glm_lupdf) |base| {
        base(&vec![y; x.nrows()][..], x, alpha, beta, sigma)
    }
}

propto_pair! {
    /// [`normal_id_glm_lpdf`] with `row_vector x`: every observation uses the predictors `x`. `y` is a
    /// scalar or a vector; the number of observations is the common length of `y`, `alpha`, and
    /// `sigma`.
    normal_id_glm_lpdf_row, normal_id_glm_lupdf_row <Y, X, A, T, S>(
        y: Y,
        x: &[X],
        alpha: A,
        beta: &[T],
        sigma: S,
    ) -> Result<T>
    where {
        Y: Arg,
        Y::Elem: Lift<T>,
        X: Lift<T> + Copy,
        A: Arg,
        A::Elem: Lift<T>,
        T: Real,
        S: Arg,
        S::Elem: Lift<T>,
    }
    base(normal_id_glm_lpdf, normal_id_glm_lupdf) |base| {
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
        base(&ys[..], &repeat_row(x, n), alpha, beta, sigma)
    }
}

propto_pair! {
    /// [`normal_id_glm_lpdf`] with `vector x`, new in Stan 2.40. `x` is read as an `N` by 1 design
    /// matrix, one predictor value per observation, so `beta` has one element; `y` is a scalar or a
    /// vector of length `N`. This reading is **unconfirmed**: the Functions Reference only says
    /// "location alpha + x * beta".
    ///
    /// # Errors
    ///
    /// As [`normal_id_glm_lpdf`], and [`Error::DimensionMismatch`] when a vector `y` does not have
    /// one element per entry of `x`.
    normal_id_glm_lpdf_vector_x, normal_id_glm_lupdf_vector_x <Y, X, A, T, S>(
        y: Y,
        x: &[X],
        alpha: A,
        beta: &[T],
        sigma: S,
    ) -> Result<T>
    where {
        Y: Arg,
        Y::Elem: Lift<T>,
        X: Lift<T> + Copy,
        A: Arg,
        A::Elem: Lift<T>,
        T: Real,
        S: Arg,
        S::Elem: Lift<T>,
    }
    base(normal_id_glm_lpdf, normal_id_glm_lupdf) |base| {
        let n = broadcast_len("normal_id_glm_lpdf", &[y.arg_len(), Some(x.len())])?;
        let ys: Vec<T> = (0..n)
            .map(|i| {
                let v: T = y.arg_at(i).lift();
                v
            })
            .collect();
        let design = Matrix::from_fn(n, 1, |i, _| x[i]);
        base(&ys[..], &design, alpha, beta, sigma)
    }
}

propto_pair! {
    /// [`categorical_logit_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`.
    /// The number of observations is the length of `y`, or one for a scalar `y`.
    categorical_logit_glm_lpmf_row, categorical_logit_glm_lupmf_row <N, X, A, T>(
        y: N,
        x: &[X],
        alpha: &[A],
        beta: &Matrix<T>,
    ) -> Result<T>
    where { N: IntArg, X: Lift<T> + Copy, A: Lift<T> + Copy, T: Real }
    base(categorical_logit_glm_lpmf, categorical_logit_glm_lupmf) |base| {
        let n = y.int_len().unwrap_or(1);
        base(y, &repeat_row(x, n), alpha, beta)
    }
}

propto_pair! {
    /// [`ordered_logistic_glm_lpmf`] with `row_vector x`: every observation uses the predictors `x`.
    /// The number of observations is the length of `y`, or one for a scalar `y`.
    ordered_logistic_glm_lpmf_row, ordered_logistic_glm_lupmf_row <N, X, T, C>(
        y: N,
        x: &[X],
        beta: &[T],
        cuts: &[C],
    ) -> Result<T>
    where { N: IntArg, X: Lift<T> + Copy, T: Real, C: Lift<T> + Copy }
    base(ordered_logistic_glm_lpmf, ordered_logistic_glm_lupmf) |base| {
        let n = y.int_len().unwrap_or(1);
        base(y, &repeat_row(x, n), beta, cuts)
    }
}

// Gaussian dynamic linear model with a diagonal observation covariance.

propto_pair! {
    /// [`gaussian_dlm_obs_lpdf`] with `vector V`: the observation covariance is the diagonal matrix
    /// with diagonal `v`.
    gaussian_dlm_obs_lpdf_vector_v, gaussian_dlm_obs_lupdf_vector_v <T>(
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
    where { T: Real, (Y, A, T0, W, V, SV, f64, f64): WienerArgs<8> }
    base(wiener_full_prec_lpdf, wiener_full_prec_lupdf) |base| {
        base(y, a, t0, w, v, sv, 0.0, 0.0, precision)
    }
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
