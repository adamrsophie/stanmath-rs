//! Multivariate distributions.
//!
//! Arguments that are often data (`y`, `mu`, `alpha`) accept any element type that lifts to the
//! scalar type of the parameters, so `&[f64]` data can be mixed with differentiable parameters.

use stanmath_rs_autodiff::functions::lgamma;
use stanmath_rs_autodiff::{Lift, Real};
use stanmath_rs_core::special::{HALF_LN_2PI, LN_PI};
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::{cholesky_decompose, mdivide_left_tri_low};

use crate::kernel::{is_param, keep};

/// Tolerance for simplex checks, as Stan's `CONSTRAINT_TOLERANCE`.
const TOL: f64 = 1e-8;

fn lift_all<T: Real, E: Lift<T> + Copy>(xs: &[E]) -> Vec<T> {
    xs.iter().map(|&x| x.lift()).collect()
}

fn check_vector_sizes(function: &'static str, k: usize, lens: &[(&str, usize)]) -> Result<()> {
    for &(name, len) in lens {
        if len != k {
            return Err(Error::dimension(
                function,
                format!("{name} has length {len}, expected {k}"),
            ));
        }
    }
    Ok(())
}

propto_pair! {
    /// Dirichlet log density of a simplex `theta` with concentrations `alpha`.
    ///
    /// Requires `alpha` positive finite and `theta` nonnegative summing to one within `1e-8`.
    /// Uses `0 ln 0 = 0`.
    dirichlet_lpdf, dirichlet_lupdf <T, A>(theta: &[T], alpha: &[A]) -> Result<T>
    where { T: Real, A: Lift<T> + Copy }
    |propto| { dirichlet_with(theta, alpha, propto) }
}

fn dirichlet_with<T: Real, A: Lift<T> + Copy>(theta: &[T], alpha: &[A], propto: bool) -> Result<T> {
    const F: &str = "dirichlet_lpdf";
    check_vector_sizes(F, theta.len(), &[("alpha", alpha.len())])?;
    let alpha: Vec<T> = lift_all(alpha);
    for a in &alpha {
        if !(a.value() > 0.0 && a.is_finite()) {
            return Err(Error::parameter(F, "alpha", a.value(), "positive finite"));
        }
    }
    let total: f64 = theta.iter().map(Real::value).sum();
    if theta.iter().any(|t| t.is_nan() || t.value() < 0.0) || (total - 1.0).abs() > TOL {
        return Err(Error::observation(F, "theta", total, "a simplex"));
    }
    let pa = is_param::<T, A>();
    let mut lp = T::zero();
    if keep(propto, &[pa]) {
        lp += lgamma(alpha.iter().copied().sum::<T>());
        for &a in &alpha {
            lp -= lgamma(a);
        }
    }
    if keep(propto, &[is_param::<T, T>(), pa]) {
        for (&t, &a) in theta.iter().zip(&alpha) {
            let am1 = a - 1.0;
            if !(am1.value() == 0.0 && t.value() == 0.0) {
                lp += am1 * t.ln();
            }
        }
    }
    Ok(lp)
}

/// The multivariate normal log density with covariance `L Lᵀ`; `params` flags whether `y` and
/// `mu` are parameters.
pub(crate) fn multi_normal_cholesky_core<T: Real>(
    function: &'static str,
    y: &[T],
    mu: &[T],
    l: &Matrix<T>,
    propto: bool,
    [py, pm]: [bool; 2],
) -> Result<T> {
    let k = y.len();
    check_vector_sizes(
        function,
        k,
        &[
            ("mu", mu.len()),
            ("L rows", l.nrows()),
            ("L columns", l.ncols()),
        ],
    )?;
    if let Some(v) = y.iter().find(|v| v.is_nan()) {
        return Err(Error::observation(function, "y", v.value(), "not NaN"));
    }
    if let Some(v) = mu.iter().find(|v| !v.is_finite()) {
        return Err(Error::parameter(function, "mu", v.value(), "finite"));
    }
    if k == 0 {
        return Ok(T::zero());
    }
    let pl = is_param::<T, T>();
    let mut lp = T::zero();
    if keep(propto, &[py, pm, pl]) {
        let diff: Vec<T> = y.iter().zip(mu).map(|(&a, &b)| a - b).collect();
        let z = mdivide_left_tri_low(l, &Matrix::column_vector(diff))?;
        lp -= z.as_slice().iter().map(|v| v.square()).sum::<T>() * 0.5;
    }
    if keep(propto, &[pl]) {
        lp -= (0..k).map(|i| l[(i, i)].ln()).sum::<T>();
    }
    if keep(propto, &[]) {
        lp -= HALF_LN_2PI * k as f64;
    }
    Ok(lp)
}

propto_pair! {
    /// Multivariate normal log density with location `mu` and covariance `L Lᵀ`, given the lower
    /// Cholesky factor `L` (only its lower triangle is read).
    multi_normal_cholesky_lpdf, multi_normal_cholesky_lupdf <T, Y, M>(
        y: &[Y],
        mu: &[M],
        l: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    |propto| {
        let params = [is_param::<T, Y>(), is_param::<T, M>()];
        let (y, mu) = (lift_all(y), lift_all(mu));
        multi_normal_cholesky_core("multi_normal_cholesky_lpdf", &y, &mu, l, propto, params)
    }
}

propto_pair! {
    /// Multivariate normal log density with location `mu` and covariance `sigma`.
    ///
    /// # Errors
    ///
    /// [`Error::NotSymmetric`] or [`Error::NotPositiveDefinite`] for an invalid covariance, plus
    /// the argument checks of [`multi_normal_cholesky_lpdf`].
    multi_normal_lpdf, multi_normal_lupdf <T, Y, M>(y: &[Y], mu: &[M], sigma: &Matrix<T>)
        -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    |propto| { multi_normal_with(y, mu, sigma, propto) }
}

fn multi_normal_with<T, Y, M>(y: &[Y], mu: &[M], sigma: &Matrix<T>, propto: bool) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    M: Lift<T> + Copy,
{
    const F: &str = "multi_normal_lpdf";
    if !sigma.is_square() || sigma.nrows() != y.len() {
        return Err(Error::dimension(
            F,
            "covariance does not match the length of y",
        ));
    }
    let l = cholesky_decompose(sigma)?;
    let params = [is_param::<T, Y>(), is_param::<T, M>()];
    multi_normal_cholesky_core(F, &lift_all(y), &lift_all(mu), &l, propto, params)
}

/// Log normalizing constant of the LKJ density, as in Stan Math (Lewandowski, Kurowicka, and Joe
/// 2009).
pub(crate) fn lkj_constant<T: Real>(eta: T, k: usize) -> T {
    let km1 = (k - 1) as f64;
    let mut c = lgamma(eta + 0.5 * km1) * km1;
    for i in 1..k {
        let i = i as f64;
        c -= lgamma(eta + 0.5 * (km1 - i)) + 0.5 * i * LN_PI;
    }
    c
}

propto_pair! {
    /// LKJ log density of the Cholesky factor `L` of a correlation matrix, with shape `eta`.
    ///
    /// `eta = 1` is uniform over correlation matrices. Requires `eta > 0` and `L` lower triangular
    /// with positive diagonal.
    lkj_corr_cholesky_lpdf, lkj_corr_cholesky_lupdf <T, E>(l: &Matrix<T>, eta: E) -> Result<T>
    where { T: Real, E: Lift<T> }
    |propto| { lkj_corr_cholesky_with(l, eta, propto) }
}

fn lkj_corr_cholesky_with<T: Real, E: Lift<T>>(l: &Matrix<T>, eta: E, propto: bool) -> Result<T> {
    const F: &str = "lkj_corr_cholesky_lpdf";
    let pe = is_param::<T, E>();
    let eta: T = eta.lift();
    if !(eta.value() > 0.0) {
        return Err(Error::parameter(F, "eta", eta.value(), "positive"));
    }
    if !l.is_square() {
        return Err(Error::dimension(F, "expected a square matrix"));
    }
    let k = l.nrows();
    for i in 0..k {
        if !(l[(i, i)].value() > 0.0) || (i + 1..k).any(|j| l[(i, j)].value() != 0.0) {
            return Err(Error::observation(
                F,
                "L",
                l[(i, i)].value(),
                "a Cholesky factor",
            ));
        }
    }
    if k == 0 {
        return Ok(T::zero());
    }
    let mut lp = T::zero();
    if keep(propto, &[pe]) {
        lp += lkj_constant(eta, k);
    }
    if keep(propto, &[is_param::<T, T>(), pe]) {
        for i in 1..k {
            let coef = (k - i - 1) as f64;
            lp += l[(i, i)].ln() * (eta * 2.0 - 2.0 + coef);
        }
    }
    Ok(lp)
}
