//! Multivariate distributions.
//!
//! Arguments that are often data (`y`, `mu`, `alpha`) accept any element type that lifts to the
//! scalar type of the parameters, so `&[f64]` data can be mixed with differentiable parameters.

use probgrad_autodiff::functions::lgamma;
use probgrad_autodiff::{Lift, Real};
use probgrad_core::special::{HALF_LN_2PI, LN_PI};
use probgrad_core::{Error, Matrix, Result};
use probgrad_linalg::{cholesky_decompose, mdivide_left_tri_low};

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

/// Dirichlet log density of a simplex `theta` with concentrations `alpha`.
///
/// Requires `alpha` positive finite and `theta` nonnegative summing to one within `1e-8`. Uses
/// `0 ln 0 = 0`.
pub fn dirichlet_lpdf<T: Real, A: Lift<T> + Copy>(theta: &[T], alpha: &[A]) -> Result<T> {
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
    let mut lp = lgamma(alpha.iter().copied().sum::<T>());
    for (&t, &a) in theta.iter().zip(&alpha) {
        lp -= lgamma(a);
        let am1 = a - 1.0;
        if !(am1.value() == 0.0 && t.value() == 0.0) {
            lp += am1 * t.ln();
        }
    }
    Ok(lp)
}

fn multi_normal_cholesky_core<T: Real>(
    function: &'static str,
    y: &[T],
    mu: &[T],
    l: &Matrix<T>,
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
    let diff: Vec<T> = y.iter().zip(mu).map(|(&a, &b)| a - b).collect();
    let z = mdivide_left_tri_low(l, &Matrix::column_vector(diff))?;
    let half_sq: T = z.as_slice().iter().map(|v| v.square()).sum::<T>() * 0.5;
    let log_det: T = (0..k).map(|i| l[(i, i)].ln()).sum();
    Ok(-half_sq - log_det - HALF_LN_2PI * k as f64)
}

/// Multivariate normal log density with location `mu` and covariance `L Lᵀ`, given the lower
/// Cholesky factor `L` (only its lower triangle is read).
pub fn multi_normal_cholesky_lpdf<T, Y, M>(y: &[Y], mu: &[M], l: &Matrix<T>) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    M: Lift<T> + Copy,
{
    multi_normal_cholesky_core("multi_normal_cholesky_lpdf", &lift_all(y), &lift_all(mu), l)
}

/// Multivariate normal log density with location `mu` and covariance `sigma`.
///
/// # Errors
///
/// [`Error::NotSymmetric`] or [`Error::NotPositiveDefinite`] for an invalid covariance, plus the
/// argument checks of [`multi_normal_cholesky_lpdf`].
pub fn multi_normal_lpdf<T, Y, M>(y: &[Y], mu: &[M], sigma: &Matrix<T>) -> Result<T>
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
    multi_normal_cholesky_core(F, &lift_all(y), &lift_all(mu), &l)
}

/// Log normalizing constant of the LKJ density, as in Stan Math (Lewandowski, Kurowicka, and Joe
/// 2009).
fn lkj_constant<T: Real>(eta: T, k: usize) -> T {
    let km1 = (k - 1) as f64;
    let mut c = lgamma(eta + 0.5 * km1) * km1;
    for i in 1..k {
        let i = i as f64;
        c -= lgamma(eta + 0.5 * (km1 - i)) + 0.5 * i * LN_PI;
    }
    c
}

/// LKJ log density of the Cholesky factor `L` of a correlation matrix, with shape `eta`.
///
/// `eta = 1` is uniform over correlation matrices. Requires `eta > 0` and `L` lower triangular with
/// positive diagonal.
pub fn lkj_corr_cholesky_lpdf<T: Real, E: Lift<T>>(l: &Matrix<T>, eta: E) -> Result<T> {
    const F: &str = "lkj_corr_cholesky_lpdf";
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
    let mut lp = lkj_constant(eta, k);
    for i in 1..k {
        let coef = (k - i - 1) as f64;
        lp += l[(i, i)].ln() * (eta * 2.0 - 2.0 + coef);
    }
    Ok(lp)
}
