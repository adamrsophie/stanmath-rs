//! Gaussian process covariance functions.
//!
//! Each function builds the covariance matrix of a set of points `x` (or the cross covariance of
//! two sets `x1`, `x2`). Points are vectors of equal length. The stationary kernels depend on the
//! Euclidean distance `d` between two points; at `d = 0` they return `sigma^2` directly, so the
//! diagonal and coincident points never differentiate `sqrt` at zero.
//!
//! | Function | `k(x, x')` |
//! | --- | --- |
//! | `gp_exp_quad_cov` | `sigma^2 exp(-d^2 / (2 l^2))` |
//! | `gp_matern32_cov` | `sigma^2 (1 + sqrt(3) d / l) exp(-sqrt(3) d / l)` |
//! | `gp_matern52_cov` | `sigma^2 (1 + sqrt(5) d / l + 5 d^2 / (3 l^2)) exp(-sqrt(5) d / l)` |
//! | `gp_exponential_cov` | `sigma^2 exp(-d / l)` |
//! | `gp_periodic_cov` | `sigma^2 exp(-2 sin^2(pi d / p) / l^2)` |
//! | `gp_dot_prod_cov` | `sigma^2 + xᵀ x'` |

use core::f64::consts::PI;

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Matrix, Result};

/// Checks that every point has the same length and no NaN coordinate; returns the length.
fn check_points<T: Real>(
    function: &'static str,
    name: &'static str,
    x: &[Vec<T>],
) -> Result<usize> {
    let dim = x.first().map_or(0, Vec::len);
    for p in x {
        if p.len() != dim {
            return Err(Error::dimension(
                function,
                format!("{name} has points of lengths {dim} and {}", p.len()),
            ));
        }
        if let Some(v) = p.iter().find(|v| v.is_nan()) {
            return Err(Error::parameter(function, name, v.value(), "not NaN"));
        }
    }
    Ok(dim)
}

fn positive_finite<T: Real>(function: &'static str, name: &'static str, v: T) -> Result<()> {
    let v = v.value();
    if v > 0.0 && v.is_finite() {
        Ok(())
    } else {
        Err(Error::parameter(function, name, v, "positive finite"))
    }
}

fn squared_dist<T: Real>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).map(|(&u, &v)| (u - v).square()).sum()
}

/// Builds `K[i, j] = k(x1[i], x2[j])`, with `x2 = x1` when `x2` is `None`.
fn build<T: Real>(
    function: &'static str,
    x1: &[Vec<T>],
    x2: Option<&[Vec<T>]>,
    k: impl Fn(&[T], &[T]) -> T,
) -> Result<Matrix<T>> {
    let d1 = check_points(function, "x1", x1)?;
    match x2 {
        None => Ok(Matrix::from_fn(x1.len(), x1.len(), |i, j| {
            k(&x1[i], &x1[j])
        })),
        Some(x2) => {
            let d2 = check_points(function, "x2", x2)?;
            if !x1.is_empty() && !x2.is_empty() && d1 != d2 {
                return Err(Error::dimension(
                    function,
                    format!("points of length {d1} and {d2}"),
                ));
            }
            Ok(Matrix::from_fn(x1.len(), x2.len(), |i, j| {
                k(&x1[i], &x2[j])
            }))
        }
    }
}

/// A stationary kernel `sigma^2 g(d^2)`, equal to `sigma^2` at `d = 0`.
fn stationary<T: Real>(sigma: T, g: impl Fn(T) -> T) -> impl Fn(&[T], &[T]) -> T {
    move |a: &[T], b: &[T]| {
        let d2 = squared_dist(a, b);
        if d2.value() == 0.0 {
            sigma.square()
        } else {
            sigma.square() * g(d2)
        }
    }
}

fn exp_quad<T: Real>(
    x1: &[Vec<T>],
    x2: Option<&[Vec<T>]>,
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    const F: &str = "gp_exp_quad_cov";
    positive_finite(F, "sigma", sigma)?;
    positive_finite(F, "length_scale", length_scale)?;
    let inv = (length_scale.square() * 2.0).recip();
    build(F, x1, x2, stationary(sigma, move |d2| (-(d2 * inv)).exp()))
}

/// Squared exponential covariance of the points `x`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] unless `sigma` and `length_scale` are positive finite and the
/// points are not NaN; [`Error::DimensionMismatch`] for points of different lengths.
pub fn gp_exp_quad_cov<T: Real>(x: &[Vec<T>], sigma: T, length_scale: T) -> Result<Matrix<T>> {
    exp_quad(x, None, sigma, length_scale)
}

/// Squared exponential cross covariance between `x1` and `x2`.
///
/// # Errors
///
/// As [`gp_exp_quad_cov`].
pub fn gp_exp_quad_cov_cross<T: Real>(
    x1: &[Vec<T>],
    x2: &[Vec<T>],
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    exp_quad(x1, Some(x2), sigma, length_scale)
}

fn matern32<T: Real>(
    x1: &[Vec<T>],
    x2: Option<&[Vec<T>]>,
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    const F: &str = "gp_matern32_cov";
    positive_finite(F, "sigma", sigma)?;
    positive_finite(F, "length_scale", length_scale)?;
    let c = length_scale.recip() * 3f64.sqrt();
    build(
        F,
        x1,
        x2,
        stationary(sigma, move |d2| {
            let r = d2.sqrt() * c;
            (r + 1.0) * (-r).exp()
        }),
    )
}

/// Matérn covariance with smoothness 3/2 of the points `x`.
///
/// # Errors
///
/// As [`gp_exp_quad_cov`].
pub fn gp_matern32_cov<T: Real>(x: &[Vec<T>], sigma: T, length_scale: T) -> Result<Matrix<T>> {
    matern32(x, None, sigma, length_scale)
}

/// Matérn 3/2 cross covariance between `x1` and `x2`.
///
/// # Errors
///
/// As [`gp_exp_quad_cov`].
pub fn gp_matern32_cov_cross<T: Real>(
    x1: &[Vec<T>],
    x2: &[Vec<T>],
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    matern32(x1, Some(x2), sigma, length_scale)
}

fn matern52<T: Real>(
    x1: &[Vec<T>],
    x2: Option<&[Vec<T>]>,
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    const F: &str = "gp_matern52_cov";
    positive_finite(F, "sigma", sigma)?;
    positive_finite(F, "length_scale", length_scale)?;
    let c = length_scale.recip() * 5f64.sqrt();
    build(
        F,
        x1,
        x2,
        stationary(sigma, move |d2| {
            let r = d2.sqrt() * c;
            (r + r.square() / 3.0 + 1.0) * (-r).exp()
        }),
    )
}

/// Matérn covariance with smoothness 5/2 of the points `x`.
///
/// # Errors
///
/// As [`gp_exp_quad_cov`].
pub fn gp_matern52_cov<T: Real>(x: &[Vec<T>], sigma: T, length_scale: T) -> Result<Matrix<T>> {
    matern52(x, None, sigma, length_scale)
}

/// Matérn 5/2 cross covariance between `x1` and `x2`.
///
/// # Errors
///
/// As [`gp_exp_quad_cov`].
pub fn gp_matern52_cov_cross<T: Real>(
    x1: &[Vec<T>],
    x2: &[Vec<T>],
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    matern52(x1, Some(x2), sigma, length_scale)
}

fn exponential<T: Real>(
    x1: &[Vec<T>],
    x2: Option<&[Vec<T>]>,
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    const F: &str = "gp_exponential_cov";
    positive_finite(F, "sigma", sigma)?;
    positive_finite(F, "length_scale", length_scale)?;
    let inv = length_scale.recip();
    build(
        F,
        x1,
        x2,
        stationary(sigma, move |d2| (-(d2.sqrt() * inv)).exp()),
    )
}

/// Exponential (Matérn 1/2) covariance of the points `x`.
///
/// # Errors
///
/// As [`gp_exp_quad_cov`].
pub fn gp_exponential_cov<T: Real>(x: &[Vec<T>], sigma: T, length_scale: T) -> Result<Matrix<T>> {
    exponential(x, None, sigma, length_scale)
}

/// Exponential cross covariance between `x1` and `x2`.
///
/// # Errors
///
/// As [`gp_exp_quad_cov`].
pub fn gp_exponential_cov_cross<T: Real>(
    x1: &[Vec<T>],
    x2: &[Vec<T>],
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    exponential(x1, Some(x2), sigma, length_scale)
}

fn periodic<T: Real>(
    x1: &[Vec<T>],
    x2: Option<&[Vec<T>]>,
    sigma: T,
    length_scale: T,
    period: T,
) -> Result<Matrix<T>> {
    const F: &str = "gp_periodic_cov";
    positive_finite(F, "sigma", sigma)?;
    positive_finite(F, "length_scale", length_scale)?;
    positive_finite(F, "period", period)?;
    let w = period.recip() * PI;
    let c = length_scale.square().recip() * -2.0;
    build(
        F,
        x1,
        x2,
        stationary(sigma, move |d2| ((d2.sqrt() * w).sin().square() * c).exp()),
    )
}

/// Periodic covariance of the points `x` with period `period`.
///
/// # Errors
///
/// As [`gp_exp_quad_cov`], and `period` must be positive finite.
pub fn gp_periodic_cov<T: Real>(
    x: &[Vec<T>],
    sigma: T,
    length_scale: T,
    period: T,
) -> Result<Matrix<T>> {
    periodic(x, None, sigma, length_scale, period)
}

/// Periodic cross covariance between `x1` and `x2`.
///
/// # Errors
///
/// As [`gp_periodic_cov`].
pub fn gp_periodic_cov_cross<T: Real>(
    x1: &[Vec<T>],
    x2: &[Vec<T>],
    sigma: T,
    length_scale: T,
    period: T,
) -> Result<Matrix<T>> {
    periodic(x1, Some(x2), sigma, length_scale, period)
}

fn dot_prod<T: Real>(x1: &[Vec<T>], x2: Option<&[Vec<T>]>, sigma: T) -> Result<Matrix<T>> {
    const F: &str = "gp_dot_prod_cov";
    let s = sigma.value();
    if !(s >= 0.0 && s.is_finite()) {
        return Err(Error::parameter(F, "sigma", s, "nonnegative finite"));
    }
    let s2 = sigma.square();
    build(F, x1, x2, move |a, b| {
        a.iter().zip(b).fold(s2, |acc, (&u, &v)| acc + u * v)
    })
}

/// Dot product covariance `sigma^2 + xᵀ x'` of the points `x`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] unless `sigma` is nonnegative finite and the points are not NaN;
/// [`Error::DimensionMismatch`] for points of different lengths.
pub fn gp_dot_prod_cov<T: Real>(x: &[Vec<T>], sigma: T) -> Result<Matrix<T>> {
    dot_prod(x, None, sigma)
}

/// Dot product cross covariance between `x1` and `x2`.
///
/// # Errors
///
/// As [`gp_dot_prod_cov`].
pub fn gp_dot_prod_cov_cross<T: Real>(x1: &[Vec<T>], x2: &[Vec<T>], sigma: T) -> Result<Matrix<T>> {
    dot_prod(x1, Some(x2), sigma)
}
