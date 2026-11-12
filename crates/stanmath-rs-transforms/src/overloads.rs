//! Scalar bound transforms with one bound per element.
//!
//! Stan accepts, besides a scalar bound, a container of bounds with the same shape as the
//! constrained container. The `_vec` forms take one bound per element of a vector, row vector,
//! or array; for a matrix pass its column-major [`Matrix::as_slice`](stanmath_rs_core::Matrix)
//! for both the values and the bounds. A scalar bound on a container is the scalar transform
//! mapped over the elements; for a mixed pair (say vector lower and scalar upper bounds) repeat
//! the scalar into a slice. The `_lp` forms add every element's Jacobian term to one `lp`.

use stanmath_rs_autodiff::{Lift, Real};
use stanmath_rs_core::{Error, Result};

use crate::scalar::{
    lb_constrain, lb_constrain_lp, lb_free, lub_constrain, lub_constrain_lp, lub_free,
    offset_multiplier_constrain, offset_multiplier_constrain_lp, offset_multiplier_free,
    ub_constrain, ub_constrain_lp, ub_free,
};

fn check_len(function: &'static str, name: &str, expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::dimension(
            function,
            format!("{name} has length {got}, expected {expected}"),
        ))
    }
}

/// [`lb_constrain`] with lower bound `lb[i]` for element `i`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `lb` and `y` differ in length.
pub fn lb_constrain_vec<T: Real, L: Lift<T> + Copy>(y: &[T], lb: &[L]) -> Result<Vec<T>> {
    check_len("lb_constrain", "lb", y.len(), lb.len())?;
    Ok(y.iter()
        .zip(lb)
        .map(|(&y, &l)| lb_constrain(y, l))
        .collect())
}

/// [`lb_constrain_vec`] adding every element's Jacobian term to `lp`.
///
/// # Errors
///
/// As [`lb_constrain_vec`].
pub fn lb_constrain_vec_lp<T: Real, L: Lift<T> + Copy>(
    y: &[T],
    lb: &[L],
    lp: &mut T,
) -> Result<Vec<T>> {
    check_len("lb_constrain", "lb", y.len(), lb.len())?;
    Ok(y.iter()
        .zip(lb)
        .map(|(&y, &l)| lb_constrain_lp(y, l, lp))
        .collect())
}

/// Inverse of [`lb_constrain_vec`].
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the lengths differ, and as [`lb_free`].
pub fn lb_free_vec<T: Real, L: Lift<T> + Copy>(x: &[T], lb: &[L]) -> Result<Vec<T>> {
    check_len("lb_free", "lb", x.len(), lb.len())?;
    x.iter().zip(lb).map(|(&x, &l)| lb_free(x, l)).collect()
}

/// [`ub_constrain`] with upper bound `ub[i]` for element `i`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `ub` and `y` differ in length.
pub fn ub_constrain_vec<T: Real, U: Lift<T> + Copy>(y: &[T], ub: &[U]) -> Result<Vec<T>> {
    check_len("ub_constrain", "ub", y.len(), ub.len())?;
    Ok(y.iter()
        .zip(ub)
        .map(|(&y, &u)| ub_constrain(y, u))
        .collect())
}

/// [`ub_constrain_vec`] adding every element's Jacobian term to `lp`.
///
/// # Errors
///
/// As [`ub_constrain_vec`].
pub fn ub_constrain_vec_lp<T: Real, U: Lift<T> + Copy>(
    y: &[T],
    ub: &[U],
    lp: &mut T,
) -> Result<Vec<T>> {
    check_len("ub_constrain", "ub", y.len(), ub.len())?;
    Ok(y.iter()
        .zip(ub)
        .map(|(&y, &u)| ub_constrain_lp(y, u, lp))
        .collect())
}

/// Inverse of [`ub_constrain_vec`].
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the lengths differ, and as [`ub_free`].
pub fn ub_free_vec<T: Real, U: Lift<T> + Copy>(x: &[T], ub: &[U]) -> Result<Vec<T>> {
    check_len("ub_free", "ub", x.len(), ub.len())?;
    x.iter().zip(ub).map(|(&x, &u)| ub_free(x, u)).collect()
}

/// [`lub_constrain`] with bounds `lb[i]` and `ub[i]` for element `i`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when a bound slice differs in length from `y`, and as
/// [`lub_constrain`].
pub fn lub_constrain_vec<T: Real, L: Lift<T> + Copy, U: Lift<T> + Copy>(
    y: &[T],
    lb: &[L],
    ub: &[U],
) -> Result<Vec<T>> {
    check_len("lub_constrain", "lb", y.len(), lb.len())?;
    check_len("lub_constrain", "ub", y.len(), ub.len())?;
    y.iter()
        .zip(lb.iter().zip(ub))
        .map(|(&y, (&l, &u))| lub_constrain(y, l, u))
        .collect()
}

/// [`lub_constrain_vec`] adding every element's Jacobian term to `lp`.
///
/// # Errors
///
/// As [`lub_constrain_vec`].
pub fn lub_constrain_vec_lp<T: Real, L: Lift<T> + Copy, U: Lift<T> + Copy>(
    y: &[T],
    lb: &[L],
    ub: &[U],
    lp: &mut T,
) -> Result<Vec<T>> {
    check_len("lub_constrain", "lb", y.len(), lb.len())?;
    check_len("lub_constrain", "ub", y.len(), ub.len())?;
    y.iter()
        .zip(lb.iter().zip(ub))
        .map(|(&y, (&l, &u))| lub_constrain_lp(y, l, u, lp))
        .collect()
}

/// Inverse of [`lub_constrain_vec`].
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the lengths differ, and as [`lub_free`].
pub fn lub_free_vec<T: Real, L: Lift<T> + Copy, U: Lift<T> + Copy>(
    x: &[T],
    lb: &[L],
    ub: &[U],
) -> Result<Vec<T>> {
    check_len("lub_free", "lb", x.len(), lb.len())?;
    check_len("lub_free", "ub", x.len(), ub.len())?;
    x.iter()
        .zip(lb.iter().zip(ub))
        .map(|(&x, (&l, &u))| lub_free(x, l, u))
        .collect()
}

/// [`offset_multiplier_constrain`] with offset `mu[i]` and multiplier `sigma[i]` for element
/// `i`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `mu` or `sigma` differs in length from `y`, and as
/// [`offset_multiplier_constrain`].
pub fn offset_multiplier_constrain_vec<T: Real, M: Lift<T> + Copy, S: Lift<T> + Copy>(
    y: &[T],
    mu: &[M],
    sigma: &[S],
) -> Result<Vec<T>> {
    check_len("offset_multiplier_constrain", "mu", y.len(), mu.len())?;
    check_len("offset_multiplier_constrain", "sigma", y.len(), sigma.len())?;
    y.iter()
        .zip(mu.iter().zip(sigma))
        .map(|(&y, (&m, &s))| offset_multiplier_constrain(y, m, s))
        .collect()
}

/// [`offset_multiplier_constrain_vec`] adding every element's Jacobian term to `lp`.
///
/// # Errors
///
/// As [`offset_multiplier_constrain_vec`].
pub fn offset_multiplier_constrain_vec_lp<T: Real, M: Lift<T> + Copy, S: Lift<T> + Copy>(
    y: &[T],
    mu: &[M],
    sigma: &[S],
    lp: &mut T,
) -> Result<Vec<T>> {
    check_len("offset_multiplier_constrain", "mu", y.len(), mu.len())?;
    check_len("offset_multiplier_constrain", "sigma", y.len(), sigma.len())?;
    y.iter()
        .zip(mu.iter().zip(sigma))
        .map(|(&y, (&m, &s))| offset_multiplier_constrain_lp(y, m, s, lp))
        .collect()
}

/// Inverse of [`offset_multiplier_constrain_vec`].
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the lengths differ, and as [`offset_multiplier_free`].
pub fn offset_multiplier_free_vec<T: Real, M: Lift<T> + Copy, S: Lift<T> + Copy>(
    x: &[T],
    mu: &[M],
    sigma: &[S],
) -> Result<Vec<T>> {
    check_len("offset_multiplier_free", "mu", x.len(), mu.len())?;
    check_len("offset_multiplier_free", "sigma", x.len(), sigma.len())?;
    x.iter()
        .zip(mu.iter().zip(sigma))
        .map(|(&x, (&m, &s))| offset_multiplier_free(x, m, s))
        .collect()
}
