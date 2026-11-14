//! Matrix functions: exponential, integer powers, and the generalized inverse.
//!
//! `matrix_exp` uses the degree 13 Padé approximant with scaling and squaring (Higham 2005,
//! "The scaling and squaring method for the matrix exponential revisited"). It is written with
//! the differentiable products and solves of this crate, so its derivatives are those of the
//! approximant, which matches the exponential to the same relative accuracy.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Matrix, Result};

use crate::basic::identity_matrix;
use crate::lu::{inverse, mdivide_left};
use crate::{add, mdivide_left_spd, multiply, require_square, scale, subtract};

/// Coefficients `b_0, ..., b_13` of the degree 13 Padé approximant to `exp`.
const PADE13: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

/// Largest 1-norm for which the degree 13 approximant meets double precision unscaled.
const THETA13: f64 = 5.371_920_351_148_152;

/// `sum_i c_i M_i` for square matrices of equal size.
fn combine<T: Real>(terms: &[(&Matrix<T>, f64)]) -> Matrix<T> {
    let n = terms[0].0.nrows();
    Matrix::from_fn(n, n, |i, j| {
        terms
            .iter()
            .fold(T::zero(), |acc, &(m, c)| acc + m[(i, j)] * c)
    })
}

/// Matrix exponential `exp(A)` of a square matrix.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input and [`Error::InvalidParameter`] for
/// non-finite entries.
pub fn matrix_exp<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    const F: &str = "matrix_exp";
    require_square(F, a)?;
    let n = a.nrows();
    if n == 0 {
        return Ok(a.clone());
    }
    let norm1 = (0..n)
        .map(|j| (0..n).map(|i| a[(i, j)].value().abs()).sum::<f64>())
        .fold(0.0, f64::max);
    if !norm1.is_finite() {
        return Err(Error::parameter(F, "A", norm1, "finite"));
    }
    let s = if norm1 > THETA13 {
        (norm1 / THETA13).log2().ceil() as i32
    } else {
        0
    };
    let a = scale(a, T::from_f64(0.5f64.powi(s)));
    let eye = identity_matrix::<T>(n);
    let a2 = multiply(&a, &a)?;
    let a4 = multiply(&a2, &a2)?;
    let a6 = multiply(&a4, &a2)?;
    let b = &PADE13;
    let u_high = multiply(&a6, &combine(&[(&a6, b[13]), (&a4, b[11]), (&a2, b[9])]))?;
    let u_low = combine(&[(&a6, b[7]), (&a4, b[5]), (&a2, b[3]), (&eye, b[1])]);
    let u = multiply(&a, &add(&u_high, &u_low)?)?;
    let v_high = multiply(&a6, &combine(&[(&a6, b[12]), (&a4, b[10]), (&a2, b[8])]))?;
    let v_low = combine(&[(&a6, b[6]), (&a4, b[4]), (&a2, b[2]), (&eye, b[0])]);
    let v = add(&v_high, &v_low)?;
    let mut r = mdivide_left(&subtract(&v, &u)?, &add(&v, &u)?)?;
    for _ in 0..s {
        r = multiply(&r, &r)?;
    }
    Ok(r)
}

/// `exp(A) B`.
///
/// # Errors
///
/// As [`matrix_exp`] and [`multiply`].
pub fn matrix_exp_multiply<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    multiply(&matrix_exp(a)?, b)
}

/// `exp(t A) B` for a scalar `t`.
///
/// # Errors
///
/// As [`matrix_exp`] and [`multiply`].
pub fn scale_matrix_exp_multiply<T: Real>(t: T, a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    multiply(&matrix_exp(&scale(a, t))?, b)
}

/// `A^n` for a square matrix and a nonnegative integer `n`, by repeated squaring.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input and [`Error::InvalidParameter`] for a
/// negative `n`.
pub fn matrix_power<T: Real>(a: &Matrix<T>, n: i32) -> Result<Matrix<T>> {
    const F: &str = "matrix_power";
    require_square(F, a)?;
    if n < 0 {
        return Err(Error::parameter(F, "n", f64::from(n), "nonnegative"));
    }
    let mut result = identity_matrix::<T>(a.nrows());
    let mut base = a.clone();
    let mut e = n;
    while e > 0 {
        if e & 1 == 1 {
            result = multiply(&result, &base)?;
        }
        e >>= 1;
        if e > 0 {
            base = multiply(&base, &base)?;
        }
    }
    Ok(result)
}

/// Moore-Penrose generalized inverse of a matrix of full rank.
///
/// A square matrix gives its inverse, a wide matrix (`m < n`) gives `Aᵀ (A Aᵀ)⁻¹`, and a tall
/// matrix gives `(Aᵀ A)⁻¹ Aᵀ`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] or [`Error::NotPositiveDefinite`] when `A` does not have full rank.
pub fn generalized_inverse<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    let (m, n) = a.shape();
    let at = a.transpose();
    if m == n {
        inverse(a)
    } else if m < n {
        Ok(mdivide_left_spd(&multiply(a, &at)?, a)?.transpose())
    } else {
        mdivide_left_spd(&multiply(&at, a)?, &at)
    }
}
