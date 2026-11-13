//! Decompositions that return all their factors at once, as Stan's tuple-returning `svd`, `qr`,
//! and `qr_thin` do.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Matrix, Result};

use crate::{qr_Q, qr_R, qr_thin_Q, qr_thin_R, singular_values, svd_U, svd_V};

/// Thin singular value decomposition `A = U diag(σ) Vᵀ`, returned as `(U, σ, V)` with `σ` in
/// descending order. `U` is `m` by `k` and `V` is `n` by `k` with `k = min(m, n)`.
///
/// # Errors
///
/// As [`singular_values`].
pub fn svd<T: Real>(a: &Matrix<T>) -> Result<(Matrix<T>, Vec<T>, Matrix<T>)> {
    // ponytail: runs the eigendecomposition three times; expose the shared `svd_parts` of
    // eigen.rs to this module if the cost matters.
    Ok((svd_U(a)?, singular_values(a)?, svd_V(a)?))
}

/// Full QR decomposition `A = Q R`, returned as `(Q, R)` with `Q` orthogonal (`m` by `m`) and
/// `R` upper triangular (`m` by `n`) with a nonnegative diagonal.
pub fn qr<T: Real>(a: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    // ponytail: factors twice; return both factors from qr.rs's `householder` if the cost
    // matters.
    (qr_Q(a), qr_R(a))
}

/// Thin QR decomposition `A = Q R`, returned as `(Q, R)` with `Q` of size `m` by `k` and `R` of
/// size `k` by `n`, `k = min(m, n)`.
pub fn qr_thin<T: Real>(a: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    // ponytail: factors twice, as `qr`.
    (qr_thin_Q(a), qr_thin_R(a))
}
