//! QR decomposition by Householder reflections.
//!
//! One generic algorithm (Golub and Van Loan, algorithm 5.2.1) serves every scalar type; the sign
//! of each reflection is chosen on the values, so all modes take the same branch. The factors are
//! normalized so that the diagonal of `R` is nonnegative, which makes them unique for a matrix of
//! full column rank.

#![allow(non_snake_case)] // Stan's names qr_Q, qr_R, qr_thin_Q, qr_thin_R

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::Matrix;

/// Full QR factors: `Q` is `m` by `m` and orthogonal, `R` is `m` by `n` and upper triangular.
fn householder<T: Real>(a: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    let (m, n) = a.shape();
    let mut r = a.clone();
    let mut q = Matrix::from_fn(m, m, |i, j| if i == j { T::one() } else { T::zero() });
    for k in 0..n.min(m.saturating_sub(1)) {
        let norm2: f64 = (k..m).map(|i| r[(i, k)].value().powi(2)).sum();
        if norm2 == 0.0 {
            continue;
        }
        let norm = (k..m).map(|i| r[(i, k)].square()).sum::<T>().sqrt();
        let alpha = if r[(k, k)].value() >= 0.0 {
            -norm
        } else {
            norm
        };
        let mut v: Vec<T> = (k..m).map(|i| r[(i, k)]).collect();
        v[0] -= alpha;
        let vnorm2: T = v.iter().map(|&x| x.square()).sum();
        if vnorm2.value() == 0.0 {
            continue;
        }
        let tau = vnorm2.recip() * 2.0;
        // R[k.., j] -= tau v (vᵀ R[k.., j])
        for j in k..n {
            let mut s = T::zero();
            for (l, &vl) in v.iter().enumerate() {
                s += vl * r[(k + l, j)];
            }
            let f = s * tau;
            for (l, &vl) in v.iter().enumerate() {
                r[(k + l, j)] -= vl * f;
            }
        }
        // Q[i, k..] -= tau (Q[i, k..] v) vᵀ
        for i in 0..m {
            let mut s = T::zero();
            for (l, &vl) in v.iter().enumerate() {
                s += q[(i, k + l)] * vl;
            }
            let f = s * tau;
            for (l, &vl) in v.iter().enumerate() {
                q[(i, k + l)] -= f * vl;
            }
        }
    }
    for i in 0..m.min(n) {
        if r[(i, i)].value() < 0.0 {
            for j in 0..n {
                r[(i, j)] = -r[(i, j)];
            }
            for l in 0..m {
                q[(l, i)] = -q[(l, i)];
            }
        }
    }
    // Entries below the diagonal of R are rounding errors.
    for j in 0..n {
        for i in j + 1..m {
            r[(i, j)] = T::zero();
        }
    }
    (q, r)
}

/// The orthogonal factor `Q` (`m` by `m`) of `A = Q R`, with the diagonal of `R` nonnegative.
pub fn qr_Q<T: Real>(a: &Matrix<T>) -> Matrix<T> {
    householder(a).0
}

/// The upper triangular factor `R` (`m` by `n`) of `A = Q R`, with a nonnegative diagonal.
pub fn qr_R<T: Real>(a: &Matrix<T>) -> Matrix<T> {
    householder(a).1
}

/// The first `min(m, n)` columns of `Q`, as in the thin factorization `A = Q R`.
pub fn qr_thin_Q<T: Real>(a: &Matrix<T>) -> Matrix<T> {
    let k = a.nrows().min(a.ncols());
    let q = householder(a).0;
    Matrix::from_fn(q.nrows(), k, |i, j| q[(i, j)])
}

/// The first `min(m, n)` rows of `R`, as in the thin factorization `A = Q R`.
pub fn qr_thin_R<T: Real>(a: &Matrix<T>) -> Matrix<T> {
    let k = a.nrows().min(a.ncols());
    let r = householder(a).1;
    Matrix::from_fn(k, r.ncols(), |i, j| r[(i, j)])
}
