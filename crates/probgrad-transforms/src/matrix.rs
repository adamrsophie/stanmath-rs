//! Matrix constraints: Cholesky factors of correlation and covariance matrices, and covariance
//! matrices.

use probgrad_autodiff::Real;
use probgrad_autodiff::functions::log1m;
use probgrad_core::{Error, Matrix, Result};
use probgrad_linalg::{cholesky_decompose, multiply_lower_tri_self_transpose};

use crate::vector::CONSTRAINT_TOLERANCE;

fn size_error(function: &'static str, got: usize, want: usize) -> Error {
    Error::dimension(
        function,
        format!("expected {want} unconstrained values, got {got}"),
    )
}

fn build_corr_cholesky<T: Real>(y: &[T], k: usize, mut lp: Option<&mut T>) -> Result<Matrix<T>> {
    const F: &str = "cholesky_factor_corr_constrain";
    let want = k * k.saturating_sub(1) / 2;
    if y.len() != want {
        return Err(size_error(F, y.len(), want));
    }
    let mut l = Matrix::from_fn(k, k, |_, _| T::zero());
    if k == 0 {
        return Ok(l);
    }
    // Canonical partial correlations z = tanh(y), with ln |dz/dy| = ln(1 - z^2).
    let z: Vec<T> = y.iter().map(|v| v.tanh()).collect();
    if let Some(lp) = lp.as_deref_mut() {
        for &zi in &z {
            *lp += log1m(zi.square());
        }
    }
    l[(0, 0)] = T::one();
    let mut pos = 0;
    for i in 1..k {
        l[(i, 0)] = z[pos];
        pos += 1;
        let mut sum_sqs = l[(i, 0)].square();
        for j in 1..i {
            if let Some(lp) = lp.as_deref_mut() {
                *lp += log1m(sum_sqs) * 0.5;
            }
            l[(i, j)] = z[pos] * (-sum_sqs + 1.0).sqrt();
            pos += 1;
            sum_sqs += l[(i, j)].square();
        }
        l[(i, i)] = (-sum_sqs + 1.0).sqrt();
    }
    Ok(l)
}

/// Cholesky factor of a `k` by `k` correlation matrix from `k (k - 1) / 2` unconstrained values,
/// through canonical partial correlations `tanh(y)`, in Stan's row-wise order.
pub fn cholesky_factor_corr_constrain<T: Real>(y: &[T], k: usize) -> Result<Matrix<T>> {
    build_corr_cholesky(y, k, None)
}

/// [`cholesky_factor_corr_constrain`] adding its log Jacobian to `lp`.
pub fn cholesky_factor_corr_constrain_lp<T: Real>(
    y: &[T],
    k: usize,
    lp: &mut T,
) -> Result<Matrix<T>> {
    build_corr_cholesky(y, k, Some(lp))
}

/// Inverse of [`cholesky_factor_corr_constrain`]. Requires a lower triangular `L` with positive
/// diagonal and unit-length rows.
pub fn cholesky_factor_corr_free<T: Real>(l: &Matrix<T>) -> Result<Vec<T>> {
    const F: &str = "cholesky_factor_corr_free";
    if !l.is_square() {
        return Err(Error::dimension(F, "expected a square matrix"));
    }
    let k = l.nrows();
    for i in 0..k {
        let norm: f64 = (0..=i).map(|j| l[(i, j)].value().powi(2)).sum();
        let upper_zero = (i + 1..k).all(|j| l[(i, j)].value() == 0.0);
        if !upper_zero
            || !(l[(i, i)].value() > 0.0)
            || !((norm - 1.0).abs() <= CONSTRAINT_TOLERANCE)
        {
            return Err(Error::parameter(
                F,
                "L",
                norm,
                "a Cholesky factor of a correlation matrix",
            ));
        }
    }
    let mut y = Vec::with_capacity(k * k.saturating_sub(1) / 2);
    for i in 1..k {
        let mut sum_sqs = l[(i, 0)].square();
        y.push(l[(i, 0)].atanh());
        for j in 1..i {
            let z = l[(i, j)] / (-sum_sqs + 1.0).sqrt();
            y.push(z.atanh());
            sum_sqs += l[(i, j)].square();
        }
    }
    Ok(y)
}

fn build_cholesky_factor<T: Real>(
    x: &[T],
    m: usize,
    n: usize,
    lp: Option<&mut T>,
) -> Result<Matrix<T>> {
    const F: &str = "cholesky_factor_cov_constrain";
    if m < n {
        return Err(Error::dimension(
            F,
            format!("{m} rows is fewer than {n} columns"),
        ));
    }
    let want = n * (n + 1) / 2 + (m - n) * n;
    if x.len() != want {
        return Err(size_error(F, x.len(), want));
    }
    let mut l = Matrix::from_fn(m, n, |_, _| T::zero());
    let mut pos = 0;
    let mut log_diag = T::zero();
    for r in 0..n {
        for c in 0..r {
            l[(r, c)] = x[pos];
            pos += 1;
        }
        log_diag += x[pos];
        l[(r, r)] = x[pos].exp();
        pos += 1;
    }
    for r in n..m {
        for c in 0..n {
            l[(r, c)] = x[pos];
            pos += 1;
        }
    }
    if let Some(lp) = lp {
        *lp += log_diag;
    }
    Ok(l)
}

/// Cholesky factor of a covariance matrix: `m` by `n` (`m >= n`), lower triangular with positive
/// diagonal, from `n (n + 1) / 2 + (m - n) n` unconstrained values (diagonal entries on the log
/// scale, row-wise order as in Stan).
pub fn cholesky_factor_cov_constrain<T: Real>(x: &[T], m: usize, n: usize) -> Result<Matrix<T>> {
    build_cholesky_factor(x, m, n, None)
}

/// [`cholesky_factor_cov_constrain`] adding the sum of the log diagonal to `lp`.
pub fn cholesky_factor_cov_constrain_lp<T: Real>(
    x: &[T],
    m: usize,
    n: usize,
    lp: &mut T,
) -> Result<Matrix<T>> {
    build_cholesky_factor(x, m, n, Some(lp))
}

/// Inverse of [`cholesky_factor_cov_constrain`].
pub fn cholesky_factor_cov_free<T: Real>(l: &Matrix<T>) -> Result<Vec<T>> {
    const F: &str = "cholesky_factor_cov_free";
    let (m, n) = l.shape();
    if m < n {
        return Err(Error::dimension(F, "fewer rows than columns"));
    }
    let mut x = Vec::new();
    for r in 0..m {
        for c in 0..n.min(r + 1) {
            if r == c {
                if !(l[(r, r)].value() > 0.0) {
                    return Err(Error::parameter(
                        F,
                        "L",
                        l[(r, r)].value(),
                        "with positive diagonal",
                    ));
                }
                x.push(l[(r, r)].ln());
            } else {
                x.push(l[(r, c)]);
            }
        }
        if (r + 1..n).any(|c| l[(r, c)].value() != 0.0) {
            return Err(Error::parameter(F, "L", f64::NAN, "lower triangular"));
        }
    }
    Ok(x)
}

/// Covariance matrix `L Lᵀ` with `L` from [`cholesky_factor_cov_constrain`] (`k` by `k`).
pub fn cov_matrix_constrain<T: Real>(x: &[T], k: usize) -> Result<Matrix<T>> {
    multiply_lower_tri_self_transpose(&build_cholesky_factor(x, k, k, None)?)
}

/// [`cov_matrix_constrain`] adding `k ln 2 + sum_i (k - i + 1) ln L_ii` (0-based `i`) to `lp`.
pub fn cov_matrix_constrain_lp<T: Real>(x: &[T], k: usize, lp: &mut T) -> Result<Matrix<T>> {
    let l = build_cholesky_factor(x, k, k, None)?;
    *lp += core::f64::consts::LN_2 * k as f64;
    for i in 0..k {
        *lp += l[(i, i)].ln() * (k - i + 1) as f64;
    }
    multiply_lower_tri_self_transpose(&l)
}

/// Inverse of [`cov_matrix_constrain`]. Requires a symmetric positive definite matrix.
pub fn cov_matrix_free<T: Real>(sigma: &Matrix<T>) -> Result<Vec<T>> {
    cholesky_factor_cov_free(&cholesky_decompose(sigma)?)
}
