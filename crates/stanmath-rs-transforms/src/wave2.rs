//! Remaining constraining transforms of Stan Math: the identity, positive, probability, and
//! correlation scalars; the `cholesky_corr_*` and `cholesky_factor_*` names; the covariance matrix
//! transform through canonical partial correlations and standard deviations
//! (`cov_matrix_constrain_lkj`); and the helpers behind it (`read_*`, `factor_*`, `make_nu`).
//!
//! Canonical partial correlations (CPCs) follow Lewandowski, Kurowicka and Joe (2009). Every
//! function here reads and writes them in the same order as [`crate::corr_matrix_constrain`]:
//! column-major order of the strict lower triangle of the Cholesky factor (column 0 first).

#![allow(non_snake_case)] // Stan's names read_corr_L, read_cov_L, factor_U

use stanmath_rs_autodiff::Real;
use stanmath_rs_autodiff::functions::{
    inv_logit, log_inv_logit, log_sum_exp, log1m, log1m_inv_logit, logit,
};
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::{cholesky_decompose, multiply_lower_tri_self_transpose};

use crate::matrix::{
    cholesky_factor_corr_constrain, cholesky_factor_corr_constrain_lp, cholesky_factor_corr_free,
    cholesky_factor_cov_constrain, cholesky_factor_cov_constrain_lp, cholesky_factor_cov_free,
};

fn size_error(function: &'static str, got: usize, want: usize) -> Error {
    Error::dimension(
        function,
        format!("expected {want} unconstrained values, got {got}"),
    )
}

/// The identity transform, for unconstrained parameters.
pub fn identity_constrain<T>(x: T) -> T {
    x
}

/// Inverse of [`identity_constrain`].
pub fn identity_free<T>(y: T) -> T {
    y
}

/// Positive scalar `x = exp(y)`.
pub fn positive_constrain<T: Real>(y: T) -> T {
    y.exp()
}

/// [`positive_constrain`] adding `ln |dx/dy| = y` to `lp`.
pub fn positive_constrain_lp<T: Real>(y: T, lp: &mut T) -> T {
    *lp += y;
    y.exp()
}

/// Inverse of [`positive_constrain`]: `y = ln x`. Requires `x > 0`.
pub fn positive_free<T: Real>(x: T) -> Result<T> {
    if !(x.value() > 0.0) {
        return Err(Error::parameter(
            "positive_free",
            "x",
            x.value(),
            "positive",
        ));
    }
    Ok(x.ln())
}

/// Probability `x = inv_logit(y)` in `(0, 1)`.
pub fn prob_constrain<T: Real>(y: T) -> T {
    inv_logit(y)
}

/// [`prob_constrain`] adding `ln inv_logit(y) + ln(1 - inv_logit(y))` to `lp`.
pub fn prob_constrain_lp<T: Real>(y: T, lp: &mut T) -> T {
    *lp += log_inv_logit(y) + log1m_inv_logit(y);
    inv_logit(y)
}

/// Inverse of [`prob_constrain`]: `y = logit(x)`. Requires `0 <= x <= 1`.
pub fn prob_free<T: Real>(x: T) -> Result<T> {
    if !(x.value() >= 0.0 && x.value() <= 1.0) {
        return Err(Error::parameter("prob_free", "x", x.value(), "in [0, 1]"));
    }
    Ok(logit(x))
}

/// Correlation `x = tanh(y)` in `(-1, 1)`.
pub fn corr_constrain<T: Real>(y: T) -> T {
    y.tanh()
}

/// [`corr_constrain`] adding `ln(1 - tanh(y)^2)` to `lp`, evaluated as
/// `2 ln 2 - 2 ln(exp(y) + exp(-y))` so that it stays finite for large `|y|`.
pub fn corr_constrain_lp<T: Real>(y: T, lp: &mut T) -> T {
    *lp += log_sum_exp(y, -y) * -2.0 + core::f64::consts::LN_2 * 2.0;
    y.tanh()
}

/// Inverse of [`corr_constrain`]: `y = atanh(x)`. Requires `-1 <= x <= 1`.
pub fn corr_free<T: Real>(x: T) -> Result<T> {
    if !(x.value().abs() <= 1.0) {
        return Err(Error::parameter("corr_free", "x", x.value(), "in [-1, 1]"));
    }
    Ok(x.atanh())
}

/// Stan's name for [`cholesky_factor_corr_constrain`].
pub fn cholesky_corr_constrain<T: Real>(y: &[T], k: usize) -> Result<Matrix<T>> {
    cholesky_factor_corr_constrain(y, k)
}

/// Stan's name for [`cholesky_factor_corr_constrain_lp`].
pub fn cholesky_corr_constrain_lp<T: Real>(y: &[T], k: usize, lp: &mut T) -> Result<Matrix<T>> {
    cholesky_factor_corr_constrain_lp(y, k, lp)
}

/// Stan's name for [`cholesky_factor_corr_free`].
pub fn cholesky_corr_free<T: Real>(l: &Matrix<T>) -> Result<Vec<T>> {
    cholesky_factor_corr_free(l)
}

/// Stan's name for [`cholesky_factor_cov_constrain`] (an `m` by `n` factor, `m >= n`).
pub fn cholesky_factor_constrain<T: Real>(x: &[T], m: usize, n: usize) -> Result<Matrix<T>> {
    cholesky_factor_cov_constrain(x, m, n)
}

/// Stan's name for [`cholesky_factor_cov_constrain_lp`].
pub fn cholesky_factor_constrain_lp<T: Real>(
    x: &[T],
    m: usize,
    n: usize,
    lp: &mut T,
) -> Result<Matrix<T>> {
    cholesky_factor_cov_constrain_lp(x, m, n, lp)
}

/// Stan's name for [`cholesky_factor_cov_free`].
pub fn cholesky_factor_free<T: Real>(l: &Matrix<T>) -> Result<Vec<T>> {
    cholesky_factor_cov_free(l)
}

/// Cholesky factor of a `k` by `k` correlation matrix from CPCs in `[-1, 1]`. When `lp` is given
/// it receives the log Jacobian of the map from the CPCs to the strict lower triangle of the
/// correlation matrix, `sum (k - j - 2) / 2 ln(1 - z(i, j)^2)` over `i > j` (0-based column `j`).
fn corr_cholesky<T: Real>(
    function: &'static str,
    cpcs: &[T],
    k: usize,
    mut lp: Option<&mut T>,
) -> Result<Matrix<T>> {
    let want = k * k.saturating_sub(1) / 2;
    if cpcs.len() != want {
        return Err(size_error(function, cpcs.len(), want));
    }
    if let Some(z) = cpcs.iter().find(|z| !(z.value().abs() <= 1.0)) {
        return Err(Error::parameter(function, "CPCs", z.value(), "in [-1, 1]"));
    }
    let mut l = Matrix::from_fn(k, k, |_, _| T::zero());
    let mut remaining = vec![T::one(); k];
    let mut pos = 0;
    for j in 0..k {
        l[(j, j)] = remaining[j].sqrt();
        // The last CPC column has power zero; skipping it keeps `z = ±1` from adding NaN.
        let power = (k - j).saturating_sub(2) as f64 * 0.5;
        let mut column_lp = if power > 0.0 { lp.as_deref_mut() } else { None };
        for i in j + 1..k {
            let z = cpcs[pos];
            pos += 1;
            if let Some(lp) = column_lp.as_deref_mut() {
                *lp += log1m(z.square()) * power;
            }
            l[(i, j)] = z * remaining[i].sqrt();
            remaining[i] *= -z.square() + 1.0;
        }
    }
    Ok(l)
}

/// Cholesky factor of a `k` by `k` correlation matrix from `k (k - 1) / 2` canonical partial
/// correlations in `[-1, 1]`, in the order of [`crate::corr_matrix_constrain`].
pub fn read_corr_L<T: Real>(cpcs: &[T], k: usize) -> Result<Matrix<T>> {
    corr_cholesky("read_corr_L", cpcs, k, None)
}

/// [`read_corr_L`] adding to `lp` the log Jacobian of the map from the CPCs to the strict lower
/// triangle of the correlation matrix: `sum (k - j - 2) / 2 ln(1 - z(i, j)^2)` (0-based column
/// `j`).
pub fn read_corr_L_lp<T: Real>(cpcs: &[T], k: usize, lp: &mut T) -> Result<Matrix<T>> {
    corr_cholesky("read_corr_L", cpcs, k, Some(lp))
}

/// Correlation matrix `L Lᵀ` with `L` from [`read_corr_L`].
pub fn read_corr_matrix<T: Real>(cpcs: &[T], k: usize) -> Result<Matrix<T>> {
    multiply_lower_tri_self_transpose(&corr_cholesky("read_corr_matrix", cpcs, k, None)?)
}

/// [`read_corr_matrix`] adding the log Jacobian of [`read_corr_L_lp`] to `lp`.
pub fn read_corr_matrix_lp<T: Real>(cpcs: &[T], k: usize, lp: &mut T) -> Result<Matrix<T>> {
    multiply_lower_tri_self_transpose(&corr_cholesky("read_corr_matrix", cpcs, k, Some(lp))?)
}

/// `diag(sds) L` with `L` from the CPCs. When `lp` is given it receives the log Jacobian of the map
/// from `(CPCs, sds)` to the lower triangle of the covariance matrix `Σ = D R D`: the CPC term of
/// [`read_corr_L_lp`] plus `k ln 2 + k sum ln sds`. (`Σ_ii = sd_i^2` contributes `ln 2 + ln sd_i`
/// and `Σ_ij = sd_i sd_j R_ij` contributes `ln sd_i + ln sd_j`; the Jacobian is block triangular.)
fn cov_cholesky<T: Real>(
    function: &'static str,
    cpcs: &[T],
    sds: &[T],
    mut lp: Option<&mut T>,
) -> Result<Matrix<T>> {
    if let Some(s) = sds.iter().find(|s| !(s.value() > 0.0)) {
        return Err(Error::parameter(function, "sds", s.value(), "positive"));
    }
    let k = sds.len();
    let mut l = corr_cholesky(function, cpcs, k, lp.as_deref_mut())?;
    if let Some(lp) = lp {
        let log_sds: T = sds.iter().map(|s| s.ln()).sum();
        *lp += log_sds * k as f64 + core::f64::consts::LN_2 * k as f64;
    }
    for i in 0..k {
        for j in 0..=i {
            l[(i, j)] *= sds[i];
        }
    }
    Ok(l)
}

/// Cholesky factor `diag(sds) L` of the covariance matrix with correlations from the CPCs (as
/// [`read_corr_L`]) and standard deviations `sds`. The size is `sds.len()`.
pub fn read_cov_L<T: Real>(cpcs: &[T], sds: &[T]) -> Result<Matrix<T>> {
    cov_cholesky("read_cov_L", cpcs, sds, None)
}

/// [`read_cov_L`] adding to `lp` the log Jacobian of the map from `(CPCs, sds)` to the lower
/// triangle of the covariance matrix: the CPC term of [`read_corr_L_lp`] plus
/// `k ln 2 + k sum ln sds`.
pub fn read_cov_L_lp<T: Real>(cpcs: &[T], sds: &[T], lp: &mut T) -> Result<Matrix<T>> {
    cov_cholesky("read_cov_L", cpcs, sds, Some(lp))
}

/// Covariance matrix `D R D` with `R` from the CPCs and `D = diag(sds)`.
pub fn read_cov_matrix<T: Real>(cpcs: &[T], sds: &[T]) -> Result<Matrix<T>> {
    multiply_lower_tri_self_transpose(&cov_cholesky("read_cov_matrix", cpcs, sds, None)?)
}

/// [`read_cov_matrix`] adding the log Jacobian of [`read_cov_L_lp`] to `lp`.
pub fn read_cov_matrix_lp<T: Real>(cpcs: &[T], sds: &[T], lp: &mut T) -> Result<Matrix<T>> {
    multiply_lower_tri_self_transpose(&cov_cholesky("read_cov_matrix", cpcs, sds, Some(lp))?)
}

/// `atanh` of the CPCs of the correlation matrix whose Cholesky factor is `l`, in the order of
/// [`read_corr_L`].
fn unconstrained_cpcs<T: Real>(l: &Matrix<T>) -> Vec<T> {
    let k = l.nrows();
    let mut remaining = vec![T::one(); k];
    let mut y = Vec::with_capacity(k * k.saturating_sub(1) / 2);
    for j in 0..k {
        for i in j + 1..k {
            let z = l[(i, j)] / remaining[i].sqrt();
            remaining[i] -= l[(i, j)].square();
            y.push(z.atanh());
        }
    }
    y
}

/// Unconstrained CPCs (`atanh` of the canonical partial correlations) of the correlation matrix
/// `Uᵀ U`, where `U` is its upper triangular Cholesky factor. Inverse of `tanh` followed by
/// [`read_corr_L`] (up to the transpose).
pub fn factor_U<T: Real>(u: &Matrix<T>) -> Result<Vec<T>> {
    if !u.is_square() {
        return Err(Error::dimension("factor_U", "expected a square matrix"));
    }
    Ok(unconstrained_cpcs(&u.transpose()))
}

/// Splits a symmetric positive definite matrix `Σ = D R D` into unconstrained CPCs of `R`
/// (`atanh` scale, order of [`read_corr_L`]) and log standard deviations `ln sqrt(Σ_ii)`.
pub fn factor_cov_matrix<T: Real>(sigma: &Matrix<T>) -> Result<(Vec<T>, Vec<T>)> {
    const F: &str = "factor_cov_matrix";
    if !sigma.is_square() {
        return Err(Error::dimension(F, "expected a square matrix"));
    }
    let k = sigma.nrows();
    if let Some(i) = (0..k).find(|&i| !(sigma[(i, i)].value() > 0.0)) {
        return Err(Error::parameter(
            F,
            "Sigma",
            sigma[(i, i)].value(),
            "with positive diagonal",
        ));
    }
    let sds: Vec<T> = (0..k).map(|i| sigma[(i, i)].sqrt()).collect();
    let r = Matrix::from_fn(k, k, |i, j| sigma[(i, j)] / (sds[i] * sds[j]));
    let cpcs = unconstrained_cpcs(&cholesky_decompose(&r)?);
    Ok((cpcs, sds.iter().map(|s| s.ln()).collect()))
}

/// Shape parameters of the CPCs under an LKJ(`eta`) distribution on `k` by `k` correlation
/// matrices: the CPC in 0-based column `j` follows a Beta(`nu`, `nu`) distribution stretched to
/// `(-1, 1)` with `nu = eta + (k - 2 - j) / 2`. One entry per CPC, in the order of
/// [`read_corr_L`]. Requires `eta > 0`.
pub fn make_nu<T: Real>(eta: T, k: usize) -> Result<Vec<T>> {
    if !(eta.value() > 0.0) {
        return Err(Error::parameter("make_nu", "eta", eta.value(), "positive"));
    }
    let mut nu = Vec::with_capacity(k * k.saturating_sub(1) / 2);
    for j in 0..k {
        for _ in j + 1..k {
            nu.push(eta + (k - 2 - j) as f64 * 0.5);
        }
    }
    Ok(nu)
}

fn split_lkj<'a, T>(function: &'static str, x: &'a [T], k: usize) -> Result<(&'a [T], &'a [T])> {
    let m = k * k.saturating_sub(1) / 2;
    if x.len() != m + k {
        return Err(size_error(function, x.len(), m + k));
    }
    Ok(x.split_at(m))
}

/// Covariance matrix of size `k` from `k (k - 1) / 2 + k` unconstrained values: first the CPCs on
/// the `atanh` scale (order of [`read_corr_L`]), then the log standard deviations. The result is
/// [`read_cov_matrix`] of `tanh` of the first part and `exp` of the second.
pub fn cov_matrix_constrain_lkj<T: Real>(x: &[T], k: usize) -> Result<Matrix<T>> {
    let (y, u) = split_lkj("cov_matrix_constrain_lkj", x, k)?;
    let cpcs: Vec<T> = y.iter().map(|&v| corr_constrain(v)).collect();
    let sds: Vec<T> = u.iter().map(|&v| positive_constrain(v)).collect();
    read_cov_matrix(&cpcs, &sds)
}

/// [`cov_matrix_constrain_lkj`] adding to `lp` the log Jacobian with respect to the lower
/// triangle of the covariance matrix: [`corr_constrain_lp`] and [`positive_constrain_lp`] terms
/// plus the [`read_cov_matrix_lp`] term.
pub fn cov_matrix_constrain_lkj_lp<T: Real>(x: &[T], k: usize, lp: &mut T) -> Result<Matrix<T>> {
    let (y, u) = split_lkj("cov_matrix_constrain_lkj", x, k)?;
    let cpcs: Vec<T> = y.iter().map(|&v| corr_constrain_lp(v, lp)).collect();
    let sds: Vec<T> = u.iter().map(|&v| positive_constrain_lp(v, lp)).collect();
    read_cov_matrix_lp(&cpcs, &sds, lp)
}

/// Inverse of [`cov_matrix_constrain_lkj`]. Requires a symmetric positive definite matrix.
pub fn cov_matrix_free_lkj<T: Real>(sigma: &Matrix<T>) -> Result<Vec<T>> {
    let (mut x, log_sds) = factor_cov_matrix(sigma)?;
    x.extend(log_sds);
    Ok(x)
}
