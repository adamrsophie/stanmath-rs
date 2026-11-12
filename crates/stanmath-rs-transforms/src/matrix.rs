//! Matrix constraints: Cholesky factors of correlation and covariance matrices, correlation and
//! covariance matrices, sum-to-zero matrices, and column and row stochastic matrices.

use stanmath_rs_autodiff::Real;
use stanmath_rs_autodiff::functions::log1m;
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::{cholesky_decompose, multiply_lower_tri_self_transpose};

use crate::vector::{
    CONSTRAINT_TOLERANCE, simplex_constrain, simplex_constrain_lp, simplex_free,
    sum_to_zero_constrain, sum_to_zero_free,
};

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

/// Cholesky factor of a correlation matrix from canonical partial correlations `z = tanh(y)` taken
/// in column-major order of the strict lower triangle (see [`corr_matrix_constrain`]).
///
/// With `r_i = prod_{m < j} (1 - z(i, m)^2)`, the factor is `L(j, j) = sqrt(r_j)` and
/// `L(i, j) = z(i, j) sqrt(r_i)` for `i > j`. When `lp` is given it receives the log Jacobian of
/// `y` to the strict lower triangle of `L Lᵀ`: `ln(1 - z^2)` for each `tanh`, plus
/// `(k - j - 2) / 2 ln(1 - z(i, j)^2)` for the map from partial correlations to correlations.
/// The second term follows because, ordering the correlations column by column, the Jacobian is
/// triangular with diagonal `dR(i, j)/dz(i, j) = L(j, j) sqrt(r_i)`, a product of square roots of
/// `1 - z^2` over earlier columns; each `z(a, b)` then appears `(a - b - 1) + (k - 1 - a)` times.
fn corr_cholesky_by_columns<T: Real>(
    y: &[T],
    k: usize,
    mut lp: Option<&mut T>,
) -> Result<Matrix<T>> {
    const F: &str = "corr_matrix_constrain";
    let want = k * k.saturating_sub(1) / 2;
    if y.len() != want {
        return Err(size_error(F, y.len(), want));
    }
    let mut l = Matrix::from_fn(k, k, |_, _| T::zero());
    let mut remaining = vec![T::one(); k];
    let mut pos = 0;
    for j in 0..k {
        l[(j, j)] = remaining[j].sqrt();
        for i in j + 1..k {
            let z = y[pos].tanh();
            pos += 1;
            if let Some(lp) = lp.as_deref_mut() {
                *lp += log1m(z.square()) * ((k - j - 2) as f64 * 0.5 + 1.0);
            }
            l[(i, j)] = z * remaining[i].sqrt();
            remaining[i] *= -z.square() + 1.0;
        }
    }
    Ok(l)
}

/// Correlation matrix of size `k` from `k (k - 1) / 2` unconstrained values, as Stan's
/// `corr_matrix_constrain`.
///
/// The values become canonical partial correlations `z = tanh(y)`, which fill the strict lower
/// triangle of a Cholesky factor `L` in column-major order: the first `k - 1` values are column 0
/// (rows 1 to `k - 1`), the next `k - 2` are column 1, and so on. This is the same as filling the
/// strict upper triangle of the partial correlation array row by row. It differs from
/// [`cholesky_factor_corr_constrain`], which fills the lower triangle row by row. The result is
/// `L Lᵀ`.
pub fn corr_matrix_constrain<T: Real>(y: &[T], k: usize) -> Result<Matrix<T>> {
    multiply_lower_tri_self_transpose(&corr_cholesky_by_columns(y, k, None)?)
}

/// [`corr_matrix_constrain`] adding to `lp` the log Jacobian with respect to the strict lower
/// triangle of the correlation matrix: `sum (1 + (k - j - 2) / 2) ln(1 - z(i, j)^2)` over
/// `i > j` (0-based column `j`).
pub fn corr_matrix_constrain_lp<T: Real>(y: &[T], k: usize, lp: &mut T) -> Result<Matrix<T>> {
    multiply_lower_tri_self_transpose(&corr_cholesky_by_columns(y, k, Some(lp))?)
}

/// Inverse of [`corr_matrix_constrain`]. Requires a symmetric positive definite matrix with unit
/// diagonal (within `1e-8`).
pub fn corr_matrix_free<T: Real>(sigma: &Matrix<T>) -> Result<Vec<T>> {
    const F: &str = "corr_matrix_free";
    if !sigma.is_square() {
        return Err(Error::dimension(F, "expected a square matrix"));
    }
    let k = sigma.nrows();
    for i in 0..k {
        let d = sigma[(i, i)].value();
        if !((d - 1.0).abs() <= CONSTRAINT_TOLERANCE) {
            return Err(Error::parameter(
                F,
                "Sigma",
                d,
                "a correlation matrix with unit diagonal",
            ));
        }
    }
    let l = cholesky_decompose(sigma)?;
    let mut remaining = vec![T::one(); k];
    let mut y = Vec::with_capacity(k * k.saturating_sub(1) / 2);
    for j in 0..k {
        for i in j + 1..k {
            let z = l[(i, j)] / remaining[i].sqrt();
            remaining[i] -= l[(i, j)].square();
            y.push(z.atanh());
        }
    }
    Ok(y)
}

/// Sum-to-zero matrix of size `(n + 1)` by `(m + 1)` from an `n` by `m` unconstrained matrix: every
/// row and every column sums to zero.
///
/// The vector transform [`sum_to_zero_constrain`] is applied to every column and then to every row,
/// so the result is `V_n Y V_mᵀ` with the orthonormal bases `V` of the vector transform. The map is
/// an isometry onto its image and adds no Jacobian term.
pub fn sum_to_zero_matrix_constrain<T: Real>(y: &Matrix<T>) -> Matrix<T> {
    let (n, m) = y.shape();
    let cols: Vec<T> = (0..m)
        .flat_map(|j| sum_to_zero_constrain(y.column(j)))
        .collect();
    let w = Matrix::from_column_major(n + 1, m, cols).transpose();
    let rows: Vec<T> = (0..=n)
        .flat_map(|i| sum_to_zero_constrain(w.column(i)))
        .collect();
    Matrix::from_column_major(m + 1, n + 1, rows).transpose()
}

/// [`sum_to_zero_matrix_constrain`]; the transform is an isometry, so `lp` is unchanged.
pub fn sum_to_zero_matrix_constrain_lp<T: Real>(y: &Matrix<T>, _lp: &mut T) -> Matrix<T> {
    sum_to_zero_matrix_constrain(y)
}

/// Inverse of [`sum_to_zero_matrix_constrain`]. Requires at least one row and one column, and
/// every row and column sum within `1e-8` of zero.
pub fn sum_to_zero_matrix_free<T: Real>(x: &Matrix<T>) -> Result<Matrix<T>> {
    const F: &str = "sum_to_zero_matrix_free";
    let (n, m) = x.shape();
    if n == 0 || m == 0 {
        return Err(Error::dimension(
            F,
            "expected at least one row and one column",
        ));
    }
    let col_sums = (0..m).map(|j| x.column(j).iter().map(Real::value).sum::<f64>());
    let row_sums = (0..n).map(|i| x.row(i).map(Real::value).sum::<f64>());
    if let Some(total) = col_sums
        .chain(row_sums)
        .find(|s| !(s.abs() <= CONSTRAINT_TOLERANCE))
    {
        return Err(Error::parameter(
            F,
            "x",
            total,
            "rows and columns summing to zero",
        ));
    }
    let xt = x.transpose();
    let mut w = Vec::with_capacity(n * (m - 1));
    for i in 0..n {
        w.extend(sum_to_zero_free(xt.column(i))?);
    }
    let w = Matrix::from_column_major(m - 1, n, w).transpose();
    let mut y = Vec::with_capacity((n - 1) * (m - 1));
    for j in 0..m - 1 {
        y.extend(sum_to_zero_free(w.column(j))?);
    }
    Ok(Matrix::from_column_major(n - 1, m - 1, y))
}

/// Column stochastic matrix of size `(n + 1)` by `m` from an `n` by `m` unconstrained matrix:
/// [`simplex_constrain`] applied to every column.
pub fn stochastic_column_constrain<T: Real>(y: &Matrix<T>) -> Matrix<T> {
    let (n, m) = y.shape();
    let data: Vec<T> = (0..m)
        .flat_map(|j| simplex_constrain(y.column(j)))
        .collect();
    Matrix::from_column_major(n + 1, m, data)
}

/// [`stochastic_column_constrain`] adding the sum of the columns' simplex log Jacobians
/// ([`simplex_constrain_lp`]) to `lp`.
pub fn stochastic_column_constrain_lp<T: Real>(y: &Matrix<T>, lp: &mut T) -> Matrix<T> {
    let (n, m) = y.shape();
    let mut data = Vec::with_capacity((n + 1) * m);
    for j in 0..m {
        data.extend(simplex_constrain_lp(y.column(j), lp));
    }
    Matrix::from_column_major(n + 1, m, data)
}

/// Inverse of [`stochastic_column_constrain`]. Requires every column to be a simplex with positive
/// entries.
pub fn stochastic_column_free<T: Real>(x: &Matrix<T>) -> Result<Matrix<T>> {
    let (n, m) = x.shape();
    let mut data = Vec::with_capacity(n.saturating_sub(1) * m);
    for j in 0..m {
        data.extend(simplex_free(x.column(j))?);
    }
    Ok(Matrix::from_column_major(n.saturating_sub(1), m, data))
}

/// Row stochastic matrix of size `n` by `(m + 1)` from an `n` by `m` unconstrained matrix:
/// [`simplex_constrain`] applied to every row.
pub fn stochastic_row_constrain<T: Real>(y: &Matrix<T>) -> Matrix<T> {
    stochastic_column_constrain(&y.transpose()).transpose()
}

/// [`stochastic_row_constrain`] adding the sum of the rows' simplex log Jacobians to `lp`.
pub fn stochastic_row_constrain_lp<T: Real>(y: &Matrix<T>, lp: &mut T) -> Matrix<T> {
    stochastic_column_constrain_lp(&y.transpose(), lp).transpose()
}

/// Inverse of [`stochastic_row_constrain`]. Requires every row to be a simplex with positive
/// entries.
pub fn stochastic_row_free<T: Real>(x: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(stochastic_column_free(&x.transpose())?.transpose())
}
