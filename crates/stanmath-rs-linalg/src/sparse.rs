//! Sparse matrices in compressed sparse row (CSR) form, following Stan's `csr_*` functions.
//!
//! A CSR matrix with `m` rows is held as three arrays: the nonzero values `w` in row order, the
//! column index `v` of each value, and the row starts `u` (length `m + 1`), so that the values of
//! row `i` are `w[u[i]..u[i + 1]]`. Unlike Stan, the indices here are zero based.

use stanmath_rs_autodiff::Real;
use stanmath_rs_autodiff::functions::dot_product;
use stanmath_rs_core::{Error, Matrix, Result};

/// A sparse matrix in compressed sparse row form.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix<T> {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Nonzero values in row order (Stan's `w`).
    pub values: Vec<T>,
    /// Column index of each value, zero based (Stan's `v`, which is one based).
    pub columns: Vec<usize>,
    /// Start of each row in `values`, plus the total count at the end (Stan's `u`, one based in
    /// Stan).
    pub row_starts: Vec<usize>,
}

fn check_csr<T>(function: &'static str, a: &CsrMatrix<T>) -> Result<()> {
    if a.row_starts.len() != a.nrows + 1 {
        return Err(Error::dimension(
            function,
            format!("{} row starts for {} rows", a.row_starts.len(), a.nrows),
        ));
    }
    if a.columns.len() != a.values.len() {
        return Err(Error::dimension(
            function,
            format!(
                "{} column indices for {} values",
                a.columns.len(),
                a.values.len()
            ),
        ));
    }
    if a.row_starts[0] != 0 || a.row_starts[a.nrows] != a.values.len() {
        return Err(Error::dimension(
            function,
            "row starts must begin at 0 and end at the number of values",
        ));
    }
    if a.row_starts.windows(2).any(|w| w[1] < w[0]) {
        return Err(Error::dimension(
            function,
            "row starts must be nondecreasing",
        ));
    }
    if let Some(&c) = a.columns.iter().find(|&&c| c >= a.ncols) {
        return Err(Error::dimension(
            function,
            format!("column index {c} for {} columns", a.ncols),
        ));
    }
    Ok(())
}

/// Extracts the nonzero entries of a dense matrix in CSR form (Stan's `csr_extract`). Entries
/// whose primal value is zero are left out.
pub fn csr_extract<T: Real>(a: &Matrix<T>) -> CsrMatrix<T> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut values = Vec::new();
    let mut columns = Vec::new();
    let mut row_starts = Vec::with_capacity(m + 1);
    row_starts.push(0);
    for i in 0..m {
        for j in 0..n {
            let x = a[(i, j)];
            if x.value() != 0.0 {
                values.push(x);
                columns.push(j);
            }
        }
        row_starts.push(values.len());
    }
    CsrMatrix {
        nrows: m,
        ncols: n,
        values,
        columns,
        row_starts,
    }
}

/// The nonzero values of `a` in CSR order (Stan's `csr_extract_w`).
pub fn csr_extract_w<T: Real>(a: &Matrix<T>) -> Vec<T> {
    csr_extract(a).values
}

/// The zero-based column indices of the nonzero values of `a` (Stan's `csr_extract_v`, which is
/// one based).
pub fn csr_extract_v<T: Real>(a: &Matrix<T>) -> Vec<usize> {
    csr_extract(a).columns
}

/// The zero-based row starts of the CSR form of `a` (Stan's `csr_extract_u`, which is one based).
pub fn csr_extract_u<T: Real>(a: &Matrix<T>) -> Vec<usize> {
    csr_extract(a).row_starts
}

/// Expands a CSR matrix into a dense matrix (Stan's `csr_to_dense_matrix`).
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for inconsistent CSR arrays.
pub fn csr_to_dense_matrix<T: Real>(a: &CsrMatrix<T>) -> Result<Matrix<T>> {
    check_csr("csr_to_dense_matrix", a)?;
    let mut out = Matrix::from_fn(a.nrows, a.ncols, |_, _| T::zero());
    for i in 0..a.nrows {
        for k in a.row_starts[i]..a.row_starts[i + 1] {
            out[(i, a.columns[k])] += a.values[k];
        }
    }
    Ok(out)
}

/// Sparse matrix times dense vector, `A b` (Stan's `csr_matrix_times_vector`). Each output entry
/// is recorded as one fused dot product in reverse mode.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for inconsistent CSR arrays or when `b` does not have `ncols`
/// entries.
pub fn csr_matrix_times_vector<T: Real>(a: &CsrMatrix<T>, b: &[T]) -> Result<Vec<T>> {
    const F: &str = "csr_matrix_times_vector";
    check_csr(F, a)?;
    if b.len() != a.ncols {
        return Err(Error::dimension(
            F,
            format!("{} columns times a vector of length {}", a.ncols, b.len()),
        ));
    }
    Ok((0..a.nrows)
        .map(|i| {
            let range = a.row_starts[i]..a.row_starts[i + 1];
            let gathered: Vec<T> = a.columns[range.clone()].iter().map(|&c| b[c]).collect();
            if gathered.is_empty() {
                T::zero()
            } else {
                dot_product(&a.values[range], &gathered)
            }
        })
        .collect())
}
