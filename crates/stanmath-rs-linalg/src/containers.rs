//! Container constructors, slicing, concatenation, and conversions with Stan's names.
//!
//! Representation follows the rest of this crate: vectors, row vectors, and one dimensional
//! arrays are slices (`&[T]`) in and `Vec<T>` out; two and three dimensional arrays are nested
//! `Vec`s in row-major order; matrices are column-major [`Matrix<T>`]. Integer data is `i64`.
//! Where Stan has several overloads under one name, each overload here has its own name.
//!
//! # Indices
//!
//! Start indices (`block`, `segment`, `sub_col`, `sub_row`, `one_hot_*`) are **one based**, as in
//! Stan, so code translated from a Stan program keeps its arguments. Sizes are `usize`, which
//! rules out Stan's negative size errors at the type level. These functions only move values, so
//! they work for any element type, including reverse-mode variables.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Matrix, Result};

use crate::shape::Dims;

/// Checks that the one based range `start..start + len` lies inside `1..=size`; `start` itself
/// must be a valid index.
fn check_range(
    function: &'static str,
    what: &str,
    start: usize,
    len: usize,
    size: usize,
) -> Result<()> {
    if (1..=size).contains(&start) && start - 1 + len <= size {
        Ok(())
    } else {
        Err(Error::dimension(
            function,
            format!("{what} {start} with {len} entries is out of range for size {size}"),
        ))
    }
}

/// Concatenation of two arrays (also two vectors for `append_row`, or two row vectors for
/// `append_col`, since all of these are slices here).
pub fn append_array<T: Clone>(x: &[T], y: &[T]) -> Vec<T> {
    [x, y].concat()
}

/// The columns of `a` followed by the columns of `b`. Pass a column vector as an `n` by 1
/// matrix.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the row counts differ.
pub fn append_col<T: Clone>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    if a.nrows() != b.nrows() {
        return Err(Error::dimension(
            "append_col",
            format!("{} rows and {} rows", a.nrows(), b.nrows()),
        ));
    }
    // Column-major storage: appending columns is appending data.
    Ok(Matrix::from_column_major(
        a.nrows(),
        a.ncols() + b.ncols(),
        [a.as_slice(), b.as_slice()].concat(),
    ))
}

/// The rows of `a` followed by the rows of `b`. Pass a row vector as a 1 by `n` matrix.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the column counts differ.
pub fn append_row<T: Clone>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    if a.ncols() != b.ncols() {
        return Err(Error::dimension(
            "append_row",
            format!("{} columns and {} columns", a.ncols(), b.ncols()),
        ));
    }
    let ra = a.nrows();
    Ok(Matrix::from_fn(ra + b.nrows(), a.ncols(), |i, j| {
        if i < ra {
            a[(i, j)].clone()
        } else {
            b[(i - ra, j)].clone()
        }
    }))
}

/// The `n_rows` by `n_cols` submatrix of `m` whose top left entry is `(i, j)` (one based).
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the block does not fit inside `m`.
pub fn block<T: Clone>(
    m: &Matrix<T>,
    i: usize,
    j: usize,
    n_rows: usize,
    n_cols: usize,
) -> Result<Matrix<T>> {
    check_range("block", "row", i, n_rows, m.nrows())?;
    check_range("block", "column", j, n_cols, m.ncols())?;
    Ok(Matrix::from_fn(n_rows, n_cols, |r, c| {
        m[(i - 1 + r, j - 1 + c)].clone()
    }))
}

/// The first `n` elements of `v`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `n` exceeds the length of `v`.
pub fn head<T: Clone>(v: &[T], n: usize) -> Result<Vec<T>> {
    if n > v.len() {
        return Err(Error::dimension(
            "head",
            format!("{n} elements requested from {}", v.len()),
        ));
    }
    Ok(v[..n].to_vec())
}

/// The last `n` elements of `v`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `n` exceeds the length of `v`.
pub fn tail<T: Clone>(v: &[T], n: usize) -> Result<Vec<T>> {
    if n > v.len() {
        return Err(Error::dimension(
            "tail",
            format!("{n} elements requested from {}", v.len()),
        ));
    }
    Ok(v[v.len() - n..].to_vec())
}

/// The `n` elements of `v` starting at index `i` (one based).
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the segment does not fit inside `v`.
pub fn segment<T: Clone>(v: &[T], i: usize, n: usize) -> Result<Vec<T>> {
    check_range("segment", "start", i, n, v.len())?;
    Ok(v[i - 1..i - 1 + n].to_vec())
}

/// The column vector of `n_rows` entries of column `j` of `m`, starting at row `i` (one based).
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the range does not fit inside `m`.
pub fn sub_col<T: Clone>(m: &Matrix<T>, i: usize, j: usize, n_rows: usize) -> Result<Vec<T>> {
    check_range("sub_col", "row", i, n_rows, m.nrows())?;
    check_range("sub_col", "column", j, 1, m.ncols())?;
    Ok(m.column(j - 1)[i - 1..i - 1 + n_rows].to_vec())
}

/// The row vector of `n_cols` entries of row `i` of `m`, starting at column `j` (one based).
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the range does not fit inside `m`.
pub fn sub_row<T: Clone>(m: &Matrix<T>, i: usize, j: usize, n_cols: usize) -> Result<Vec<T>> {
    check_range("sub_row", "row", i, 1, m.nrows())?;
    check_range("sub_row", "column", j, n_cols, m.ncols())?;
    Ok((0..n_cols).map(|c| m[(i - 1, j - 1 + c)].clone()).collect())
}

/// The elements of `v` in reverse order.
pub fn reverse<T: Clone>(v: &[T]) -> Vec<T> {
    v.iter().rev().cloned().collect()
}

/// Array of `n` copies of `x`.
pub fn rep_array<T: Clone>(x: T, n: usize) -> Vec<T> {
    vec![x; n]
}

/// `m` by `n` array (row-major nesting) of copies of `x`.
pub fn rep_array_2d<T: Clone>(x: T, m: usize, n: usize) -> Vec<Vec<T>> {
    vec![vec![x; n]; m]
}

/// `k` by `m` by `n` array of copies of `x`.
pub fn rep_array_3d<T: Clone>(x: T, k: usize, m: usize, n: usize) -> Vec<Vec<Vec<T>>> {
    vec![vec![vec![x; n]; m]; k]
}

/// Vector of `n` copies of `x`.
pub fn rep_vector<T: Clone>(x: T, n: usize) -> Vec<T> {
    vec![x; n]
}

/// Row vector of `n` copies of `x`.
pub fn rep_row_vector<T: Clone>(x: T, n: usize) -> Vec<T> {
    vec![x; n]
}

/// `m` by `n` matrix with every entry equal to `x`.
pub fn rep_matrix<T: Clone>(x: T, m: usize, n: usize) -> Matrix<T> {
    Matrix::from_column_major(m, n, vec![x; m * n])
}

/// Matrix whose `n` columns are each the vector `v`.
pub fn rep_matrix_vector<T: Clone>(v: &[T], n: usize) -> Matrix<T> {
    Matrix::from_fn(v.len(), n, |i, _| v[i].clone())
}

/// Matrix whose `m` rows are each the row vector `rv`.
pub fn rep_matrix_row_vector<T: Clone>(rv: &[T], m: usize) -> Matrix<T> {
    Matrix::from_fn(m, rv.len(), |_, j| rv[j].clone())
}

/// Vector of `n` ones.
pub fn ones_vector<T: Real>(n: usize) -> Vec<T> {
    vec![T::one(); n]
}

/// Row vector of `n` ones.
pub fn ones_row_vector<T: Real>(n: usize) -> Vec<T> {
    vec![T::one(); n]
}

/// Real array of `n` ones.
pub fn ones_array<T: Real>(n: usize) -> Vec<T> {
    vec![T::one(); n]
}

/// Integer array of `n` ones.
pub fn ones_int_array(n: usize) -> Vec<i64> {
    vec![1; n]
}

/// Vector of `n` zeros.
pub fn zeros_vector<T: Real>(n: usize) -> Vec<T> {
    vec![T::zero(); n]
}

/// Row vector of `n` zeros.
pub fn zeros_row_vector<T: Real>(n: usize) -> Vec<T> {
    vec![T::zero(); n]
}

/// Real array of `n` zeros.
pub fn zeros_array<T: Real>(n: usize) -> Vec<T> {
    vec![T::zero(); n]
}

/// Integer array of `n` zeros.
pub fn zeros_int_array(n: usize) -> Vec<i64> {
    vec![0; n]
}

/// `k` evenly spaced values from `low` to `high`; a single value is `high`.
fn linspaced<T: Real>(function: &'static str, k: usize, low: f64, high: f64) -> Result<Vec<T>> {
    if !low.is_finite() {
        return Err(Error::parameter(function, "low", low, "finite"));
    }
    if !high.is_finite() {
        return Err(Error::parameter(function, "high", high, "finite"));
    }
    if !(high >= low) {
        return Err(Error::parameter(
            function,
            "high",
            high,
            "greater than or equal to low",
        ));
    }
    let values = match k {
        0 => Vec::new(),
        1 => vec![high],
        _ => {
            let step = (high - low) / (k - 1) as f64;
            // Count from the nearer end so both endpoints are exact.
            (0..k)
                .map(|i| {
                    if i < k / 2 {
                        low + i as f64 * step
                    } else {
                        high - (k - 1 - i) as f64 * step
                    }
                })
                .collect()
        }
    };
    Ok(values.into_iter().map(T::from_f64).collect())
}

/// Vector of `k` evenly spaced values from `low` to `high` inclusive; `k = 1` gives `[high]`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `low` or `high` is not finite or `high < low`.
pub fn linspaced_vector<T: Real>(k: usize, low: f64, high: f64) -> Result<Vec<T>> {
    linspaced("linspaced_vector", k, low, high)
}

/// Row vector of `k` evenly spaced values; see [`linspaced_vector`].
///
/// # Errors
///
/// As [`linspaced_vector`].
pub fn linspaced_row_vector<T: Real>(k: usize, low: f64, high: f64) -> Result<Vec<T>> {
    linspaced("linspaced_row_vector", k, low, high)
}

/// Real array of `k` evenly spaced values; see [`linspaced_vector`].
///
/// # Errors
///
/// As [`linspaced_vector`].
pub fn linspaced_array<T: Real>(k: usize, low: f64, high: f64) -> Result<Vec<T>> {
    linspaced("linspaced_array", k, low, high)
}

/// Increasing integer array of length `k` starting at `low` with integer spacing.
///
/// When `high - low` is a multiple of `k - 1` the values run from `low` to `high` in equal steps.
/// When there are fewer distinct integers in `[low, high]` than `k`, each value is repeated
/// equally. Otherwise `high` is lowered to the largest value that allows one of these two layouts
/// (the step is `(high - low) / (k - 1)` rounded down). `k = 1` gives `[high]`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `high < low`.
pub fn linspaced_int_array(k: usize, low: i64, high: i64) -> Result<Vec<i64>> {
    if high < low {
        return Err(Error::parameter(
            "linspaced_int_array",
            "high",
            high as f64,
            "greater than or equal to low",
        ));
    }
    if k == 1 {
        return Ok(vec![high]);
    }
    let k = k as i64;
    let range = high - low;
    Ok(if k > 1 && range + 1 < k {
        let repeat = (k + range) / (range + 1);
        (0..k).map(|i| low + i / repeat).collect()
    } else {
        let step = range / (k - 1).max(1);
        (0..k).map(|i| low + i * step).collect()
    })
}

fn check_one_hot(function: &'static str, n: usize, k: usize) -> Result<()> {
    if n == 0 {
        return Err(Error::parameter(function, "n", 0.0, "positive"));
    }
    if !(1..=n).contains(&k) {
        return Err(Error::parameter(function, "k", k as f64, "between 1 and n"));
    }
    Ok(())
}

fn one_hot<T: Clone>(function: &'static str, n: usize, k: usize, off: T, on: T) -> Result<Vec<T>> {
    check_one_hot(function, n, k)?;
    let mut v = vec![off; n];
    v[k - 1] = on;
    Ok(v)
}

/// Vector of length `n` with a one at index `k` (one based) and zeros elsewhere.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `n = 0` or `k` is not in `1..=n`.
pub fn one_hot_vector<T: Real>(n: usize, k: usize) -> Result<Vec<T>> {
    one_hot("one_hot_vector", n, k, T::zero(), T::one())
}

/// Row vector form of [`one_hot_vector`].
///
/// # Errors
///
/// As [`one_hot_vector`].
pub fn one_hot_row_vector<T: Real>(n: usize, k: usize) -> Result<Vec<T>> {
    one_hot("one_hot_row_vector", n, k, T::zero(), T::one())
}

/// Real array form of [`one_hot_vector`].
///
/// # Errors
///
/// As [`one_hot_vector`].
pub fn one_hot_array<T: Real>(n: usize, k: usize) -> Result<Vec<T>> {
    one_hot("one_hot_array", n, k, T::zero(), T::one())
}

/// Integer array form of [`one_hot_vector`].
///
/// # Errors
///
/// As [`one_hot_vector`].
pub fn one_hot_int_array(n: usize, k: usize) -> Result<Vec<i64>> {
    one_hot("one_hot_int_array", n, k, 0, 1)
}

/// The simplex of length `n` with every entry `1 / n`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `n = 0`.
pub fn uniform_simplex<T: Real>(n: usize) -> Result<Vec<T>> {
    if n == 0 {
        return Err(Error::parameter("uniform_simplex", "n", 0.0, "positive"));
    }
    Ok(vec![T::from_f64(1.0 / n as f64); n])
}

/// Copy of a vector, row vector, or one dimensional array as a vector.
pub fn to_vector<T: Clone>(x: &[T]) -> Vec<T> {
    x.to_vec()
}

/// The entries of `m` in column-major order, as a vector.
pub fn to_vector_matrix<T: Clone>(m: &Matrix<T>) -> Vec<T> {
    m.as_slice().to_vec()
}

/// Integer array converted to a real vector.
pub fn to_vector_int<T: Real>(x: &[i64]) -> Vec<T> {
    x.iter().map(|&v| T::from_f64(v as f64)).collect()
}

/// Copy of a vector, row vector, or one dimensional array as a row vector.
pub fn to_row_vector<T: Clone>(x: &[T]) -> Vec<T> {
    x.to_vec()
}

/// The entries of `m` in column-major order, as a row vector.
pub fn to_row_vector_matrix<T: Clone>(m: &Matrix<T>) -> Vec<T> {
    to_vector_matrix(m)
}

/// Integer array converted to a real row vector.
pub fn to_row_vector_int<T: Real>(x: &[i64]) -> Vec<T> {
    to_vector_int(x)
}

fn check_reshape(len: usize, m: usize, n: usize) -> Result<()> {
    if m * n == len {
        Ok(())
    } else {
        Err(Error::dimension(
            "to_matrix",
            format!("{len} elements cannot fill a {m}x{n} matrix"),
        ))
    }
}

/// `m` by `n` matrix filled from `x` in column-major order (Stan's `to_matrix(v, m, n)` for a
/// vector, row vector, or array).
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `x` does not have `m n` elements.
pub fn to_matrix<T: Clone>(x: &[T], m: usize, n: usize) -> Result<Matrix<T>> {
    check_reshape(x.len(), m, n)?;
    Ok(Matrix::from_column_major(m, n, x.to_vec()))
}

/// `m` by `n` matrix filled from `x` in row-major order (Stan's `to_matrix(v, m, n, 0)`).
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `x` does not have `m n` elements.
pub fn to_matrix_row_major<T: Clone>(x: &[T], m: usize, n: usize) -> Result<Matrix<T>> {
    check_reshape(x.len(), m, n)?;
    Ok(Matrix::from_fn(m, n, |i, j| x[i * n + j].clone()))
}

/// `a` reshaped to `m` by `n`, keeping column-major order.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `a` does not have `m n` elements.
pub fn to_matrix_reshape<T: Clone>(a: &Matrix<T>, m: usize, n: usize) -> Result<Matrix<T>> {
    to_matrix(a.as_slice(), m, n)
}

/// A vector as an `n` by 1 matrix.
pub fn to_matrix_from_vector<T: Clone>(v: &[T]) -> Matrix<T> {
    Matrix::column_vector(v.to_vec())
}

/// A row vector as a 1 by `n` matrix.
pub fn to_matrix_from_row_vector<T: Clone>(v: &[T]) -> Matrix<T> {
    Matrix::from_column_major(1, v.len(), v.to_vec())
}

/// A two dimensional array (a list of rows) as a matrix.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the rows have different lengths.
pub fn to_matrix_from_array_2d<T: Clone>(a: &[Vec<T>]) -> Result<Matrix<T>> {
    let n = a.first().map_or(0, Vec::len);
    if a.iter().any(|row| row.len() != n) {
        return Err(Error::dimension("to_matrix", "rows have different lengths"));
    }
    Ok(Matrix::from_rows(a))
}

/// A two dimensional integer array as a real matrix.
///
/// # Errors
///
/// As [`to_matrix_from_array_2d`].
pub fn to_matrix_from_int_array_2d<T: Real>(a: &[Vec<i64>]) -> Result<Matrix<T>> {
    let rows: Vec<Vec<T>> = a.iter().map(|row| to_vector_int(row)).collect();
    to_matrix_from_array_2d(&rows)
}

/// The entries of `m` in column-major order, as an array.
pub fn to_array_1d<T: Clone>(m: &Matrix<T>) -> Vec<T> {
    to_vector_matrix(m)
}

/// A two dimensional array flattened in row-major order.
pub fn to_array_1d_from_array_2d<T: Clone>(a: &[Vec<T>]) -> Vec<T> {
    a.concat()
}

/// A three dimensional array flattened in row-major order.
pub fn to_array_1d_from_array_3d<T: Clone>(a: &[Vec<Vec<T>>]) -> Vec<T> {
    a.iter().flat_map(|plane| plane.concat()).collect()
}

/// The rows of `m` as a two dimensional array.
pub fn to_array_2d<T: Clone>(m: &Matrix<T>) -> Vec<Vec<T>> {
    m.to_rows()
}

/// `x` truncated toward zero to an integer.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `x` is NaN or outside the range of a 32-bit integer (Stan's
/// `int`), so results agree with Stan.
pub fn to_int(x: f64) -> Result<i64> {
    if (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&x) {
        Ok(x.trunc() as i64)
    } else {
        Err(Error::parameter(
            "to_int",
            "x",
            x,
            "within the range of a 32-bit integer",
        ))
    }
}

/// Elementwise [`to_int`].
///
/// # Errors
///
/// As [`to_int`], for the first element out of range.
pub fn to_int_array(x: &[f64]) -> Result<Vec<i64>> {
    x.iter().map(|&v| to_int(v)).collect()
}

/// Total number of scalars in `x`, at any nesting depth: a vector of length `n` has `n`, a matrix
/// has `rows * cols`, and an array of matrices has the sum over its matrices. A scalar has 1.
pub fn num_elements<D: Dims + ?Sized>(x: &D) -> usize {
    x.element_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slicing_is_one_based() {
        let m = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(block(&m, 2, 2, 1, 2).unwrap().as_slice(), &[5.0, 6.0]);
        assert_eq!(sub_col(&m, 1, 3, 2).unwrap(), vec![3.0, 6.0]);
        assert_eq!(sub_row(&m, 2, 1, 3).unwrap(), vec![4.0, 5.0, 6.0]);
        assert_eq!(segment(&[1, 2, 3, 4], 2, 2).unwrap(), vec![2, 3]);
        assert!(segment(&[1, 2, 3], 3, 2).is_err());
        assert!(block(&m, 0, 1, 1, 1).is_err());
        assert_eq!(head(&[1, 2, 3], 2).unwrap(), vec![1, 2]);
        assert_eq!(tail(&[1, 2, 3], 2).unwrap(), vec![2, 3]);
    }

    #[test]
    fn appends_and_reshapes() {
        let a = Matrix::from_rows(&[[1.0], [2.0]]);
        let b = Matrix::from_rows(&[[3.0, 4.0], [5.0, 6.0]]);
        let c = append_col(&a, &b).unwrap();
        assert_eq!(c.to_rows(), vec![vec![1.0, 3.0, 4.0], vec![2.0, 5.0, 6.0]]);
        let r = append_row(&b, &Matrix::from_rows(&[[7.0, 8.0]])).unwrap();
        assert_eq!(r.shape(), (3, 2));
        assert_eq!(r[(2, 1)], 8.0);
        let x = [1, 2, 3, 4, 5, 6];
        assert_eq!(to_matrix(&x, 2, 3).unwrap()[(1, 0)], 2);
        assert_eq!(to_matrix_row_major(&x, 2, 3).unwrap()[(1, 0)], 4);
        assert!(to_matrix(&x, 4, 2).is_err());
    }

    #[test]
    fn constructors() {
        let v: Vec<f64> = linspaced_vector(5, 0.0, 1.0).unwrap();
        assert_eq!(v, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspaced_vector::<f64>(1, 0.0, 3.0).unwrap(), vec![3.0]);
        assert_eq!(linspaced_int_array(5, 1, 10).unwrap(), vec![1, 3, 5, 7, 9]);
        assert_eq!(
            linspaced_int_array(8, 1, 4).unwrap(),
            vec![1, 1, 2, 2, 3, 3, 4, 4]
        );
        assert_eq!(one_hot_int_array(3, 2).unwrap(), vec![0, 1, 0]);
        assert!(one_hot_int_array(3, 4).is_err());
        assert_eq!(uniform_simplex::<f64>(4).unwrap(), vec![0.25; 4]);
        assert_eq!(to_int(-2.7).unwrap(), -2);
        assert!(to_int(f64::NAN).is_err());
        assert!(to_int(3.0e9).is_err());
    }
}
