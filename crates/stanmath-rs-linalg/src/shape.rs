//! Size queries, row and column extraction, and vector norms with Stan's names.
//!
//! Vectors are slices, so `rows` and `cols` take matrices only; for a vector use `size`. Row and
//! column indices are one based, as in Stan and in `containers.rs`.

use stanmath_rs_autodiff::{Complex, Dual, Real, Var};
use stanmath_rs_core::{Error, Matrix, Result};

/// Number of rows of `m`.
pub fn rows<T>(m: &Matrix<T>) -> usize {
    m.nrows()
}

/// Number of columns of `m`.
pub fn cols<T>(m: &Matrix<T>) -> usize {
    m.ncols()
}

/// Number of elements of a vector, row vector, or one dimensional array.
pub fn size<T>(v: &[T]) -> usize {
    v.len()
}

/// Shape of a Stan value held in Rust containers: scalars, slices and `Vec`s (vectors, row vectors,
/// and arrays), fixed size arrays, and [`Matrix`], nested to any depth.
///
/// Rust does not record the declared sizes of an empty container, so the inner dimensions of an
/// empty array are missing: `dims` of an empty `Vec<Vec<f64>>` is `[0]`, where Stan reports the
/// declared inner size as well. Ragged nesting reports the inner dimensions of the first element.
pub trait Dims {
    /// Appends the dimensions of `self`, outermost first, to `out`. A scalar appends nothing.
    fn dims_into(&self, out: &mut Vec<usize>);

    /// Total number of scalars in `self`.
    fn element_count(&self) -> usize;
}

macro_rules! scalar_dims {
    ($($t:ty),*) => {$(
        impl Dims for $t {
            fn dims_into(&self, _out: &mut Vec<usize>) {}
            fn element_count(&self) -> usize {
                1
            }
        }
    )*};
}

scalar_dims!(f64, i64);

impl Dims for Var<'_> {
    fn dims_into(&self, _out: &mut Vec<usize>) {}
    fn element_count(&self) -> usize {
        1
    }
}

impl<T: Real> Dims for Dual<T> {
    fn dims_into(&self, _out: &mut Vec<usize>) {}
    fn element_count(&self) -> usize {
        1
    }
}

impl<T: Real> Dims for Complex<T> {
    fn dims_into(&self, _out: &mut Vec<usize>) {}
    fn element_count(&self) -> usize {
        1
    }
}

impl<T: Dims> Dims for [T] {
    fn dims_into(&self, out: &mut Vec<usize>) {
        out.push(self.len());
        if let Some(first) = self.first() {
            first.dims_into(out);
        }
    }
    fn element_count(&self) -> usize {
        self.iter().map(Dims::element_count).sum()
    }
}

impl<T: Dims, const N: usize> Dims for [T; N] {
    fn dims_into(&self, out: &mut Vec<usize>) {
        self.as_slice().dims_into(out);
    }
    fn element_count(&self) -> usize {
        self.as_slice().element_count()
    }
}

impl<T: Dims> Dims for Vec<T> {
    fn dims_into(&self, out: &mut Vec<usize>) {
        self.as_slice().dims_into(out);
    }
    fn element_count(&self) -> usize {
        self.as_slice().element_count()
    }
}

impl<T> Dims for Matrix<T> {
    fn dims_into(&self, out: &mut Vec<usize>) {
        out.push(self.nrows());
        out.push(self.ncols());
    }
    fn element_count(&self) -> usize {
        self.len()
    }
}

/// Dimensions of `x`, outermost first, as Stan's `dims`: empty for a scalar, `[n]` for a vector or
/// one dimensional array, `[rows, cols]` for a matrix, and `[n, rows, cols]` for an array of
/// matrices. See [`Dims`] for empty containers.
pub fn dims<D: Dims + ?Sized>(x: &D) -> Vec<usize> {
    let mut out = Vec::new();
    x.dims_into(&mut out);
    out
}

/// Row `i` (one based) of `m`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `i` is not in `1..=rows(m)`.
pub fn row<T: Clone>(m: &Matrix<T>, i: usize) -> Result<Vec<T>> {
    if !(1..=m.nrows()).contains(&i) {
        return Err(Error::dimension(
            "row",
            format!("row index {i} is outside 1..={}", m.nrows()),
        ));
    }
    Ok(m.row(i - 1).cloned().collect())
}

/// Column `j` (one based) of `m`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `j` is not in `1..=cols(m)`.
pub fn col<T: Clone>(m: &Matrix<T>, j: usize) -> Result<Vec<T>> {
    if !(1..=m.ncols()).contains(&j) {
        return Err(Error::dimension(
            "col",
            format!("column index {j} is outside 1..={}", m.ncols()),
        ));
    }
    Ok(m.column(j - 1).to_vec())
}

/// Sum of absolute values. The derivative of `|x|` is taken to be zero at zero.
pub fn norm1<T: Real>(v: &[T]) -> T {
    v.iter().map(|&x| x.abs()).sum()
}

/// Euclidean norm. The derivative is undefined at the zero vector, as in Stan.
pub fn norm2<T: Real>(v: &[T]) -> T {
    v.iter().map(|&x| x.square()).sum::<T>().sqrt()
}

/// The columns of `m`, each as a vector. Stan's `to_vector_array`.
pub fn to_vector_array<T: Clone>(m: &Matrix<T>) -> Vec<Vec<T>> {
    (0..m.ncols()).map(|j| m.column(j).to_vec()).collect()
}

/// The rows of `m`, each as a row vector. Stan's `to_row_vector_array`.
pub fn to_row_vector_array<T: Clone>(m: &Matrix<T>) -> Vec<Vec<T>> {
    (0..m.nrows())
        .map(|i| m.row(i).cloned().collect())
        .collect()
}

/// `trace(A B)`, computed as the sum of `A[i, k] B[k, i]` without forming the product. Stan's
/// `trace_dot`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `A` is `n` by `k` and `B` is `k` by `n`.
pub fn trace_dot<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<T> {
    if a.ncols() != b.nrows() || a.nrows() != b.ncols() {
        return Err(Error::dimension(
            "trace_dot",
            format!(
                "A is {}x{} and B is {}x{}; B must be {}x{}",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols(),
                a.ncols(),
                a.nrows()
            ),
        ));
    }
    Ok((0..a.nrows())
        .flat_map(|i| (0..a.ncols()).map(move |k| (i, k)))
        .map(|(i, k)| a[(i, k)] * b[(k, i)])
        .sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::num_elements;

    #[test]
    fn dims_of_nested_containers() {
        let ms = vec![Matrix::<f64>::zeros(2, 3); 4];
        assert_eq!(dims(&ms), [4, 2, 3]);
        assert_eq!(num_elements(&ms), 24);
        let empty: Vec<Vec<f64>> = Vec::new();
        assert_eq!(dims(&empty), [0]);
        assert_eq!(dims(&1.5), Vec::<usize>::new());
        assert_eq!(dims(&[[1i64, 2], [3, 4], [5, 6]]), [3, 2]);
        let ragged = vec![vec![1.0], vec![2.0, 3.0, 4.0], vec![]];
        assert_eq!(num_elements(&ragged), 4);
        assert_eq!(dims(&ragged), [3, 1]);
    }
}
