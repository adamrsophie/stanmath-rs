//! A dense, column-major matrix container.
//!
//! [`Matrix`] places no bounds on its element type. That matters because reverse-mode variables
//! borrow their tape and are therefore not `'static`, which rules out containers such as
//! `nalgebra::DMatrix` (its `Scalar` bound requires `'static`). Numerical kernels live in
//! `probgrad-linalg`; this type only stores and indexes elements.

use core::ops::{Index, IndexMut};

/// Dense matrix stored in column-major order (the same layout as Eigen and Stan Math).
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix<T> {
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Wraps column-major `data` as an `nrows` by `ncols` matrix.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    pub fn from_column_major(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            nrows * ncols,
            "Matrix::from_column_major: expected {} elements for a {nrows}x{ncols} matrix, got {}",
            nrows * ncols,
            data.len()
        );
        Matrix { nrows, ncols, data }
    }

    /// Builds a matrix by evaluating `f(row, col)` for every element.
    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(f(i, j));
            }
        }
        Matrix { nrows, ncols, data }
    }

    /// Builds a matrix from row slices.
    ///
    /// # Panics
    ///
    /// Panics if the rows have different lengths.
    pub fn from_rows<R: AsRef<[T]>>(rows: &[R]) -> Self
    where
        T: Clone,
    {
        let nrows = rows.len();
        let ncols = rows.first().map_or(0, |r| r.as_ref().len());
        assert!(
            rows.iter().all(|r| r.as_ref().len() == ncols),
            "Matrix::from_rows: rows have different lengths"
        );
        Matrix::from_fn(nrows, ncols, |i, j| rows[i].as_ref()[j].clone())
    }

    /// Builds an `n` by 1 column vector.
    pub fn column_vector(data: Vec<T>) -> Self {
        let n = data.len();
        Matrix::from_column_major(n, 1, data)
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// True when the matrix has as many rows as columns.
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the matrix has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Elements in column-major order.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable elements in column-major order.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the matrix and returns its column-major elements.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Element at `(row, col)`, or `None` when out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.nrows && col < self.ncols {
            Some(&self.data[col * self.nrows + row])
        } else {
            None
        }
    }

    /// Column `col` as a contiguous slice.
    pub fn column(&self, col: usize) -> &[T] {
        &self.data[col * self.nrows..(col + 1) * self.nrows]
    }

    /// Row `row` as an iterator over its elements.
    pub fn row(&self, row: usize) -> impl Iterator<Item = &T> + '_ {
        assert!(row < self.nrows, "Matrix::row: row {row} out of bounds");
        (0..self.ncols).map(move |j| &self.data[j * self.nrows + row])
    }

    /// Transposed copy.
    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        Matrix::from_fn(self.ncols, self.nrows, |i, j| self[(j, i)].clone())
    }

    /// Applies `f` to every element.
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> Matrix<U> {
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            data: self.data.iter().map(f).collect(),
        }
    }

    /// Copies the matrix into a vector of rows.
    pub fn to_rows(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        (0..self.nrows)
            .map(|i| self.row(i).cloned().collect())
            .collect()
    }
}

impl Matrix<f64> {
    /// Matrix of zeros.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Matrix::from_column_major(nrows, ncols, vec![0.0; nrows * ncols])
    }

    /// Identity matrix of size `n`.
    pub fn identity(n: usize) -> Self {
        Matrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 })
    }

    /// Largest absolute elementwise difference, or infinity when shapes differ.
    pub fn max_abs_diff(&self, other: &Matrix<f64>) -> f64 {
        if self.shape() != other.shape() {
            return f64::INFINITY;
        }
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &T {
        assert!(
            row < self.nrows && col < self.ncols,
            "Matrix index ({row}, {col}) out of bounds for {}x{} matrix",
            self.nrows,
            self.ncols
        );
        &self.data[col * self.nrows + row]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(
            row < self.nrows && col < self.ncols,
            "Matrix index ({row}, {col}) out of bounds for {}x{} matrix",
            self.nrows,
            self.ncols
        );
        &mut self.data[col * self.nrows + row]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_column_major() {
        let m = Matrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(m[(1, 2)], 6.0);
        assert_eq!(m.column(1), &[2.0, 5.0]);
        assert_eq!(m.row(0).copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t[(2, 1)], 6.0);
        assert_eq!(
            t.to_rows(),
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        assert_eq!(m.get(2, 0), None);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_checks_bounds() {
        let m = Matrix::identity(2);
        let _ = m[(0, 2)];
    }
}
