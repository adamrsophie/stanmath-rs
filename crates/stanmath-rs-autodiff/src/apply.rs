//! Elementwise application of scalar functions over containers: Stan's `apply_scalar_unary` and
//! `apply_scalar_binary`.
//!
//! Stan vectorizes most scalar functions over vectors, row vectors, matrices, and arrays of any
//! depth (`exp(matrix)`, `pow(array[] real, real)`). In Rust a scalar function is one generic
//! function, so the container forms go through these two helpers:
//!
//! ```
//! use stanmath_rs_autodiff::functions::{apply_scalar_binary, apply_scalar_unary, exp, pow};
//!
//! let x = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
//! assert_eq!(apply_scalar_unary(&x, exp)[1][0], 2.0_f64.exp());
//! // A scalar on one side is captured by the closure.
//! let squares = apply_scalar_unary(&x, |v| pow(v, 2.0));
//! let sums = apply_scalar_binary(&x, &squares, |a, b| a + b).unwrap();
//! assert_eq!(sums[1][1], 12.0);
//! ```

use stanmath_rs_core::{Error, Matrix, Result};

use crate::dual::Dual;
use crate::real::Real;
use crate::tape::Var;

/// A scalar or a container of scalars, nested to any depth: `Vec` for Stan's arrays, vectors,
/// and row vectors, and [`Matrix`] for its matrices.
pub trait ScalarContainer {
    /// The scalar type at the leaves.
    type Scalar: Copy;
    /// The same shape with leaves of type `U`.
    type Mapped<U>;
    /// Applies `f` to every scalar, keeping the shape.
    fn map_scalars<U>(&self, f: &mut impl FnMut(Self::Scalar) -> U) -> Self::Mapped<U>;
    /// Applies `f` to every pair of scalars at the same position.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] when the shapes differ.
    fn zip_scalars<U>(
        &self,
        other: &Self,
        f: &mut impl FnMut(Self::Scalar, Self::Scalar) -> U,
    ) -> Result<Self::Mapped<U>>;
}

macro_rules! leaf {
    ($($t:ty $(, $g:tt)*);*) => {$(
        impl<$($g,)*> ScalarContainer for $t {
            type Scalar = Self;
            type Mapped<U> = U;
            fn map_scalars<U>(&self, f: &mut impl FnMut(Self) -> U) -> U {
                f(*self)
            }
            fn zip_scalars<U>(&self, other: &Self, f: &mut impl FnMut(Self, Self) -> U) -> Result<U> {
                Ok(f(*self, *other))
            }
        }
    )*};
}

leaf!(f64; i64; Var<'t>, 't);

impl<T: Real> ScalarContainer for Dual<T> {
    type Scalar = Self;
    type Mapped<U> = U;
    fn map_scalars<U>(&self, f: &mut impl FnMut(Self) -> U) -> U {
        f(*self)
    }
    fn zip_scalars<U>(&self, other: &Self, f: &mut impl FnMut(Self, Self) -> U) -> Result<U> {
        Ok(f(*self, *other))
    }
}

impl<C: ScalarContainer> ScalarContainer for Vec<C> {
    type Scalar = C::Scalar;
    type Mapped<U> = Vec<C::Mapped<U>>;
    fn map_scalars<U>(&self, f: &mut impl FnMut(C::Scalar) -> U) -> Self::Mapped<U> {
        self.iter().map(|c| c.map_scalars(f)).collect()
    }
    fn zip_scalars<U>(
        &self,
        other: &Self,
        f: &mut impl FnMut(C::Scalar, C::Scalar) -> U,
    ) -> Result<Self::Mapped<U>> {
        if self.len() != other.len() {
            return Err(Error::dimension(
                "apply_scalar_binary",
                format!("sizes {} and {}", self.len(), other.len()),
            ));
        }
        self.iter()
            .zip(other)
            .map(|(a, b)| a.zip_scalars(b, f))
            .collect()
    }
}

impl<T: Copy> ScalarContainer for Matrix<T> {
    type Scalar = T;
    type Mapped<U> = Matrix<U>;
    fn map_scalars<U>(&self, f: &mut impl FnMut(T) -> U) -> Matrix<U> {
        self.map(|&v| f(v))
    }
    fn zip_scalars<U>(&self, other: &Self, f: &mut impl FnMut(T, T) -> U) -> Result<Matrix<U>> {
        if self.shape() != other.shape() {
            return Err(Error::dimension(
                "apply_scalar_binary",
                format!("shapes {:?} and {:?}", self.shape(), other.shape()),
            ));
        }
        let (r, c) = self.shape();
        let data = self
            .as_slice()
            .iter()
            .zip(other.as_slice())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Ok(Matrix::from_column_major(r, c, data))
    }
}

/// `f` applied to every scalar of `x`, keeping its shape (Stan's vectorized unary functions).
pub fn apply_scalar_unary<C: ScalarContainer, U>(
    x: &C,
    mut f: impl FnMut(C::Scalar) -> U,
) -> C::Mapped<U> {
    x.map_scalars(&mut f)
}

/// `f` applied to the scalars of `a` and `b` at each position (Stan's vectorized binary functions
/// over two containers of one shape). For a scalar on one side, capture it in a closure passed to
/// [`apply_scalar_unary`].
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the shapes differ.
pub fn apply_scalar_binary<C: ScalarContainer, U>(
    a: &C,
    b: &C,
    mut f: impl FnMut(C::Scalar, C::Scalar) -> U,
) -> Result<C::Mapped<U>> {
    a.zip_scalars(b, &mut f)
}
