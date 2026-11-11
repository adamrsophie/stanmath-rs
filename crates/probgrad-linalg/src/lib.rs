//! Differentiable dense linear algebra.
//!
//! Functions take [`Matrix<T>`](probgrad_core::Matrix) for any [`Real`] `T` and follow Stan Math's
//! names. Each core operation (`multiply`, `cholesky_decompose`, the triangular solves) has a
//! structured reverse-mode adjoint: in reverse mode it records one custom tape operation computed
//! with `f64` kernels, so it never forms a dense Jacobian and the tape grows with the size of the
//! output, not with the operation count. In forward mode, including forward over reverse for
//! Hessians, the same generic algorithm runs on dual numbers, which keeps every derivative order
//! exact. The other functions are compositions of these.
//!
//! ```
//! use probgrad_autodiff::grad;
//! use probgrad_core::Matrix;
//! use probgrad_linalg::log_determinant_spd;
//!
//! // d/dA log det(A) = A⁻¹; on the lower-triangle parameterization, off-diagonals count twice.
//! let g = grad(
//!     |p| {
//!         let a = Matrix::from_rows(&[[p[0], p[1]], [p[1], p[2]]]);
//!         log_determinant_spd(&a).unwrap()
//!     },
//!     &[2.0, 0.5, 1.0],
//! );
//! let det = 2.0 * 1.0 - 0.25;
//! assert!((g[0] - 1.0 / det).abs() < 1e-15);
//! assert!((g[1] - 2.0 * (-0.5 / det)).abs() < 1e-15);
//! ```
//!
//! # Conventions
//!
//! `cholesky_decompose` and everything built on it read only the lower triangle of a symmetric
//! input, after checking symmetry on the values to an absolute tolerance of `1e-8` (as Stan Math
//! does). Derivatives therefore flow to the lower triangle; entries above the diagonal have zero
//! derivative. For a matrix built symmetrically from parameters this gives the correct total
//! derivative.

mod ops;

use probgrad_autodiff::Real;
use probgrad_core::{Error, Matrix, Result};

use ops::{Cholesky, Multiply, TriangularSolve};

fn require_square<T>(function: &'static str, a: &Matrix<T>) -> Result<()> {
    if a.is_square() {
        Ok(())
    } else {
        Err(Error::dimension(
            function,
            format!("expected a square matrix, got {}x{}", a.nrows(), a.ncols()),
        ))
    }
}

/// Matrix product `A B`.
pub fn multiply<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    if a.ncols() != b.nrows() {
        return Err(Error::dimension(
            "multiply",
            format!(
                "{}x{} times {}x{}",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols()
            ),
        ));
    }
    let f = Multiply {
        m: a.nrows(),
        k: a.ncols(),
        n: b.ncols(),
    };
    let inputs: Vec<T> = a.as_slice().iter().chain(b.as_slice()).copied().collect();
    let c = T::map_multi(&inputs, &f)?;
    Ok(Matrix::from_column_major(f.m, f.n, c))
}

/// Matrix-vector product `A x`.
pub fn multiply_vector<T: Real>(a: &Matrix<T>, x: &[T]) -> Result<Vec<T>> {
    Ok(multiply(a, &Matrix::column_vector(x.to_vec()))?.into_vec())
}

/// Lower Cholesky factor `L` of a symmetric positive definite matrix, `A = L Lᵀ`.
///
/// # Errors
///
/// [`Error::NotSymmetric`] or [`Error::NotPositiveDefinite`]; a non-square input gives
/// [`Error::DimensionMismatch`].
pub fn cholesky_decompose<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    require_square("cholesky_decompose", a)?;
    let n = a.nrows();
    let l = T::map_multi(a.as_slice(), &Cholesky { n })?;
    Ok(Matrix::from_column_major(n, n, l))
}

fn tri_solve<T: Real>(l: &Matrix<T>, b: &Matrix<T>, transpose: bool) -> Result<Matrix<T>> {
    const F: &str = "mdivide_left_tri_low";
    require_square(F, l)?;
    if b.nrows() != l.nrows() {
        return Err(Error::dimension(
            F,
            format!(
                "{}x{} triangular factor with {} right-hand side rows",
                l.nrows(),
                l.ncols(),
                b.nrows()
            ),
        ));
    }
    let f = TriangularSolve {
        n: l.nrows(),
        m: b.ncols(),
        transpose,
    };
    let inputs: Vec<T> = l.as_slice().iter().chain(b.as_slice()).copied().collect();
    let x = T::map_multi(&inputs, &f)?;
    Ok(Matrix::from_column_major(f.n, f.m, x))
}

/// `L⁻¹ B` for lower triangular `L` (only its lower triangle is read).
pub fn mdivide_left_tri_low<T: Real>(l: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    tri_solve(l, b, false)
}

/// `L⁻ᵀ B` for lower triangular `L` (only its lower triangle is read).
pub fn mdivide_left_tri_low_transpose<T: Real>(l: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    tri_solve(l, b, true)
}

/// `A⁻¹ B` for symmetric positive definite `A`, by two triangular solves.
pub fn mdivide_left_spd<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    let l = cholesky_decompose(a)?;
    mdivide_left_tri_low_transpose(&l, &mdivide_left_tri_low(&l, b)?)
}

/// `A⁻¹` for symmetric positive definite `A`.
pub fn inverse_spd<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    let n = a.nrows();
    let eye = Matrix::from_fn(n, n, |i, j| if i == j { T::one() } else { T::zero() });
    mdivide_left_spd(a, &eye)
}

/// `ln det(A)` for symmetric positive definite `A`, as `2 sum ln L_ii`.
pub fn log_determinant_spd<T: Real>(a: &Matrix<T>) -> Result<T> {
    let l = cholesky_decompose(a)?;
    Ok((0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<T>() * 2.0)
}

/// Quadratic form `xᵀ A x`.
pub fn quad_form<T: Real>(a: &Matrix<T>, x: &[T]) -> Result<T> {
    if !a.is_square() || a.nrows() != x.len() {
        return Err(Error::dimension(
            "quad_form",
            format!(
                "{}x{} matrix with a vector of length {}",
                a.nrows(),
                a.ncols(),
                x.len()
            ),
        ));
    }
    let ax = multiply_vector(a, x)?;
    Ok(probgrad_autodiff::functions::dot_product(x, &ax))
}

/// `L Lᵀ` for lower triangular `L` (entries above the diagonal are treated as zero).
pub fn multiply_lower_tri_self_transpose<T: Real>(l: &Matrix<T>) -> Result<Matrix<T>> {
    require_square("multiply_lower_tri_self_transpose", l)?;
    let n = l.nrows();
    let lower = Matrix::from_fn(n, n, |i, j| if i >= j { l[(i, j)] } else { T::zero() });
    multiply(&lower, &lower.transpose())
}

fn elementwise<T: Real>(
    function: &'static str,
    a: &Matrix<T>,
    b: &Matrix<T>,
    op: impl Fn(T, T) -> T,
) -> Result<Matrix<T>> {
    if a.shape() != b.shape() {
        return Err(Error::dimension(
            function,
            format!("{:?} and {:?}", a.shape(), b.shape()),
        ));
    }
    Ok(Matrix::from_fn(a.nrows(), a.ncols(), |i, j| {
        op(a[(i, j)], b[(i, j)])
    }))
}

/// Elementwise sum.
pub fn add<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    elementwise("add", a, b, |x, y| x + y)
}

/// Elementwise difference.
pub fn subtract<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    elementwise("subtract", a, b, |x, y| x - y)
}

/// `c A` for a scalar `c`.
pub fn scale<T: Real>(a: &Matrix<T>, c: T) -> Matrix<T> {
    a.map(|&x| x * c)
}
