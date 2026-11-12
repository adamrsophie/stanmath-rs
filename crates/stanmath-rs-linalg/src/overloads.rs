//! Structural overloads of Stan's matrix functions that the main signatures cannot express.
//!
//! * Vector forms of products, solves, quadratic forms, and concatenation. Vectors and row
//!   vectors are both slices, so one function serves both where Stan has two signatures.
//! * Gaussian process inputs given as an array of scalars, and a length scale per dimension.
//! * A diagonal `D`, given as a vector, for `trace_gen_inv_quad_form_ldlt`.
//! * Complex matrices and vectors (the `_complex` suffix). The arithmetic functions are generic
//!   over [`Ring`], so they also accept real scalars and `i64`; the decompositions build on the
//!   complex Schur decomposition of `complex_linalg.rs`.
//!
//! The Stan Functions Reference leaves three complex conventions open; these are the readings
//! used here. Complex dot products do not conjugate either argument (`sum_i x_i y_i`).
//! `symmetrize_from_lower_tri` makes `A = Aᵀ`, without conjugation. The `_sym` decompositions
//! read "symmetric" as Hermitian (`A = Aᴴ`), which gives real eigenvalues in ascending order.

#![allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices

use core::f64::consts::SQRT_2;
use core::iter::Sum;
use core::ops::{Add, Mul, Sub};

use stanmath_rs_autodiff::{Complex, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::complex_linalg::{schur, schur_vector};
use crate::ops::{SYMMETRY_TOL, at};
use crate::{
    LdltFactor, append_col, append_row, diag_matrix, gp_exp_quad_cov, gp_exp_quad_cov_cross,
    gp_exponential_cov, gp_exponential_cov_cross, gp_matern32_cov, gp_matern32_cov_cross,
    gp_matern52_cov, gp_matern52_cov_cross, mdivide_left, mdivide_left_spd, mdivide_left_tri_low,
    mdivide_right, mdivide_right_spd, mdivide_right_tri_low, multiply, quad_form_sym,
    require_square, to_matrix_from_row_vector, to_matrix_from_vector, trace_gen_inv_quad_form_ldlt,
};

fn length_mismatch(function: &'static str, what: &str, expected: usize, got: usize) -> Error {
    Error::dimension(
        function,
        format!("{what} has length {got}, expected {expected}"),
    )
}

// Real vector forms.

/// `x A` for a row vector `x`. Stan's `multiply(row_vector, matrix)`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `x` has one element per row of `A`.
pub fn multiply_row_vector<T: Real>(x: &[T], a: &Matrix<T>) -> Result<Vec<T>> {
    Ok(multiply(&to_matrix_from_row_vector(x), a)?
        .as_slice()
        .to_vec())
}

/// Outer product `x yᵀ` of a vector and a row vector. Stan's `multiply(vector, row_vector)`;
/// generic over [`Ring`], so it also serves complex vectors.
pub fn multiply_outer<R: Ring>(x: &[R], y: &[R]) -> Matrix<R> {
    Matrix::from_fn(x.len(), y.len(), |i, j| x[i] * y[j])
}

/// `A⁻¹ b` for a vector `b`. Stan's `mdivide_left(matrix, vector)`.
///
/// # Errors
///
/// As [`mdivide_left`].
pub fn mdivide_left_vector<T: Real>(a: &Matrix<T>, b: &[T]) -> Result<Vec<T>> {
    Ok(mdivide_left(a, &to_matrix_from_vector(b))?
        .as_slice()
        .to_vec())
}

/// `A⁻¹ b` for a symmetric positive definite `A` and a vector `b`. Stan's
/// `mdivide_left_spd(matrix, vector)`.
///
/// # Errors
///
/// As [`mdivide_left_spd`].
pub fn mdivide_left_spd_vector<T: Real>(a: &Matrix<T>, b: &[T]) -> Result<Vec<T>> {
    Ok(mdivide_left_spd(a, &to_matrix_from_vector(b))?
        .as_slice()
        .to_vec())
}

/// `L⁻¹ b` for a lower triangular `L` and a vector `b`. Stan's
/// `mdivide_left_tri_low(matrix, vector)`.
///
/// # Errors
///
/// As [`mdivide_left_tri_low`].
pub fn mdivide_left_tri_low_vector<T: Real>(l: &Matrix<T>, b: &[T]) -> Result<Vec<T>> {
    Ok(mdivide_left_tri_low(l, &to_matrix_from_vector(b))?
        .as_slice()
        .to_vec())
}

/// `b A⁻¹` for a row vector `b`. Stan's `mdivide_right(row_vector, matrix)`.
///
/// # Errors
///
/// As [`mdivide_right`].
pub fn mdivide_right_row_vector<T: Real>(b: &[T], a: &Matrix<T>) -> Result<Vec<T>> {
    Ok(mdivide_right(&to_matrix_from_row_vector(b), a)?
        .as_slice()
        .to_vec())
}

/// `b A⁻¹` for a row vector `b` and a symmetric positive definite `A`. Stan's
/// `mdivide_right_spd(row_vector, matrix)`.
///
/// # Errors
///
/// As [`mdivide_right_spd`].
pub fn mdivide_right_spd_row_vector<T: Real>(b: &[T], a: &Matrix<T>) -> Result<Vec<T>> {
    Ok(mdivide_right_spd(&to_matrix_from_row_vector(b), a)?
        .as_slice()
        .to_vec())
}

/// `b L⁻¹` for a row vector `b` and a lower triangular `L`. Stan's
/// `mdivide_right_tri_low(row_vector, matrix)`.
///
/// # Errors
///
/// As [`mdivide_right_tri_low`].
pub fn mdivide_right_tri_low_row_vector<T: Real>(b: &[T], l: &Matrix<T>) -> Result<Vec<T>> {
    Ok(mdivide_right_tri_low(&to_matrix_from_row_vector(b), l)?
        .as_slice()
        .to_vec())
}

/// `Bᵀ A B` for a square `A`. Stan's `quad_form(matrix, matrix)`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `A` is not square or `B` does not have one row per column
/// of `A`.
pub fn quad_form_matrix<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    require_square("quad_form", a)?;
    multiply(&b.transpose(), &multiply(a, b)?)
}

/// `xᵀ A x` for a symmetric `A`. Stan's `quad_form_sym(matrix, vector)`.
///
/// # Errors
///
/// As [`quad_form_sym`].
pub fn quad_form_sym_vector<T: Real>(a: &Matrix<T>, x: &[T]) -> Result<T> {
    Ok(quad_form_sym(a, &to_matrix_from_vector(x))?[(0, 0)])
}

/// `trace(diag(d) Bᵀ A⁻¹ B)` for a factored symmetric positive definite `A` and the diagonal
/// of `D` given as a vector.
///
/// # Errors
///
/// As [`trace_gen_inv_quad_form_ldlt`].
pub fn trace_gen_inv_quad_form_ldlt_diag<T: Real>(
    d: &[T],
    a: &LdltFactor<T>,
    b: &Matrix<T>,
) -> Result<T> {
    // ponytail: forms the dense diagonal matrix; a weighted column sum avoids the n^2 if it matters.
    trace_gen_inv_quad_form_ldlt(&diag_matrix(d), a, b)
}

// Concatenation with scalars and vectors; `T: Clone`, so complex elements work too.

/// `[x, v...]`. Stan's `append_row(real, vector)` and `append_col(real, row_vector)`.
pub fn prepend_scalar<T: Clone>(x: T, v: &[T]) -> Vec<T> {
    core::iter::once(x).chain(v.iter().cloned()).collect()
}

/// `[v..., x]`. Stan's `append_row(vector, real)` and `append_col(row_vector, real)`.
pub fn append_scalar<T: Clone>(v: &[T], x: T) -> Vec<T> {
    v.iter().cloned().chain(core::iter::once(x)).collect()
}

/// `[A, v]`: the column `v` added on the right of `A`. Stan's `append_col(matrix, vector)`.
///
/// # Errors
///
/// As [`append_col`].
pub fn append_col_matrix_vector<T: Clone>(a: &Matrix<T>, v: &[T]) -> Result<Matrix<T>> {
    append_col(a, &to_matrix_from_vector(v))
}

/// `[v, A]`: the column `v` added on the left of `A`. Stan's `append_col(vector, matrix)`.
///
/// # Errors
///
/// As [`append_col`].
pub fn append_col_vector_matrix<T: Clone>(v: &[T], a: &Matrix<T>) -> Result<Matrix<T>> {
    append_col(&to_matrix_from_vector(v), a)
}

/// The two column matrix `[x, y]`. Stan's `append_col(vector, vector)`.
///
/// # Errors
///
/// As [`append_col`].
pub fn append_col_vectors<T: Clone>(x: &[T], y: &[T]) -> Result<Matrix<T>> {
    append_col(&to_matrix_from_vector(x), &to_matrix_from_vector(y))
}

/// `A` with the row `v` added below. Stan's `append_row(matrix, row_vector)`.
///
/// # Errors
///
/// As [`append_row`].
pub fn append_row_matrix_row_vector<T: Clone>(a: &Matrix<T>, v: &[T]) -> Result<Matrix<T>> {
    append_row(a, &to_matrix_from_row_vector(v))
}

/// `A` with the row `v` added above. Stan's `append_row(row_vector, matrix)`.
///
/// # Errors
///
/// As [`append_row`].
pub fn append_row_row_vector_matrix<T: Clone>(v: &[T], a: &Matrix<T>) -> Result<Matrix<T>> {
    append_row(&to_matrix_from_row_vector(v), a)
}

/// The two row matrix with rows `x` and `y`. Stan's `append_row(row_vector, row_vector)`.
///
/// # Errors
///
/// As [`append_row`].
pub fn append_row_row_vectors<T: Clone>(x: &[T], y: &[T]) -> Result<Matrix<T>> {
    append_row(&to_matrix_from_row_vector(x), &to_matrix_from_row_vector(y))
}

/// The matrix whose columns are the vectors `vs`. Stan's `to_matrix(array[] vector)`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the vectors differ in length.
pub fn to_matrix_from_vector_array<T: Clone>(vs: &[Vec<T>]) -> Result<Matrix<T>> {
    let rows = vs.first().map_or(0, Vec::len);
    if let Some(v) = vs.iter().find(|v| v.len() != rows) {
        return Err(length_mismatch("to_matrix", "a column", rows, v.len()));
    }
    Ok(Matrix::from_fn(rows, vs.len(), |i, j| vs[j][i].clone()))
}

// Gaussian process inputs.

/// The one element points `[x_i]` of a one dimensional input, so the `gp_*_cov` functions take
/// Stan's `array[] real` signatures: `gp_exp_quad_cov(&gp_points_1d(&x), sigma, l)`.
pub fn gp_points_1d<T: Clone>(x: &[T]) -> Vec<Vec<T>> {
    x.iter().map(|v| vec![v.clone()]).collect()
}

/// Divides coordinate `d` of every point by `length_scale[d]`.
fn ard_scale<T: Real>(
    function: &'static str,
    x: &[Vec<T>],
    length_scale: &[T],
) -> Result<Vec<Vec<T>>> {
    if let Some(l) = length_scale
        .iter()
        .find(|l| !(l.value() > 0.0 && l.value().is_finite()))
    {
        return Err(Error::parameter(
            function,
            "length_scale",
            l.value(),
            "positive and finite",
        ));
    }
    x.iter()
        .map(|p| {
            if p.len() != length_scale.len() {
                return Err(length_mismatch(
                    function,
                    "a point",
                    length_scale.len(),
                    p.len(),
                ));
            }
            Ok(p.iter()
                .zip(length_scale)
                .map(|(&v, &l)| v / l)
                .collect::<Vec<T>>())
        })
        .collect()
}

/// Generates the covariance and cross covariance forms with one length scale per dimension.
macro_rules! ard_kernel {
    ($iso:ident, $iso_cross:ident, $ard:ident, $ard_cross:ident, $name:literal) => {
        #[doc = concat!("[`", stringify!($iso), "`] with one length scale per dimension: ")]
        /// coordinate `d` is divided by `length_scale[d]` before the kernel is applied with unit
        /// length scale.
        ///
        /// # Errors
        ///
        #[doc = concat!("As [`", stringify!($iso), "`]; also [`Error::DimensionMismatch`]")]
        /// unless every point has one coordinate per length scale, and
        /// [`Error::InvalidParameter`] for a length scale that is not positive and finite.
        pub fn $ard<T: Real>(x: &[Vec<T>], sigma: T, length_scale: &[T]) -> Result<Matrix<T>> {
            $iso(&ard_scale($name, x, length_scale)?, sigma, T::from_f64(1.0))
        }

        #[doc = concat!("[`", stringify!($iso_cross), "`] with one length scale per dimension;")]
        #[doc = concat!(" see [`", stringify!($ard), "`].")]
        ///
        /// # Errors
        ///
        #[doc = concat!("As [`", stringify!($ard), "`].")]
        pub fn $ard_cross<T: Real>(
            x1: &[Vec<T>],
            x2: &[Vec<T>],
            sigma: T,
            length_scale: &[T],
        ) -> Result<Matrix<T>> {
            $iso_cross(
                &ard_scale($name, x1, length_scale)?,
                &ard_scale($name, x2, length_scale)?,
                sigma,
                T::from_f64(1.0),
            )
        }
    };
}

ard_kernel!(
    gp_exp_quad_cov,
    gp_exp_quad_cov_cross,
    gp_exp_quad_cov_ard,
    gp_exp_quad_cov_ard_cross,
    "gp_exp_quad_cov"
);
ard_kernel!(
    gp_matern32_cov,
    gp_matern32_cov_cross,
    gp_matern32_cov_ard,
    gp_matern32_cov_ard_cross,
    "gp_matern32_cov"
);
ard_kernel!(
    gp_matern52_cov,
    gp_matern52_cov_cross,
    gp_matern52_cov_ard,
    gp_matern52_cov_ard_cross,
    "gp_matern52_cov"
);
ard_kernel!(
    gp_exponential_cov,
    gp_exponential_cov_cross,
    gp_exponential_cov_ard,
    gp_exponential_cov_ard_cross,
    "gp_exponential_cov"
);

// Complex (and generic ring) arithmetic.

/// Element types with the operations the `_complex` functions need: real scalars, complex
/// numbers, and `i64` all qualify. The empty sum is the zero.
pub trait Ring: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Sum {}

impl<R: Copy + Add<Output = R> + Sub<Output = R> + Mul<Output = R> + Sum> Ring for R {}

fn zero<R: Ring>() -> R {
    core::iter::empty::<R>().sum()
}

/// The dot product convention of every `_complex` function: `sum_i x_i y_i`, no conjugation.
fn dot<'a, R: Ring + 'a>(
    x: impl IntoIterator<Item = &'a R>,
    y: impl IntoIterator<Item = &'a R>,
) -> R {
    x.into_iter().zip(y).map(|(&a, &b)| a * b).sum()
}

fn same_shape<R>(function: &'static str, a: &Matrix<R>, b: &Matrix<R>) -> Result<()> {
    if a.shape() == b.shape() {
        Ok(())
    } else {
        Err(Error::dimension(
            function,
            format!("{:?} and {:?}", a.shape(), b.shape()),
        ))
    }
}

/// Matrix product `A B`. Stan's `multiply` for complex matrices.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `A` has one column per row of `B`.
pub fn multiply_complex<R: Ring>(a: &Matrix<R>, b: &Matrix<R>) -> Result<Matrix<R>> {
    if a.ncols() != b.nrows() {
        return Err(Error::dimension(
            "multiply",
            format!("{:?} times {:?}", a.shape(), b.shape()),
        ));
    }
    Ok(Matrix::from_fn(a.nrows(), b.ncols(), |i, j| {
        dot(a.row(i), b.column(j))
    }))
}

/// `A x` for a vector `x`. Stan's `multiply(complex_matrix, complex_vector)`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `x` has one element per column of `A`.
pub fn multiply_vector_complex<R: Ring>(a: &Matrix<R>, x: &[R]) -> Result<Vec<R>> {
    if x.len() != a.ncols() {
        return Err(length_mismatch("multiply", "x", a.ncols(), x.len()));
    }
    Ok((0..a.nrows()).map(|i| dot(a.row(i), x)).collect())
}

/// `x A` for a row vector `x`. Stan's `multiply(complex_row_vector, complex_matrix)`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `x` has one element per row of `A`.
pub fn multiply_row_vector_complex<R: Ring>(x: &[R], a: &Matrix<R>) -> Result<Vec<R>> {
    if x.len() != a.nrows() {
        return Err(length_mismatch("multiply", "x", a.nrows(), x.len()));
    }
    Ok((0..a.ncols()).map(|j| dot(x, a.column(j))).collect())
}

/// `sum_i x_i y_i`, without conjugation. Stan's `dot_product` and `multiply(row_vector, vector)`
/// for complex vectors.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the lengths differ.
pub fn dot_product_complex<R: Ring>(x: &[R], y: &[R]) -> Result<R> {
    if x.len() != y.len() {
        return Err(length_mismatch("dot_product", "y", x.len(), y.len()));
    }
    Ok(dot(x, y))
}

/// `sum_i x_i x_i`, without conjugation. Stan's `dot_self` for complex vectors.
pub fn dot_self_complex<R: Ring>(x: &[R]) -> R {
    dot(x, x)
}

/// Dot products of matching columns. Stan's `columns_dot_product` for complex matrices; for
/// two vectors pass `to_matrix_from_vector` of each.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the shapes differ.
pub fn columns_dot_product_complex<R: Ring>(a: &Matrix<R>, b: &Matrix<R>) -> Result<Vec<R>> {
    same_shape("columns_dot_product", a, b)?;
    Ok((0..a.ncols())
        .map(|j| dot(a.column(j), b.column(j)))
        .collect())
}

/// Dot products of matching rows. Stan's `rows_dot_product` for complex matrices.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the shapes differ.
pub fn rows_dot_product_complex<R: Ring>(a: &Matrix<R>, b: &Matrix<R>) -> Result<Vec<R>> {
    same_shape("rows_dot_product", a, b)?;
    Ok((0..a.nrows()).map(|i| dot(a.row(i), b.row(i))).collect())
}

/// Dot product of each column with itself. Stan's `columns_dot_self` for complex matrices.
pub fn columns_dot_self_complex<R: Ring>(a: &Matrix<R>) -> Vec<R> {
    (0..a.ncols())
        .map(|j| dot(a.column(j), a.column(j)))
        .collect()
}

/// Dot product of each row with itself. Stan's `rows_dot_self` for complex matrices.
pub fn rows_dot_self_complex<R: Ring>(a: &Matrix<R>) -> Vec<R> {
    (0..a.nrows()).map(|i| dot(a.row(i), a.row(i))).collect()
}

/// Sum of the diagonal; zero for an empty matrix. Stan's `trace` for complex matrices.
pub fn trace_complex<R: Ring>(a: &Matrix<R>) -> R {
    (0..a.nrows().min(a.ncols())).map(|i| a[(i, i)]).sum()
}

/// The square matrix with diagonal `v`. Stan's `diag_matrix` for complex vectors.
pub fn diag_matrix_complex<R: Ring>(v: &[R]) -> Matrix<R> {
    let n = v.len();
    Matrix::from_fn(n, n, |i, j| if i == j { v[i] } else { zero() })
}

/// The diagonal of `A`. Stan's `diagonal` for complex matrices.
pub fn diagonal_complex<R: Ring>(a: &Matrix<R>) -> Vec<R> {
    (0..a.nrows().min(a.ncols())).map(|i| a[(i, i)]).collect()
}

/// `A` with `c` added to every diagonal element. Stan's `add_diag(complex_matrix, complex)`.
pub fn add_diag_complex<R: Ring>(a: &Matrix<R>, c: R) -> Matrix<R> {
    Matrix::from_fn(a.nrows(), a.ncols(), |i, j| {
        if i == j { a[(i, j)] + c } else { a[(i, j)] }
    })
}

/// `A` with `d` added to its diagonal. Stan's `add_diag` with a complex vector or row vector.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `d` has one element per diagonal entry of `A`.
pub fn add_diag_vector_complex<R: Ring>(a: &Matrix<R>, d: &[R]) -> Result<Matrix<R>> {
    let n = a.nrows().min(a.ncols());
    if d.len() != n {
        return Err(length_mismatch("add_diag", "d", n, d.len()));
    }
    Ok(Matrix::from_fn(a.nrows(), a.ncols(), |i, j| {
        if i == j { a[(i, j)] + d[i] } else { a[(i, j)] }
    }))
}

/// `diag(v) M`. Stan's `diag_pre_multiply` for complex arguments.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `v` has one element per row of `M`.
pub fn diag_pre_multiply_complex<R: Ring>(v: &[R], m: &Matrix<R>) -> Result<Matrix<R>> {
    if v.len() != m.nrows() {
        return Err(length_mismatch(
            "diag_pre_multiply",
            "v",
            m.nrows(),
            v.len(),
        ));
    }
    Ok(Matrix::from_fn(m.nrows(), m.ncols(), |i, j| {
        v[i] * m[(i, j)]
    }))
}

/// `M diag(v)`. Stan's `diag_post_multiply` for complex arguments.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `v` has one element per column of `M`.
pub fn diag_post_multiply_complex<R: Ring>(m: &Matrix<R>, v: &[R]) -> Result<Matrix<R>> {
    if v.len() != m.ncols() {
        return Err(length_mismatch(
            "diag_post_multiply",
            "v",
            m.ncols(),
            v.len(),
        ));
    }
    Ok(Matrix::from_fn(m.nrows(), m.ncols(), |i, j| {
        m[(i, j)] * v[j]
    }))
}

/// Running sums `y_n = x_1 + ... + x_n`. Stan's `cumulative_sum` for complex arrays and vectors.
pub fn cumulative_sum_complex<R: Ring>(x: &[R]) -> Vec<R> {
    let mut acc = zero::<R>();
    x.iter()
        .map(|&v| {
            acc = acc + v;
            acc
        })
        .collect()
}

/// Running sums of an integer array. Stan's `cumulative_sum(array[] int)`.
pub fn cumulative_sum_int(x: &[i64]) -> Vec<i64> {
    cumulative_sum_complex(x)
}

/// Sum of the elements; zero when empty. Stan's `sum` for complex containers (for a matrix pass
/// [`Matrix::as_slice`]).
pub fn sum_complex<R: Ring>(x: &[R]) -> R {
    x.iter().copied().sum()
}

/// Product of the elements; one when empty. Stan's `prod` for complex containers (for a matrix
/// pass [`Matrix::as_slice`]).
pub fn prod_complex<T: Real>(x: &[Complex<T>]) -> Complex<T> {
    x.iter().fold(Complex::one(), |p, &z| p * z)
}

/// The matrix with `M`'s lower triangle (diagonal included) mirrored above the diagonal, without
/// conjugation. Stan's `symmetrize_from_lower_tri` for complex matrices.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input.
pub fn symmetrize_from_lower_tri_complex<T: Clone>(m: &Matrix<T>) -> Result<Matrix<T>> {
    require_square("symmetrize_from_lower_tri", m)?;
    let n = m.nrows();
    Ok(Matrix::from_fn(n, n, |i, j| {
        if i >= j {
            m[(i, j)].clone()
        } else {
            m[(j, i)].clone()
        }
    }))
}

/// `B A⁻¹` for complex matrices, by Gaussian elimination with partial pivoting on `Aᵀ Xᵀ = Bᵀ`.
/// Stan's `mdivide_right(complex_matrix, complex_matrix)`; for a row vector `b` pass
/// `to_matrix_from_row_vector(b)`.
///
/// Derivatives flow through the elimination in `T`; there is no custom reverse rule.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `A` is not square or `B` does not have one column per row
/// of `A`, and [`Error::InvalidParameter`] when `A` is singular.
pub fn mdivide_right_complex<T: Real>(
    b: &Matrix<Complex<T>>,
    a: &Matrix<Complex<T>>,
) -> Result<Matrix<Complex<T>>> {
    const F: &str = "mdivide_right";
    require_square(F, a)?;
    let n = a.nrows();
    if b.ncols() != n {
        return Err(Error::dimension(
            F,
            format!("B is {:?} and A is {n}x{n}", b.shape()),
        ));
    }
    let r = b.nrows();
    // m holds Aᵀ and x holds Bᵀ (n by r), both column major.
    let mut m = a.transpose().as_slice().to_vec();
    let mut x = b.transpose().as_slice().to_vec();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| {
                m[at(n, i, k)]
                    .value()
                    .abs()
                    .total_cmp(&m[at(n, j, k)].value().abs())
            })
            .unwrap_or(k);
        if !(m[at(n, p, k)].value().abs() > 0.0) {
            return Err(Error::parameter(F, "A", 0.0, "nonsingular"));
        }
        if p != k {
            for j in 0..n {
                m.swap(at(n, p, j), at(n, k, j));
            }
            for j in 0..r {
                x.swap(at(n, p, j), at(n, k, j));
            }
        }
        let pivot = m[at(n, k, k)];
        for i in k + 1..n {
            let f = m[at(n, i, k)] / pivot;
            for j in k..n {
                let t = f * m[at(n, k, j)];
                m[at(n, i, j)] -= t;
            }
            for j in 0..r {
                let t = f * x[at(n, k, j)];
                x[at(n, i, j)] -= t;
            }
        }
    }
    for j in 0..r {
        for i in (0..n).rev() {
            let mut s = x[at(n, i, j)];
            for k in i + 1..n {
                s -= m[at(n, i, k)] * x[at(n, k, j)];
            }
            x[at(n, i, j)] = s / m[at(n, i, i)];
        }
    }
    Ok(Matrix::from_column_major(n, r, x).transpose())
}

// Complex decompositions.

#[allow(clippy::type_complexity)]
fn schur_complex<T: Real>(
    function: &'static str,
    a: &Matrix<Complex<T>>,
) -> Result<(Matrix<Complex<T>>, Matrix<Complex<T>>)> {
    require_square(function, a)?;
    let n = a.nrows();
    let (u, t, _) = schur(function, a.as_slice(), n)?;
    Ok((
        Matrix::from_column_major(n, n, u),
        Matrix::from_column_major(n, n, t),
    ))
}

/// Complex Schur decomposition `A = U T Uᴴ` of a complex square matrix: returns `(U, T)` with
/// `U` unitary and `T` upper triangular. Stan's `complex_schur_decompose(complex_matrix)`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input, [`Error::InvalidParameter`] for a
/// non-finite entry, and [`Error::NoConvergence`] when the QR iteration does not converge.
#[allow(clippy::type_complexity)]
pub fn complex_schur_decompose_complex<T: Real>(
    a: &Matrix<Complex<T>>,
) -> Result<(Matrix<Complex<T>>, Matrix<Complex<T>>)> {
    schur_complex("complex_schur_decompose", a)
}

/// Unitary factor `U` of [`complex_schur_decompose_complex`].
///
/// # Errors
///
/// As [`complex_schur_decompose_complex`].
pub fn complex_schur_decompose_u_complex<T: Real>(
    a: &Matrix<Complex<T>>,
) -> Result<Matrix<Complex<T>>> {
    Ok(schur_complex("complex_schur_decompose_u", a)?.0)
}

/// Upper triangular factor `T` of [`complex_schur_decompose_complex`].
///
/// # Errors
///
/// As [`complex_schur_decompose_complex`].
pub fn complex_schur_decompose_t_complex<T: Real>(
    a: &Matrix<Complex<T>>,
) -> Result<Matrix<Complex<T>>> {
    Ok(schur_complex("complex_schur_decompose_t", a)?.1)
}

/// Eigenvalues (the Schur diagonal) and, when asked, unit eigenvectors stacked column major.
#[allow(clippy::type_complexity)]
fn eigen_complex<T: Real>(
    function: &'static str,
    a: &Matrix<Complex<T>>,
    with_vectors: bool,
) -> Result<(Vec<Complex<T>>, Vec<Complex<T>>)> {
    require_square(function, a)?;
    let n = a.nrows();
    let (u, t, diag0) = schur(function, a.as_slice(), n)?;
    let scale = diag0
        .iter()
        .fold(0.0f64, |m, z| m.max(z.abs()))
        .max(f64::MIN_POSITIVE);
    let small = f64::EPSILON * scale;
    let values = (0..n).map(|k| t[at(n, k, k)]).collect();
    let vectors = if with_vectors {
        (0..n)
            .flat_map(|k| schur_vector(&u, &t, n, k, small))
            .collect()
    } else {
        Vec::new()
    };
    Ok((values, vectors))
}

/// Eigenvectors (as columns) and eigenvalues of a complex square matrix. Stan's
/// `eigendecompose(complex_matrix)`. Eigenvalues follow the Schur diagonal, unsorted as in Stan;
/// eigenvectors have unit norm with their largest component real and positive.
///
/// # Errors
///
/// As [`complex_schur_decompose_complex`].
#[allow(clippy::type_complexity)]
pub fn eigendecompose_complex<T: Real>(
    a: &Matrix<Complex<T>>,
) -> Result<(Matrix<Complex<T>>, Vec<Complex<T>>)> {
    let n = a.nrows();
    let (values, vectors) = eigen_complex("eigendecompose", a, true)?;
    Ok((Matrix::from_column_major(n, n, vectors), values))
}

/// Eigenvalues of a complex square matrix; see [`eigendecompose_complex`].
///
/// # Errors
///
/// As [`complex_schur_decompose_complex`].
pub fn eigenvalues_complex<T: Real>(a: &Matrix<Complex<T>>) -> Result<Vec<Complex<T>>> {
    Ok(eigen_complex("eigenvalues", a, false)?.0)
}

/// Eigenvectors (as columns) of a complex square matrix; see [`eigendecompose_complex`].
///
/// # Errors
///
/// As [`complex_schur_decompose_complex`].
pub fn eigenvectors_complex<T: Real>(a: &Matrix<Complex<T>>) -> Result<Matrix<Complex<T>>> {
    Ok(eigendecompose_complex(a)?.0)
}

fn check_hermitian<T: Real>(function: &'static str, a: &Matrix<Complex<T>>) -> Result<()> {
    let n = a.nrows();
    for j in 0..n {
        for i in j..n {
            let gap = (a[(i, j)].value() - a[(j, i)].value().conj()).abs();
            if !(gap <= SYMMETRY_TOL) {
                return Err(Error::NotSymmetric {
                    function,
                    row: i,
                    col: j,
                });
            }
        }
    }
    Ok(())
}

/// Hermitian eigendecomposition through the complex Schur form, which is diagonal for a
/// Hermitian matrix: eigenvalues ascending (imaginary parts exactly zero) and the matching
/// columns of `U`.
#[allow(clippy::type_complexity)]
fn eigen_hermitian<T: Real>(
    function: &'static str,
    a: &Matrix<Complex<T>>,
) -> Result<(Matrix<Complex<T>>, Vec<Complex<T>>)> {
    require_square(function, a)?;
    check_hermitian(function, a)?;
    let n = a.nrows();
    let (u, t, diag0) = schur(function, a.as_slice(), n)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| diag0[i].re.total_cmp(&diag0[j].re));
    let values = order
        .iter()
        .map(|&k| Complex::from_real(t[at(n, k, k)].re))
        .collect();
    let vectors = Matrix::from_fn(n, n, |i, c| u[at(n, i, order[c])]);
    Ok((vectors, values))
}

/// Eigenvectors (as columns) and eigenvalues (ascending) of a Hermitian matrix. Stan's
/// `eigendecompose_sym(complex_matrix)`, reading "symmetric" as Hermitian.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input, [`Error::NotSymmetric`] when `A` is not
/// Hermitian, and as [`complex_schur_decompose_complex`].
#[allow(clippy::type_complexity)]
pub fn eigendecompose_sym_complex<T: Real>(
    a: &Matrix<Complex<T>>,
) -> Result<(Matrix<Complex<T>>, Vec<Complex<T>>)> {
    eigen_hermitian("eigendecompose_sym", a)
}

/// Eigenvalues (ascending) of a Hermitian matrix; see [`eigendecompose_sym_complex`].
///
/// # Errors
///
/// As [`eigendecompose_sym_complex`].
pub fn eigenvalues_sym_complex<T: Real>(a: &Matrix<Complex<T>>) -> Result<Vec<Complex<T>>> {
    Ok(eigen_hermitian("eigenvalues_sym", a)?.1)
}

/// Eigenvectors of a Hermitian matrix, ordered by ascending eigenvalue; see
/// [`eigendecompose_sym_complex`].
///
/// # Errors
///
/// As [`eigendecompose_sym_complex`].
pub fn eigenvectors_sym_complex<T: Real>(a: &Matrix<Complex<T>>) -> Result<Matrix<Complex<T>>> {
    Ok(eigen_hermitian("eigenvectors_sym", a)?.0)
}

/// Thin singular value decomposition `A = U diag(σ) Vᴴ` of a complex matrix, `σ` descending.
///
/// It uses the Hermitian eigendecomposition of `[[0, A], [Aᴴ, 0]]`, whose eigenvalues are `±σ_i`
/// (and zeros), with eigenvectors `(u_i, ±v_i) / sqrt(2)`. As for real input, the singular
/// vectors of zero singular values are not reliable.
#[allow(clippy::type_complexity)]
fn svd_complex_parts<T: Real>(
    function: &'static str,
    a: &Matrix<Complex<T>>,
) -> Result<(Vec<T>, Matrix<Complex<T>>, Matrix<Complex<T>>)> {
    let (m, n) = a.shape();
    let k = m.min(n);
    if k == 0 {
        return Ok((
            Vec::new(),
            Matrix::from_column_major(m, 0, Vec::new()),
            Matrix::from_column_major(n, 0, Vec::new()),
        ));
    }
    let size = m + n;
    let aug = Matrix::from_fn(size, size, |i, j| {
        if i < m && j >= m {
            a[(i, j - m)]
        } else if i >= m && j < m {
            a[(j, i - m)].conj()
        } else {
            Complex::zero()
        }
    });
    let (vectors, values) = eigen_hermitian(function, &aug)?;
    let col = |r: usize| size - 1 - r;
    let root2 = T::from_f64(SQRT_2);
    let sigma = (0..k).map(|r| values[col(r)].re).collect();
    let u = Matrix::from_fn(m, k, |i, r| vectors[(i, col(r))] * root2);
    let v = Matrix::from_fn(n, k, |i, r| vectors[(m + i, col(r))] * root2);
    Ok((sigma, u, v))
}

/// `(U, σ, V)` with `A = U diag(σ) Vᴴ`, `σ` real and descending. Stan's `svd(complex_matrix)`.
///
/// # Errors
///
/// [`Error::NoConvergence`] when the underlying Schur iteration does not converge, and
/// [`Error::InvalidParameter`] for a non-finite entry.
#[allow(clippy::type_complexity)]
pub fn svd_complex<T: Real>(
    a: &Matrix<Complex<T>>,
) -> Result<(Matrix<Complex<T>>, Vec<T>, Matrix<Complex<T>>)> {
    let (sigma, u, v) = svd_complex_parts("svd", a)?;
    Ok((u, sigma, v))
}

/// Singular values of a complex matrix, real and descending. Stan's
/// `singular_values(complex_matrix)`.
///
/// # Errors
///
/// As [`svd_complex`].
pub fn singular_values_complex<T: Real>(a: &Matrix<Complex<T>>) -> Result<Vec<T>> {
    Ok(svd_complex_parts("singular_values", a)?.0)
}

/// Left singular vectors of a complex matrix (`m` by `min(m, n)`). Stan's
/// `svd_U(complex_matrix)`.
///
/// # Errors
///
/// As [`svd_complex`].
#[allow(non_snake_case)]
pub fn svd_U_complex<T: Real>(a: &Matrix<Complex<T>>) -> Result<Matrix<Complex<T>>> {
    Ok(svd_complex_parts("svd_U", a)?.1)
}

/// Right singular vectors of a complex matrix (`n` by `min(m, n)`). Stan's
/// `svd_V(complex_matrix)`.
///
/// # Errors
///
/// As [`svd_complex`].
#[allow(non_snake_case)]
pub fn svd_V_complex<T: Real>(a: &Matrix<Complex<T>>) -> Result<Matrix<Complex<T>>> {
    Ok(svd_complex_parts("svd_V", a)?.2)
}
