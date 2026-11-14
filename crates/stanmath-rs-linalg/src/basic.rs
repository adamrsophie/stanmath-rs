//! Products, quadratic forms, reductions, and elementwise operations.
//!
//! These are compositions of the structured operations of this crate and of the fused n-ary
//! functions of `stanmath-rs-autodiff` (`dot_product`, `sum`), so in reverse mode a dot product of
//! length `n` is one tape node.

#![allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices

use stanmath_rs_autodiff::Real;
use stanmath_rs_autodiff::functions::{dot_product, log_sum_exp_slice, sum};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::ops::check_symmetric;
use crate::{elementwise, multiply, require_square};

fn length_mismatch(function: &'static str, what: &str, expected: usize, got: usize) -> Error {
    Error::dimension(
        function,
        format!("{what} has length {got}, expected {expected}"),
    )
}

/// Transposed copy of `A`.
pub fn transpose<T: Real>(a: &Matrix<T>) -> Matrix<T> {
    a.transpose()
}

/// The `n` by `n` identity matrix.
pub fn identity_matrix<T: Real>(n: usize) -> Matrix<T> {
    Matrix::from_fn(n, n, |i, j| if i == j { T::one() } else { T::zero() })
}

/// Square matrix with `v` on the diagonal and zeros elsewhere.
pub fn diag_matrix<T: Real>(v: &[T]) -> Matrix<T> {
    let n = v.len();
    Matrix::from_fn(n, n, |i, j| if i == j { v[i] } else { T::zero() })
}

/// The diagonal of `A` (its first `min(m, n)` diagonal entries).
pub fn diagonal<T: Real>(a: &Matrix<T>) -> Vec<T> {
    (0..a.nrows().min(a.ncols())).map(|i| a[(i, i)]).collect()
}

/// Sum of the diagonal entries.
pub fn trace<T: Real>(a: &Matrix<T>) -> T {
    sum(&diagonal(a))
}

/// `A` with `c` added to every diagonal entry.
pub fn add_diag<T: Real>(a: &Matrix<T>, c: T) -> Matrix<T> {
    let mut out = a.clone();
    for i in 0..a.nrows().min(a.ncols()) {
        out[(i, i)] += c;
    }
    out
}

/// `A` with `d[i]` added to diagonal entry `i`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `d` does not have `min(m, n)` entries.
pub fn add_diag_vector<T: Real>(a: &Matrix<T>, d: &[T]) -> Result<Matrix<T>> {
    let k = a.nrows().min(a.ncols());
    if d.len() != k {
        return Err(length_mismatch("add_diag", "diagonal", k, d.len()));
    }
    let mut out = a.clone();
    for (i, &di) in d.iter().enumerate() {
        out[(i, i)] += di;
    }
    Ok(out)
}

/// `diag(v) M`: row `i` of `M` scaled by `v[i]`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `v` does not have one entry per row.
pub fn diag_pre_multiply<T: Real>(v: &[T], m: &Matrix<T>) -> Result<Matrix<T>> {
    if v.len() != m.nrows() {
        return Err(length_mismatch(
            "diag_pre_multiply",
            "v",
            m.nrows(),
            v.len(),
        ));
    }
    Ok(Matrix::from_fn(m.nrows(), m.ncols(), |i, j| {
        m[(i, j)] * v[i]
    }))
}

/// `M diag(v)`: column `j` of `M` scaled by `v[j]`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `v` does not have one entry per column.
pub fn diag_post_multiply<T: Real>(m: &Matrix<T>, v: &[T]) -> Result<Matrix<T>> {
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

/// `diag(v) M diag(v)` for square `M`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square `M` or a `v` of the wrong length.
pub fn quad_form_diag<T: Real>(m: &Matrix<T>, v: &[T]) -> Result<Matrix<T>> {
    require_square("quad_form_diag", m)?;
    if v.len() != m.nrows() {
        return Err(length_mismatch("quad_form_diag", "v", m.nrows(), v.len()));
    }
    Ok(Matrix::from_fn(m.nrows(), m.ncols(), |i, j| {
        m[(i, j)] * v[i] * v[j]
    }))
}

/// `Bᵀ A B` for symmetric `A`, symmetrized as `(R + Rᵀ) / 2` so that the result is exactly
/// symmetric.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for incompatible shapes and [`Error::NotSymmetric`].
pub fn quad_form_sym<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    const F: &str = "quad_form_sym";
    require_square(F, a)?;
    let values: Vec<f64> = a.as_slice().iter().map(Real::value).collect();
    check_symmetric(F, &values, a.nrows())?;
    let r = multiply(&b.transpose(), &multiply(a, b)?)?;
    Ok(Matrix::from_fn(r.nrows(), r.ncols(), |i, j| {
        (r[(i, j)] + r[(j, i)]) * 0.5
    }))
}

/// `trace(Bᵀ A B)`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for incompatible shapes.
pub fn trace_quad_form<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<T> {
    require_square("trace_quad_form", a)?;
    let ab = multiply(a, b)?;
    Ok(dot_product(b.as_slice(), ab.as_slice()))
}

/// `trace(D Bᵀ A B)`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for incompatible shapes.
pub fn trace_gen_quad_form<T: Real>(d: &Matrix<T>, a: &Matrix<T>, b: &Matrix<T>) -> Result<T> {
    const F: &str = "trace_gen_quad_form";
    require_square(F, a)?;
    require_square(F, d)?;
    let r = multiply(&b.transpose(), &multiply(a, b)?)?;
    if r.shape() != d.shape() {
        return Err(Error::dimension(
            F,
            format!("D is {:?}, Bᵀ A B is {:?}", d.shape(), r.shape()),
        ));
    }
    // trace(D R) = sum_ij D_ij R_ji
    Ok(dot_product(d.as_slice(), r.transpose().as_slice()))
}

/// `Aᵀ A`.
///
/// # Errors
///
/// Never fails for valid matrices; the `Result` comes from [`multiply`].
pub fn crossprod<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    multiply(&a.transpose(), a)
}

/// `A Aᵀ`.
///
/// # Errors
///
/// Never fails for valid matrices; the `Result` comes from [`multiply`].
pub fn tcrossprod<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    multiply(a, &a.transpose())
}

fn same_shape<T>(function: &'static str, a: &Matrix<T>, b: &Matrix<T>) -> Result<()> {
    if a.shape() == b.shape() {
        Ok(())
    } else {
        Err(Error::dimension(
            function,
            format!("{:?} and {:?}", a.shape(), b.shape()),
        ))
    }
}

/// Dot products of corresponding columns of `A` and `B`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the shapes differ.
pub fn columns_dot_product<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Vec<T>> {
    same_shape("columns_dot_product", a, b)?;
    Ok((0..a.ncols())
        .map(|j| dot_product(a.column(j), b.column(j)))
        .collect())
}

/// Dot products of corresponding rows of `A` and `B`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the shapes differ.
pub fn rows_dot_product<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Vec<T>> {
    same_shape("rows_dot_product", a, b)?;
    Ok((0..a.nrows())
        .map(|i| {
            let ra: Vec<T> = a.row(i).copied().collect();
            let rb: Vec<T> = b.row(i).copied().collect();
            dot_product(&ra, &rb)
        })
        .collect())
}

/// Sum of squares of each column.
pub fn columns_dot_self<T: Real>(a: &Matrix<T>) -> Vec<T> {
    (0..a.ncols()).map(|j| dot_self(a.column(j))).collect()
}

/// Sum of squares of each row.
pub fn rows_dot_self<T: Real>(a: &Matrix<T>) -> Vec<T> {
    (0..a.nrows())
        .map(|i| dot_self(&a.row(i).copied().collect::<Vec<T>>()))
        .collect()
}

/// Sum of squares `xᵀ x`.
pub fn dot_self<T: Real>(x: &[T]) -> T {
    dot_product(x, x)
}

/// Squared Euclidean distance between two vectors.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the lengths differ.
pub fn squared_distance<T: Real>(x: &[T], y: &[T]) -> Result<T> {
    if x.len() != y.len() {
        return Err(length_mismatch("squared_distance", "y", x.len(), y.len()));
    }
    Ok(x.iter().zip(y).map(|(&a, &b)| (a - b).square()).sum())
}

/// Euclidean distance between two vectors. The derivative is not finite where `x = y`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the lengths differ.
pub fn distance<T: Real>(x: &[T], y: &[T]) -> Result<T> {
    Ok(squared_distance(x, y)?.sqrt())
}

/// Running sums `x[0], x[0] + x[1], ...`.
pub fn cumulative_sum<T: Real>(x: &[T]) -> Vec<T> {
    let mut acc = T::zero();
    x.iter()
        .map(|&v| {
            acc += v;
            acc
        })
        .collect()
}

/// Elementwise product.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the shapes differ.
pub fn elt_multiply<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    elementwise("elt_multiply", a, b, |x, y| x * y)
}

/// Elementwise quotient.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the shapes differ.
pub fn elt_divide<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    elementwise("elt_divide", a, b, |x, y| x / y)
}

fn nonempty<T>(function: &'static str, x: &[T]) -> Result<()> {
    if x.is_empty() {
        Err(Error::dimension(function, "empty sequence"))
    } else {
        Ok(())
    }
}

/// Arithmetic mean. For a matrix pass `a.as_slice()`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for an empty sequence.
pub fn mean<T: Real>(x: &[T]) -> Result<T> {
    nonempty("mean", x)?;
    Ok(sum(x) / x.len() as f64)
}

/// Sample variance with divisor `n - 1`; zero for a single element.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for an empty sequence.
pub fn variance<T: Real>(x: &[T]) -> Result<T> {
    nonempty("variance", x)?;
    if x.len() == 1 {
        return Ok(T::zero());
    }
    let m = mean(x)?;
    let ss: T = x.iter().map(|&v| (v - m).square()).sum();
    Ok(ss / (x.len() - 1) as f64)
}

/// Sample standard deviation, the square root of [`variance`]; zero for a single element.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for an empty sequence.
pub fn sd<T: Real>(x: &[T]) -> Result<T> {
    nonempty("sd", x)?;
    if x.len() == 1 {
        return Ok(T::zero());
    }
    Ok(variance(x)?.sqrt())
}

/// `ln(sum(exp(A)))` over every entry of a matrix.
pub fn log_sum_exp_matrix<T: Real>(a: &Matrix<T>) -> T {
    log_sum_exp_slice(a.as_slice())
}
