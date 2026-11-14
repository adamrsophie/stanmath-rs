//! Functions on a factored symmetric positive definite matrix (Stan's `LDLT_factor`).
//!
//! Stan accepts an `LDLT_factor` only when every pivot of `A = L D Lᵀ` is positive, that is, when
//! `A` is positive definite. For such a matrix the factorization is the Cholesky factorization
//! `A = C Cᵀ` with `C = L D^(1/2)`, so [`LdltFactor`] stores `C` and every function here is a
//! composition of [`cholesky_decompose`] and the triangular solves, whose structured adjoints
//! already exist. Solves follow Golub and Van Loan, section 4.2; `chol2inv` uses
//! `(L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹` (Higham, "Accuracy and Stability of Numerical Algorithms", section 14.3).

use stanmath_rs_autodiff::Real;
use stanmath_rs_autodiff::functions::dot_product;
use stanmath_rs_core::{Error, Matrix, Result};

use crate::{
    cholesky_decompose, crossprod, identity_matrix, mdivide_left_tri_low,
    mdivide_left_tri_low_transpose, require_square,
};

/// A symmetric positive definite matrix in factored form, the counterpart of Stan's
/// `LDLT_factor`.
///
/// Build it once with [`LdltFactor::new`] and pass it to the `*_ldlt` functions, which then
/// reuse the factorization.
#[derive(Clone, Debug)]
pub struct LdltFactor<T> {
    /// Lower Cholesky factor `C` with `A = C Cᵀ`.
    l: Matrix<T>,
}

impl<T: Real> LdltFactor<T> {
    /// Factors the symmetric positive definite matrix `A`. Only its lower triangle is read after
    /// the symmetry check, as in [`cholesky_decompose`].
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] for a non-square input, [`Error::NotSymmetric`], and
    /// [`Error::NotPositiveDefinite`] when a pivot is not positive.
    pub fn new(a: &Matrix<T>) -> Result<Self> {
        Ok(LdltFactor {
            l: cholesky_decompose(a)?,
        })
    }

    /// Number of rows (and columns) of the factored matrix.
    pub fn rows(&self) -> usize {
        self.l.nrows()
    }

    /// `A⁻¹ B` for the factored `A`.
    fn solve(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        mdivide_left_tri_low_transpose(&self.l, &mdivide_left_tri_low(&self.l, b)?)
    }
}

fn rows_mismatch(function: &'static str, n: usize, what: &str, got: usize) -> Error {
    Error::dimension(
        function,
        format!("{n}x{n} factored matrix with {what} {got}"),
    )
}

/// `A⁻¹ B` for a factored symmetric positive definite `A`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `B` does not have one row per row of `A`.
pub fn mdivide_left_ldlt<T: Real>(a: &LdltFactor<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    if b.nrows() != a.rows() {
        return Err(rows_mismatch(
            "mdivide_left_ldlt",
            a.rows(),
            "right-hand side rows",
            b.nrows(),
        ));
    }
    a.solve(b)
}

/// `B A⁻¹` for a factored symmetric positive definite `A`, computed as `(A⁻¹ Bᵀ)ᵀ`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `B` does not have one column per row of `A`.
pub fn mdivide_right_ldlt<T: Real>(b: &Matrix<T>, a: &LdltFactor<T>) -> Result<Matrix<T>> {
    if b.ncols() != a.rows() {
        return Err(rows_mismatch(
            "mdivide_right_ldlt",
            a.rows(),
            "left-hand side columns",
            b.ncols(),
        ));
    }
    Ok(a.solve(&b.transpose())?.transpose())
}

/// `ln det(A)` for a factored symmetric positive definite `A`, as `2 sum ln C_ii`.
pub fn log_determinant_ldlt<T: Real>(a: &LdltFactor<T>) -> T {
    (0..a.rows()).map(|i| a.l[(i, i)].ln()).sum::<T>() * 2.0
}

/// `trace(Bᵀ A⁻¹ B)` for a factored symmetric positive definite `A`, computed as the squared
/// Frobenius norm of `C⁻¹ B`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `B` does not have one row per row of `A`.
pub fn trace_inv_quad_form_ldlt<T: Real>(a: &LdltFactor<T>, b: &Matrix<T>) -> Result<T> {
    if b.nrows() != a.rows() {
        return Err(rows_mismatch(
            "trace_inv_quad_form_ldlt",
            a.rows(),
            "B rows",
            b.nrows(),
        ));
    }
    let y = mdivide_left_tri_low(&a.l, b)?;
    Ok(dot_product(y.as_slice(), y.as_slice()))
}

/// `trace(D Bᵀ A⁻¹ B)` for a factored symmetric positive definite `A` and square `D`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `D` is not square, when `B` does not have one row per row
/// of `A`, or when `D` does not have one row per column of `B`.
pub fn trace_gen_inv_quad_form_ldlt<T: Real>(
    d: &Matrix<T>,
    a: &LdltFactor<T>,
    b: &Matrix<T>,
) -> Result<T> {
    const F: &str = "trace_gen_inv_quad_form_ldlt";
    require_square(F, d)?;
    if b.nrows() != a.rows() {
        return Err(rows_mismatch(F, a.rows(), "B rows", b.nrows()));
    }
    if d.nrows() != b.ncols() {
        return Err(Error::dimension(
            F,
            format!(
                "D is {}x{} but B has {} columns",
                d.nrows(),
                d.ncols(),
                b.ncols()
            ),
        ));
    }
    // M = Bᵀ A⁻¹ B = Yᵀ Y with Y = C⁻¹ B, and trace(D M) = sum_ij D_ij M_ji.
    let m = crossprod(&mdivide_left_tri_low(&a.l, b)?)?;
    Ok(dot_product(d.as_slice(), m.transpose().as_slice()))
}

/// `(L Lᵀ)⁻¹` for a lower triangular `L`, the inverse of the matrix whose Cholesky factor is `L`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input, and [`Error::InvalidParameter`] when an
/// entry above the diagonal is nonzero or a diagonal entry is zero or not finite.
pub fn chol2inv<T: Real>(l: &Matrix<T>) -> Result<Matrix<T>> {
    const F: &str = "chol2inv";
    require_square(F, l)?;
    let n = l.nrows();
    for j in 0..n {
        for i in 0..j {
            let v = l[(i, j)].value();
            if v != 0.0 {
                return Err(Error::parameter(F, "L", v, "lower triangular"));
            }
        }
    }
    // (L Lᵀ)⁻¹ = L⁻ᵀ L⁻¹ = (L⁻¹)ᵀ (L⁻¹).
    let l_inv = mdivide_left_tri_low(l, &identity_matrix(n))?;
    crossprod(&l_inv)
}

fn symmetrize<T: Real>(
    function: &'static str,
    m: &Matrix<T>,
    from_lower: bool,
) -> Result<Matrix<T>> {
    require_square(function, m)?;
    let n = m.nrows();
    Ok(Matrix::from_fn(n, n, |i, j| {
        // (big, small) indexes the lower triangle, (small, big) the upper one.
        let (big, small) = if i >= j { (i, j) } else { (j, i) };
        if from_lower {
            m[(big, small)]
        } else {
            m[(small, big)]
        }
    }))
}

/// The symmetric matrix whose lower triangle (diagonal included) is that of `M`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input.
pub fn symmetrize_from_lower_tri<T: Real>(m: &Matrix<T>) -> Result<Matrix<T>> {
    symmetrize("symmetrize_from_lower_tri", m, true)
}

/// The symmetric matrix whose upper triangle (diagonal included) is that of `M`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input.
pub fn symmetrize_from_upper_tri<T: Real>(m: &Matrix<T>) -> Result<Matrix<T>> {
    symmetrize("symmetrize_from_upper_tri", m, false)
}
