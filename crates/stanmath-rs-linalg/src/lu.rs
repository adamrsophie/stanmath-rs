//! LU factorization with partial pivoting: general solves, inverses, and determinants.
//!
//! The factorization `P A = L U` (Golub and Van Loan, algorithm 3.4.1) is one generic algorithm
//! that serves `f64` and every forward-mode type; pivots are chosen on the values, so every mode
//! takes the same branch. Reverse mode records one custom operation per call with the adjoints
//! below (Giles 2008, "Collected matrix derivative results"):
//!
//! * `X = A⁻¹ B`: `B̄ = A⁻ᵀ X̄`, `Ā = -B̄ Xᵀ`;
//! * `Y = A⁻¹`: `Ā = -Yᵀ Ȳ Yᵀ`;
//! * `d = det A`: `Ā = d̄ d A⁻ᵀ` (the transposed adjugate when `A` is singular);
//! * `l = ln |det A|`: `Ā = l̄ A⁻ᵀ`.

#![allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices

use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::ops::at;
use crate::{
    mdivide_left_spd, mdivide_left_tri_low, mdivide_left_tri_low_transpose, require_square,
};

/// Packed factors of `P A = L U` (`L` unit lower triangular, `U` upper triangular) and the row
/// interchanges, in the order they were applied.
struct Lu<S> {
    n: usize,
    lu: Vec<S>,
    piv: Vec<usize>,
    /// Determinant of the permutation: `1` or `-1`.
    sign: f64,
}

/// Factors the `n` by `n` column-major matrix `a`. `None` when a pivot is zero or not finite.
fn factor<S: Real>(a: &[S], n: usize) -> Option<Lu<S>> {
    let mut lu = a.to_vec();
    let mut piv = Vec::with_capacity(n);
    let mut sign = 1.0;
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| {
                lu[at(n, i, k)]
                    .value()
                    .abs()
                    .total_cmp(&lu[at(n, j, k)].value().abs())
            })
            .unwrap_or(k);
        let pivot = lu[at(n, p, k)];
        if pivot.value() == 0.0 || !pivot.is_finite() {
            return None;
        }
        piv.push(p);
        if p != k {
            sign = -sign;
            for j in 0..n {
                lu.swap(at(n, k, j), at(n, p, j));
            }
        }
        for i in k + 1..n {
            lu[at(n, i, k)] /= pivot;
        }
        for j in k + 1..n {
            let ukj = lu[at(n, k, j)];
            for i in k + 1..n {
                let lik = lu[at(n, i, k)];
                lu[at(n, i, j)] -= lik * ukj;
            }
        }
    }
    Some(Lu { n, lu, piv, sign })
}

impl<S: Real> Lu<S> {
    /// Overwrites `b` with `A⁻¹ b`.
    fn solve(&self, b: &mut [S]) {
        let n = self.n;
        for (k, &p) in self.piv.iter().enumerate() {
            b.swap(k, p);
        }
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= self.lu[at(n, i, k)] * b[k];
            }
            b[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            for k in i + 1..n {
                s -= self.lu[at(n, i, k)] * b[k];
            }
            b[i] = s / self.lu[at(n, i, i)];
        }
    }

    /// Overwrites `b` with `A⁻ᵀ b`, using `Aᵀ = Uᵀ Lᵀ P`.
    fn solve_transpose(&self, b: &mut [S]) {
        let n = self.n;
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= self.lu[at(n, k, i)] * b[k];
            }
            b[i] = s / self.lu[at(n, i, i)];
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            for k in i + 1..n {
                s -= self.lu[at(n, k, i)] * b[k];
            }
            b[i] = s;
        }
        for (k, &p) in self.piv.iter().enumerate().rev() {
            b.swap(k, p);
        }
    }

    fn determinant(&self) -> S {
        let mut d = S::from_f64(self.sign);
        for i in 0..self.n {
            d *= self.lu[at(self.n, i, i)];
        }
        d
    }

    fn log_abs_determinant(&self) -> S {
        (0..self.n)
            .map(|i| self.lu[at(self.n, i, i)].abs().ln())
            .sum()
    }
}

fn singular(function: &'static str) -> Error {
    Error::parameter(function, "A", 0.0, "nonsingular")
}

/// `X = A⁻¹ B` with `A` of size `n` by `n` and `B` of size `n` by `m`; inputs `[A, B]`.
#[derive(Clone, Copy, Debug)]
struct LuSolve {
    function: &'static str,
    n: usize,
    m: usize,
}

impl MultiFn for LuSolve {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        let (n, m) = (self.n, self.m);
        let (a, b) = x.split_at(n * n);
        let lu = factor(a, n).ok_or_else(|| singular(self.function))?;
        let mut out = b.to_vec();
        for c in 0..m {
            lu.solve(&mut out[c * n..(c + 1) * n]);
        }
        Ok(out)
    }

    fn reverse(&self, x: &[f64], xs: &[f64], x_bar: &[f64], needed: &[bool], adj: &mut [f64]) {
        let (n, m) = (self.n, self.m);
        let Some(lu) = factor(&x[..n * n], n) else {
            return;
        };
        let mut b_bar = x_bar.to_vec();
        for c in 0..m {
            lu.solve_transpose(&mut b_bar[c * n..(c + 1) * n]);
        }
        let (a_adj, b_adj) = adj.split_at_mut(n * n);
        let (a_needed, b_needed) = needed.split_at(n * n);
        if b_needed.iter().any(|&v| v) {
            for (a, v) in b_adj.iter_mut().zip(&b_bar) {
                *a += v;
            }
        }
        if a_needed.iter().any(|&v| v) {
            for j in 0..n {
                for i in 0..n {
                    let mut s = 0.0;
                    for c in 0..m {
                        s += b_bar[at(n, i, c)] * xs[at(n, j, c)];
                    }
                    a_adj[at(n, i, j)] -= s;
                }
            }
        }
    }
}

/// `Y = A⁻¹`; input `A` (`n` by `n`).
#[derive(Clone, Copy, Debug)]
struct Inverse {
    n: usize,
}

impl MultiFn for Inverse {
    fn eval<S: Real>(&self, a: &[S]) -> Result<Vec<S>> {
        let n = self.n;
        let lu = factor(a, n).ok_or_else(|| singular("inverse"))?;
        let mut out = vec![S::zero(); n * n];
        for c in 0..n {
            out[at(n, c, c)] = S::one();
            lu.solve(&mut out[c * n..(c + 1) * n]);
        }
        Ok(out)
    }

    fn reverse(&self, _a: &[f64], y: &[f64], y_bar: &[f64], needed: &[bool], adj: &mut [f64]) {
        if !needed.iter().any(|&v| v) {
            return;
        }
        let n = self.n;
        // T = Ȳ Yᵀ, then Ā = -Yᵀ T.
        let mut t = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += y_bar[at(n, i, k)] * y[at(n, j, k)];
                }
                t[at(n, i, j)] = s;
            }
        }
        for j in 0..n {
            for i in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += y[at(n, k, i)] * t[at(n, k, j)];
                }
                adj[at(n, i, j)] -= s;
            }
        }
    }
}

/// `det A` (`log = false`) or `ln |det A|` (`log = true`); input `A` (`n` by `n`).
#[derive(Clone, Copy, Debug)]
struct Determinant {
    n: usize,
    log: bool,
}

/// Determinant in `f64`, zero for a singular matrix.
fn determinant_f64(a: &[f64], n: usize) -> f64 {
    factor(a, n).map_or(0.0, |lu| lu.determinant())
}

/// Cofactor `(-1)^(i + j) det(A without row i and column j)`.
fn cofactor(a: &[f64], n: usize, i: usize, j: usize) -> f64 {
    let m = n - 1;
    let mut minor = Vec::with_capacity(m * m);
    for c in (0..n).filter(|&c| c != j) {
        for r in (0..n).filter(|&r| r != i) {
            minor.push(a[at(n, r, c)]);
        }
    }
    let d = determinant_f64(&minor, m);
    if (i + j) % 2 == 0 { d } else { -d }
}

impl MultiFn for Determinant {
    fn eval<S: Real>(&self, a: &[S]) -> Result<Vec<S>> {
        let value = match factor(a, self.n) {
            Some(lu) if self.log => lu.log_abs_determinant(),
            Some(lu) => lu.determinant(),
            None if self.log => S::from_f64(f64::NEG_INFINITY),
            None => S::zero(),
        };
        Ok(vec![value])
    }

    fn reverse(&self, a: &[f64], y: &[f64], y_bar: &[f64], needed: &[bool], adj: &mut [f64]) {
        if !needed.iter().any(|&v| v) {
            return;
        }
        let n = self.n;
        match factor(a, n) {
            Some(lu) => {
                let scale = if self.log { y_bar[0] } else { y_bar[0] * y[0] };
                // Column j of A⁻ᵀ solves Aᵀ z = e_j.
                let mut e = vec![0.0; n];
                for j in 0..n {
                    e.iter_mut().for_each(|v| *v = 0.0);
                    e[j] = 1.0;
                    lu.solve_transpose(&mut e);
                    for i in 0..n {
                        adj[at(n, i, j)] += scale * e[i];
                    }
                }
            }
            // ponytail: O(n^5) cofactor expansion, only reached at an exactly singular matrix.
            None if !self.log && n > 0 => {
                for j in 0..n {
                    for i in 0..n {
                        adj[at(n, i, j)] += y_bar[0] * cofactor(a, n, i, j);
                    }
                }
            }
            None => {}
        }
    }
}

fn lu_solve<T: Real>(function: &'static str, a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    require_square(function, a)?;
    if b.nrows() != a.nrows() {
        return Err(Error::dimension(
            function,
            format!(
                "{}x{} matrix with {} right-hand side rows",
                a.nrows(),
                a.ncols(),
                b.nrows()
            ),
        ));
    }
    let f = LuSolve {
        function,
        n: a.nrows(),
        m: b.ncols(),
    };
    let inputs: Vec<T> = a.as_slice().iter().chain(b.as_slice()).copied().collect();
    let x = T::map_multi(&inputs, &f)?;
    Ok(Matrix::from_column_major(f.n, f.m, x))
}

/// `A⁻¹ B` for a square nonsingular `A`, by LU factorization with partial pivoting.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for incompatible shapes and [`Error::InvalidParameter`] when `A`
/// is singular.
pub fn mdivide_left<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    lu_solve("mdivide_left", a, b)
}

/// `B A⁻¹` for a square nonsingular `A`, computed as `(A⁻ᵀ Bᵀ)ᵀ`.
///
/// # Errors
///
/// As [`mdivide_left`].
pub fn mdivide_right<T: Real>(b: &Matrix<T>, a: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(lu_solve("mdivide_right", &a.transpose(), &b.transpose())?.transpose())
}

/// `B A⁻¹` for symmetric positive definite `A`.
///
/// # Errors
///
/// As [`mdivide_left_spd`].
pub fn mdivide_right_spd<T: Real>(b: &Matrix<T>, a: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(mdivide_left_spd(a, &b.transpose())?.transpose())
}

/// `B L⁻¹` for lower triangular `L` (only its lower triangle is read).
///
/// # Errors
///
/// As [`mdivide_left_tri_low_transpose`].
pub fn mdivide_right_tri_low<T: Real>(b: &Matrix<T>, l: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(mdivide_left_tri_low_transpose(l, &b.transpose())?.transpose())
}

/// `U⁻¹ B` for upper triangular `U` (only its upper triangle is read).
///
/// # Errors
///
/// As [`mdivide_left_tri_low_transpose`], which it calls on `Uᵀ`.
pub fn mdivide_left_tri_upper<T: Real>(u: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    mdivide_left_tri_low_transpose(&u.transpose(), b)
}

/// The triangle of a matrix that a triangular solve reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Triangle {
    /// The lower triangle, diagonal included.
    Lower,
    /// The upper triangle, diagonal included.
    Upper,
}

/// `A⁻¹ B` for triangular `A`, reading only `triangle` of `A`. Stan Math's `mdivide_left_tri`,
/// whose template argument picks the triangle.
///
/// # Errors
///
/// As [`mdivide_left_tri_low`] and [`mdivide_left_tri_upper`].
pub fn mdivide_left_tri<T: Real>(
    a: &Matrix<T>,
    b: &Matrix<T>,
    triangle: Triangle,
) -> Result<Matrix<T>> {
    match triangle {
        Triangle::Lower => mdivide_left_tri_low(a, b),
        Triangle::Upper => mdivide_left_tri_upper(a, b),
    }
}

/// `B A⁻¹` for triangular `A`, reading only `triangle` of `A`. Stan Math's `mdivide_right_tri`.
///
/// # Errors
///
/// As [`mdivide_left_tri_low`].
pub fn mdivide_right_tri<T: Real>(
    b: &Matrix<T>,
    a: &Matrix<T>,
    triangle: Triangle,
) -> Result<Matrix<T>> {
    match triangle {
        Triangle::Lower => mdivide_right_tri_low(b, a),
        // B U⁻¹ = ((Uᵀ)⁻¹ Bᵀ)ᵀ, and Uᵀ is lower triangular.
        Triangle::Upper => Ok(mdivide_left_tri_low(&a.transpose(), &b.transpose())?.transpose()),
    }
}

/// `A⁻¹` for a square nonsingular `A`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input and [`Error::InvalidParameter`] when `A`
/// is singular.
pub fn inverse<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    require_square("inverse", a)?;
    let n = a.nrows();
    let y = T::map_multi(a.as_slice(), &Inverse { n })?;
    Ok(Matrix::from_column_major(n, n, y))
}

/// `det A` for a square `A`; zero when `A` is singular.
///
/// At an exactly singular matrix the reverse-mode gradient is the transposed adjugate, but
/// forward-mode tangents are zero.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input.
pub fn determinant<T: Real>(a: &Matrix<T>) -> Result<T> {
    require_square("determinant", a)?;
    let d = T::map_multi(
        a.as_slice(),
        &Determinant {
            n: a.nrows(),
            log: false,
        },
    )?;
    Ok(d[0])
}

/// `ln |det A|` for a square `A`; negative infinity when `A` is singular.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input.
pub fn log_determinant<T: Real>(a: &Matrix<T>) -> Result<T> {
    require_square("log_determinant", a)?;
    let d = T::map_multi(
        a.as_slice(),
        &Determinant {
            n: a.nrows(),
            log: true,
        },
    )?;
    Ok(d[0])
}

#[cfg(test)]
mod triangle_tests {
    use super::*;

    #[test]
    fn right_upper_triangular_solve_inverts_the_product() {
        // The lower part (9.0) must be ignored.
        let u = Matrix::from_fn(3, 3, |i, j| {
            if i <= j {
                1.0 + (i + 2 * j) as f64
            } else {
                9.0
            }
        });
        let b = Matrix::from_fn(2, 3, |i, j| (i * 3 + j) as f64 - 1.5);
        let x = mdivide_right_tri(&b, &u, Triangle::Upper).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                let s: f64 = (0..=j).map(|k| x[(i, k)] * u[(k, j)]).sum();
                assert!((s - b[(i, j)]).abs() < 1e-12);
            }
        }
    }
}
