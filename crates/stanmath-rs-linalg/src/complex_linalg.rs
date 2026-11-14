//! Complex linear algebra: discrete Fourier transforms, the complex Schur decomposition, and the
//! eigendecomposition of real nonsymmetric matrices.
//!
//! # Fourier transforms
//!
//! `fft` computes `y_k = sum_n v_n exp(-2 pi i n k / N)` and `inv_fft` its inverse, including the
//! factor `1 / N`. Lengths that are powers of two use the iterative radix 2 algorithm of Cooley
//! and Tukey; other lengths use Bluestein's chirp z transform on a power of two grid. Both are
//! linear in the input, so generic arithmetic on [`Real`] gives exact derivatives in every mode.
//!
//! # Schur form and eigenvalues
//!
//! The complex Schur form `A = U T U^H` is computed in `f64` by Householder reduction to upper
//! Hessenberg form followed by the shifted QR algorithm with Wilkinson shifts and Givens rotations
//! (Golub and Van Loan, sections 7.4 and 7.5). For a differentiable scalar type the `f64` factor
//! `U` is then refined by Newton steps in that type: each step solves the triangular Sylvester
//! equation for a skew Hermitian correction `X` that removes the strictly lower part of
//! `U^H A U`, and reorthonormalizes `U (I + X)` by modified Gram and Schmidt. Starting from the
//! converged values, `k` steps make derivatives exact to order `2^k - 1`.
//!
//! Eigenvalues are the diagonal of `T`; eigenvectors come from back substitution in `T`, mapped
//! through `U`, scaled to unit Euclidean norm, and rotated so that their largest component (by
//! primal magnitude) is real and positive. For a real input, eigenvalues whose imaginary part is
//! below `1000 eps max |lambda|` are returned exactly real with real eigenvectors, and complex
//! eigenvalues are returned in adjacent conjugate pairs, positive imaginary part first, the second
//! member being the exact conjugate of the first.
//!
//! Repeated eigenvalues (closer than `1000 eps max |lambda|`) have no unique Schur vectors or
//! eigenvectors; their coupling terms are left out of the refinement and the back substitution
//! uses a tiny pivot, so values are usable but derivatives are not.

#![allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices

use core::f64::consts::PI;

use stanmath_rs_autodiff::{Complex, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::ops::at;
use crate::require_square;

/// Relative gap below which two eigenvalues are treated as repeated.
const CLUSTER: f64 = 1e3 * f64::EPSILON;

/// Refinement steps needed in a type of derivative order `order`: the smallest `k` with
/// `2^k - 1 >= order`.
fn refinement_steps(order: usize) -> usize {
    let mut k = 0;
    while (1usize << k) - 1 < order {
        k += 1;
    }
    k
}

#[inline]
fn lift<T: Real>(z: Complex<f64>) -> Complex<T> {
    Complex::from_f64(z.re, z.im)
}

/// `z w` for a constant `w`.
#[inline]
fn mul_const<T: Real>(z: Complex<T>, w: Complex<f64>) -> Complex<T> {
    Complex::new(z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re)
}

// Fourier transforms.

/// In place radix 2 transform of a power of two length with kernel `exp(sign 2 pi i n k / N)`,
/// `sign = -1` forward and `+1` inverse; unscaled.
fn fft_pow2<T: Real>(x: &mut [Complex<T>], inverse: bool) {
    let n = x.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = sign * 2.0 * PI / len as f64;
        for k in 0..half {
            let angle = step * k as f64;
            let w = Complex {
                re: angle.cos(),
                im: angle.sin(),
            };
            for start in (0..n).step_by(len) {
                let u = x[start + k];
                let v = mul_const(x[start + k + half], w);
                x[start + k] = u + v;
                x[start + k + half] = u - v;
            }
        }
        len <<= 1;
    }
}

/// Bluestein's algorithm for any length `n > 1`: with the chirp `w_j = exp(sign pi i j^2 / n)`,
/// `y_k = w_k sum_m (x_m w_m) conj(w_{k - m})`, a convolution done by power of two transforms.
fn bluestein<T: Real>(x: &[Complex<T>], inverse: bool) -> Vec<Complex<T>> {
    let n = x.len();
    let m = (2 * n - 1).next_power_of_two();
    let sign = if inverse { 1.0 } else { -1.0 };
    // k^2 mod 2n keeps the angle small, so the chirp is accurate for large k.
    let chirp: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let angle = sign * PI * ((k * k) % (2 * n)) as f64 / n as f64;
            Complex {
                re: angle.cos(),
                im: angle.sin(),
            }
        })
        .collect();
    let mut a = vec![Complex::<T>::zero(); m];
    for k in 0..n {
        a[k] = mul_const(x[k], chirp[k]);
    }
    let mut b = vec![Complex::<f64>::zero(); m];
    b[0] = chirp[0].conj();
    for k in 1..n {
        b[k] = chirp[k].conj();
        b[m - k] = chirp[k].conj();
    }
    fft_pow2(&mut a, false);
    fft_pow2(&mut b, false);
    for (ak, bk) in a.iter_mut().zip(&b) {
        *ak = mul_const(*ak, *bk);
    }
    fft_pow2(&mut a, true);
    let inv_m = 1.0 / m as f64;
    (0..n)
        .map(|k| mul_const(a[k], chirp[k]).scale(inv_m))
        .collect()
}

/// Unscaled discrete Fourier transform, forward or inverse kernel.
fn dft<T: Real>(x: &[Complex<T>], inverse: bool) -> Vec<Complex<T>> {
    let n = x.len();
    if n <= 1 {
        x.to_vec()
    } else if n.is_power_of_two() {
        let mut y = x.to_vec();
        fft_pow2(&mut y, inverse);
        y
    } else {
        bluestein(x, inverse)
    }
}

/// Discrete Fourier transform, `y_k = sum_n v_n exp(-2 pi i n k / N)`.
pub fn fft<T: Real>(v: &[Complex<T>]) -> Vec<Complex<T>> {
    dft(v, false)
}

/// Inverse discrete Fourier transform, `v_n = (1 / N) sum_k y_k exp(2 pi i n k / N)`.
pub fn inv_fft<T: Real>(y: &[Complex<T>]) -> Vec<Complex<T>> {
    let s = 1.0 / y.len() as f64;
    dft(y, true).into_iter().map(|z| z.scale(s)).collect()
}

fn dft2<T: Real>(m: &Matrix<Complex<T>>, inverse: bool) -> Matrix<Complex<T>> {
    let (r, c) = m.shape();
    let mut data = Vec::with_capacity(r * c);
    for j in 0..c {
        data.extend(dft(m.column(j), inverse));
    }
    let mut out = Matrix::from_column_major(r, c, data);
    for i in 0..r {
        let row: Vec<Complex<T>> = out.row(i).copied().collect();
        for (j, z) in dft(&row, inverse).into_iter().enumerate() {
            out[(i, j)] = z;
        }
    }
    out
}

/// Two dimensional discrete Fourier transform: [`fft`] of every column, then of every row.
pub fn fft2<T: Real>(m: &Matrix<Complex<T>>) -> Matrix<Complex<T>> {
    dft2(m, false)
}

/// Inverse of [`fft2`], including the factor `1 / (rows cols)`.
pub fn inv_fft2<T: Real>(m: &Matrix<Complex<T>>) -> Matrix<Complex<T>> {
    let s = 1.0 / m.len() as f64;
    dft2(m, true).map(|z| z.scale(s))
}

// Complex Schur decomposition.

/// `A B` for `n` by `n` column-major complex matrices.
fn cmatmul<T: Real>(a: &[Complex<T>], b: &[Complex<T>], n: usize) -> Vec<Complex<T>> {
    let mut out = vec![Complex::zero(); n * n];
    for j in 0..n {
        for k in 0..n {
            let bkj = b[at(n, k, j)];
            for i in 0..n {
                let t = a[at(n, i, k)] * bkj;
                out[at(n, i, j)] += t;
            }
        }
    }
    out
}

/// Conjugate transpose of an `n` by `n` column-major complex matrix.
fn adjoint<T: Real>(a: &[Complex<T>], n: usize) -> Vec<Complex<T>> {
    (0..n * n).map(|k| a[at(n, k / n, k % n)].conj()).collect()
}

/// Modified Gram and Schmidt on the columns of an `n` by `n` matrix.
fn orthonormalize<T: Real>(u: &mut [Complex<T>], n: usize) {
    for j in 0..n {
        for k in 0..j {
            let mut r = Complex::zero();
            for i in 0..n {
                r += u[at(n, i, k)].conj() * u[at(n, i, j)];
            }
            for i in 0..n {
                let t = u[at(n, i, k)] * r;
                u[at(n, i, j)] -= t;
            }
        }
        let norm = (0..n).map(|i| u[at(n, i, j)].norm()).sum::<T>().sqrt();
        for i in 0..n {
            u[at(n, i, j)] = u[at(n, i, j)] / norm;
        }
    }
}

/// Complex Givens rotation `G = [[c, s], [-conj(s), c]]` with real `c` and `G [a, b]^T = [r, 0]^T`.
fn givens(a: Complex<f64>, b: Complex<f64>) -> (f64, Complex<f64>) {
    let (aa, bb) = (a.abs(), b.abs());
    if bb == 0.0 {
        return (1.0, Complex::zero());
    }
    if aa == 0.0 {
        return (0.0, b.conj().scale(1.0 / bb));
    }
    let r = aa.hypot(bb);
    (aa / r, a.scale(1.0 / aa) * b.conj().scale(1.0 / r))
}

/// Eigenvalue of `[[a, b], [c, d]]` closer to `d`.
fn wilkinson(a: Complex<f64>, b: Complex<f64>, c: Complex<f64>, d: Complex<f64>) -> Complex<f64> {
    let p = (a - d).scale(0.5);
    let bc = b * c;
    let disc = (p * p + bc).sqrt();
    let (q1, q2) = (p + disc, p - disc);
    let q = if q1.abs() >= q2.abs() { q1 } else { q2 };
    if q.abs() == 0.0 { d } else { d - bc / q }
}

/// Two column-major `n` by `n` complex matrices.
type ComplexPair = (Vec<Complex<f64>>, Vec<Complex<f64>>);

/// Complex Schur form in `f64`: `(U, T)` with `A = U T U^H`, `U` unitary, `T` upper triangular.
fn schur_f64(function: &'static str, a: &[Complex<f64>], n: usize) -> Result<ComplexPair> {
    let mut h = a.to_vec();
    let mut u: Vec<Complex<f64>> = (0..n * n)
        .map(|k| {
            if k % n == k / n {
                Complex::one()
            } else {
                Complex::zero()
            }
        })
        .collect();

    // Householder reduction to upper Hessenberg form.
    for k in 0..n.saturating_sub(2) {
        let norm = (k + 1..n)
            .map(|i| h[at(n, i, k)].norm())
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            continue;
        }
        let x0 = h[at(n, k + 1, k)];
        let x0_abs = x0.abs();
        let phase = if x0_abs == 0.0 {
            Complex::one()
        } else {
            x0.scale(1.0 / x0_abs)
        };
        let mut v: Vec<Complex<f64>> = (k + 1..n).map(|i| h[at(n, i, k)]).collect();
        v[0] += phase.scale(norm);
        let vnorm = v.iter().map(|z| z.norm()).sum::<f64>().sqrt();
        for z in &mut v {
            *z = z.scale(1.0 / vnorm);
        }
        // H = P H with P = I - 2 v v^H acting on rows k + 1 and below.
        for j in 0..n {
            let mut s = Complex::zero();
            for (r, vi) in v.iter().enumerate() {
                s += vi.conj() * h[at(n, k + 1 + r, j)];
            }
            for (r, vi) in v.iter().enumerate() {
                h[at(n, k + 1 + r, j)] -= *vi * s.scale(2.0);
            }
        }
        // H = H P and U = U P on columns k + 1 and right of it.
        for mat in [&mut h, &mut u] {
            for i in 0..n {
                let mut s = Complex::zero();
                for (r, vi) in v.iter().enumerate() {
                    s += mat[at(n, i, k + 1 + r)] * *vi;
                }
                for (r, vi) in v.iter().enumerate() {
                    mat[at(n, i, k + 1 + r)] -= s * vi.conj().scale(2.0);
                }
            }
        }
        for i in k + 2..n {
            h[at(n, i, k)] = Complex::zero();
        }
    }

    // Shifted QR iteration on the active block [l, hi].
    let anorm = h
        .iter()
        .map(|z| z.abs())
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let max_iterations = 30 * n.max(1);
    let mut total = 0;
    let mut its = 0;
    let mut hi = n.saturating_sub(1);
    while hi > 0 {
        let mut l = hi;
        while l > 0 {
            let s = h[at(n, l - 1, l - 1)].abs() + h[at(n, l, l)].abs();
            let s = if s == 0.0 { anorm } else { s };
            if h[at(n, l, l - 1)].abs() <= f64::EPSILON * s {
                h[at(n, l, l - 1)] = Complex::zero();
                break;
            }
            l -= 1;
        }
        if l == hi {
            hi -= 1;
            its = 0;
            continue;
        }
        if total >= max_iterations {
            return Err(Error::NoConvergence {
                function,
                iterations: total,
            });
        }
        total += 1;
        its += 1;
        let shift = if its % 10 == 0 {
            // Exceptional shift to break cycles.
            let e = h[at(n, hi, hi - 1)].re.abs()
                + if hi >= 2 {
                    h[at(n, hi - 1, hi - 2)].re.abs()
                } else {
                    0.0
                };
            h[at(n, hi, hi)] + Complex::from_f64(e, 0.0)
        } else {
            wilkinson(
                h[at(n, hi - 1, hi - 1)],
                h[at(n, hi - 1, hi)],
                h[at(n, hi, hi - 1)],
                h[at(n, hi, hi)],
            )
        };
        for k in l..=hi {
            h[at(n, k, k)] -= shift;
        }
        // `H - shift I = Q R`, by rotations from the left.
        let mut rotations = Vec::with_capacity(hi - l);
        for k in l..hi {
            let (c, s) = givens(h[at(n, k, k)], h[at(n, k + 1, k)]);
            for j in k..n {
                let (x, y) = (h[at(n, k, j)], h[at(n, k + 1, j)]);
                h[at(n, k, j)] = x.scale(c) + s * y;
                h[at(n, k + 1, j)] = y.scale(c) - s.conj() * x;
            }
            h[at(n, k + 1, k)] = Complex::zero();
            rotations.push((c, s));
        }
        // R Q, applied to every row above, and accumulated into U.
        for (idx, &(c, s)) in rotations.iter().enumerate() {
            let k = l + idx;
            for i in 0..=k + 1 {
                let (x, y) = (h[at(n, i, k)], h[at(n, i, k + 1)]);
                h[at(n, i, k)] = x.scale(c) + y * s.conj();
                h[at(n, i, k + 1)] = y.scale(c) - x * s;
            }
            for i in 0..n {
                let (x, y) = (u[at(n, i, k)], u[at(n, i, k + 1)]);
                u[at(n, i, k)] = x.scale(c) + y * s.conj();
                u[at(n, i, k + 1)] = y.scale(c) - x * s;
            }
        }
        for k in l..=hi {
            h[at(n, k, k)] += shift;
        }
    }
    for j in 0..n {
        for i in j + 1..n {
            h[at(n, i, j)] = Complex::zero();
        }
    }
    Ok((u, h))
}

/// Complex Schur form of a complex `n` by `n` matrix, with derivatives exact to `T::ORDER`.
/// Returns `(U, T, primal diagonal of T)`.
#[allow(clippy::type_complexity)]
pub(crate) fn schur<T: Real>(
    function: &'static str,
    a: &[Complex<T>],
    n: usize,
) -> Result<(Vec<Complex<T>>, Vec<Complex<T>>, Vec<Complex<f64>>)> {
    let a0: Vec<Complex<f64>> = a.iter().map(Complex::value).collect();
    for z in &a0 {
        if !z.re.is_finite() {
            return Err(Error::parameter(function, "A", z.re, "finite"));
        }
        if !z.im.is_finite() {
            return Err(Error::parameter(function, "A", z.im, "finite"));
        }
    }
    let (u0, t0) = schur_f64(function, &a0, n)?;
    let diag0: Vec<Complex<f64>> = (0..n).map(|i| t0[at(n, i, i)]).collect();
    let mut u: Vec<Complex<T>> = u0.iter().map(|&z| lift(z)).collect();
    let steps = refinement_steps(T::ORDER);
    if steps == 0 {
        return Ok((u, t0.iter().map(|&z| lift(z)).collect(), diag0));
    }
    let scale = diag0.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    let gap = CLUSTER * scale;
    for _ in 0..steps {
        let b = cmatmul(&adjoint(&u, n), &cmatmul(a, &u, n), n);
        // Strictly lower X with lower([upper(B), X]) = -lower(B), solved from the bottom row up
        // and from the left column right.
        let mut x = vec![Complex::<T>::zero(); n * n];
        for i in (1..n).rev() {
            for j in 0..i {
                if (diag0[i] - diag0[j]).abs() <= gap {
                    continue;
                }
                let mut rhs = -b[at(n, i, j)];
                for k in i + 1..n {
                    rhs -= b[at(n, i, k)] * x[at(n, k, j)];
                }
                for k in 0..j {
                    rhs += x[at(n, i, k)] * b[at(n, k, j)];
                }
                x[at(n, i, j)] = rhs / (b[at(n, i, i)] - b[at(n, j, j)]);
            }
        }
        // I + X with X skew Hermitian.
        for j in 0..n {
            for i in j + 1..n {
                x[at(n, j, i)] = -x[at(n, i, j)].conj();
            }
            x[at(n, j, j)] = Complex::one();
        }
        u = cmatmul(&u, &x, n);
        orthonormalize(&mut u, n);
    }
    let mut t = cmatmul(&adjoint(&u, n), &cmatmul(a, &u, n), n);
    for j in 0..n {
        for i in j + 1..n {
            t[at(n, i, j)] = Complex::zero();
        }
    }
    Ok((u, t, diag0))
}

fn real_to_complex<T: Real>(function: &'static str, a: &Matrix<T>) -> Result<Vec<Complex<T>>> {
    require_square(function, a)?;
    Ok(a.as_slice()
        .iter()
        .map(|&x| Complex::from_real(x))
        .collect())
}

#[allow(clippy::type_complexity)]
fn schur_matrices<T: Real>(
    function: &'static str,
    a: &Matrix<T>,
) -> Result<(Matrix<Complex<T>>, Matrix<Complex<T>>)> {
    let ac = real_to_complex(function, a)?;
    let n = a.nrows();
    let (u, t, _) = schur(function, &ac, n)?;
    Ok((
        Matrix::from_column_major(n, n, u),
        Matrix::from_column_major(n, n, t),
    ))
}

/// Complex Schur decomposition `A = U T U^H` of a real square matrix: returns `(U, T)` with `U`
/// unitary and `T` upper triangular, eigenvalues on its diagonal.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input, [`Error::InvalidParameter`] for a
/// non-finite entry, and [`Error::NoConvergence`] when the QR iteration does not converge.
#[allow(clippy::type_complexity)]
pub fn complex_schur_decompose<T: Real>(
    a: &Matrix<T>,
) -> Result<(Matrix<Complex<T>>, Matrix<Complex<T>>)> {
    schur_matrices("complex_schur_decompose", a)
}

/// Unitary factor `U` of [`complex_schur_decompose`].
///
/// # Errors
///
/// As [`complex_schur_decompose`].
pub fn complex_schur_decompose_u<T: Real>(a: &Matrix<T>) -> Result<Matrix<Complex<T>>> {
    Ok(schur_matrices("complex_schur_decompose_u", a)?.0)
}

/// Upper triangular factor `T` of [`complex_schur_decompose`].
///
/// # Errors
///
/// As [`complex_schur_decompose`].
pub fn complex_schur_decompose_t<T: Real>(a: &Matrix<T>) -> Result<Matrix<Complex<T>>> {
    Ok(schur_matrices("complex_schur_decompose_t", a)?.1)
}

// Nonsymmetric eigendecomposition.

/// One output position: a real eigenvalue, or a conjugate pair that expands to two outputs. The
/// index is the Schur diagonal position (for a pair, of the member with positive imaginary part).
#[derive(Clone, Copy, Debug)]
enum Slot {
    Real(usize),
    Pair(usize),
}

/// Groups the Schur diagonal of a real matrix into real eigenvalues and conjugate pairs, in order
/// of first appearance.
fn eigen_slots(diag: &[Complex<f64>], scale: f64) -> Vec<Slot> {
    let n = diag.len();
    let tol = CLUSTER * scale;
    let mut used = vec![false; n];
    let mut slots = Vec::with_capacity(n);
    for i in 0..n {
        if used[i] {
            continue;
        }
        used[i] = true;
        if diag[i].im.abs() <= tol {
            slots.push(Slot::Real(i));
            continue;
        }
        let target = diag[i].conj();
        let partner = (0..n)
            .filter(|&j| !used[j])
            .min_by(|&j, &k| {
                (diag[j] - target)
                    .abs()
                    .total_cmp(&(diag[k] - target).abs())
            })
            .filter(|&j| (diag[j] - target).abs() <= 0.5 * diag[i].im.abs());
        match partner {
            Some(j) => {
                used[j] = true;
                slots.push(Slot::Pair(if diag[i].im > 0.0 { i } else { j }));
            }
            // Only reachable when rounding hides a partner; keep the value as computed.
            None => slots.push(Slot::Real(i)),
        }
    }
    slots
}

/// Unit eigenvector for the Schur diagonal entry `k`: back substitution in `T`, mapped by `U`,
/// rotated so its largest component is real and positive.
pub(crate) fn schur_vector<T: Real>(
    u: &[Complex<T>],
    t: &[Complex<T>],
    n: usize,
    k: usize,
    small: f64,
) -> Vec<Complex<T>> {
    let lambda = t[at(n, k, k)];
    let mut y = vec![Complex::<T>::zero(); n];
    y[k] = Complex::one();
    for i in (0..k).rev() {
        let mut s = Complex::zero();
        for j in i + 1..=k {
            s += t[at(n, i, j)] * y[j];
        }
        let mut d = t[at(n, i, i)] - lambda;
        if d.value().abs() <= small {
            d = Complex::from_f64(small, 0.0);
        }
        y[i] = -s / d;
    }
    let mut v = vec![Complex::<T>::zero(); n];
    for j in 0..=k {
        for i in 0..n {
            let p = u[at(n, i, j)] * y[j];
            v[i] += p;
        }
    }
    let m = (0..n)
        .max_by(|&a, &b| v[a].value().abs().total_cmp(&v[b].value().abs()))
        .unwrap_or(0);
    let vm = v[m];
    let phase = vm.conj() / vm.abs();
    let v: Vec<Complex<T>> = v.into_iter().map(|z| z * phase).collect();
    let norm = v.iter().map(|z| z.norm()).sum::<T>().sqrt();
    v.into_iter().map(|z| z / norm).collect()
}

/// Eigenvalues of each slot (and eigenvectors when asked) of a real square matrix.
#[allow(clippy::type_complexity)]
fn eigen_parts<T: Real>(
    function: &'static str,
    a: &Matrix<T>,
    with_vectors: bool,
) -> Result<(Vec<Slot>, Vec<Complex<T>>, Vec<Vec<Complex<T>>>)> {
    let ac = real_to_complex(function, a)?;
    let n = a.nrows();
    let (u, t, diag0) = schur(function, &ac, n)?;
    let scale = diag0
        .iter()
        .fold(0.0f64, |m, z| m.max(z.abs()))
        .max(f64::MIN_POSITIVE);
    let small = f64::EPSILON * scale;
    let slots = eigen_slots(&diag0, scale);
    let mut values = Vec::with_capacity(slots.len());
    let mut vectors = Vec::new();
    for slot in &slots {
        let (Slot::Real(k) | Slot::Pair(k)) = *slot;
        let mut lambda = t[at(n, k, k)];
        let real = matches!(slot, Slot::Real(_));
        if real {
            lambda.im = T::zero();
        }
        values.push(lambda);
        if with_vectors {
            let mut v = schur_vector(&u, &t, n, k, small);
            if real {
                for z in &mut v {
                    z.im = T::zero();
                }
            }
            vectors.push(v);
        }
    }
    Ok((slots, values, vectors))
}

/// Expands conjugate pairs into two outputs.
#[allow(clippy::type_complexity)]
fn expand<T: Real>(
    slots: &[Slot],
    values: Vec<Complex<T>>,
    vectors: Vec<Vec<Complex<T>>>,
) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
    let mut out_values = Vec::new();
    let mut out_vectors = Vec::new();
    let mut vectors = vectors.into_iter();
    for (slot, lambda) in slots.iter().zip(values) {
        let v = vectors.next();
        out_values.push(lambda);
        if let Some(v) = &v {
            out_vectors.extend_from_slice(v);
        }
        if let Slot::Pair(_) = slot {
            out_values.push(lambda.conj());
            if let Some(v) = &v {
                out_vectors.extend(v.iter().map(|z| z.conj()));
            }
        }
    }
    (out_values, out_vectors)
}

/// Eigenvectors (as columns) and eigenvalues of a real square matrix, both complex.
///
/// Real eigenvalues have exactly zero imaginary part and real eigenvectors; complex eigenvalues
/// come in adjacent conjugate pairs, positive imaginary part first. Eigenvectors have unit norm and
/// their largest component is real and positive. The order follows the Schur form and need not
/// match Stan's.
///
/// # Errors
///
/// As [`complex_schur_decompose`].
#[allow(clippy::type_complexity)]
pub fn eigendecompose<T: Real>(a: &Matrix<T>) -> Result<(Matrix<Complex<T>>, Vec<Complex<T>>)> {
    let n = a.nrows();
    let (slots, values, vectors) = eigen_parts("eigendecompose", a, true)?;
    let (values, vectors) = expand(&slots, values, vectors);
    Ok((Matrix::from_column_major(n, n, vectors), values))
}

/// Complex eigenvalues of a real square matrix; see [`eigendecompose`] for the order.
///
/// # Errors
///
/// As [`complex_schur_decompose`].
pub fn eigenvalues<T: Real>(a: &Matrix<T>) -> Result<Vec<Complex<T>>> {
    let (slots, values, vectors) = eigen_parts("eigenvalues", a, false)?;
    Ok(expand(&slots, values, vectors).0)
}

/// Complex eigenvectors (as columns) of a real square matrix; see [`eigendecompose`].
///
/// # Errors
///
/// As [`complex_schur_decompose`].
pub fn eigenvectors<T: Real>(a: &Matrix<T>) -> Result<Matrix<Complex<T>>> {
    Ok(eigendecompose(a)?.0)
}

/// Real pseudo eigendecomposition `A V = V D`: `D` is block diagonal with a `1 x 1` block for each
/// real eigenvalue and a `2 x 2` block `[[a, b], [-b, a]]` for each pair `a +/- i b`; the matching
/// columns of `V` are the real eigenvector, or the real and imaginary parts of the eigenvector of
/// `a + i b`.
fn pseudo<T: Real>(function: &'static str, a: &Matrix<T>) -> Result<(Matrix<T>, Matrix<T>)> {
    let n = a.nrows();
    let (slots, values, vectors) = eigen_parts(function, a, true)?;
    let mut v = Vec::with_capacity(n * n);
    let mut d = Matrix::from_fn(n, n, |_, _| T::zero());
    let mut col = 0;
    for ((slot, lambda), vector) in slots.iter().zip(values).zip(vectors) {
        v.extend(vector.iter().map(|z| z.re));
        d[(col, col)] = lambda.re;
        if let Slot::Pair(_) = slot {
            v.extend(vector.iter().map(|z| z.im));
            d[(col, col + 1)] = lambda.im;
            d[(col + 1, col)] = -lambda.im;
            d[(col + 1, col + 1)] = lambda.re;
            col += 2;
        } else {
            col += 1;
        }
    }
    Ok((Matrix::from_column_major(n, n, v), d))
}

/// Real pseudo eigenvectors and block diagonal pseudo eigenvalues, `(V, D)` with `A V = V D`.
///
/// # Errors
///
/// As [`complex_schur_decompose`].
pub fn pseudo_eigendecompose<T: Real>(a: &Matrix<T>) -> Result<(Matrix<T>, Matrix<T>)> {
    pseudo("pseudo_eigendecompose", a)
}

/// Block diagonal pseudo eigenvalue matrix `D` of [`pseudo_eigendecompose`].
///
/// # Errors
///
/// As [`complex_schur_decompose`].
pub fn pseudo_eigenvalues<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(pseudo("pseudo_eigenvalues", a)?.1)
}

/// Real pseudo eigenvector matrix `V` of [`pseudo_eigendecompose`].
///
/// # Errors
///
/// As [`complex_schur_decompose`].
pub fn pseudo_eigenvectors<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(pseudo("pseudo_eigenvectors", a)?.0)
}
