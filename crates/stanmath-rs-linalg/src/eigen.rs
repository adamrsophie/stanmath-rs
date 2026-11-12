//! Symmetric eigendecomposition and the singular value decomposition built on it.
//!
//! The eigendecomposition `A = Q Λ Qᵀ` is computed in `f64` by the cyclic Jacobi method (Golub
//! and Van Loan, section 8.5). For a differentiable scalar type the `f64` eigenvectors are then
//! refined in that type by the iteration of Ogita and Aishima (2018, "Iterative refinement for
//! symmetric eigenvalue decomposition"), which converges quadratically. Starting from the
//! converged values, `k` refinement steps make derivatives exact to order `2^k - 1`, the same
//! argument as for Newton steps, so the number of steps covers [`Real::ORDER`].
//!
//! Reverse mode records one custom operation with the adjoint of Giles (2008):
//! `Ā = Q (Λ̄ + F ∘ (Qᵀ Q̄)) Qᵀ` with `F_ij = 1 / (λ_j - λ_i)` off the diagonal.
//!
//! Eigenvalues closer than `1000 ε max |λ|` are treated as repeated: their eigenvectors, and
//! derivatives involving them, are not unique, and the corresponding terms of `F` are set to zero.

#![allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices

use core::f64::consts::SQRT_2;

use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::ops::{at, check_symmetric};
use crate::require_square;

const MAX_SWEEPS: usize = 100;

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

/// Cyclic Jacobi eigenvalue algorithm in `f64`. Returns the eigenvalues in ascending order and
/// the matching orthonormal eigenvectors as the columns of an `n` by `n` column-major matrix.
fn jacobi(function: &'static str, a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut a: Vec<f64> = (0..n * n)
        .map(|k| {
            let (i, j) = (k % n, k / n);
            0.5 * (a[at(n, i, j)] + a[at(n, j, i)])
        })
        .collect();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[at(n, i, i)] = 1.0;
    }
    let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut converged = false;
    for _ in 0..MAX_SWEEPS {
        let mut off = 0.0;
        for j in 0..n {
            for i in 0..n {
                if i != j {
                    off += a[at(n, i, j)] * a[at(n, i, j)];
                }
            }
        }
        if off.sqrt() <= f64::EPSILON * norm {
            converged = true;
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[at(n, p, q)];
                if apq == 0.0 {
                    continue;
                }
                let tau = (a[at(n, q, q)] - a[at(n, p, p)]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + tau.hypot(1.0))
                } else {
                    -1.0 / (-tau + tau.hypot(1.0))
                };
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[at(n, k, p)], a[at(n, k, q)]);
                    a[at(n, k, p)] = c * akp - s * akq;
                    a[at(n, k, q)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[at(n, p, k)], a[at(n, q, k)]);
                    a[at(n, p, k)] = c * apk - s * aqk;
                    a[at(n, q, k)] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[at(n, k, p)], v[at(n, k, q)]);
                    v[at(n, k, p)] = c * vkp - s * vkq;
                    v[at(n, k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    if !converged {
        return Err(Error::NoConvergence {
            function,
            iterations: MAX_SWEEPS,
        });
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[at(n, i, i)].total_cmp(&a[at(n, j, j)]));
    let values = order.iter().map(|&i| a[at(n, i, i)]).collect();
    let mut vectors = Vec::with_capacity(n * n);
    for &i in &order {
        vectors.extend_from_slice(&v[i * n..(i + 1) * n]);
    }
    Ok((values, vectors))
}

/// `A X` for `n` by `n` column-major matrices.
fn square_product<S: Real>(a: &[S], x: &[S], n: usize) -> Vec<S> {
    let mut out = vec![S::zero(); n * n];
    for j in 0..n {
        for k in 0..n {
            let xkj = x[at(n, k, j)];
            for i in 0..n {
                out[at(n, i, j)] += a[at(n, i, k)] * xkj;
            }
        }
    }
    out
}

/// Eigenvalues (ascending) and eigenvectors of symmetric `a`, with derivatives exact to
/// `S::ORDER`.
fn eigen_sym<S: Real>(function: &'static str, a: &[S], n: usize) -> Result<(Vec<S>, Vec<S>)> {
    let values: Vec<f64> = a.iter().map(Real::value).collect();
    check_symmetric(function, &values, n)?;
    let (lam0, q0) = jacobi(function, &values, n)?;
    let scale = lam0.iter().fold(0.0f64, |m, l| m.max(l.abs()));
    let gap = CLUSTER * scale;
    let repeated = |i: usize, j: usize| (lam0[j] - lam0[i]).abs() <= gap;
    let mut x: Vec<S> = q0.iter().map(|&v| S::from_f64(v)).collect();
    for _ in 0..refinement_steps(S::ORDER) {
        let ax = square_product(a, &x, n);
        // R = I - XᵀX and S = XᵀAX.
        let mut r = vec![S::zero(); n * n];
        let mut s = vec![S::zero(); n * n];
        for j in 0..n {
            for i in 0..n {
                let (mut xx, mut xax) = (S::zero(), S::zero());
                for k in 0..n {
                    xx += x[at(n, k, i)] * x[at(n, k, j)];
                    xax += x[at(n, k, i)] * ax[at(n, k, j)];
                }
                r[at(n, i, j)] = if i == j { -xx + 1.0 } else { -xx };
                s[at(n, i, j)] = xax;
            }
        }
        let lam: Vec<S> = (0..n)
            .map(|i| s[at(n, i, i)] / (-r[at(n, i, i)] + 1.0))
            .collect();
        let mut e = vec![S::zero(); n * n];
        for j in 0..n {
            for i in 0..n {
                e[at(n, i, j)] = if i == j || repeated(i, j) {
                    r[at(n, i, j)] * 0.5
                } else {
                    (s[at(n, i, j)] + lam[j] * r[at(n, i, j)]) / (lam[j] - lam[i])
                };
            }
        }
        let xe = square_product(&x, &e, n);
        for (xi, d) in x.iter_mut().zip(xe) {
            *xi += d;
        }
    }
    // Rayleigh quotients of the refined eigenvectors.
    let ax = square_product(a, &x, n);
    let lam = (0..n)
        .map(|j| {
            let (mut num, mut den) = (S::zero(), S::zero());
            for k in 0..n {
                num += x[at(n, k, j)] * ax[at(n, k, j)];
                den += x[at(n, k, j)].square();
            }
            num / den
        })
        .collect();
    Ok((lam, x))
}

/// Symmetric eigendecomposition; input `A` (`n` by `n`), output `[λ (n), Q (n by n)]`.
#[derive(Clone, Copy, Debug)]
struct EigenSym {
    function: &'static str,
    n: usize,
}

impl MultiFn for EigenSym {
    fn eval<S: Real>(&self, a: &[S]) -> Result<Vec<S>> {
        let (mut lam, q) = eigen_sym(self.function, a, self.n)?;
        lam.extend(q);
        Ok(lam)
    }

    fn reverse(&self, _a: &[f64], y: &[f64], y_bar: &[f64], needed: &[bool], adj: &mut [f64]) {
        if !needed.iter().any(|&v| v) {
            return;
        }
        let n = self.n;
        let (lam, q) = y.split_at(n);
        let (lam_bar, q_bar) = y_bar.split_at(n);
        let scale = lam.iter().fold(0.0f64, |m, l| m.max(l.abs()));
        let gap = CLUSTER * scale;
        // inner = Λ̄ + F ∘ (Qᵀ Q̄)
        let mut inner = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                inner[at(n, i, j)] = if i == j {
                    lam_bar[i]
                } else if (lam[j] - lam[i]).abs() <= gap {
                    0.0
                } else {
                    let mut m = 0.0;
                    for k in 0..n {
                        m += q[at(n, k, i)] * q_bar[at(n, k, j)];
                    }
                    m / (lam[j] - lam[i])
                };
            }
        }
        // Ā = Q inner Qᵀ
        let qi = square_product(q, &inner, n);
        for j in 0..n {
            for i in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += qi[at(n, i, k)] * q[at(n, j, k)];
                }
                adj[at(n, i, j)] += s;
            }
        }
    }
}

fn decompose<T: Real>(function: &'static str, a: &Matrix<T>) -> Result<(Matrix<T>, Vec<T>)> {
    require_square(function, a)?;
    let n = a.nrows();
    let mut out = T::map_multi(a.as_slice(), &EigenSym { function, n })?;
    let q = out.split_off(n);
    Ok((Matrix::from_column_major(n, n, q), out))
}

/// Eigenvectors (as columns) and eigenvalues (ascending) of a symmetric matrix.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for a non-square input, [`Error::NotSymmetric`], and
/// [`Error::NoConvergence`] when the Jacobi iteration does not converge (for example with NaN
/// entries).
pub fn eigendecompose_sym<T: Real>(a: &Matrix<T>) -> Result<(Matrix<T>, Vec<T>)> {
    decompose("eigendecompose_sym", a)
}

/// Eigenvalues of a symmetric matrix, in ascending order.
///
/// # Errors
///
/// As [`eigendecompose_sym`].
pub fn eigenvalues_sym<T: Real>(a: &Matrix<T>) -> Result<Vec<T>> {
    Ok(decompose("eigenvalues_sym", a)?.1)
}

/// Eigenvectors of a symmetric matrix as columns, ordered by ascending eigenvalue.
///
/// # Errors
///
/// As [`eigendecompose_sym`].
pub fn eigenvectors_sym<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(decompose("eigenvectors_sym", a)?.0)
}

/// Thin singular value decomposition `A = U diag(σ) Vᵀ` with `σ` in descending order.
///
/// It uses the eigendecomposition of the symmetric matrix `[[0, A], [Aᵀ, 0]]`, whose eigenvalues
/// are `±σ_i` (and zeros), with eigenvectors `(u_i, ±v_i) / sqrt(2)`. This avoids squaring the
/// condition number as `AᵀA` would. Zero singular values cluster with the zero eigenvalues, so
/// their singular vectors are not reliable.
fn svd_parts<T: Real>(
    function: &'static str,
    a: &Matrix<T>,
) -> Result<(Vec<T>, Matrix<T>, Matrix<T>)> {
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
            a[(j, i - m)]
        } else {
            T::zero()
        }
    });
    let (vectors, values) = decompose(function, &aug)?;
    let col = |r: usize| size - 1 - r;
    let sigma = (0..k).map(|r| values[col(r)]).collect();
    let u = Matrix::from_fn(m, k, |i, r| vectors[(i, col(r))] * SQRT_2);
    let v = Matrix::from_fn(n, k, |i, r| vectors[(m + i, col(r))] * SQRT_2);
    Ok((sigma, u, v))
}

/// Singular values of `A` in descending order.
///
/// # Errors
///
/// [`Error::NoConvergence`] when the underlying eigenvalue iteration does not converge.
pub fn singular_values<T: Real>(a: &Matrix<T>) -> Result<Vec<T>> {
    Ok(svd_parts("singular_values", a)?.0)
}

/// Left singular vectors of `A` (`m` by `min(m, n)`), ordered by descending singular value.
///
/// # Errors
///
/// As [`singular_values`].
#[allow(non_snake_case)]
pub fn svd_U<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(svd_parts("svd_U", a)?.1)
}

/// Right singular vectors of `A` (`n` by `min(m, n)`), ordered by descending singular value.
///
/// # Errors
///
/// As [`singular_values`].
#[allow(non_snake_case)]
pub fn svd_V<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    Ok(svd_parts("svd_V", a)?.2)
}
