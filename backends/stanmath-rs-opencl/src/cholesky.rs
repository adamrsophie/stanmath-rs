//! Cholesky decomposition on the device: Stan's OpenCL `cholesky_decompose`.
//!
//! The factor is computed right looking, one column per pair of launches: the first scales the
//! column below the diagonal, the second takes the diagonal's square root and updates the
//! trailing lower triangle.

// The negated comparisons also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::device::{Arg, DeviceBuffer, launch};
use crate::host_product;
use crate::tri::device_solve;

pub(crate) const FUNCTION: &str = "cholesky_decompose";

/// Stan's tolerance for symmetry.
const SYMMETRY_TOL: f64 = 1e-8;

const CHOLESKY: &str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void scale(__global double* a, const ulong n, const ulong j) {
  const size_t i = j + 1 + get_global_id(0);
  a[i + j * n] /= sqrt(a[j + j * n]);
}
__kernel void update(__global double* a, const ulong n, const ulong j) {
  const size_t r = j + get_global_id(0);
  const size_t c = j + get_global_id(1);
  if (r == j && c == j) {
    a[j + j * n] = sqrt(a[j + j * n]);
  } else if (c > j && r >= c) {
    a[r + c * n] -= a[r + j * n] * a[c + j * n];
  }
}
";

/// Overwrites the lower triangle of the `n` by `n` column-major `buffer` with its Cholesky
/// factor; the upper triangle is left as it was.
pub(crate) fn factor_in_place(buffer: &DeviceBuffer, n: usize) -> Result<()> {
    for j in 0..n {
        let args = [Arg::Buffer(buffer), Arg::Size(n), Arg::Size(j)];
        launch(
            FUNCTION,
            "cholesky",
            "scale",
            || CHOLESKY.into(),
            &args,
            &[n - j - 1],
        )?;
        launch(
            FUNCTION,
            "cholesky",
            "update",
            || CHOLESKY.into(),
            &args,
            &[n - j, n - j],
        )?;
    }
    Ok(())
}

/// The lower Cholesky factor of the `n` by `n` column-major `a`, on the device.
fn device_factor(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let buffer = DeviceBuffer::from_slice(a)?;
    factor_in_place(&buffer, n)?;
    let mut l = buffer.to_vec()?;
    for j in 0..n {
        if !(l[j + j * n] > 0.0) {
            return Err(Error::NotPositiveDefinite { function: FUNCTION });
        }
        l[j * n..j * n + j].fill(0.0);
    }
    Ok(l)
}

#[derive(Clone)]
struct Cholesky {
    n: usize,
}

impl MultiFn for Cholesky {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        if S::ORDER > 0 {
            let a = Matrix::from_column_major(self.n, self.n, x.to_vec());
            return Ok(stanmath_rs_linalg::cholesky_decompose(&a)?.into_vec());
        }
        let v: Vec<f64> = x.iter().map(Real::value).collect();
        Ok(device_factor(&v, self.n)?
            .into_iter()
            .map(S::from_f64)
            .collect())
    }

    fn reverse(&self, _x: &[f64], l: &[f64], l_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        let n = self.n;
        // A failed kernel leaves NaN gradients.
        let s = adjoint(l, l_adj, n).unwrap_or_else(|_| vec![f64::NAN; n * n]);
        // Only the lower triangle of A is read, so it takes the symmetric part of S.
        for j in 0..n {
            for i in j..n {
                let g = if i == j {
                    s[i + i * n]
                } else {
                    s[i + j * n] + s[j + i * n]
                };
                if needed[i + j * n] {
                    x_adj[i + j * n] += g;
                }
            }
        }
    }
}

/// `S = L⁻ᵀ Φ(Lᵀ L̄) L⁻¹`, where Φ keeps the lower triangle and halves the diagonal, with the
/// product and both triangular solves on the device.
fn adjoint(l: &[f64], l_adj: &[f64], n: usize) -> Result<Vec<f64>> {
    // L is constant above the diagonal, so only the lower triangle of L̄ counts.
    let lower: Vec<f64> = (0..n * n)
        .map(|k| if k % n >= k / n { l_adj[k] } else { 0.0 })
        .collect();
    let m = host_product(l, &lower, n, n, n, true, false)?;
    // Pᵀ, the transpose of Φ(M), as the right-hand side of the first solve.
    let mut pt = vec![0.0; n * n];
    for j in 0..n {
        for i in j..n {
            pt[j + i * n] = if i == j {
                0.5 * m[i + i * n]
            } else {
                m[i + j * n]
            };
        }
    }
    // Wᵀ = L⁻ᵀ Pᵀ gives W = P L⁻¹; then S = L⁻ᵀ W.
    let wt = device_solve(l, &pt, n, n, true)?;
    let mut w = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            w[i + j * n] = wt[j + i * n];
        }
    }
    device_solve(l, &w, n, n, true)
}

/// The lower Cholesky factor `L` of a symmetric positive definite matrix, `A = L Lᵀ`, with the
/// factorization on the device, in any scalar type. Forward-mode scalars are factored on the
/// host; in reverse mode the adjoint's product and triangular solves run on the device.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `a` is square, [`Error::NotSymmetric`], and
/// [`Error::NotPositiveDefinite`], as the host `cholesky_decompose`; [`Error::Backend`] when a
/// kernel fails.
pub fn cholesky_decompose_on_device<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(Error::dimension(
            FUNCTION,
            format!("expected a square matrix, got {}x{}", n, a.ncols()),
        ));
    }
    let v = a.as_slice();
    for j in 0..n {
        for i in j + 1..n {
            let (lo, hi) = (v[i + j * n].value(), v[j + i * n].value());
            if !((lo - hi).abs() <= SYMMETRY_TOL) {
                return Err(Error::NotSymmetric {
                    function: FUNCTION,
                    row: i,
                    col: j,
                });
            }
        }
    }
    if n == 0 {
        return Ok(Matrix::from_column_major(0, 0, Vec::new()));
    }
    let l = T::map_multi(v, &Cholesky { n })?;
    Ok(Matrix::from_column_major(n, n, l))
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::{Dual, Var, value_and_grad};

    /// A symmetric positive definite `n` by `n` matrix.
    fn spd(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i + j * n] =
                    1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { n as f64 } else { 0.0 };
            }
        }
        a
    }

    #[test]
    fn cholesky_matches_the_host() {
        for n in [1, 2, 5, 17] {
            let a = spd(n);
            let m = Matrix::from_column_major(n, n, a.clone());
            let got = cholesky_decompose_on_device(&m).unwrap();
            let want = stanmath_rs_linalg::cholesky_decompose(&m).unwrap();
            for (g, w) in got.as_slice().iter().zip(want.as_slice()) {
                assert!((g - w).abs() <= 1e-12, "n = {n}: {g} vs {w}");
            }
            let weights: Vec<f64> = (0..n * n).map(|k| 0.3 + 0.1 * k as f64).collect();
            let run = |on_device: bool| {
                value_and_grad(
                    |v: &[Var<'_>]| {
                        let m = Matrix::from_column_major(n, n, v.to_vec());
                        let l = if on_device {
                            cholesky_decompose_on_device(&m).unwrap()
                        } else {
                            stanmath_rs_linalg::cholesky_decompose(&m).unwrap()
                        };
                        l.as_slice()
                            .iter()
                            .zip(&weights)
                            .map(|(&x, &w)| x * w)
                            .sum()
                    },
                    &a,
                )
            };
            let (dev, host) = (run(true), run(false));
            for (g, w) in dev.gradient.iter().zip(&host.gradient) {
                assert!(
                    (g - w).abs() <= 1e-10 * (1.0 + w.abs()),
                    "n = {n}: {g} vs {w}"
                );
            }
            let d: Vec<Dual<f64>> = a.iter().map(|&x| Dual::constant(x)).collect();
            let dm = Matrix::from_column_major(n, n, d);
            assert!(cholesky_decompose_on_device(&dm).is_ok());
        }
    }

    #[test]
    fn cholesky_rejects_bad_matrices() {
        let not_square = Matrix::from_column_major(2, 3, vec![1.0; 6]);
        assert!(cholesky_decompose_on_device(&not_square).is_err());
        let asym = Matrix::from_column_major(2, 2, vec![2.0, 1.0, 0.0, 2.0]);
        assert!(matches!(
            cholesky_decompose_on_device(&asym),
            Err(Error::NotSymmetric { .. })
        ));
        let indefinite = Matrix::from_column_major(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        assert!(matches!(
            cholesky_decompose_on_device(&indefinite),
            Err(Error::NotPositiveDefinite { .. })
        ));
    }
}
