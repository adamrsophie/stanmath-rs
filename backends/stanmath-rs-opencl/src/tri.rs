//! Triangular solves on the device: Stan's OpenCL `mdivide_left_tri_low`.
//!
//! One work item per right-hand side column runs the substitution, so the device pays off for
//! many columns.

use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::device::{Arg, DeviceBuffer, launch};
use crate::host_product;

const FUNCTION: &str = "mdivide_left_tri_low";

const SOLVE: &str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// Solves L x = b (t == 0) or L' x = b (t == 1) for column get_global_id(0) of b, in place.
__kernel void solve(__global const double* l, __global double* b, const ulong n, const ulong t) {
  __global double* x = b + get_global_id(0) * n;
  if (t == 0) {
    for (size_t i = 0; i < n; i++) {
      double s = x[i];
      for (size_t k = 0; k < i; k++) s -= l[i + k * n] * x[k];
      x[i] = s / l[i + i * n];
    }
  } else {
    for (size_t i = n; i-- > 0;) {
      double s = x[i];
      for (size_t k = i + 1; k < n; k++) s -= l[k + i * n] * x[k];
      x[i] = s / l[i + i * n];
    }
  }
}
";

/// `L⁻¹ B` (or `L⁻ᵀ B` when `transpose`) of column-major host slices, on the device.
fn device_solve(l: &[f64], b: &[f64], n: usize, m: usize, transpose: bool) -> Result<Vec<f64>> {
    let (lb, bb) = (DeviceBuffer::from_slice(l)?, DeviceBuffer::from_slice(b)?);
    launch(
        FUNCTION,
        "tri_solve",
        "solve",
        || SOLVE.into(),
        &[
            Arg::Buffer(&lb),
            Arg::Buffer(&bb),
            Arg::Size(n),
            Arg::Size(usize::from(transpose)),
        ],
        &[m],
    )?;
    bb.to_vec()
}

#[derive(Clone)]
struct Solve {
    n: usize,
    m: usize,
}

impl MultiFn for Solve {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        let (l, b) = x.split_at(self.n * self.n);
        if S::ORDER > 0 {
            let l = Matrix::from_column_major(self.n, self.n, l.to_vec());
            let b = Matrix::from_column_major(self.n, self.m, b.to_vec());
            return Ok(stanmath_rs_linalg::mdivide_left_tri_low(&l, &b)?.into_vec());
        }
        let l: Vec<f64> = l.iter().map(Real::value).collect();
        let b: Vec<f64> = b.iter().map(Real::value).collect();
        let x = device_solve(&l, &b, self.n, self.m, false)?;
        Ok(x.into_iter().map(S::from_f64).collect())
    }

    fn reverse(&self, x: &[f64], y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        let (n, m) = (self.n, self.m);
        let l = &x[..n * n];
        // b_adj = L⁻ᵀ y_adj and L_adj = -b_adj yᵀ, lower triangle; a failed kernel leaves NaN.
        let grads = device_solve(l, y_adj, n, m, true).and_then(|b_adj| {
            let l_adj = host_product(&b_adj, y, n, m, n, false, true)?;
            Ok((l_adj, b_adj))
        });
        let (l_adj, b_adj) =
            grads.unwrap_or_else(|_| (vec![f64::NAN; n * n], vec![f64::NAN; n * m]));
        for (k, g) in l_adj.iter().enumerate() {
            if k % n >= k / n && needed[k] {
                x_adj[k] -= g;
            }
        }
        for (k, g) in b_adj.iter().enumerate() {
            if needed[n * n + k] {
                x_adj[n * n + k] += g;
            }
        }
    }
}

/// `L⁻¹ B` for lower triangular `L` (only its lower triangle is read), solved on the device, in
/// any scalar type. In reverse mode the adjoint solve and product also run on the device;
/// forward-mode scalars are solved on the host.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `L` is square with as many rows as `B`;
/// [`Error::Backend`] when a kernel fails.
pub fn mdivide_left_tri_low_on_device<T: Real>(l: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    let n = l.nrows();
    if l.ncols() != n || b.nrows() != n {
        return Err(Error::dimension(
            FUNCTION,
            format!(
                "{}x{} triangular factor with {} right-hand side rows",
                n,
                l.ncols(),
                b.nrows()
            ),
        ));
    }
    let m = b.ncols();
    if n == 0 || m == 0 {
        return Ok(Matrix::from_column_major(n, m, Vec::new()));
    }
    let x: Vec<T> = l.as_slice().iter().chain(b.as_slice()).copied().collect();
    let out = T::map_multi(&x, &Solve { n, m })?;
    Ok(Matrix::from_column_major(n, m, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::{Var, value_and_grad};

    #[test]
    fn solve_matches_the_host() {
        for (n, m) in [(1, 1), (3, 2), (9, 5)] {
            // The upper triangle holds junk to check that it is not read.
            let l: Vec<f64> = (0..n * n)
                .map(|k| {
                    let (i, j) = (k % n, k / n);
                    if i == j {
                        2.0 + 0.1 * i as f64
                    } else {
                        0.3 + 0.05 * (i + 2 * j) as f64
                    }
                })
                .collect();
            let b: Vec<f64> = (0..n * m).map(|k| 1.0 - 0.15 * k as f64).collect();
            let x: Vec<f64> = l.iter().chain(&b).copied().collect();
            let run = |on_device: bool| {
                value_and_grad(
                    |v: &[Var<'_>]| {
                        let lm = Matrix::from_column_major(n, n, v[..n * n].to_vec());
                        let bm = Matrix::from_column_major(n, m, v[n * n..].to_vec());
                        let s = if on_device {
                            mdivide_left_tri_low_on_device(&lm, &bm).unwrap()
                        } else {
                            stanmath_rs_linalg::mdivide_left_tri_low(&lm, &bm).unwrap()
                        };
                        s.as_slice()
                            .iter()
                            .enumerate()
                            .map(|(k, &e)| e * (1.0 + k as f64))
                            .sum()
                    },
                    &x,
                )
            };
            let (dev, host) = (run(true), run(false));
            assert!((dev.value - host.value).abs() <= 1e-12 * (1.0 + host.value.abs()));
            for (g, w) in dev.gradient.iter().zip(&host.gradient) {
                assert!(
                    (g - w).abs() <= 1e-10 * (1.0 + w.abs()),
                    "({n}, {m}): {g} vs {w}"
                );
            }
        }
        let l = Matrix::from_column_major(2, 2, vec![1.0; 4]);
        assert!(
            mdivide_left_tri_low_on_device(&l, &Matrix::from_column_major(3, 1, vec![1.0; 3]))
                .is_err()
        );
    }
}
