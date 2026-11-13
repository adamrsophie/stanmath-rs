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
pub(crate) fn device_solve(
    l: &[f64],
    b: &[f64],
    n: usize,
    m: usize,
    transpose: bool,
) -> Result<Vec<f64>> {
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
    /// Solve with `Lᵀ` instead of `L`.
    transpose: bool,
}

impl MultiFn for Solve {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        let (l, b) = x.split_at(self.n * self.n);
        if S::ORDER > 0 {
            let l = Matrix::from_column_major(self.n, self.n, l.to_vec());
            let b = Matrix::from_column_major(self.n, self.m, b.to_vec());
            let x = if self.transpose {
                stanmath_rs_linalg::mdivide_left_tri_low_transpose(&l, &b)?
            } else {
                stanmath_rs_linalg::mdivide_left_tri_low(&l, &b)?
            };
            return Ok(x.into_vec());
        }
        let l: Vec<f64> = l.iter().map(Real::value).collect();
        let b: Vec<f64> = b.iter().map(Real::value).collect();
        let x = device_solve(&l, &b, self.n, self.m, self.transpose)?;
        Ok(x.into_iter().map(S::from_f64).collect())
    }

    fn reverse(&self, x: &[f64], y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        let (n, m) = (self.n, self.m);
        let l = &x[..n * n];
        // For L⁻¹: b_adj = L⁻ᵀ y_adj and L_adj = -b_adj yᵀ. For L⁻ᵀ: b_adj = L⁻¹ y_adj and
        // L_adj = -y b_adjᵀ. Both keep the lower triangle; a failed kernel leaves NaN.
        let grads = device_solve(l, y_adj, n, m, !self.transpose).and_then(|b_adj| {
            let l_adj = if self.transpose {
                host_product(y, &b_adj, n, m, n, false, true)?
            } else {
                host_product(&b_adj, y, n, m, n, false, true)?
            };
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
    let out = T::map_multi(
        &x,
        &Solve {
            n,
            m,
            transpose: false,
        },
    )?;
    Ok(Matrix::from_column_major(n, m, out))
}

/// `B L⁻¹` for lower triangular `L` (only its lower triangle is read), solved on the device as
/// `(L⁻ᵀ Bᵀ)ᵀ`, in any scalar type, with the adjoint on the device in reverse mode.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `L` is square with as many columns as `B`;
/// [`Error::Backend`] when a kernel fails.
pub fn mdivide_right_tri_low_on_device<T: Real>(b: &Matrix<T>, l: &Matrix<T>) -> Result<Matrix<T>> {
    const F: &str = "mdivide_right_tri_low";
    let n = l.nrows();
    if l.ncols() != n || b.ncols() != n {
        return Err(Error::dimension(
            F,
            format!(
                "{}x{} triangular factor with {} right-hand side columns",
                n,
                l.ncols(),
                b.ncols()
            ),
        ));
    }
    let m = b.nrows();
    if n == 0 || m == 0 {
        return Ok(Matrix::from_column_major(m, n, Vec::new()));
    }
    let bt = stanmath_rs_linalg::transpose(b);
    let x: Vec<T> = l.as_slice().iter().chain(bt.as_slice()).copied().collect();
    let out = T::map_multi(
        &x,
        &Solve {
            n,
            m,
            transpose: true,
        },
    )?;
    Ok(stanmath_rs_linalg::transpose(&Matrix::from_column_major(
        n, m, out,
    )))
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
            // B L⁻¹ with an m by n B, from the same values.
            let right = |on_device: bool| {
                value_and_grad(
                    |v: &[Var<'_>]| {
                        let lm = Matrix::from_column_major(n, n, v[..n * n].to_vec());
                        let bm = Matrix::from_column_major(m, n, v[n * n..].to_vec());
                        let s = if on_device {
                            mdivide_right_tri_low_on_device(&bm, &lm).unwrap()
                        } else {
                            stanmath_rs_linalg::mdivide_right_tri_low(&bm, &lm).unwrap()
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
            let (rd, rh) = (right(true), right(false));
            assert!((rd.value - rh.value).abs() <= 1e-12 * (1.0 + rh.value.abs()));
            for (g, w) in rd.gradient.iter().zip(&rh.gradient) {
                assert!(
                    (g - w).abs() <= 1e-10 * (1.0 + w.abs()),
                    "right ({n}, {m}): {g} vs {w}"
                );
            }
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
