//! The squared exponential covariance on the device: Stan's OpenCL `gp_exp_quad_cov`.

use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::device::{Arg, DeviceBuffer, launch};
use crate::special::PRELUDE;

const FUNCTION: &str = "gp_exp_quad_cov";

const KERNEL: &str =
    "__kernel void cov(__global const double* x, __global double* k, const ulong n,
                  const ulong d, const double sigma, const double l) {
  const size_t i = get_global_id(0);
  const size_t j = get_global_id(1);
  double s = 0.0;
  for (size_t c = 0; c < d; c++) {
    const double t = x[i * d + c] - x[j * d + c];
    s += t * t;
  }
  k[i + j * n] = sigma * sigma * exp(-0.5 * s / (l * l));
}
";

/// Inputs: the `n` points of dimension `d` row by row, then sigma and the length scale.
#[derive(Clone)]
struct Cov {
    n: usize,
    d: usize,
}

impl Cov {
    fn device(&self, x: &[f64]) -> Result<Vec<f64>> {
        let (n, d) = (self.n, self.d);
        let points = DeviceBuffer::from_slice(&x[..n * d])?;
        let k = DeviceBuffer::new(n * n)?;
        launch(
            FUNCTION,
            FUNCTION,
            "cov",
            || format!("{PRELUDE}{KERNEL}"),
            &[
                Arg::Buffer(&points),
                Arg::Buffer(&k),
                Arg::Size(n),
                Arg::Size(d),
                Arg::Real(x[n * d]),
                Arg::Real(x[n * d + 1]),
            ],
            &[n, n],
        )?;
        k.to_vec()
    }
}

impl MultiFn for Cov {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        let (n, d) = (self.n, self.d);
        if S::ORDER > 0 {
            let points: Vec<Vec<S>> = x[..n * d].chunks(d.max(1)).map(<[S]>::to_vec).collect();
            let points = if d == 0 { vec![Vec::new(); n] } else { points };
            return Ok(
                stanmath_rs_linalg::gp_exp_quad_cov(&points, x[n * d], x[n * d + 1])?.into_vec(),
            );
        }
        let v: Vec<f64> = x.iter().map(Real::value).collect();
        Ok(self.device(&v)?.into_iter().map(S::from_f64).collect())
    }

    fn reverse(&self, x: &[f64], y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        // ponytail: the adjoint is one host pass over the n² entries, as cheap as the forward.
        let (n, d) = (self.n, self.d);
        let (sigma, l) = (x[n * d], x[n * d + 1]);
        let (mut ds, mut dl) = (0.0, 0.0);
        for j in 0..n {
            for i in 0..n {
                let w = y_adj[i + j * n] * y[i + j * n];
                if w == 0.0 {
                    continue;
                }
                let mut s = 0.0;
                for c in 0..d {
                    let t = x[i * d + c] - x[j * d + c];
                    s += t * t;
                    // d/dx_i of exp(-s / 2l²) is -(x_i - x_j) / l² times the entry.
                    let g = w * t / (l * l);
                    if needed[i * d + c] {
                        x_adj[i * d + c] -= g;
                    }
                    if needed[j * d + c] {
                        x_adj[j * d + c] += g;
                    }
                }
                ds += 2.0 * w / sigma;
                dl += w * s / (l * l * l);
            }
        }
        if needed[n * d] {
            x_adj[n * d] += ds;
        }
        if needed[n * d + 1] {
            x_adj[n * d + 1] += dl;
        }
    }
}

/// Stan's `gp_exp_quad_cov(x, sigma, length_scale)`, `sigma² exp(-|x_i - x_j|² / (2 l²))`, with
/// the matrix computed on the device, in any scalar type. In reverse mode the adjoint runs on
/// the host; forward-mode scalars are computed on the host.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when the points differ in dimension; [`Error::InvalidParameter`]
/// unless `sigma` and the length scale are positive and finite and the points are not NaN;
/// [`Error::Backend`] when a kernel fails.
pub fn gp_exp_quad_cov_on_device<T: Real>(
    x: &[Vec<T>],
    sigma: T,
    length_scale: T,
) -> Result<Matrix<T>> {
    let n = x.len();
    let d = x.first().map_or(0, Vec::len);
    if x.iter().any(|p| p.len() != d) {
        return Err(Error::dimension(FUNCTION, "points differ in dimension"));
    }
    for (name, v) in [
        ("sigma", sigma.value()),
        ("length_scale", length_scale.value()),
    ] {
        if !(v > 0.0 && v.is_finite()) {
            return Err(Error::parameter(FUNCTION, name, v, "positive finite"));
        }
    }
    if let Some(v) = x.iter().flatten().map(Real::value).find(|v| v.is_nan()) {
        return Err(Error::parameter(FUNCTION, "x", v, "not NaN"));
    }
    if n == 0 {
        return Ok(Matrix::from_column_major(0, 0, Vec::new()));
    }
    let inputs: Vec<T> = x
        .iter()
        .flatten()
        .copied()
        .chain([sigma, length_scale])
        .collect();
    let k = T::map_multi(&inputs, &Cov { n, d })?;
    Ok(Matrix::from_column_major(n, n, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::{Var, value_and_grad};

    #[test]
    fn covariance_matches_the_host() {
        for (n, d) in [(1, 1), (4, 1), (6, 3)] {
            let mut p: Vec<f64> = (0..n * d)
                .map(|i| ((i * 5 % 7) as f64 - 3.0) / 4.0)
                .collect();
            p.extend([1.3, 0.8]);
            let run = |on_device: bool| {
                value_and_grad(
                    |v: &[Var<'_>]| {
                        let points: Vec<Vec<Var<'_>>> =
                            v[..n * d].chunks(d).map(<[Var<'_>]>::to_vec).collect();
                        let (s, l) = (v[n * d], v[n * d + 1]);
                        let k = if on_device {
                            gp_exp_quad_cov_on_device(&points, s, l).unwrap()
                        } else {
                            stanmath_rs_linalg::gp_exp_quad_cov(&points, s, l).unwrap()
                        };
                        k.as_slice()
                            .iter()
                            .enumerate()
                            .map(|(i, &e)| e * (0.5 + 0.1 * i as f64))
                            .sum()
                    },
                    &p,
                )
            };
            let (dev, host) = (run(true), run(false));
            assert!((dev.value - host.value).abs() <= 1e-12 * (1.0 + host.value.abs()));
            for (g, w) in dev.gradient.iter().zip(&host.gradient) {
                assert!(
                    (g - w).abs() <= 1e-10 * (1.0 + w.abs()),
                    "({n}, {d}): {g} vs {w}"
                );
            }
        }
        assert!(gp_exp_quad_cov_on_device(&[vec![0.0]], -1.0, 1.0).is_err());
    }
}
