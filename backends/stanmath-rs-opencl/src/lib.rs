//! OpenCL backend of stanmath-rs, after Stan Math's OpenCL support.
//!
//! The device is chosen as in Stan: platform `OPENCL_PLATFORM_ID` and device `OPENCL_DEVICE_ID`,
//! read from the environment and zero when unset. The device must support double precision
//! (`cl_khr_fp64`). A [`MatrixCl`] is a matrix in device memory, stored column major like
//! [`Matrix`]; move matrices with [`to_matrix_cl`] and [`from_matrix_cl`].

mod density;
mod device;
mod elementwise;

pub use density::{Density, lpdf_on_device};

pub use elementwise::{Binary, Unary, binary_on_device, unary_on_device};

use device::{Arg, DeviceBuffer, launch};
use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::{Error, Matrix, Result};

/// A matrix of doubles in device memory, stored column major. Stan's `matrix_cl<double>`.
pub struct MatrixCl {
    rows: usize,
    cols: usize,
    buffer: DeviceBuffer,
}

impl MatrixCl {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// The name of the OpenCL device in use.
///
/// # Errors
///
/// [`Error::Backend`] when no device can be opened or it lacks double precision.
pub fn device_name() -> Result<String> {
    device::device_name()
}

/// Copies `m` to the device. Stan's `to_matrix_cl`.
///
/// # Errors
///
/// [`Error::Backend`] when the device cannot be opened or the copy fails.
pub fn to_matrix_cl(m: &Matrix<f64>) -> Result<MatrixCl> {
    Ok(MatrixCl {
        rows: m.nrows(),
        cols: m.ncols(),
        buffer: DeviceBuffer::from_slice(m.as_slice())?,
    })
}

/// Copies `m` back to the host. Stan's `from_matrix_cl`.
///
/// # Errors
///
/// [`Error::Backend`] when the copy fails.
pub fn from_matrix_cl(m: &MatrixCl) -> Result<Matrix<f64>> {
    Ok(Matrix::from_column_major(
        m.rows,
        m.cols,
        m.buffer.to_vec()?,
    ))
}

const MULTIPLY: &str = r"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// c = op(a) op(b), with op(a) n by k and op(b) k by m, both column major; ta and tb transpose.
__kernel void multiply(__global const double* a, __global const double* b, __global double* c,
                       const ulong n, const ulong k, const ulong m, const ulong ta,
                       const ulong tb) {
  const ulong i = get_global_id(0);
  const ulong j = get_global_id(1);
  double s = 0.0;
  for (ulong l = 0; l < k; l++) {
    const double x = ta ? a[l + i * k] : a[i + l * n];
    const double y = tb ? b[j + l * m] : b[l + j * k];
    s += x * y;
  }
  c[i + j * n] = s;
}
";

/// `op(a) op(b)` on the device for buffers holding `op(a)` (`n` by `k`) and `op(b)` (`k` by `m`),
/// transposed in storage when `ta` or `tb` is set.
#[allow(clippy::too_many_arguments)]
fn product(
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    n: usize,
    k: usize,
    m: usize,
    ta: bool,
    tb: bool,
) -> Result<DeviceBuffer> {
    if k == 0 {
        return DeviceBuffer::from_slice(&vec![0.0; n * m]);
    }
    let c = DeviceBuffer::new(n * m)?;
    launch(
        "multiply",
        "multiply",
        "multiply",
        || MULTIPLY.to_string(),
        &[
            Arg::Buffer(a),
            Arg::Buffer(b),
            Arg::Buffer(&c),
            Arg::Size(n),
            Arg::Size(k),
            Arg::Size(m),
            Arg::Size(usize::from(ta)),
            Arg::Size(usize::from(tb)),
        ],
        &[n, m],
    )?;
    Ok(c)
}

fn check_product(a: (usize, usize), b: (usize, usize)) -> Result<()> {
    if a.1 == b.0 {
        return Ok(());
    }
    Err(Error::dimension(
        "multiply",
        format!("a is {}x{} but b is {}x{}", a.0, a.1, b.0, b.1),
    ))
}

/// The matrix product `a b` on the device. Stan's `multiply` for `matrix_cl`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `a` has as many columns as `b` has rows, and
/// [`Error::Backend`] when the kernel fails.
pub fn multiply(a: &MatrixCl, b: &MatrixCl) -> Result<MatrixCl> {
    check_product((a.rows, a.cols), (b.rows, b.cols))?;
    Ok(MatrixCl {
        rows: a.rows,
        cols: b.cols,
        buffer: product(&a.buffer, &b.buffer, a.rows, a.cols, b.cols, false, false)?,
    })
}

/// `op(a) op(b)` of host slices, computed on the device.
#[allow(clippy::too_many_arguments)]
fn host_product(
    a: &[f64],
    b: &[f64],
    n: usize,
    k: usize,
    m: usize,
    ta: bool,
    tb: bool,
) -> Result<Vec<f64>> {
    let (a, b) = (DeviceBuffer::from_slice(a)?, DeviceBuffer::from_slice(b)?);
    product(&a, &b, n, k, m, ta, tb)?.to_vec()
}

/// `a b` for an `n` by `k` matrix `a` and a `k` by `m` matrix `b`, flattened column major.
#[derive(Clone)]
struct Product {
    n: usize,
    k: usize,
    m: usize,
}

impl MultiFn for Product {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        let (a, b) = x.split_at(self.n * self.k);
        if S::ORDER > 0 {
            // Nested derivatives stay on the host, in the caller's scalar type.
            return Ok((0..self.n * self.m)
                .map(|c| {
                    let (i, j) = (c % self.n, c / self.n);
                    (0..self.k)
                        .map(|l| a[i + l * self.n] * b[l + j * self.k])
                        .sum()
                })
                .collect());
        }
        let a: Vec<f64> = a.iter().map(Real::value).collect();
        let b: Vec<f64> = b.iter().map(Real::value).collect();
        let c = host_product(&a, &b, self.n, self.k, self.m, false, false)?;
        Ok(c.into_iter().map(S::from_f64).collect())
    }

    fn reverse(&self, x: &[f64], _y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        let (n, k, m) = (self.n, self.k, self.m);
        let (a, b) = x.split_at(n * k);
        // a_adj = c_adj bᵀ and b_adj = aᵀ c_adj; a failed kernel leaves NaN gradients.
        let grads = [
            (0, host_product(y_adj, b, n, m, k, false, true), n * k),
            (n * k, host_product(a, y_adj, k, n, m, true, false), k * m),
        ];
        for (start, g, len) in grads {
            let g = g.unwrap_or_else(|_| vec![f64::NAN; len]);
            for (o, (gi, &need)) in x_adj[start..start + len]
                .iter_mut()
                .zip(g.iter().zip(&needed[start..start + len]))
            {
                if need {
                    *o += gi;
                }
            }
        }
    }
}

/// The matrix product `a b` of host matrices, computed on the device, in any scalar type. In
/// reverse mode both adjoint products also run on the device; forward-mode scalars are multiplied
/// on the host.
///
/// # Errors
///
/// As [`multiply`].
pub fn multiply_on_device<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    check_product(a.shape(), b.shape())?;
    let f = Product {
        n: a.nrows(),
        k: a.ncols(),
        m: b.ncols(),
    };
    let x: Vec<T> = a.as_slice().iter().chain(b.as_slice()).copied().collect();
    let c = T::map_multi(&x, &f)?;
    Ok(Matrix::from_column_major(f.n, f.m, c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
        Matrix::from_fn(a.nrows(), b.ncols(), |i, j| {
            (0..a.ncols())
                .map(|l| a.as_slice()[i + l * a.nrows()] * b.column(j)[l])
                .sum()
        })
    }

    #[test]
    fn device_has_double_precision() {
        println!("OpenCL device: {}", device_name().unwrap());
    }

    #[test]
    fn round_trip_keeps_values() {
        let m = Matrix::from_fn(3, 2, |i, j| 0.1 * i as f64 - 1.7 * j as f64);
        let back = from_matrix_cl(&to_matrix_cl(&m).unwrap()).unwrap();
        assert_eq!(back, m);
        let empty = Matrix::<f64>::zeros(0, 4);
        assert_eq!(
            from_matrix_cl(&to_matrix_cl(&empty).unwrap()).unwrap(),
            empty
        );
    }

    #[test]
    fn multiply_matches_host() {
        let a = Matrix::from_fn(5, 3, |i, j| (i as f64 + 1.0).sin() * (j as f64 - 0.5));
        let b = Matrix::from_fn(3, 4, |i, j| 0.3 * i as f64 - (j as f64).cos());
        let c = from_matrix_cl(
            &multiply(&to_matrix_cl(&a).unwrap(), &to_matrix_cl(&b).unwrap()).unwrap(),
        )
        .unwrap();
        assert!(c.max_abs_diff(&reference(&a, &b)) < 1e-14);
        let z = Matrix::<f64>::zeros(2, 0);
        let w = Matrix::<f64>::zeros(0, 3);
        let zw = from_matrix_cl(
            &multiply(&to_matrix_cl(&z).unwrap(), &to_matrix_cl(&w).unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(zw, Matrix::zeros(2, 3));
        assert!(multiply(&to_matrix_cl(&a).unwrap(), &to_matrix_cl(&a).unwrap()).is_err());
    }

    #[test]
    fn multiply_on_device_gradients_match_host() {
        use stanmath_rs_autodiff::{Dual, grad};
        let x: Vec<f64> = (0..14).map(|i| (i as f64 * 0.7).sin()).collect();
        // Loss: the sum of squares of a b, for a 2x4 and b 4x1, plus a term outside the product.
        let loss = |v: &[f64]| -> f64 {
            let a = Matrix::from_column_major(2, 4, v[..8].to_vec());
            let b = Matrix::from_column_major(4, 1, v[8..12].to_vec());
            reference(&a, &b)
                .as_slice()
                .iter()
                .map(|c| c * c)
                .sum::<f64>()
                + v[12] * v[13]
        };
        let g = grad(
            |v| {
                let a = Matrix::from_column_major(2, 4, v[..8].to_vec());
                let b = Matrix::from_column_major(4, 1, v[8..12].to_vec());
                let c = multiply_on_device(&a, &b).unwrap();
                c.as_slice()
                    .iter()
                    .map(|&c| c * c)
                    .sum::<stanmath_rs_autodiff::Var<'_>>()
                    + v[12] * v[13]
            },
            &x,
        );
        for k in 0..x.len() {
            let h = 1e-6;
            let (mut up, mut down) = (x.clone(), x.clone());
            up[k] += h;
            down[k] -= h;
            let fd = (loss(&up) - loss(&down)) / (2.0 * h);
            assert!((g[k] - fd).abs() < 1e-7, "gradient {k}: {} vs {fd}", g[k]);
        }
        // Forward mode runs on the host and agrees with reverse mode.
        let duals: Vec<Dual<f64>> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| Dual::new(v, if i == 3 { 1.0 } else { 0.0 }))
            .collect();
        let a = Matrix::from_column_major(2, 4, duals[..8].to_vec());
        let b = Matrix::from_column_major(4, 1, duals[8..12].to_vec());
        let c = multiply_on_device(&a, &b).unwrap();
        let d: f64 = c.as_slice().iter().map(|&c| (c * c).eps).sum();
        assert!((d - g[3]).abs() < 1e-12, "{d} vs {}", g[3]);
    }
}
