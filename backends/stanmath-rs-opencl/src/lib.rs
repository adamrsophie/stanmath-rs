//! OpenCL backend of stanmath-rs, after Stan Math's OpenCL support.
//!
//! The device is chosen as in Stan: platform `OPENCL_PLATFORM_ID` and device `OPENCL_DEVICE_ID`,
//! read from the environment and zero when unset. The device must support double precision
//! (`cl_khr_fp64`). A [`MatrixCl`] is a matrix in device memory, stored column major like
//! [`Matrix`]; move matrices with [`to_matrix_cl`] and [`from_matrix_cl`].

mod device;

use device::{Arg, DeviceBuffer, launch};
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
__kernel void multiply(__global const double* a, __global const double* b, __global double* c,
                       const ulong n, const ulong k) {
  const ulong i = get_global_id(0);
  const ulong j = get_global_id(1);
  double s = 0.0;
  for (ulong l = 0; l < k; l++) {
    s += a[i + l * n] * b[l + j * k];
  }
  c[i + j * n] = s;
}
";

/// The matrix product `a b` on the device. Stan's `multiply` for `matrix_cl`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `a` has as many columns as `b` has rows, and
/// [`Error::Backend`] when the kernel fails.
pub fn multiply(a: &MatrixCl, b: &MatrixCl) -> Result<MatrixCl> {
    const F: &str = "multiply";
    if a.cols != b.rows {
        return Err(Error::dimension(
            F,
            format!("a is {}x{} but b is {}x{}", a.rows, a.cols, b.rows, b.cols),
        ));
    }
    if a.cols == 0 {
        return to_matrix_cl(&Matrix::zeros(a.rows, b.cols));
    }
    let c = DeviceBuffer::new(a.rows * b.cols)?;
    launch(
        F,
        "multiply",
        MULTIPLY,
        &[
            Arg::Buffer(&a.buffer),
            Arg::Buffer(&b.buffer),
            Arg::Buffer(&c),
            Arg::Size(a.rows),
            Arg::Size(a.cols),
        ],
        &[a.rows, b.cols],
    )?;
    Ok(MatrixCl {
        rows: a.rows,
        cols: b.cols,
        buffer: c,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_product(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
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
        assert!(c.max_abs_diff(&host_product(&a, &b)) < 1e-14);
        let z = Matrix::<f64>::zeros(2, 0);
        let w = Matrix::<f64>::zeros(0, 3);
        let zw = from_matrix_cl(
            &multiply(&to_matrix_cl(&z).unwrap(), &to_matrix_cl(&w).unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(zw, Matrix::zeros(2, 3));
        assert!(multiply(&to_matrix_cl(&a).unwrap(), &to_matrix_cl(&a).unwrap()).is_err());
    }
}
