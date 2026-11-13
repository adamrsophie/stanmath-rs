//! Stan Math library functions that the Stan language does not expose.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Matrix, Result};

use crate::{gp_exp_quad_cov, matrix_exp};

/// Deprecated name of [`gp_exp_quad_cov`].
///
/// # Errors
///
/// As [`gp_exp_quad_cov`].
pub fn cov_exp_quad<T: Real>(x: &[Vec<T>], sigma: T, length_scale: T) -> Result<Matrix<T>> {
    gp_exp_quad_cov(x, sigma, length_scale)
}

/// The matrix exponential of a 2 by 2 matrix.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] unless `a` is 2 by 2; otherwise as [`matrix_exp`].
pub fn matrix_exp_2x2<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    if (a.nrows(), a.ncols()) != (2, 2) {
        return Err(Error::dimension(
            "matrix_exp_2x2",
            format!("expected 2x2, got {}x{}", a.nrows(), a.ncols()),
        ));
    }
    matrix_exp(a)
}

/// The matrix exponential by Padé approximation with scaling and squaring, which is how
/// [`matrix_exp`] computes it.
///
/// # Errors
///
/// As [`matrix_exp`].
pub fn matrix_exp_pade<T: Real>(a: &Matrix<T>) -> Result<Matrix<T>> {
    matrix_exp(a)
}

/// Each vector of `x` divided elementwise by `v`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when a vector of `x` differs in length from `v`.
pub fn divide_columns<T: Real>(x: &[Vec<T>], v: &[T]) -> Result<Vec<Vec<T>>> {
    x.iter()
        .map(|c| {
            if c.len() == v.len() {
                Ok(c.iter().zip(v).map(|(&a, &b)| a / b).collect())
            } else {
                Err(Error::dimension(
                    "divide_columns",
                    format!(
                        "vector of size {} with divisor of size {}",
                        c.len(),
                        v.len()
                    ),
                ))
            }
        })
        .collect()
}

/// The integers `low, low + 1, ..., high`; empty when `high < low`.
pub fn unitspaced_array(low: i64, high: i64) -> Vec<i64> {
    (low..=high).collect()
}

/// The number of nonzeros in row `i` (zero based) of a compressed sparse row matrix with row
/// starts `u`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `i + 1` is not an index of `u` or the row starts decrease.
pub fn csr_u_to_z(u: &[usize], i: usize) -> Result<usize> {
    match (u.get(i), u.get(i + 1)) {
        (Some(&a), Some(&b)) if b >= a => Ok(b - a),
        _ => Err(Error::dimension(
            "csr_u_to_z",
            format!("row {i} with {} row starts", u.len()),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_helpers() {
        let a = Matrix::from_column_major(2, 2, vec![0.1, 0.2, -0.3, 0.4]);
        assert_eq!(
            matrix_exp_2x2(&a).unwrap().as_slice(),
            matrix_exp(&a).unwrap().as_slice()
        );
        assert!(matrix_exp_2x2(&Matrix::from_column_major(1, 1, vec![1.0])).is_err());
        let d = divide_columns(&[vec![2.0, 6.0]], &[2.0, 3.0]).unwrap();
        assert_eq!(d, [[1.0, 2.0]]);
        assert!(divide_columns(&[vec![1.0]], &[1.0, 2.0]).is_err());
        assert_eq!(unitspaced_array(2, 4), [2, 3, 4]);
        assert!(unitspaced_array(3, 2).is_empty());
        assert_eq!(csr_u_to_z(&[0, 2, 5], 1).unwrap(), 3);
        assert!(csr_u_to_z(&[0, 2], 1).is_err());
        let x = [vec![0.0], vec![1.0]];
        assert_eq!(
            cov_exp_quad(&x, 1.0, 1.0).unwrap().as_slice(),
            gp_exp_quad_cov(&x, 1.0, 1.0).unwrap().as_slice()
        );
    }
}
