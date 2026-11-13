//! Generalized linear models on the device: Stan's OpenCL `normal_id_glm_lpdf` and the rest.
//!
//! The linear predictor `x beta + alpha` is the device matrix product, and the elementwise
//! density runs on the device as [`lpmf_on_device`]. The categorical and ordered logistic
//! models compute their predictor on the device and their density on the host.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_distributions as d;

use crate::{Density, lpmf_on_device, multiply_on_device};

/// `x beta + alpha` for an `n` by `k` matrix `x`, with `alpha` of length one or `n`.
fn predictor<T: Real>(
    function: &'static str,
    x: &Matrix<T>,
    alpha: &[T],
    beta: &Matrix<T>,
) -> Result<Matrix<T>> {
    let n = x.nrows();
    if alpha.len() != 1 && alpha.len() != beta.ncols() && !(beta.ncols() == 1 && alpha.len() == n) {
        return Err(Error::dimension(
            function,
            format!("alpha has {} elements", alpha.len()),
        ));
    }
    let eta = multiply_on_device(x, beta)?;
    let m = beta.ncols();
    let data = eta
        .as_slice()
        .iter()
        .enumerate()
        .map(|(idx, &e)| {
            // One column: alpha per row. Several (categorical): alpha per column.
            let a = match (alpha.len(), m) {
                (1, _) => alpha[0],
                (_, 1) => alpha[idx % n],
                _ => alpha[idx / n],
            };
            e + a
        })
        .collect();
    Ok(Matrix::from_column_major(n, m, data))
}

fn column<T: Real>(beta: &[T]) -> Matrix<T> {
    Matrix::from_column_major(beta.len(), 1, beta.to_vec())
}

/// Stan's `normal_id_glm_lpdf(y | x, alpha, beta, sigma)` on the device; `alpha` and `sigma`
/// have length one (broadcast) or the number of rows of `x`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for inconsistent sizes and the errors of [`lpmf_on_device`].
pub fn normal_id_glm_lpdf_on_device<T: Real>(
    y: &[T],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &[T],
    sigma: &[T],
) -> Result<T> {
    let eta = predictor("normal_id_glm_lpdf", x, alpha, &column(beta))?;
    lpmf_on_device(Density::Normal, &[], &[y, eta.as_slice(), sigma])
}

/// Stan's `bernoulli_logit_glm_lpmf(y | x, alpha, beta)` on the device.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`].
pub fn bernoulli_logit_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &[T],
) -> Result<T> {
    let eta = predictor("bernoulli_logit_glm_lpmf", x, alpha, &column(beta))?;
    lpmf_on_device(Density::BernoulliLogit, &[y], &[eta.as_slice()])
}

/// Stan's `poisson_log_glm_lpmf(y | x, alpha, beta)` on the device.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`].
pub fn poisson_log_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &[T],
) -> Result<T> {
    let eta = predictor("poisson_log_glm_lpmf", x, alpha, &column(beta))?;
    lpmf_on_device(Density::PoissonLog, &[y], &[eta.as_slice()])
}

/// Stan's `neg_binomial_2_log_glm_lpmf(y | x, alpha, beta, phi)` on the device.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`].
pub fn neg_binomial_2_log_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &[T],
    phi: &[T],
) -> Result<T> {
    let eta = predictor("neg_binomial_2_log_glm_lpmf", x, alpha, &column(beta))?;
    lpmf_on_device(Density::NegBinomial2Log, &[y], &[eta.as_slice(), phi])
}

/// Stan's `categorical_logit_glm_lpmf(y | x, alpha, beta)` with the `n` by `K` predictor
/// computed on the device; `alpha` has length `K` and `beta` is `k` by `K`.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`], and the host `categorical_logit_lpmf` errors.
pub fn categorical_logit_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &Matrix<T>,
) -> Result<T> {
    const F: &str = "categorical_logit_glm_lpmf";
    if alpha.len() != beta.ncols() {
        return Err(Error::dimension(
            F,
            format!(
                "alpha has {} elements, beta {} columns",
                alpha.len(),
                beta.ncols()
            ),
        ));
    }
    let eta = predictor(F, x, alpha, beta)?;
    let (n, k) = eta.shape();
    if y.len() != n {
        return Err(Error::dimension(
            F,
            format!("{} outcomes for {n} rows", y.len()),
        ));
    }
    let v = eta.as_slice();
    let mut total = T::zero();
    for (i, &yi) in y.iter().enumerate() {
        let row: Vec<T> = (0..k).map(|j| v[i + j * n]).collect();
        total += d::categorical_logit_lpmf(yi, &row)?;
    }
    Ok(total)
}

/// Stan's `ordered_logistic_glm_lpmf(y | x, beta, cuts)` with the predictor computed on the
/// device.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`], and the host `ordered_logistic_lpmf` errors.
pub fn ordered_logistic_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    beta: &[T],
    cuts: &[T],
) -> Result<T> {
    let eta = predictor("ordered_logistic_glm_lpmf", x, &[T::zero()], &column(beta))?;
    d::ordered_logistic_lpmf(y, eta.as_slice(), cuts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::{Var, value_and_grad};

    const N: usize = 5;
    const K: usize = 3;

    fn x_values() -> Vec<f64> {
        (0..N * K)
            .map(|i| ((i * 7 % 11) as f64 - 5.0) / 6.0)
            .collect()
    }

    /// Compares the device and host forms on `params` = x, then the model's own parameters.
    fn compare(name: &str, params: &[f64], f: impl for<'t> Fn(&[Var<'t>], bool) -> Var<'t>) {
        let dev = value_and_grad(|v| f(v, true), params);
        let host = value_and_grad(|v| f(v, false), params);
        assert!(
            (dev.value - host.value).abs() <= 1e-12 * (1.0 + host.value.abs()),
            "{name}: {} vs {}",
            dev.value,
            host.value
        );
        for (k, (a, b)) in dev.gradient.iter().zip(&host.gradient).enumerate() {
            assert!(
                (a - b).abs() <= 1e-10 * (1.0 + b.abs()),
                "{name} gradient {k}: {a} vs {b}"
            );
        }
    }

    fn split<'a, 't>(v: &'a [Var<'t>]) -> (Matrix<Var<'t>>, &'a [Var<'t>]) {
        (
            Matrix::from_column_major(N, K, v[..N * K].to_vec()),
            &v[N * K..],
        )
    }

    #[test]
    fn glms_match_the_host() {
        let ys = [0.3, -1.2, 0.8, 2.0, -0.4];
        let counts = [0, 2, 1, 5, 3];
        let binary = [1, 0, 0, 1, 1];
        let mut p = x_values();
        p.extend([0.2, 0.5, -0.3, 0.4, 1.3]); // alpha, beta (3), sigma or phi
        compare("normal_id_glm", &p, |v, dev| {
            let (x, r) = split(v);
            let y: Vec<Var<'_>> = ys.iter().map(|&y| v[0] * 0.0 + y).collect();
            if dev {
                normal_id_glm_lpdf_on_device(&y, &x, &r[..1], &r[1..4], &r[4..]).unwrap()
            } else {
                d::normal_id_glm_lpdf(&y, &x, r[0], &r[1..4], r[4]).unwrap()
            }
        });
        compare("bernoulli_logit_glm", &p[..p.len() - 1], |v, dev| {
            let (x, r) = split(v);
            if dev {
                bernoulli_logit_glm_lpmf_on_device(&binary, &x, &r[..1], &r[1..]).unwrap()
            } else {
                d::bernoulli_logit_glm_lpmf(&binary, &x, r[0], &r[1..]).unwrap()
            }
        });
        compare("poisson_log_glm", &p[..p.len() - 1], |v, dev| {
            let (x, r) = split(v);
            if dev {
                poisson_log_glm_lpmf_on_device(&counts, &x, &r[..1], &r[1..]).unwrap()
            } else {
                d::poisson_log_glm_lpmf(&counts, &x, r[0], &r[1..]).unwrap()
            }
        });
        compare("neg_binomial_2_log_glm", &p, |v, dev| {
            let (x, r) = split(v);
            if dev {
                neg_binomial_2_log_glm_lpmf_on_device(&counts, &x, &r[..1], &r[1..4], &r[4..])
                    .unwrap()
            } else {
                d::neg_binomial_2_log_glm_lpmf(&counts, &x, r[0], &r[1..4], r[4]).unwrap()
            }
        });
        let mut pc = x_values();
        pc.extend([0.1, -0.2, 0.3]); // alpha
        pc.extend((0..K * 3).map(|i| 0.1 * i as f64 - 0.4)); // beta, K by 3
        let classes = [1, 3, 2, 2, 1];
        compare("categorical_logit_glm", &pc, |v, dev| {
            let (x, r) = split(v);
            let beta = Matrix::from_column_major(K, 3, r[3..].to_vec());
            if dev {
                categorical_logit_glm_lpmf_on_device(&classes, &x, &r[..3], &beta).unwrap()
            } else {
                d::categorical_logit_glm_lpmf(&classes, &x, &r[..3], &beta).unwrap()
            }
        });
        let mut po = x_values();
        po.extend([0.4, -0.6, 0.9, -0.5, 0.7]); // beta (3), cuts (2)
        compare("ordered_logistic_glm", &po, |v, dev| {
            let (x, r) = split(v);
            if dev {
                ordered_logistic_glm_lpmf_on_device(&classes, &x, &r[..3], &r[3..]).unwrap()
            } else {
                d::ordered_logistic_glm_lpmf(&classes, &x, &r[..3], &r[3..]).unwrap()
            }
        });
    }
}
